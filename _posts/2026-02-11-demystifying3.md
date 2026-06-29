---
layout: post
toc:
  sidebar: left
title: "Demystifying Multimodal Learning: Speculative Decoding in Multimodal Architectures"
date: 2026-07-25 14:14:00
description: A blogpost series on the nuts and bolts of Multimodal Learning
tags: Multimodal-Learning Inference-Optimization
# thumbnail: assets/img/TODO-speculative-decoding-thumbnail.png
thumbnail: https://cdn-uploads.huggingface.co/production/uploads/661d4e74b8f13412f6d48a50/SfOleyYtgr6UtQ4lT8jv7.png

community_article_url: https://huggingface.co/blog/MatteoNulli/de-mystifying-multimodal-learning-speculative-decoding
blogpost_url: https://matteonulli.github.io/blog/2026/demystifying3/
math: true
mermaid: true
_styles: >
  .mermaid svg { 
    max-width: 100%; 
    height: auto; 
  }
---


##### <b>Matteo Nulli</b>
<!-- ###### eBay
###### <img src="https://upload.wikimedia.org/wikipedia/commons/1/1b/EBay_logo.svg" alt="eBay" height="24"/> &nbsp;  -->
###### <a href="https://huggingface.co/blog/MatteoNulli/de-mystifying-multimodal-learning-speculative-decoding" title="Community Article"><i class="fa-brands fa-hugging-face" style="font-size: 1.75em;"></i></a> <a href="https://matteonulli.github.io/blog/2026/demystifying3/" title="Blogpost"><i class="fa-regular fa-newspaper" style="font-size: 1.75em;"></i></a>
<br>

## Introduction

Across the previous installments of `Demystifying Multimodal Learning`, we have built a clear mental model of the cost of vision. We defined  <abbr title="Click here for our previous blogpost.">[what a Visual Token (VT) is](https://huggingface.co/blog/MatteoNulli/de-mystifying-multimodal-learning-enabiling-vision)</abbr>, derived formulas to  <abbr title="Click here for our previous blogpost.">[calculate # Visual Tokens ( \\( V \\)) across architectures](https://huggingface.co/blog/MatteoNulli/de-mystifying-multimodal-learning-hidden-ineff)</abbr>, and dissected their  <abbr title="Click here for our previous blogpost.">[impact on inference latency, context windows and VRAM](https://huggingface.co/blog/MatteoNulli/de-mystifying-multimodal-learning-impact-vt-laten)</abbr>.

So far, every lever we pulled was about *spending fewer tokens*. But there is an orthogonal question, one that lives at the very end of the pipeline, in the autoregressive decoding loop:

<p align="center"><code>Can we make a VLMs generate faster, without retraining it and without changing a single output token?</code></p>

The text-only world answered this years ago with **Speculative Decoding (SD)** ([Leviathan et al., 2023](#specdec-2023), [Chen et al., 2023](#specsample-2023)): a *lossless* trick that routinely doubles LLM throughput. The natural follow-up is whether this free lunch survives the jump to images and video. As we will see, the answer is a qualified *yes*, the naive recipe already works, but the visual tokens we have spent three blogposts worrying about come back to haunt us, and squeezing out the full speedup requires rethinking SD with vision in mind.

In this installment we will first recap [how Speculative Decoding works](#speculative-decoding-in-a-nutshell), then ask [whether it applies natively to VLMs](#can-we-apply-speculative-decoding-natively-to-vlms) and what breaks, and finally walk through two representative approaches that fix it, [SpecVLM](#approach-1-specvlm-compress-the-vision-distill-the-draft) for images and [ParallelVLM](#approach-2-parallelvlm-aligning-and-parallelizing-for-video) for video, before zooming out to [the broader landscape](#the-broader-landscape) of multimodal SD.

## Speculative Decoding in a Nutshell

Autoregressive generation is slow for a structural reason: each token requires a full forward pass of a large model, and these passes are *memory-bound*. The GPU spends most of its time shuffling weights and KV Cache in and out of memory, not computing. Decoding one token uses roughly the same wall-clock time as decoding many in parallel.

Speculative Decoding exploits exactly this asymmetry with two models:

- A small, fast **draft** model \\( q \\) that autoregressively proposes a chunk of \\( k \\) candidate tokens.
- A large, accurate **target** model \\( p \\) that verifies all \\( k \\) candidates in a **single parallel forward pass**.

<a id="figure-1"></a>
<figure style="width: 80%; margin: auto; text-align: center;">
  <img src="https://developer-blogs.nvidia.com/wp-content/uploads/2025/09/speculative-decoding-draft-target-approach.gif"
       alt="Speculative decoding: autoregressive drafting then parallel verification"
       style="width: 100%;">
  <figcaption style="margin-top: 10px; font-style: italic; color: #555;">
    Figure 1: <b>The two phases of Speculative Decoding.</b> Step 1: the draft model cheaply autoregresses a chunk of tokens. Step 2: the target model verifies the whole chunk in one parallel pass, accepting the longest correct prefix and rejecting the rest. Video from <a href="https://developer.nvidia.com/blog/an-introduction-to-speculative-decoding-for-reducing-latency-in-ai-inference/">NVIDIA Developer blog</a>.
  </figcaption>
</figure>


The magic is in the verification step. Rather than blindly trusting the draft, SD applies **rejection sampling**: each drafted token \\( x \\) is accepted with probability

$$ \alpha(x) = \min\left(1, \frac{p(x)}{q(x)}\right) $$

and on the first rejection, the token is resampled from the residual distribution \\( (p - q)_+ \\) (renormalized). This rule is not a heuristic, it is provably equivalent to sampling directly from the target \\( p \\) ([Leviathan et al., 2023](#specdec-2023)). In other words, SD is **lossless**: the output distribution is *identical* to running the target model alone. We pay nothing in quality, we only gain speed, exactly as long as the cheap draft tends to agree with the expensive target.

A short worked example: suppose the draft proposes *"dogs love chasing after"*. The target verifies in parallel and, comparing \\( p(x) \\) against \\( q(x) \\) token by token, it accepts *"dogs"*, *"love"*, *"chasing"*, but rejects *"after"* (because \\( p(\text{after}) \ll q(\text{after}) \\)). We keep the 3 accepted tokens, resample the fourth from the target, and start the next round, all for the price of **one** target forward pass instead of four.

Two metrics govern how much we actually win, and we will see both throughout the rest of this post:

- **Wall-clock speedup ( \\( \tau \\) ):** end-to-end latency relative to the autoregressive target baseline. This is the number that pays the bills.
- **Mean accepted length ( \\( \sigma \\), sometimes \\( M \\) ):** the average number of tokens accepted by the target per speculative round. Higher \\( \sigma \\) means the draft is better aligned with the target, the lever that drives \\( \tau \\). A closely related quantity is the **token acceptance ratio ( \\( A \\) )**, the fraction of drafted tokens that survive verification.

The whole game of *good* speculative decoding is maximizing \\( \sigma \\) (a well-aligned, fast draft) while keeping the draft itself cheap. Modern LLM methods such as Medusa ([Cai et al., 2024](#medusa-2024)) and the EAGLE family ([Li et al., 2024](#eagle-2024), [2024b](#eagle2-2024), [2025](#eagle3-2025)) push \\( \sigma \\) up by drafting at the *feature* level and reusing the target's own LM head, rather than training a fully separate small model.

## Can We Apply Speculative Decoding Natively to VLMs?

The honest first question is whether any of this even matters for multimodal models, or whether you can just bolt a standard EAGLE-style draft onto a VLM and call it a day.

<p align="center"><code>Does Speculative Decoding provide speed advantages when applied naively to VLMs?</code></p>

The answer is a clear **yes**. Building a faithful EAGLE-2-style draft for a VLM, what the SpecVLM authors call *EagleVLM* ([Huang et al., 2025](#specvlm-2025)), already delivers **1.5–2.3× end-to-end speedups** over full autoregressive inference across the LLaVA family, with no loss in output quality. The reason is intuitive: as we established in our [latency blogpost](https://huggingface.co/blog/MatteoNulli/de-mystifying-multimodal-learning-impact-vt-laten), VLM decoding is just as memory-bound as LLM decoding, so the same asymmetry SD exploits is still there.

<a id="figure-2"></a>
<figure style="width: 80%; margin: auto; text-align: center;">
  <img src="/assets/img/adapted_specvlm.png"
       alt="Naive speculative decoding speedups on LLaVA"
       style="width: 100%;">
  <figcaption style="margin-top: 10px; font-style: italic; color: #555;">
    Figure 2: <b>Naive SD already helps.</b> An EAGLE-2-style draft (EagleVLM) yields 1.9–2.3× end-to-end speedups across LLaVA v1.5/v1.6 at 7B and 13B. Figure adapted from <a href="#specvlm-2025">Huang et al., 2025</a>.
  </figcaption>
</figure>

So the free lunch survives. The trouble starts when we ask *why it isn't even faster*, and here the visual tokens we have been tracking all series long take center stage.

#### The image problem

Two coupled issues hold naive SD back on VLMs ([Huang et al., 2025](#specvlm-2025)):

1. **Visual tokens inflate the KV Cache.** Multi-image and high-resolution inputs dump thousands of visual tokens into the cache during prefill (recall the [AnyRes / multi-grid blowup](https://huggingface.co/blog/MatteoNulli/de-mystifying-multimodal-learning-hidden-ineff#:~:text=Strategy%20B%3A%20The%20Multi%2DGrid%20/%20AnyRes)). The draft model inherits this burden: it must carry the same enormous visual context, which makes the "cheap" draft far less cheap.
2. **A fat KV Cache means slow attention.** A larger cache raises per-step latency, particularly in the attention layers and the memory traffic moving keys and values around. Every extra visual token the draft drags along directly erodes the speed advantage it is supposed to provide.

The takeaway is that, in the multimodal setting, the draft's *visual* workload, not its language modeling, becomes the bottleneck. This is the single observation that motivates **vision token compression** as the central design lever for multimodal SD.

#### The video problem

Video LLMs make everything worse, because the token counts are larger by another order of magnitude. Beyond the image issues above, three additional pain points emerge ([Kong et al., 2026](#parallelvlm-2026)):

1. **Sequential execution bottleneck.** In vanilla SD the draft and target run one after the other. As video tokens grow, both prefilling latency and decoding time grow with them, and because of this sequential scheduling the hardware sits idle for roughly **20% of the prefill** span and **50% of the decode** span. The accelerators are starved precisely when there is the most work to do.
2. **Entanglement of speed ratio and alignment.** Heavier video inputs shrink the draft's relative speed advantage, the obvious fix is to *prune* the draft's visual tokens, but a draft running on aggressively pruned windows can no longer retain salient visual detail or coherent textual grounding. Pruning buys speed and pays for it in acceptance length. SpecVLM partially mitigates this through online distillation (more below), but the tension remains.
3. **Positional bias in attention guidance.** A tempting way to choose *which* tokens to keep is to follow the target's attention. But the target's attention over video is strongly position-biased: in one analysis, **21% of the selected video tokens fall within just 4.0% of the position width** (the first frame and the last few, frames 1 and 125–128). Pruning by raw attention therefore keeps tokens because of *where* they are, not because of *what* they carry, a biased and lossy signal.

<!-- PLACEHOLDER IMAGE — upload the "Vision-Text Attention & Positional Distribution" heatmap (presentation slide 18) to your HF CDN and replace the src. -->
<a id="figure-3"></a>
<figure style="width: 75%; margin: auto; text-align: center;">
  <img src="/assets/img/attn_guidance_spec.png"
       alt="Positional bias of target attention over video tokens"
       style="width: 100%;">
  <figcaption style="margin-top: 10px; font-style: italic; color: #555;">
    Figure 3: <b>Attention is positionally biased.</b> The target model disproportionately selects video tokens at the start and end of the sequence, regardless of content. Figure adapted from <a href="#parallelvlm-2026">Kong et al., 2026</a>.
  </figcaption>
</figure>

With the disease diagnosed, the cure is now clear. We need drafts that carry *less* visual weight, that stay *aligned* with the target despite carrying less, and, for video, that stop wasting hardware on sequential scheduling. The next two sections are two concrete answers to exactly these problems.

## SpecVLM — Compress the Vision, Distill the Draft

SpecVLM ([Huang et al., 2025](#specvlm-2025)) tackles the *image* setting head-on. It starts from the strong EagleVLM baseline above and adds two ingredients: an **elastic visual compressor** to shrink the draft's visual burden, and an **online-logit distillation** protocol to keep the slimmed-down draft aligned with the target.

#### Elastic visual compression

If the draft's problem is too many visual tokens, the obvious move is to compress them before they ever reach the draft. The catch is that the *right* compressor depends on the input: a dense OCR image and a simple scene have very different compression sweet spots. SpecVLM therefore does not commit to a single operator. It assembles a toolbox of four complementary visual compressors and chooses among them:

- **Pruning** — drop redundant tokens (random or structured).
- **Pooling** — spatially downsample groups of tokens into one.
- **Convolution** — learn a compact spatial summary.
- **Resampler** — a Q-Former-style cross-attention module ([Li et al., 2023](#blip2-2023)) that distills many tokens into a few learned queries.

<!-- PLACEHOLDER IMAGE — upload the "Different visual compressors / Elastic compressor" figure (presentation slide 8) to your HF CDN and replace the src. -->
<a id="figure-4"></a>
<figure style="width: 85%; margin: auto; text-align: center;">
  <img src="REPLACE_WITH_HF_CDN_URL"
       alt="The four visual compressors and the elastic compressor"
       style="width: 100%;">
  <figcaption style="margin-top: 10px; font-style: italic; color: #555;">
    Figure 4: <b>The visual compressor toolbox.</b> Pruning, pooling, convolution and resampler primitives, combined into an elastic compressor that trades FLOPs/parameters against accuracy per input. Figure adapted from <a href="#specvlm-2025">Huang et al., 2025</a>.
  </figcaption>
</figure>

The word *elastic* is the key. Rather than hard-wiring one strategy, the compressor adaptively selects how aggressively to compress per input, navigating the FLOPs/parameters-versus-accuracy frontier on the fly. The authors explore three ways of combining the experts:

**Strategy A: Weighted combination of experts**<br>
A question-conditioned gate produces weights and blends the outputs of all compressors.

**Strategy B: Multi-granularity feature concatenation**<br>
Each compressor runs at a pre-defined compression ratio and their outputs are concatenated, giving the draft a multi-scale view.

**Strategy C: Dynamic selection of compression experts**<br>
A question-conditioned gate picks the single best (top-1) compressor based on question difficulty and image-text relevance, the leanest option, spending compute only where it helps.

<u>Upside</u>: **the draft carries a fraction of the visual tokens, so each draft step gets genuinely cheaper**.<br>
<u>Downside</u>: a more aggressively compressed draft sees less, and risks drifting from the target, which is exactly what the next ingredient repairs.

#### Online-logit distillation

A compressed draft is only useful if it still *agrees* with the target, otherwise acceptance length \\( \sigma \\) collapses and the speedup evaporates. The usual fix is offline distillation, but building a teacher-logit corpus at multimodal scale is cumbersome and storage-heavy. SpecVLM instead distills **online**, generating the teacher's supervision on the fly during training:

- The target produces token-level logits \\( \mathbf{z}_p \\) and penultimate-layer features \\( \mathbf{f}_p \\).
- The draft produces its own \\( \mathbf{z}_q \\) and \\( \mathbf{f}_q \\).
- The draft is trained to match both, with a combined cross-entropy (on logits) and Smooth-L1 (on features) objective:

$$ \mathcal{L}_{\text{online}} = \lambda_{\text{logit}}\, \mathcal{L}_{\text{CE}}(\mathbf{z}_q, \mathbf{z}_p) + \lambda_{\text{feat}}\, \mathcal{L}_{\text{SmoothL1}}(\mathbf{f}_q, \mathbf{f}_p) $$

This eliminates the offline corpus entirely while staying compute-efficient. It also surfaces a neat empirical phenomenon, a **training-time scaling effect**: with the data and draft architecture fixed, *longer* online training monotonically *increases* the draft's mean accepted length \\( \sigma \\), and hence the speedup. Better-aligned drafts are, quite literally, a matter of training them longer.

<!-- PLACEHOLDER IMAGE — upload the SpecVLM inference + training process figure (presentation slide 11) to your HF CDN and replace the src. -->
<a id="figure-5"></a>
<figure style="width: 85%; margin: auto; text-align: center;">
  <img src="REPLACE_WITH_HF_CDN_URL"
       alt="SpecVLM inference and online-distillation training process"
       style="width: 100%;">
  <figcaption style="margin-top: 10px; font-style: italic; color: #555;">
    Figure 5: <b>SpecVLM end-to-end.</b> (a) Inference: a UV-compressed draft autoregresses while the target verifies. (b) Training: online distillation matches the draft's logits and features to the frozen target via Smooth-L1 + CE. Figure adapted from <a href="#specvlm-2025">Huang et al., 2025</a>.
  </figcaption>
</figure>

#### Results

Stacking compression and online distillation on top of EagleVLM consistently improves both \\( \tau \\) and \\( \sigma \\). On the LLaVA family, SpecVLM lifts the per-benchmark speedup above the EagleVLM baseline (e.g. from 2.09× to 2.20× on LLaVA-1.5-7B, and from 2.29× to 2.38× on LLaVA-1.6-13B at temperature 0), and the paper reports reaching **2.5–2.9× end-to-end speedups within 5 training epochs** across LLaVA and MMMU, all while remaining strictly lossless. The single-image latency breakdown is striking: a 306 ms autoregressive pass for LLaVA-1.6-7B drops to 65 ms with EagleVLM and to **46 ms** with SpecVLM.

## ParallelVLM — Aligning and Parallelizing for Video

SpecVLM compresses *what* the draft sees. ParallelVLM ([Kong et al., 2026](#parallelvlm-2026)) goes after the harder video setting by fixing *which* tokens to keep and *when* the models run. It contributes two ideas: an unbiased way to prune, and a parallel pipeline that stops wasting idle hardware.

> A naming note worth pausing on: ParallelVLM builds on an earlier, identically-named *SpecVLM* for video LLMs ([Ji et al., 2025](#specvlm-video-2025)), a training-free, verifier-guided pruning method, which is a *different* paper from the image-focused SpecVLM above. Three groups, two papers called "SpecVLM"; we disambiguate by author and modality throughout.

#### UV-Prune: unbiased verifier-guided pruning

Recall the positional-bias problem ([Figure 3](#figure-3)): pruning by the target's raw attention keeps tokens for the wrong reason. ParallelVLM reframes the question entirely. Instead of asking *"which tokens does the model attend to?"*, it asks:

<p align="center"><code>Which tokens become increasingly aligned with the text query as information flows through the target's layers?</code></p>

This is **Unbiased Verifier-Guided Pruning (UV-Prune)**. For each visual token \\( V_i \\) and text token \\( X_j \\), it measures the cosine similarity of their representations,

$$ S_{ij} = \frac{V_i \cdot X_j}{\lVert V_i \rVert\, \lVert X_j \rVert}, $$

and tracks how this vision-text alignment *changes across layers*, \\( \Delta S \\). Tokens whose alignment **increases** as they pass deeper into the target are the ones genuinely accumulating query-relevant semantics; a Top-K selection over \\( \Delta S \\) keeps them. Because the signal is the *trajectory* of alignment rather than a raw attention value, it sidesteps the start/end positional bias, "unbiased guidance" instead of "positional bias".

<!-- PLACEHOLDER IMAGE — upload the UV-Prune figure (layerwise variations + Top-K + unbiased vs biased distribution, presentation slide 20) to your HF CDN and replace the src. -->
<a id="figure-6"></a>
<figure style="width: 85%; margin: auto; text-align: center;">
  <img src="REPLACE_WITH_HF_CDN_URL"
       alt="UV-Prune: layerwise alignment variations and Top-K selection"
       style="width: 100%;">
  <figcaption style="margin-top: 10px; font-style: italic; color: #555;">
    Figure 6: <b>UV-Prune.</b> Layerwise cosine-similarity variations ( \( \Delta S \) ) drive a Top-K selection of tokens whose vision-text alignment grows through the target, yielding unbiased guidance instead of positional bias. Figure adapted from <a href="#parallelvlm-2026">Kong et al., 2026</a>.
  </figcaption>
</figure>

#### A parallel pipeline

The second contribution attacks the sequential-execution waste directly by overlapping the draft and target:

**Stage 1 — Parallel Prefilling.** The target begins prefilling the *whole* video. As soon as its intermediate layers finish, that information is immediately used to run UV-Prune, and the draft then prefills *only the pruned token set*. Because the draft's job is now so much lighter, it finishes before the target, and that leftover gap is not wasted: it is spent generating **startup tokens**, a set of candidate tokens produced *before* decoding even begins.

**Stage 2 — Parallel Decoding.** Thanks to the startup tokens, rejection sampling can begin directly, with no warmup. And because pruning makes draft decoding so much cheaper relative to target verification, the verification **window size can grow by almost 2×**, the target can afford to check nearly twice as many speculated tokens per round, lifting throughput further. A pre-rollback mechanism keeps the whole thing lossless.

<!-- PLACEHOLDER IMAGE — upload the "Overview: Vanilla SD vs SpecVLM vs ParallelVLM" timeline (presentation slide 23) to your HF CDN and replace the src. -->
<a id="figure-7"></a>
<figure style="width: 90%; margin: auto; text-align: center;">
  <img src="REPLACE_WITH_HF_CDN_URL"
       alt="Execution timelines: vanilla SD vs SpecVLM vs ParallelVLM"
       style="width: 100%;">
  <figcaption style="margin-top: 10px; font-style: italic; color: #555;">
    Figure 7: <b>Filling the idle gaps.</b> Compared to vanilla SD and SpecVLM, ParallelVLM overlaps draft and target prefill, uses the slack to emit startup tokens, and keeps the accelerators busy through decoding. Figure adapted from <a href="#parallelvlm-2026">Kong et al., 2026</a>.
  </figcaption>
</figure>

#### Results

The payoff scales with the size gap between draft and target. Averaged across five video benchmarks (VideoDetailCaption, VideoMME, MVBench, MVLU, LongVideoBench), ParallelVLM reaches **2.11×** for LLaVA-OV (0.5B draft & 7B target), **3.36×** for LLaVA-OV (7B & 72B), and **2.42×** for Qwen2.5-VL (7B & 32B), beating Vanilla SD, OnSD, SD-Tree and the video SpecVLM on every setting, with mean accepted lengths up to ~8.5 tokens per round.

Crucially, because UV-Prune lives inside a *lossless* SD framework, accuracy is preserved. Against *lossy* visual-token pruning methods such as FastV ([Chen et al., 2024](#fastv-2024)), SparseVLM ([Zhang et al., 2024](#sparsevlm-2024)) and DyCoke ([Tao et al., 2024](#dycoke-2024)) at a 10% retention ratio, ParallelVLM retains **~98–99% accuracy** while delivering a higher speedup, where the lossy baselines sit at ~83–91% accuracy and 1.4–1.6×. An ablation on the pruning rate \\( \alpha \\) shows the sweet spot at \\( \alpha = 0.9 \\) (3.36× for LLaVA-OV, 2.42× for Qwen), beyond which over-pruning ( \\( \alpha = 1.0 \\) ) starves the draft and acceptance length, and hence speed, falls off.

## The Broader Landscape

SpecVLM and ParallelVLM are two points in a fast-growing space. The recurring theme across all of it is the same: *lighten the draft's visual load, keep it aligned with the target, and never break losslessness.* A few other directions worth knowing:

- **MSD — Multimodal Speculative Decoding** ([Lin et al., 2025](#msd-2025)). Argues that text and visual tokens are different enough that the draft should process them *separately*, and trains the draft in two stages, text-only instruction tuning first, then a gradual curriculum of multimodal data, reaching up to 2.29×/2.46× on LLaVA-1.5 7B/13B.
- **ViSpec — Vision-Aware Speculative Decoding** ([Kang et al., 2025](#vispec-2025)). Adds a lightweight Q-Former-style vision adaptor to compress image tokens, then extracts a single *global* visual feature vector and injects it into every subsequent text token's hidden state, giving the draft persistent visual grounding over long generations. Combined with synthetic long-response training data, it reports up to **3.22×**, against ~1.6× for Medusa and ~2.1× for EAGLE-2 on the same models.
- **Video SpecVLM** ([Ji et al., 2025](#specvlm-video-2025)). The training-free precursor to ParallelVLM: a two-stage verifier-guided pruning that drops up to **90%** of video tokens, on the finding that the draft's speculation is remarkably insensitive to video-token pruning, for 2.68× (LLaVA-OV-72B) and 2.11× (Qwen2.5-VL-32B).
- **Spec-VLA** ([Wang et al., 2025](#specvla-2025)). Pushes SD beyond perception into *action*: for Vision-Language-Action robotics models, it relaxes the acceptance criterion using the relative distances between action tokens, recovering a 1.42× speedup over OpenVLA where naive SD barely moves the needle.

<div style="margin-top: 2rem; margin-bottom: 2rem; font-family: sans-serif; width: 100%;">
    <table style="margin-left: auto; margin-right: auto; border-collapse: collapse; border: 1px solid #dee2e6; box-shadow: 0 2px 5px rgba(0,0,0,0.05); min-width: 60%;">
        <thead>
            <tr style="background-color: #f8f9fa; border-bottom: 2px solid #dee2e6;">
                <th style="padding: 12px 20px; text-align: left; border: 1px solid #dee2e6;">Method</th>
                <th style="padding: 12px 20px; text-align: left; border: 1px solid #dee2e6;">Modality</th>
                <th style="padding: 12px 20px; text-align: left; border: 1px solid #dee2e6;">Core Idea</th>
                <th style="padding: 12px 20px; text-align: center; border: 1px solid #dee2e6;">Reported Speedup</th>
            </tr>
        </thead>
        <tbody>
            <tr style="border-bottom: 1px solid #dee2e6;">
                <td style="padding: 12px 20px; border: 1px solid #dee2e6; font-weight: bold;">EagleVLM</td>
                <td style="padding: 12px 20px; border: 1px solid #dee2e6;">Image</td>
                <td style="padding: 12px 20px; border: 1px solid #dee2e6;">EAGLE-2 draft ported to VLMs</td>
                <td style="padding: 12px 20px; border: 1px solid #dee2e6; text-align: center;">1.5–2.3×</td>
            </tr>
            <tr style="border-bottom: 1px solid #dee2e6;">
                <td style="padding: 12px 20px; border: 1px solid #dee2e6; font-weight: bold;">SpecVLM</td>
                <td style="padding: 12px 20px; border: 1px solid #dee2e6;">Image</td>
                <td style="padding: 12px 20px; border: 1px solid #dee2e6;">Elastic visual compressor + online-logit distillation</td>
                <td style="padding: 12px 20px; border: 1px solid #dee2e6; text-align: center;">2.5–2.9×</td>
            </tr>
            <tr style="border-bottom: 1px solid #dee2e6;">
                <td style="padding: 12px 20px; border: 1px solid #dee2e6; font-weight: bold;">MSD</td>
                <td style="padding: 12px 20px; border: 1px solid #dee2e6;">Image</td>
                <td style="padding: 12px 20px; border: 1px solid #dee2e6;">Separate text/visual drafting + staged training</td>
                <td style="padding: 12px 20px; border: 1px solid #dee2e6; text-align: center;">~2.3–2.5×</td>
            </tr>
            <tr style="border-bottom: 1px solid #dee2e6;">
                <td style="padding: 12px 20px; border: 1px solid #dee2e6; font-weight: bold;">ViSpec</td>
                <td style="padding: 12px 20px; border: 1px solid #dee2e6;">Image</td>
                <td style="padding: 12px 20px; border: 1px solid #dee2e6;">Vision adaptor + global feature injection</td>
                <td style="padding: 12px 20px; border: 1px solid #dee2e6; text-align: center;">up to 3.22×</td>
            </tr>
            <tr style="border-bottom: 1px solid #dee2e6;">
                <td style="padding: 12px 20px; border: 1px solid #dee2e6; font-weight: bold;">Video SpecVLM</td>
                <td style="padding: 12px 20px; border: 1px solid #dee2e6;">Video</td>
                <td style="padding: 12px 20px; border: 1px solid #dee2e6;">Training-free staged verifier-guided pruning (≤90%)</td>
                <td style="padding: 12px 20px; border: 1px solid #dee2e6; text-align: center;">up to 2.68×</td>
            </tr>
            <tr style="border-bottom: 1px solid #dee2e6;">
                <td style="padding: 12px 20px; border: 1px solid #dee2e6; font-weight: bold;">ParallelVLM</td>
                <td style="padding: 12px 20px; border: 1px solid #dee2e6;">Video</td>
                <td style="padding: 12px 20px; border: 1px solid #dee2e6;">UV-Prune + parallel prefill + startup tokens</td>
                <td style="padding: 12px 20px; border: 1px solid #dee2e6; text-align: center;">up to 3.36×</td>
            </tr>
            <tr style="border-bottom: 1px solid #dee2e6;">
                <td style="padding: 12px 20px; border: 1px solid #dee2e6; font-weight: bold;">Spec-VLA</td>
                <td style="padding: 12px 20px; border: 1px solid #dee2e6;">Action (VLA)</td>
                <td style="padding: 12px 20px; border: 1px solid #dee2e6;">Relaxed acceptance on action-token distances</td>
                <td style="padding: 12px 20px; border: 1px solid #dee2e6; text-align: center;">1.42×</td>
            </tr>
        </tbody>
    </table>
    <p style="text-align: center; margin-top: 15px; font-size: 0.9em; color: #555; font-style: italic;">
        Table 1: <b>A snapshot of Speculative Decoding for multimodal architectures.</b> Speedups are reported by each respective paper and are not directly comparable across different models, hardware and settings.
    </p>
</div>

## Conclusions & Key Takeaways

So, where are the *accuracy* tables in all of this? The answer is the quiet superpower of this whole line of work: there aren't any to report, and that is the point. Every method here lives inside a rejection-sampling framework that provably reproduces the target's output distribution. The decoding is **lossless** by construction, so the only meaningful axis is speed.

Pulling the threads together:

- *Speculative Decoding works on multimodal models out of the box.* A naive EAGLE-style draft already buys roughly 2× on VLMs, for free and without quality loss, which is a strong argument for reaching for it in production.
- *Visual tokens are still the villain.* The very token counts we spent this series quantifying are what cap the naive speedup: they bloat the draft's KV Cache, slow its attention, and, in video, starve the hardware through sequential scheduling.
- *The fix is to lighten and align the draft.* Whether through SpecVLM's elastic compression and online distillation, ParallelVLM's unbiased UV-Prune and parallel pipeline, or ViSpec's global feature injection, the recipe is the same: give the draft fewer visual tokens, but keep it faithful to the target so acceptance length stays high.

And the road ahead is wide open. A few questions I find particularly exciting:

- Can these techniques extend to **self-speculative** decoding, where one model drafts and verifies itself, removing the separate draft entirely?
- Can we dynamically **compress the draft's KV Cache** rather than only its input token window?
- Beyond pruning the *tokens*, why not prune the *model*, drafting with a structurally smaller target?
- And, as always, how far does **scaling the training and data** of the draft push the mean accepted length?

Multimodal inference is expensive, but it does not have to be slow. Speculative Decoding is one of the rare optimizations that costs us nothing in quality, and as the draft learns to see with fewer, smarter tokens, the gap between "watching a model think" and "getting the answer" keeps closing.

## Citation

If you use this work, please cite:

```bibtex
@misc{nulli2026speculativedecoding,
  title={Demystifying Multimodal Learning: Speculative Decoding in Multimodal Architectures},
  author={Nulli, Matteo},
  year={2026},
  url={https://huggingface.co/blog/MatteoNulli/de-mystifying-multimodal-learning-speculative-decoding},
  howpublished={Available at \url{https://matteonulli.github.io/blog/2026/demystifying3/} and \url{https://huggingface.co/blog/MatteoNulli/de-mystifying-multimodal-learning-speculative-decoding}},
  note={Hugging Face Blog}
}
```


<br>

**References**

<div id="references-section">

<a id="specdec-2023" class="bib-item"> Leviathan, Yaniv, Matan Kalman, and Yossi Matias. "Fast inference from transformers via speculative decoding." International Conference on Machine Learning (ICML), 2023. arXiv preprint arXiv:2211.17192. </a>

<a id="specsample-2023" class="bib-item"> Chen, Charlie, Sebastian Borgeaud, Geoffrey Irving, Jean-Baptiste Lespiau, Laurent Sifre, and John Jumper. "Accelerating large language model decoding with speculative sampling." arXiv preprint arXiv:2302.01318 (2023). </a>

<a id="medusa-2024" class="bib-item"> Cai, Tianle, Yuhong Li, Zhengyang Geng, Hongwu Peng, Jason D. Lee, Deming Chen, and Tri Dao. "Medusa: Simple LLM inference acceleration framework with multiple decoding heads." International Conference on Machine Learning (ICML), 2024. arXiv preprint arXiv:2401.10774. </a>

<a id="eagle-2024" class="bib-item"> Li, Yuhui, Fangyun Wei, Chao Zhang, and Hongyang Zhang. "EAGLE: Speculative sampling requires rethinking feature uncertainty." International Conference on Machine Learning (ICML), 2024. arXiv preprint arXiv:2401.15077. </a>

<a id="eagle2-2024" class="bib-item"> Li, Yuhui, Fangyun Wei, Chao Zhang, and Hongyang Zhang. "EAGLE-2: Faster inference of language models with dynamic draft trees." Conference on Empirical Methods in Natural Language Processing (EMNLP), 2024. arXiv preprint arXiv:2406.16858. </a>

<a id="eagle3-2025" class="bib-item"> Li, Yuhui, Fangyun Wei, Chao Zhang, and Hongyang Zhang. "EAGLE-3: Scaling up inference acceleration of large language models via training-time test." arXiv preprint arXiv:2503.01840 (2025). </a>

<a id="specvlm-2025" class="bib-item"> Huang, Haiduo, Fuwei Yang, Zhenhua Liu, Xuanwu Yin, Dong Li, Pengju Ren, and Emad Barsoum. "SpecVLM: Fast speculative decoding in vision-language models." arXiv preprint arXiv:2509.11815 (2025). </a>

<a id="parallelvlm-2026" class="bib-item"> Kong, Quan, Yuhao Shen, Yicheng Ji, Huan Li, and Cong Wang. "ParallelVLM: Lossless video-LLM acceleration with visual alignment aware parallel speculative decoding." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026. </a>

<a id="specvlm-video-2025" class="bib-item"> Ji, Yicheng, Jun Zhang, Heming Xia, Jinpeng Chen, Lidan Shou, Gang Chen, and Huan Li. "SpecVLM: Enhancing speculative decoding of video LLMs via verifier-guided token pruning." Conference on Empirical Methods in Natural Language Processing (EMNLP), 2025. arXiv preprint arXiv:2508.16201. </a>

<a id="msd-2025" class="bib-item"> Lin, Luxi, Zhihang Lin, Zhanpeng Zeng, and Rongrong Ji. "Speculative decoding reimagined for multimodal large language models." arXiv preprint arXiv:2505.14260 (2025). </a>

<a id="vispec-2025" class="bib-item"> Kang, Jialiang, Han Shu, Wenshuo Li, Yingjie Zhai, and Xinghao Chen. "ViSpec: Accelerating vision-language models with vision-aware speculative decoding." arXiv preprint arXiv:2509.15235 (2025). </a>

<a id="specvla-2025" class="bib-item"> Wang, Songsheng, Rucheng Yu, Zhihang Yuan, Chao Yu, Feng Gao, Yu Wang, and Derek F. Wong. "Spec-VLA: Speculative decoding for vision-language-action models with relaxed acceptance." Conference on Empirical Methods in Natural Language Processing (EMNLP), 2025. arXiv preprint arXiv:2507.22424. </a>

<a id="blip2-2023" class="bib-item"> Li, Junnan, Dongxu Li, Silvio Savarese, and Steven Hoi. "BLIP-2: Bootstrapping language-image pre-training with frozen image encoders and large language models." International Conference on Machine Learning (ICML), 2023. arXiv preprint arXiv:2301.12597. </a>

<a id="fastv-2024" class="bib-item"> Chen, Liang, Haozhe Zhao, Tianyu Liu, Shuai Bai, Junyang Lin, Chang Zhou, and Baobao Chang. "An image is worth 1/2 tokens after layer 2: Plug-and-play inference acceleration for large vision-language models." European Conference on Computer Vision (ECCV), 2024. arXiv preprint arXiv:2403.06764. </a>

<a id="sparsevlm-2024" class="bib-item"> Zhang, Yuan, Chun-Kai Fan, Junpeng Ma, Wenzhao Zheng, Tao Huang, Kuan Cheng, Denis Gudovskiy, Tomoyuki Okuno, Yohei Nakata, Kurt Keutzer, and Shanghang Zhang. "SparseVLM: Visual token sparsification for efficient vision-language model inference." arXiv preprint arXiv:2410.04417 (2024). </a>

<a id="dycoke-2024" class="bib-item"> Tao, Keda, Can Qin, Haoxuan You, Yang Sui, and Huan Wang. "DyCoke: Dynamic compression of tokens for fast video large language models." arXiv preprint arXiv:2411.15024 (2024). </a>

<a id="llavonevision-2024" class="bib-item">Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. 2024a. Llava-onevision: Easy visual task transfer. Preprint, arXiv:2408.03326.</a>

<a id="qwen2-5-vl-2025" class="bib-item">Bai Shuai, Chen Keqin, Liu Xuejing, Wang Jialin, Ge Wenbin, Song Sibo, Dang Kai, Wang Peng, Wang Shijie, Tang Jun, Zhong Humen, Zhu Yuanzhi, Yang Mingkun, Li Zhaohai, Wan Jianqiang, Wang Pengfei, Ding Wei, Fu Zheren, Xu Yiheng, Ye Jiabo, Zhang Xi, Xie Tianbao, Cheng Zesen, Zhang Hang, Yang Zhibo, Xu Haiyang, Lin Junyang. (2025). Qwen2.5-VL Technical Report. arXiv preprint arXiv:2502.13923.</a>

</div>

<style>
  /* Hide all references by default */
  .bib-item { display: none; }
  /* Show only the ones with the 'cited' class */
  .bib-item.cited { display: block; margin-bottom: 10px; }
</style>

{% raw %}
<script>
document.addEventListener("DOMContentLoaded", function() {
    // 1. Find all internal links in the post (usually starting with #)
    const links = document.querySelectorAll('a[href^="#"]');
    const citedIds = new Set();

    links.forEach(link => {
        // Get the ID being linked to (remove the # character)
        const id = link.getAttribute('href').substring(1);
        if (id) citedIds.add(id);
    });

    // 2. Loop through all reference items
    const refItems = document.querySelectorAll('.bib-item');
    refItems.forEach(item => {
        if (citedIds.has(item.id)) {
            item.classList.add('cited'); // This makes it visible via CSS
        }
    });
});
</script>
{% endraw %}
