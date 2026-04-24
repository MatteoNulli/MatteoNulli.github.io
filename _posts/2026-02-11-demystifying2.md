---
layout: post
title: "Demystifying Multimodal Learning: Impact of Visual Tokens on Inference Latency"
date: 2026-04-20 14:14:00
description: A blogpost series on the nuts and bolts of Multimodal Learning
tags: Multimodal-Learning Inference-Optimization
# thumbnail: assets/img/mllms_visual_tokens_wide.png
thumbnail: https://cdn-uploads.huggingface.co/production/uploads/661d4e74b8f13412f6d48a50/SfOleyYtgr6UtQ4lT8jv7.png

community_article_url: https://huggingface.co/blog/MatteoNulli/de-mystifying-multimodal-learning-impact-vt-laten
blogpost_url: https://matteonulli.github.io/blog/2025/demystifying2/
math: true
mermaid: true
_styles: >
  .mermaid svg { 
    max-width: 100%; 
    height: auto; 
  }
---


##### <b>Matteo Nulli, Marcin Mazur</b>
<!-- ###### eBay
###### <img src="https://upload.wikimedia.org/wikipedia/commons/1/1b/EBay_logo.svg" alt="eBay" height="24"/> &nbsp;  -->
###### <a href="https://huggingface.co/blog/MatteoNulli/de-mystifying-multimodal-learning-impact-vt-laten" title="Community Article"><i class="fa-brands fa-hugging-face" style="font-size: 1.75em;"></i></a> <a href="https://matteonulli.github.io/blog/2025/demystifying2/" title="Blogpost"><i class="fa-regular fa-newspaper" style="font-size: 1.75em;"></i></a>
<br>

## Introduction
In our previous iterations of `Demystifying Multimodal Learning`, we have defined  <abbr title="Click here for our previous blogpost.">[what a Visual Token (VT) is](https://huggingface.co/blog/MatteoNulli/de-mystifying-multimodal-learning-enabiling-vision)</abbr> and established formulas to  <abbr title="Click here for our previous blogpost.">[calculate # Visual Tokens ( \\( V \\)) for different architectures](https://huggingface.co/blog/MatteoNulli/de-mystifying-multimodal-learning-hidden-ineff)</abbr>.

Before we look under the hood of production engines, we need to ask a fundamental question:

<p align="center"><code>What is the true cost of <var>V</var> on inference latency, and why is it critical for Machine Learning Engineers scaling and deploying models?</code></p>


The answer is simple: Scale is incredibly expensive. As highlighted in the recent [Moondream blog](https://moondream.ai/blog/moondream-2025-04-14-release), analyzing vision at scale quickly becomes the primary bottleneck for AI applications. When your system needs to process millions of images or sift through thousands of hours of video, compute resources are drained at an alarming rate.

<a id="figure-1"></a>
<figure style="width: 70%; margin: auto; text-align: center;">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/661d4e74b8f13412f6d48a50/SfOleyYtgr6UtQ4lT8jv7.png"
       alt="model efficiency"
       style="width: 100%;">
  <figcaption style="margin-top: 10px; font-style: italic; color: #555;">
    Figure 1: <b>Figure from <a href="https://moondream.ai/blog/moondream-2025-04-14-release">Moondream blog, 2025</a></b>. Demonstratating the focus of fronteer labs on VLM efficiency.
  </figcaption>
</figure>

While frontier AI labs are intensely focused on making VLMs faster (<a href="#smolvlmhf-2025">Marafioti, Andrés, et al., 2025</a>, <a href="#paligemma2-2024">Steiner, Andreas, et al. 2024</a>, <a href="#gemma-3-2025">Gemma-Team, 2025</a>, <a href="#qwen2-5-vl-2025">Bai Shuai, et al., 2025</a>) at the architectural level (see [Figure 1](#figure-1)), far fewer people are dissecting the downstream impact of these visual tokens during inference.

Architectural efficiency is a great starting point, but if your deployment strategy is bloated with unnecessary tokens, scaling remains a bottleneck. To serve multimodal models effectively, we need to make more conscious, data-driven decisions about how we handle these tokens in production. Crucially, managing this isn't just about minimizing token counts—it's also about how our serving infrastructure processes them.

This requires tightly managing and exploring three critical pillars of Inference, [Latency](#inference-latency), [Context Windows](#context-window-budget), and [VRAM KV Cache](#the-cascading-impact-on-vram), all while leveraging [Architectural Decouplings](#decoupling-the-vision-encoder-and-prefix-caching) to maximize hardware efficiency.

## Inference Latency

In production use-cases, inference cost is heavily tied to latency. Large companies enforce strict input limits to ensure predictable response times, but Visual Tokens completely disrupt this predictability.

When inference providers like [vLLM](https://docs.vllm.ai/en/stable/) ([Kwon, Woosuk, et al., 2023](#vllm-2023)) process a multimodal request, they undergo several distinct phases, each carrying its own 'latency tax':

**Phase 0: Multi-Modal Processing**
Before the model even sees the image, engines apply Hugging Face Processors to combine the prompt text and multi-modal data. The text is tokenized, and the image sequences in the token IDs are updated with placeholder tokens (the number of placeholders equals the feature size outputted by the vision encoder).

- The Bottleneck: Vision Processors can be [notoriously](https://github.com/vllm-project/vllm/issues/9238) slow, creating a CPU bottleneck before GPU inference even begins.

- The Solution: To mitigate this, vLLM utilizes Processor Output Caching. When new data arrives, it checks the cache; missing items are processed in a single batch, cached, and then merged. This prevents redundant processing overhead for frequently seen or system-level images. More details [here](https://docs.vllm.ai/en/stable/design/mm_processing/#processor-output-caching).


**Phase 1: Vision Encoding**
The image is passed through the Vision Encoder (VE) to create the actual embeddings that will replace the placeholder tokens. As seen in [Figure 2](#figure-2) below, encoding latency for small VLMs (0.5B-3B parameters) increases massively at high resolutions compared to the LLM prefill stage. \\( ^{**} \\)


**Phase 2: LLM Prefill (Time-To-First-Token)**
The LLM processes all input tokens (Text + Visual) in parallel to compute the initial keys and values. A high VT count dramatically increases the Time-To-First-Token (TTFT), with prefill latency scaling quadratically with the sequence length:

$$\text{Latency}_{\text{prefill}} \propto (V + T_{\text{total}})^2$$

where V stands for the numeber of Visual Tokens and \\(T_{\text{total}}\\) represents the text tokens.

**Phase 3: LLM Decoding**
The model generates the output one token at a time. Each generated token must attend over the entire cached history of visual and text tokens. Therefore, per-token decode latency grows roughly linearly:

$$\text{Latency}_{\text{decoding}} \propto (V + T_{\text{total}})$$


<a id="figure-2"></a>
<figure style="width: 70%; margin: auto; text-align: center;">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/661d4e74b8f13412f6d48a50/PN6G2B6Y7ST3t_IpCZqoo.png"
       alt="token comparison"
       style="width: 100%;">
  <figcaption style="margin-top: 10px; font-style: italic; color: #555;">
    Figure 2: <b>Figure from <a href="#fastvlms-2025">Anasosalu et al., 2025</a>, from Apple MLR</b>. It illustrates how Vision latency dominates at high resolution. Breakdown of FastVLM’s time to the first token for different image resolutions. The vision encoder is <a href="https://huggingface.co/kevin510/fast-vit-hd">FastViT-HD</a>, and the LLM is Qwen2-1.5B.
  </figcaption>
</figure>


<small> \\( ^{**} \\) See [Anasosalu et al., 2025](#fastvlms-2025) for a good overview of Fast-ViT architectures. </small>


## Decoupling the Vision Encoder and Prefix Caching

When optimizing multimodal inference, treating the model as a single monolithic block is highly inefficient. State-of-the-art serving engines like vLLM have introduced a critical architectural optimization: the strict separation of the Vision Encoder (VE) and the Large Language Model (LLM).

By extracting the vision embedding process into a dedicated method (such as [`embed_multimodal`](https://docs.vllm.ai/en/stable/api/vllm/model_executor/models/interfaces/#vllm.model_executor.models.interfaces.SupportsMultiModal.embed_input_ids:~:text=embed_multimodal,-%C2%B6)), engines can run the VE and LLM asynchronously. This decoupling ensures that heavy image embeddings are computed, queued, and ready exactly when the decoder is prepared to ingest them.

More importantly, this separation solves a major parallelism mismatch. LLMs are massive and often require <abbr title="Click here for an overview of Tensor Parallelism."> [Tensor Parallelism (TP)](https://afmck.in/posts/2023-02-26-parallelism/#tensor-parallelism:~:text=avoid%20pipeline%20bubbles.-,Tensor%20Parallelism,-%23)</abbr> or Expert Parallelism to distribute their weights across multiple GPUs to fit into memory. Vision Encoders, however, are typically much smaller. Forcing a small VE to use TP introduces unnecessary cross-GPU communication overhead, which actually slows down the encoding phase. By decoupling the architectures, engineers can apply batch-level <abbr title="Click here for an overview of Data Parallelism.">[Data Parallelism](https://afmck.in/posts/2023-02-26-parallelism/#:~:text=may%20be%20necessary.-,Data,-Parallelism)</abbr> to the multi-modal encoder—processing different images on different GPUs simultaneously—while reserving TP strictly for the heavy lifting of the LLM.

**Multimodal Prefix Caching**

Prefix caching is a vital optimization from both the user and infrastructure provider perspectives, as it dramatically reduces redundant compute for shared context ([see this for more on prompt caching](https://sankalp.bearblog.dev/how-prompt-caching-works/)).
For Vision Language Models, this translates to massive performance gains by mitigating the heavy processing tax of visual tokens [Barrios, Wayner. et. al, 2026](#vlminf-2026). In serving engines like vLLM, this caching is implemented seamlessly by matching images based on their unique image hash before the Vision Encoder step even begins. If a hash match is found, the system simply retrieves the cached representation, meaning the computationally expensive vision model pass is skipped entirely, resulting in immense latency savings, ([read more here](https://docs.vllm.ai/en/stable/design/prefix_caching/)).


## Context Window Budget

Every LLM/VLM operates within a fixed input capacity known as the context window. As agentic AI systems become more prevalent, a variety of techniques have emerged to make more efficient use of this limited space:
- excluding intermediate reasoning from conversation history
- automatically pruning tool-call traces e.g. with [Dynamic Context Pruning](https://github.com/Opencode-DCP/opencode-dynamic-context-pruning)
- enforcing ultra-compact communication styles e.g. with [Caveman](https://github.com/juliusbrussee/caveman)

Image tokens, however, are significantly more difficult to optimize. In practice, once images are introduced into the context, they tend to persist in full—unlike text, which can be summarized or selectively removed.

For a <abbr title="Click here for a refresh on Visual Token Production Strategies.">[Strategy B](https://huggingface.co/blog/MatteoNulli/de-mystifying-multimodal-learning-hidden-ineff#:~:text=Strategy%20B%3A%20The%20Multi%2DGrid%20/%20AnyRes)</abbr> model (like [LLaVA-OneVision-7B](https://huggingface.co/lmms-lab/llava-onevision-qwen2-7b-ov) (<a href="#llavonevision-2024">Li Bo, et al., 2024</a>))  using a \\( 3 \times 3 \\) grid, a single image might consume \\( \approx 2900 \\) tokens. Given that the model has a context window of 32k, using 3 or 5 images can consume 30-45% of your entire input capacity. Even worse, if you are serving a model with a 4k pre-defined context limit, due to memory issues, a single image blocks 70% of the total input. These scenarios leaves little room for few-shot examples or long conversation history, potentially degrading the model's ability to follow complex instructions.

For this reason, it’s valuable to give users explicit control over how many tokens they are willing to allocate to visual inputs. One example of this approach is variable vision token limits in the <a href="//huggingface.co/google/gemma-4-31B-it#5-variable-image-resolution">Gemma 4</a> (<a href="#gemma4-2026">Farabet Clement, et al., 2026</a>) model family, which allows dynamic trade-offs between image fidelity and token usage.


## The Cascading Impact on VRAM
Perhaps the most critical "hidden" cost of Multimodal Learning is memory. When serving models, your maximum throughput is bound by how many requests can fit into GPU memory simultaneously (your Batch Size). This boundary is dictated heavily by the KV Cache.

The KV Cache stores the computed Key and Value vectors for all previous tokens in a sequence, preventing the model from recomputing them during the decoding phase. Unlike text tokens, which accumulate slowly as a user types or a model generates, Visual Tokens are dumped into the KV Cache all at once during the prefill phase.

- Higher VT Count \\( \rightarrow \\) Larger KV Cache footprint per request
- Larger KV Cache \\( \rightarrow \\) Fewer requests fit in VRAM
- Fewer Requests \\( \rightarrow \\) Smaller Batch Size

This creates a brutal cascading effect on your infrastructure costs. If a [high-resolution grid strategy](https://huggingface.co/blog/MatteoNulli/de-mystifying-multimodal-learning-hidden-ineff#:~:text=Strategy%20B%3A%20The%20Multi%2DGrid%20/%20AnyRes) increases your visual tokens by 10x, you might be forced to reduce your batch size by roughly the same factor just to avoid Out-Of-Memory (OOM) errors. You are effectively multiplying your cost per inference, as you now need significantly more GPUs to handle the same amount of user traffic.

The token calculations from [Demystifying Multimodal Learning: The Hidden Inefficiency in Vision Language Modelling](https://huggingface.co/blog/MatteoNulli/de-mystifying-multimodal-learning-hidden-ineff) are not just theoretical trivia—they are the direct levers that dictate your context limits, compute bottlenecks, and hardware budgets.


## Conclusions & Key Takeaways

As we have explored, the number of Visual Tokens a model generates is far more than an architectural quirk—it is a defining metric for production viability. While it might be tempting to chase state-of-the-art benchmark scores by feeding massive, high-resolution token grids into a VLM, doing so blindly sacrifices Latency, Context Windows and VRAM.

We saw how pushing too many tokens starves your available context limit, creates massive latency bottlenecks during the prefill phase, and monopolizes the KV Cache, which ultimately cripples your maximum batch size. Even with highly optimized serving engines like vLLM employing processor caching, decoupled parallelization strategies, and multimodal prefix caching to bypass redundant vision encoding, there is no software magic that can completely erase the hardware tax of a bloated token count.

Answering our [initial question](#introduction), to build commercially viable, scalable multimodal applications, we must treat token efficiency as a primary objective for model selection and architectural design. While caching shared visual context provides a critical safety valve for latency, moving forward, the most successful multimodal systems won't necessarily be the ones that process the most visual tokens, but the ones that compress visual reality into the fewest, smartest tokens possible.

## Citation

If you use this work, please cite:

```bibtex
@misc{nulli2026impactvisualtokens,
  title={Demystifying Multimodal Learning: Impact of Visual Tokens on Inference Latency},
  author={Nulli, Matteo and Mazur, Marcin},
  year={2026},
  url={https://huggingface.co/blog/MatteoNulli/de-mystifying-multimodal-learning-impact-vt-laten},
  howpublished={Available at \url{https://matteonulli.github.io/blog/2026/demystifying2/} and \url{https://huggingface.co/blog/MatteoNulli/de-mystifying-multimodal-learning-impact-vt-laten}},
  note={Hugging Face Blog}
}
```


<br>

**References**

<div id="references-section">

<a id="flash-attn-2022" class="bib-item"> Dao, Tri, et al. "Flashattention: Fast and memory-efficient exact attention with io-awareness." Advances in neural information processing systems 35 (2022): 16344-16359. </a>

<a id="vllm-2023" class="bib-item"> Kwon, Woosuk, et al. "Efficient memory management for large language model serving with pagedattention." Proceedings of the 29th symposium on operating systems principles. 2023. </a>

<a id="llavonevision-2024" class="bib-item">Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. 2024a. Llava-onevision: Easy visual task transfer. Preprint, arXiv:2408.03326.</a>

<a id="paligemma2-2024" class="bib-item"> Steiner, Andreas, et al. "Paligemma 2: A family of versatile vlms for transfer." arXiv preprint arXiv:2412.03555 (2024). </a>

<a id="smolvlmhf-2025" class="bib-item"> Marafioti, Andrés, et al. "Smolvlm: Redefining small and efficient multimodal models." arXiv preprint arXiv:2504.05299 (2025). </a>

<a id="gemma-3-2025" class="bib-item">Gemma-Team. (2025). Gemma 3 Technical Report. arXiv preprint arXiv:2503.19786.</a>

<a id="qwen2-5-vl-2025" class="bib-item" style="display: block; margin-bottom: 10px;">Bai Shuai, Chen Keqin, Liu Xuejing, Wang Jialin, Ge Wenbin, Song Sibo, Dang Kai, Wang Peng, Wang Shijie, Tang Jun, Zhong Humen, Zhu Yuanzhi, Yang Mingkun, Li Zhaohai, Wan Jianqiang, Wang Pengfei, Ding Wei, Fu Zheren, Xu Yiheng, Ye Jiabo, Zhang Xi, Xie Tianbao, Cheng Zesen, Zhang Hang, Yang Zhibo, Xu Haiyang, Lin Junyang. (2025). Qwen2.5-VL Technical Report. arXiv preprint arXiv:2502.13923.</a>

<a id="fastvlms-2025" class="bib-item"> Vasu, Pavan Kumar Anasosalu, et al. "Fastvlm: Efficient vision encoding for vision language models." Proceedings of the Computer Vision and Pattern Recognition Conference. 2025. </a>

<a id="vlminf-2026" class="bib-item"> Barrios, Wayner. "Native LLM and MLLM Inference at Scale on Apple Silicon." arXiv preprint arXiv:2601.19139 (2026).</a>

<a id="gemma4-2026" class="bib-item"> Farabet, Clement, and Olivier Lacombe. "Gemma 4: Byte for byte, the most capable open models." Google DeepMind, 2 Apr. 2026, https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/.</a>

</div>

<style>
  /* Hide all references by default */
  .bib-item { display: none; }
  /* Show only the ones with the 'cited' class */
  .bib-item.cited { display: block; margin-bottom: 10px; }
</style>

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