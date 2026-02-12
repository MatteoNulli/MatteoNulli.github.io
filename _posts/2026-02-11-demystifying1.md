---
layout: post
title: "<small>De-mystifying Multimodal Learning</small><br><b>The Hidden Inefficiency in Vision Language Modelling</b>"
date: 2026-02-30 14:14:00
description: A blogpost series on the nuts and bolts of Multimodal Learning
tags: Multimodal-Learning Vision-Language-Modelling
# thumbnail: assets/img/mllms_visual_tokens_wide.png
thumbnail: assets/img/token_comparison_paper_revised.png
mathjax: true
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
###### 📝 [Blogpost](https://matteonulli.github.io/blog/2025/vlmsecom/)
<br>


<script>
  window.MathJax = {
    tex: {
      inlineMath: [['$', '$'], ['\\(', '\\)']],
      displayMath: [['\\[', '\\]']],
    },
    svg: { fontCache: 'global' }
  };
</script>
<script id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js">
</script>

<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"
        onload="renderMathInElement(document.body, {
          delimiters: [
            {left: '$$', right: '$$', display: true},
            {left: '\\[', right: '\\]', display: true},
            {left: '$',  right: '$',  display: false},
            {left: '\\(', right: '\\)', display: false}
          ],
          // keep default ignore list so code/pre are skipped
        });"></script>

## Introduction
In the shift from text-only models to Vision Language Models (VLMs), we often talk about "parameters" and "emergent reasoning." However, there is a hidden currency that governs the performance, cost, and feasibility of these systems: **Visual Tokens (VT)**.

While Large Language Models (LLMs) are natively blind, they "see" by consuming images that have been decomposed, projected, and flattened into a format they can digest. This transformation isn't free. Whether you are building a real-time assistant or an OCR pipeline, the number of visual tokens your model generates is the primary lever for inference latency, VRAM consumption, and context window management. Understanding the computational overhead of these tokens is no longer just an academic exercise—it is a production necessity.

In the [our previous blogpost](https://matteonulli.github.io/blog/2026/demystifying0/), we explored the architectural anatomy of VLMs and how images are converted into language-compatible vectors. In this second installment of `De-mystifying Multimodal Learning` we focus on the mathematics and operational impact of that conversion. Specifically, we will cover:<br>

- [Calculating Visual Tokens](#calculating-visual-tokens): Presenting a practical guide to estimating token counts across different SOTA strategies—from Qwen’s dynamic merging to LLaVA’s high-resolution grids—without running a single line of inference.

- [Impact of Token Count](#visual-token-count-impact): Analysing of how these tokens impact the "three pillars" of production: Context Windows, Latency, and VRAM.

<a id="overview"></a>
<div class="row mt-3">
  <div class="col-12">
    <div class="mx-auto text-center" style="width: 80%;">
      {% include figure.liquid loading="eager" path="./assets/img/vlm_tokens.png" class="img-fluid rounded z-depth-1" %}
      <div class="caption text-center mt-2">
        Figure 1: <b>Overview of the Vision Token count estimation process.</b> The estimated counts are calculated based on Spatial Merge Size of 2, AnyResolution of 3x3 windows and Spatial Average Pooling of size 4. C, H, W represent the channels, heigth and width of the image, whereas N and P are the number of patches and the patch dimension respectively. F and D are embedding dimension sizes outputted from the vision encoder and the MLP and V is the number of visual tokens, which depends on the type of encoder, MLP and the re-sizing of images, i.e. the actual H and W. In <a href="#calculating-visual-tokens">the section below</a>, we dive deeper into the difference between each of the models reported on the right.
      </div>
    </div>
  </div>
</div>

<br>

## Calculating #Visual Tokens

As discussed in [our previous blogpost](https://matteonulli.github.io/blog/2026/demystifying0/), Visual Tokens are the fundamental units that allow LLMs to perceive visual data. Now that we understand the "what," we must address the "how much."

<p align="center"><code>How many Visual Tokens do VLMs produce given an image input size?</code></p>


#### Original Recipe

Within the first VLM architectures ([Liu et al., 2023](#visual-instruction-tuning-2023)) this estimation is straightforward. 
First generation VLMs relied on Vision Encoder which have a fixed input resolution and a patch size ($PS$). What this means is that whatever its input size, the picture would always be re-scaled to $H \times W$. This implies that, given both $H$ and $W$, the final number of visual tokens, i.e. dimenson $V$ 
<p align="center"> \[ V_{\text{original}} = (H/PS) \times (W/PS)\] </p>

#### The Resolution Trap

This came with several problems: 
1. Images resolutions were completely disregarded. Having the same amount of tokens for images of size 336^2 and 1024^2 does not make sense.
2. Not only does not make sense, it also does not work. Especially for OCR, visual compositional reasoning and small object detection tasks. 

However, simply making vision encoders which could support higher resolutions was also not something feasible. First, because now every image would have needed to be re-sized to say $1024\times1024$. Secondly, *tripling* the supported image heigth and keeping the same patch size results in almost *10x the amount of visual tokens*.


#### Modern Approaches 

**Strategy A: The Dynamic Merger**
We have to start with the game-changers: the QwenVL-2.5 and 3 series ([Bai et al. 2025](#qwen2-5-vl-2025), [QwenTeam, 2025](#qwen3-vl-2025)). These models ditched the "fixed resolution" rule entirely. Instead of squashing every image into a square, they process images at their native resolution. This sounds great, but it complicates our math: if the image size varies, so does the token count. To calculate it, we need a specific value from the model's config.json called the Spatial Merge Size ($SMS$). Think of $SMS$ as a compression factor—it tells the model how many raw image patches to pool together into one visual token.With this in mind, our formula becomes a bit more dynamic:

<p align="center"> \[ V_{\text{Qwen3}} = (H / (PS \cdot SMS)) \times (W/ (PS \cdot SMS)) \] </p>

<u>The Takeaway</u>: **perfect aspect ratios without distortion**.<br> 
<u>The Catch</u>: Large images (or several of them) can silently eat up your context window much faster than you expect.

**Strategy B: The Multi-Grid / AnyRes** 
Around the same period LLaVA-Next/OneVision ([Liu et al., 2024](#llava-next-2024), [Li et al., 2024b](#llavonevision-2024)) came up with a clever, yet expensive encoding technique called "Dynamic-High Resolution"/"Any Resolution". Depicted in [Figure 2](#high_res), it consists of splitting the image into $k\times k$ grids, with $k \in \{1, 9\}$ before the vision encoding.
This means repeating the encoding process $(k \times k) + 1$ times, with the 1 being the picture in its entirety. 

<a id="high_res"></a>
<div class="row mt-3">
  <div class="col-12">
    <div class="mx-auto text-center" style="width: 80%;">
      {% include figure.liquid loading="eager" path="./assets/img/high_res.png" class="img-fluid rounded z-depth-1" %}
      <div class="caption text-center mt-2">
        Figure 2: Illustration of Dynamic High Resolution on 2x2 grid from <a href="#llava-next-2024">LLaVA-NeXT</a> paper. 
      </div>
    </div>
  </div>
</div>

Although this results higher detail understanding given the entire focus of the encoder on smaller portions of the image, it also most crucially implies an enormous increase in Visual Token count. Given the calculations in the [original recipe](#original-recipe), we have 

<p align="center"> \[V_{\text{LLaVA-OneVision}} = V_{\text{original}} * [(k \times k) + 1] \] </p> 

In a couple of words a big trade-off: <u>Massive detail vs. Massive token count.</u>

**Strategy C: The Fixed Downsampler**
Gemma3 ([Gemma-Team, 2025](#gemma-3-2025)) family of models, the most recent open source VLM from gdm also employes a fixed input sized Vision Encoder SigLIP ([Zhai et al., 2024](#siglip-2024)).

The main difference between their technique and [Strategy A](#Strategy-a-The-Dynamic-Merger) is the easy but clever solution to handle higher resolution images, applying a spatial average pooling and therefore increasing the resizing input size of the model to 896. Thanks to the pooling, this yields a fixed amount of visual tokens which corresponds to 

<p align="center"> \[V_{\text{Gemma3}} = (H/(PS*\text{pooling})) \times (W/(PS*\text{pooling})) = (896/(14*4))^2 = 256 \] </p> 

with the pooling being applied within the modality connector.

<a id="token-count"></a>
<div class="row mt-3">
  <div class="col-12">
    <div class="mx-auto text-center" style="width: 60%;">
      {% include figure.liquid loading="eager" path="./assets/img/token_comparison_paper_revised.png" class="img-fluid rounded z-depth-1" %}
      <div class="caption text-center mt-2">
        Figure 2: We report the increase of Visual token count given the image resolution input. The image is assumed AnyRes assumes a 3x3 grid and the SMS and pooling are equal to 2. 
      </div>
    </div>
  </div>
</div>

A common denominator in all of these is the special tokens, which are added for every picturem signaling the beginning and end of the visual content. 


<br>

## Visual Token Count Impact

We have defined what a Visual Token (VT) is and established formulas to calculate $V$ for different architectures. 

<p align="center"><code>But why does this specific number matter? Why should a machine learning engineer care if an image is represented by 256 tokens (Strategy C) or 2,500 tokens (Strategy B)?</code></p>

The answer lies in the constraints of production environments: **Context Windows**, **Latency**, and **VRAM**.

**1. The Context Window Budget**

Every LLM has a hard limit on its input size, denoted as the Context Window ($L_{\text{max}}$). In text-only models, this budget is consumed solely by the system prompt, user history and input prompt. In VLMs, Visual Tokens consume this budget aggressively. If we denote the available context for reasoning and history as $C_{\text{text}}$, the relationship is effectively zero-sum: $$C_{\text{text}} = L_{\text{max}} - \sum_{i=1}^{N} V_{i}$$ 
where $N$ is the number of images.

For a "Strategy B" model (like LLaVA-Next) using a $3 \times 3$ grid, a single image might consume $\approx 2900$ tokens. If you are serving a model with a 4k or 8k context limit, a single image can consume 30-70% of your entire input capacity. 
This leaves little room for few-shot examples or long conversation history, potentially degrading the model's ability to follow complex instructions.


**2. Inference Latency:** The "Pre-fill" BottleneckIn production use-cases, inference cost is often a function of latency. Large companies typically enforce fixed input sizes to ensure predictable response times. Visual Tokens disrupt this predictability. When a VLM processes a request, it undergoes two phases: 
a. Pre-fill: The model processes all input tokens (Text + Visual) in parallel to compute the Key-Value (KV) cache.
b. Decoding: The model generates the output one token at a time. 
Visual Tokens sit in the Pre-fill phase. A high VT count dramatically increases the Time To First Token (TTFT). $$\text{Latency}_{\text{prefill}} \propto f(V_{\text{total}})^2 \quad (\text{for standard attention})$$
Even with linear attention optimizations (like FlashAttention), processing 3,000 visual tokens requires significantly more floating-point operations (FLOPs) than processing 256. If your application requires real-time responsiveness (e.g., a voice assistant "seeing" via camera), the difference between 50ms and 500ms in pre-fill latency is a dealbreaker.

<a id="high_res"></a>
<div class="row mt-3">
  <div class="col-12">
    <div class="mx-auto text-center" style="width: 60%;">
      {% include figure.liquid loading="eager" path="./assets/img/apple_latency_vlms.png" class="img-fluid rounded z-depth-1" %}
      <div class="caption text-center mt-2">
        Figure 3: Figure from <a href="#fastvlms-2025">FastVLM</a> paper from Apple MLR, illustrating how Vision latency dominates at high resolution. Breakdown of FastVLM’s time to the first token for different image resolutions. The vision encoder is <a href="#fastvlms-2025">FastViT-HD</a>, and the LLM has 1.5B parameters. 
      </div>
    </div>
  </div>
</div>

**3. The Cascading Impact on VRAM**
Perhaps the most critical "hidden" cost is memory. When serving models using high-performance engines like [vLLM](https://github.com/vllm-project/vllm) or [TensorRT-LLM](https://developer.nvidia.com/tensorrt-llm), the throughput is bound by how many requests can fit into the GPU memory simultaneously (batch size).
This depends heavily on the KV Cache—the memory required to store the attention context for every token in the sequence. Higher VT Count $\rightarrow$ Larger KV Cache per request.Larger KV Cache $\rightarrow$ Fewer requests fit in VRAM. Fewer Requests $\rightarrow$ Smaller Batch Size.T his creates a cascading effect on cost.

If a high-resolution strategy increases your visual tokens by $10\times$, you might be forced to reduce your batch size by roughly the same factor to avoid Out-Of-Memory (OOM) errors. This effectively multiplies your cost per inference, as you need more GPUs to handle the same traffic.4. Operational NecessityThis brings us back to the premise of this post. For experimental research, we often ignore these overheads in favor of higher benchmarks. But for production deployment, *predictability is king*. To optimize serving, engineers need to:
- Dynamically adjust expectations: If a client uploads a high-res image to a "Strategy A" (Dynamic) model, the system must instantly calculate $V_{\text{Qwen}}$ to check if it fits the current VRAM budget.
- Downsample intelligently: If $V_{\text{calculated}}$ exceeds the limit, the system needs to resize the input image before it hits the Vision Encoder to meet a specific token target.
This is why the formulas in Part 2 are not just theoretical trivia—they are essential tools for building robust, cost-effective Multimodal systems.


## Conclusions & Key Takeaways

<div class="row mt-3">
    <div class="col-12">
        <div class="table-responsive">
            <table class="table table-hover table-bordered">
                <thead class="thead-light">
                    <tr>
                        <th scope="col">Representative Models</th>
                        <th scope="col">Strategy</th>
                        <th scope="col">Resolution Logic</th>
                        <th scope="col">Token Efficiency</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td class="font-weight-bold">LLaVA1.5</td>
                        <td>Standard Resize</td>
                        <td>Squash to fixed $H \times W$</td>
                        <td><span class="badge badge-secondary">Fixed Count</span></td>
                    </tr>
                    <tr>
                        <td class="font-weight-bold">Qwen3-VL</td>
                        <td>Dynamic Merger</td>
                        <td>Native (Preserves Aspect Ratio)</td>
                        <td><span class="badge badge-warning">Linear Growth</span></td>
                    </tr>
                    <tr>
                        <td class="font-weight-bold">LLaVA-OneVision</td>
                        <td>AnyRes / Multi-Grid</td>
                        <td>Grid Split ($k \times k$) + Overview</td>
                        <td><span class="badge badge-danger">High Cost</span></td>
                    </tr>
                    <tr>
                        <td class="font-weight-bold">Gemma3</td>
                        <td>Fixed Downsampler</td>
                        <td>Resize + Spatial Pooling</td>
                        <td><span class="badge badge-success">Highly Compact</span></td>
                    </tr>
                </tbody>
            </table>
        </div>
        <div class="caption text-center mt-2">
            Table 1: We report a comparison of Visual Token calculation strategies across modern architectures in current SOTA VLMs.
        </div>
    </div>
</div>

Visual Tokens (VT) are the bridge between the image and language world, but they are also the primary bottleneck in VLM deployment. As we have seen, moving from a fixed-resolution model (like Gemma 3) to a dynamic one (like Qwen 2.5-VL or LLaVA-Next) can increase your input size by an order of magnitude.Here are the key takeaways to keep in mind when building multimodal systems:
- Tokens $\neq$ Pixels: High resolution doesn't always mean high cost. It depends entirely on the architecture (e.g., Fixed Downsampler vs. Multi-Grid).
- The "Pre-fill" Trap: Visual tokens are processed before the first word is generated. If your latency is high, check your image resolution before checking your LLM size.Context is Zero-Sum: Every visual token you use is one less token available for conversation history or few-shot examples.
- Calculate, Don't Guess: Use the formulas provided in Part 2 to pre-calculate token counts. This allows you to dynamically resize images or adjust batch sizes to prevent OOM errors in production. 

Multimodal learning is evolving rapidly, but the fundamental constraint remains: compute is finite. Mastering the math of Visual Tokens is the first step toward mastering VLM efficiency.


<!-- ## Additional Questions to Answer -->

<!-- D. What is the cost trade-off? -->

<!-- Context: LLaVA-OneVision uses many more tokens. -->

<!-- Question: "If Model A uses 256 tokens and Model B uses 2,800 tokens for the same image, what is the impact on 'Time to First Token' (latency) and VRAM?" -->



## Citation

If you use this work, please cite:

```bibtex
@misc{nulli2026demistifying,
title={De-mystifying Multimodal Learning: The Hidden Cost in Vision Language Modelling.},
author={Matteo Nulli},
year={2026},
url={https://matteonulli.github.io/blog/2026/demystifying1/}}
```
<br>

**References**
<div id="references-section">

<a id="conme-2024" class="bib-item">Huang Irene, Lin Wei, Mirza M. Jehanzeb, Hansen Jacob A., Doveh Sivan, Butoi Victor Ion, Herzig Roei, Arbelle Assaf, Kuehne Hilde, Darrell Trevor, Gan Chuang, Oliva Aude, Feris Rogerio, Karlinsky Leonid. (2024). Conme: Rethinking Evaluation of Compositional Reasoning for Modern VLMs. arXiv preprint arXiv:2406.08164.</a>

<a id="eyes-wide-shut-2024" class="bib-item">Tong Shengbang, Liu Zhuang, Zhai Yuexiang, Ma Yi, LeCun Yann, Xie Saining. (2024). Eyes Wide Shut? Exploring the Visual Shortcomings of Multimodal LLMs. arXiv preprint arXiv:2401.06209.</a>

<a id="visual-instruction-tuning-2023" class="bib-item">Liu Haotian, Li Chunyuan, Wu Qingyang, Lee Yong Jae. (2023). Visual Instruction Tuning. arXiv preprint arXiv:2304.08485.</a>

<a id="llavonevision-2024" class="bib-item">Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. 2024a. Llava-onevision: Easy visual task transfer. Preprint, arXiv:2408.03326.</a>

<a id="qwen2-5-vl-2025" class="bib-item">Bai Shuai, Chen Keqin, Liu Xuejing, Wang Jialin, Ge Wenbin, Song Sibo, Dang Kai, Wang Peng, Wang Shijie, Tang Jun, Zhong Humen, Zhu Yuanzhi, Yang Mingkun, Li Zhaohai, Wan Jianqiang, Wang Pengfei, Ding Wei, Fu Zheren, Xu Yiheng, Ye Jiabo, Zhang Xi, Xie Tianbao, Cheng Zesen, Zhang Hang, Yang Zhibo, Xu Haiyang, Lin Junyang. (2025). Qwen2.5-VL Technical Report. arXiv preprint arXiv:2502.13923.</a>

<a id="qwen3-vl-2025" class="bib-item">QwenTeam. 2025. Qwen3-vl: Sharper vision, deeper thought, broader action.</a>

<a id="internvl2-2024" class="bib-item">OpenGVLab-Team. (2024). InternVL2: Better Than the Best—Expanding Performance Boundaries of Open-Source Multimodal Models with the Progressive Scaling Strategy. Blog post. URL https://internvl.github.io/blog/2024-07-02-InternVL-2.0/.</a>

<a id="gemma-3-2025" class="bib-item">Gemma-Team. (2025). Gemma 3 Technical Report. arXiv preprint arXiv:2503.19786.</a>

<a id="bags-of-words-vlms-2023" class="bib-item">Yuksekgonul Mert, Bianchi Federico, Kalluri Pratyusha, Jurafsky Dan, Zou James. (2023). When and Why Vision-Language Models Behave Like Bags-of-Words, and What to Do About It? arXiv preprint arXiv:2210.01936.</a>

<a id="icl-compositional-vlms-2024" class="bib-item">Nulli Matteo, Ibrahimi Anesa, Pal Avik, Lee Hoshe, Najdenkoska Ivona. (2024). In-Context Learning Improves Compositional Understanding of Vision-Language Models. In ICML 2024 Workshop on Foundation Models in the Wild. arXiv preprint arXiv:2407.15487.</a>

<a id="nulliobjectguided-2025" class="bib-item">Matteo Nulli, Ivona Najdenkoska, Mohammad Mahdi Derakhshani, and Yuki M Asano. 2025. Objectguided visual tokens: Eliciting compositional reasoning in multimodal language models. In EurIPS 2025 Workshop on Principles of Generative Modeling (PriGM)</a>

<a id="vismin-2025" class="bib-item">Awal Rabiul, Ahmadi Saba, Zhang Le, Agrawal Aishwarya. (2025). Vismin: Visual Minimal-Change Understanding. arXiv preprint arXiv:2407.16772.</a>

<a id="cambrian-1-2024" class="bib-item">Tong Shengbang, Brown Ellis, Wu Penghao, Woo Sanghyun, Middepogu Manoj, Akula Sai Charitha, Yang Jihan, Yang Shusheng, Iyer Adithya, Pan Xichen, Wang Austin, Fergus Rob, LeCun Yann, Xie Saining. (2024). Cambrian-1: A Fully Open, Vision-Centric Exploration of Multimodal LLMs. arXiv preprint arXiv:2406.16860.</a>

<a id="llava-next-2024" class="bib-item">Liu Haotian, Li Chunyuan, Li Yuheng, Li Bo, Zhang Yuanhan, Shen Sheng, Lee Yong Jae. (2024). LLaVA-NeXT: Improved Reasoning, OCR, and World Knowledge. Blog post (January 2024). URL https://llava-vl.github.io/blog/2024-01-30-llava-next/.</a>

<a id="sam-2-2024" class="bib-item">Ravi Nikhila, Gabeur Valentin, Hu Yuan-Ting, Hu Ronghang, Ryali Chaitanya, Ma Tengyu, Khedr Haitham, Rädle Roman, Rolland Chloe, Gustafson Laura, Mintun Eric, Pan Junting, Alwala Kalyan Vasudev, Carion Nicolas, Wu Chao-Yuan, Girshick Ross, Dollár Piotr, Feichtenhofer Christoph. (2024). SAM 2: Segment Anything in Images and Videos. arXiv preprint arXiv:2408.00714.</a>

<a id="omg-seg-cvpr-2024" class="bib-item">Li Xiangtai, Yuan Haobo, Li Wei, Ding Henghui, Wu Size, Zhang Wenwei, Li Yining, Chen Kai, Loy Chen Change. (2024). OMG-Seg: Is One Model Good Enough for All Segmentation? In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 27948–27959.</a>

<a id="eagle-2-5-2025" class="bib-item">Chen Guo, Li Zhiqi, Wang Shihao, Jiang Jindong, Liu Yicheng, Lu Lidong, Huang De-An, Byeon Wonmin, Le Matthieu, Rintamaki Tuomas, Poon Tyler, Ehrlich Max, Lu Tong, Wang Limin, Catanzaro Bryan, Kautz Jan, Tao Andrew, Yu Zhiding, Liu Guilin. (2025). EAGLE 2.5: Boosting Long-Context Post-Training for Frontier Vision-Language Models. arXiv preprint arXiv:2504.15271.</a>

<a id="omg-llava-2024" class="bib-item">Zhang Tao, Li Xiangtai, Fei Hao, Yuan Haobo, Wu Shengqiong, Ji Shunping, Loy Chen Change, Yan Shuicheng. (2024). OMG-LLaVA: Bridging Image-Level, Object-Level, Pixel-Level Reasoning and Understanding. arXiv preprint arXiv:2406.19389.</a>

<a id="sa2va-2025" class="bib-item">Yuan Haobo, Li Xiangtai, Zhang Tao, Huang Zilong, Xu Shilin, Ji Shunping, Tong Yunhai, Qi Lu, Feng Jiashi, Yang Ming-Hsuan. (2025). SA2VA: Marrying SAM2 with LLaVA for Dense Grounded Understanding of Images and Videos. arXiv preprint arXiv:2501.04001.</a>

<a id="clip-2021" class="bib-item">Radford Alec, Kim Jong Wook, Hallacy Chris, Ramesh Aditya, Goh Gabriel, Agarwal Sandhini, Sastry Girish, Askell Amanda, Mishkin Pamela, Clark Jack, Krueger Gretchen, Sutskever Ilya. (2021). Learning Transferable Visual Models from Natural Language Supervision. arXiv preprint arXiv:2103.00020.</a>

<a id="improved-vit-baselines-2024" class="bib-item">Liu Haotian, Li Chunyuan, Li Yuheng, Lee Yong Jae. (2024). Improved Baselines with Visual Instruction Tuning. arXiv preprint arXiv:2310.03744.</a>

<a id="vit-2021" class="bib-item">Dosovitskiy Alexey, Beyer Lucas, Kolesnikov Alexander, Weissenborn Dirk, Zhai Xiaohua, Unterthiner Thomas, Dehghani Mostafa, Minderer Matthias, Heigold Georg, Gelly Sylvain, Uszkoreit Jakob, Houlsby Neil. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. arXiv preprint arXiv:2010.11929.</a>

<a id="llama-2-2023" class="bib-item">Touvron Hugo, Martin Louis, Stone Kevin, Albert Peter, Almahairi Amjad, Babaei Yasmine, Bashlykov Nikolay, Batra Soumya, Bhargava Prajjwal, Bhosale Shruti, Bikel Dan, Blecher Lukas, Canton Ferrer Cristian, Chen Moya, Cucurull Guillem, Esiobu David, Fernandes Jude, Fu Jeremy, Fu Wenyin, Fuller Brian, Gao Cynthia, Goswami Vedanuj, Goyal Naman, Hartshorn Anthony, Hosseini Saghar, Hou Rui, Inan Hakan, Kardas Marcin, Kerkez Viktor, Khabsa Madian, Kloumann Isabel, Korenev Artem, Koura Punit Singh, Lachaux Marie-Anne, Lavril Thibaut, Lee Jenya, Liskovich Diana, Lu Yinghai, Mao Yuning, Martinet Xavier, Mihaylov Todor, Mishra Pushkar, Molybog Igor, Nie Yixin, Poulton Andrew, Reizenstein Jeremy, Rungta Rashi, Saladi Kalyan, Schelten Alan, Silva Ruan, Smith Eric Michael, Subramanian Ranjan, Tan Xiao-qing Ellen, Tang Binh, Taylor Ross, Williams Adina, Kuan Jian Xiang, Xu Puxin, Yan Zheng, Zarov Iliyan, Zhang Yuchen, Fan Angela, Kambadur Melanie, Narang Sharan, Rodriguez Aurelien, Stojnic Robert, Edunov Sergey, Scialom Thomas. (2023). Llama 2: Open Foundation and Fine-Tuned Chat Models.</a>

<a id="llama-3-2-2024" class="bib-item">Meta. (2024). Llama 3.2: Revolutionizing Edge AI and Vision with Open, Customizable Models. Blog post. URL https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/.</a>

<a id="lora-2021" class="bib-item">Hu Edward J., Shen Yelong, Wallis Phillip, Allen-Zhu Zeyuan, Li Yuanzhi, Wang Shean, Wang Lu, Chen Weizhu. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv preprint arXiv:2106.09685.</a>

<a id="coco-2014" class="bib-item">Lin Tsung-Yi, Maire Michael, Belongie Serge, Hays James, Perona Pietro, Ramanan Deva, Dollár Piotr, Zitnick C. Lawrence. (2014). Microsoft COCO: Common Objects in Context. In Computer Vision – ECCV 2014, pages 740–755. Springer.</a>

<a id="image-descriptions-2014" class="bib-item">Young Peter, Lai Alice, Hodosh Micah, Hockenmaier Julia. (2014). From Image Descriptions to Visual Denotations: New Similarity Metrics for Semantic Inference Over Event Descriptions. Transactions of the Association for Computational Linguistics, 2:67–78.</a>

<a id="visual-genome-2017" class="bib-item">Krishna Ranjay, Zhu Yuke, Groth Oliver, Johnson Justin, Hata Kenji, Kravitz Joshua, Chen Stephanie, Kalantidis Yannis, Li Li-Jia, Shamma David A., et al. (2017). Visual Genome: Connecting Language and Vision Using Crowdsourced Dense Image Annotations. International Journal of Computer Vision, 123:32–73.</a>

<a id="gqa-2019" class="bib-item">Hudson Drew A., Manning Christopher D. (2019). GQA: A New Dataset for Real-World Visual Reasoning and Compositional Question Answering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 6700–6709.</a>

<a id="sugar-crepe-2023" class="bib-item">Hsieh Cheng-Yu, Zhang Jieyu, Ma Zixian, Kembhavi Aniruddha, Krishna Ranjay. (2023). SugarCrepe: Fixing Hackable Benchmarks for Vision-Language Compositionality. Advances in Neural Information Processing Systems, 36:31096–31116.</a>

<a id="gpt-4-technical-report-2024" class="bib-item">OpenAI. (2024). GPT-4 Technical Report. arXiv preprint arXiv:2303.08774.</a>

<a id="scaling-instruction-finetuned-2024" class="bib-item">Chung Hyung Won, Hou Le, Longpre Shayne, Zoph Barret, Tay Yi, Fedus William, Li Yunxuan, Wang Xuezhi, Dehghani Mostafa, Brahma Siddhartha, et al. (2024). Scaling Instruction-Finetuned Language Models. Journal of Machine Learning Research, 25(70):1–53.</a>

<a id="diagram-2016" class="bib-item">Kembhavi Aniruddha, Salvato Mike, Kolve Eric, Seo Minjoon, Hajishirzi Hannaneh, Farhadi Ali. (2016). A Diagram is Worth a Dozen Images. arXiv preprint arXiv:1603.07396.</a>

<a id="mme-2024" class="bib-item">Fu Chaoyou, Bird Peixian, Shen Yunhang, Qin Yulei, Zhang Mengdan, Lin Xu, Yang Jinrui, Zheng Xiawu, Li Ke, Sun Xing, Wu Yunsheng, Ji Rongrong. (2024). MME: A Comprehensive Evaluation Benchmark for Multimodal Large Language Models. arXiv preprint arXiv:2306.13394.</a>

<a id="evaluating-vlms-right-way-2024" class="bib-item">Chen Lin, Li Jinsong, Dong Xiaoyi, Zhang Pan, Zang Yuhang, Chen Zehui, Duan Haodong, Wang Jiaqi, Qiao Yu, Lin Dahua, Zhao Feng. (2024). Are We on the Right Way for Evaluating Large Vision-Language Models? arXiv preprint arXiv:2403.20330.</a>

<a id="mmbench-2024" class="bib-item">Liu Yuan, Duan Haodong, Zhang Yuanhan, Li Bo, Zhang Songyang, Zhao Wangbo, Yuan Yike, Wang Jiaqi, He Conghui, Liu Ziwei, Chen Kai, Lin Dahua. (2024). MMBench: Is Your Multi-Modal Model an All-Around Player? arXiv preprint arXiv:2307.06281.</a>

<a id="subobject-tokenization-2025" class="bib-item">Chen Delong, Cahyawijaya Samuel, Liu Jianfeng, Wang Baoyuan, Fung Pascale. (2025). Subobject-Level Image Tokenization. arXiv preprint arXiv:2402.14327.</a>

<a id="deepspeed-2020" class="bib-item">Rasley Jeff, Rajbhandari Samyam, Ruwase Olatunji, He Yuxiong. (2020). DeepSpeed: System Optimizations Enable Training Deep Learning Models with Over 100 Billion Parameters. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD ’20), pages 3505–3506. doi:10.1145/3394486.3406703.</a>

<a id="zero-2020" class="bib-item">Rajbhandari Samyam, Rasley Jeff, Ruwase Olatunji, He Yuxiong. (2020). ZeRO: Memory Optimizations Toward Training Trillion Parameter Models. In SC20: International Conference for High Performance Computing, Networking, Storage and Analysis, pages 1–16. doi:10.1109/SC41405.2020.00024.</a>

<a id="adam-2017" class="bib-item">Kingma Diederik P., Ba Jimmy. (2017). Adam: A Method for Stochastic Optimization. arXiv preprint arXiv:1412.6980.</a>

<a id="adamw-2019" class="bib-item">Loshchilov Ilya, Hutter Frank. (2019). Decoupled Weight Decay Regularization. arXiv preprint arXiv:1711.05101.</a>

<a id="bert-2019" class="bib-item">Devlin Jacob, Chang Ming-Wei, Lee Kenton, Toutanova Kristina. (2019). BERT: Pre-Training of Deep Bidirectional Transformers for Language Understanding. In Proceedings of NAACL-HLT 2019, pages 4171–4186.</a>

<a id="attention-is-all-you-need-2017" class="bib-item">Vaswani Ashish, Shazeer Noam, Parmar Niki, Uszkoreit Jakob, Jones Llion, Gomez Aidan N., Kaiser Łukasz, Polosukhin Illia. (2017). Attention Is All You Need. Advances in Neural Information Processing Systems, 30.</a>

<a id="pixtral-12b-2024" class="bib-item">Agrawal Pravesh, Antoniak Szymon, Bou Hanna Emma, Bout Baptiste, Chaplot Devendra, Chudnovsky Jessica, Costa Diogo, De Monicault Baudouin, Garg Saurabh, Gervet Theophile, Ghosh Soham, Héliou Amélie, Jacob Paul, Jiang Albert Q., Khandelwal Kartik, Lacroix Timothée, Lample Guillaume, Las Casas Diego, Lavril Thibaut, Le Scao Teven, Lo Andy, Marshall Louis, Martin Arthur, Mensch Arthur, Muddireddy Pavankumar, Nemychnikova Valera, Pellat Marie, Von Platen Patrick, Raghuraman Nikhil, Bout Rozière Baptiste, Sablayrolles Alexandre, Saulnier Lucile, Sauvestre Romain, Rozière Baptiste, Shang Wendy, Soletskyi Roman, Stewart Lawrence, Stock Pierre, Studnia Joachim, Subramanian Sandeep, Vaze Sagar, Wang Thomas, Yang Sophia. (2024). Pixtral 12B. arXiv preprint arXiv:2410.07073.</a>

<a id="roformer-2023" class="bib-item">Su Jianlin, Lu Yu, Pan Shengfeng, Murtadha Ahmed, Wen Bo, Liu Yunfeng. (2023). RoFormer: Enhanced Transformer with Rotary Position Embedding. arXiv preprint arXiv:2104.09864.</a>

<a id="blip2-2023" class="bib-item">Li J, Li D, Savarese S, Hoi S. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. InInternational conference on machine learning 2023</a>

<a id="llama-3-herd-2024" class="bib-item">Dubey Abhimanyu, et al. (2024). The Llama 3 Herd of Models. arXiv preprint arXiv:2407.21783.</a>

<a id="reproducible-scaling-laws-2023" class="bib-item">Cherti Mehdi, Beaumont Romain, Wightman Ross, Wortsman Mitchell, Ilharco Gabriel, Gordon Cade, Schuhmann Christoph, Schmidt Ludwig, Jitsev Jenia. (2023). Reproducible Scaling Laws for Contrastive Language-Image Learning. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 2818–2829. doi:10.1109/CVPR52729.2023.00276.</a>

<a id="sigmoid-loss-2023" class="bib-item">Zhai Xiaohua, Mustafa Basil, Kolesnikov Alexander, Beyer Lucas. (2023). Sigmoid Loss for Language Image Pre-Training. arXiv preprint arXiv:2303.15343.</a>

<a id="dinov2-2024" class="bib-item">Oquab Maxime, Darcet Timothée, Moutakanni Théo, Vo Huy, Szafraniec Marc, Khalidov Vasil, Fernandez Pierre, Haziza Daniel, Massa Francisco, El-Nouby Alaaeldin, Assran Mahmoud, Ballas Nicolas, Galuba Wojciech, Misra Ishan, Rabbat Michael, Sharma Vasu, Synnaeve Gabriel, Xu Hu, Jegou Hervé, Mairal Julien, Labatut Patrick, Joulin Armand, Bojanowski Piotr. (2024). DINOv2: Learning Robust Visual Features Without Supervision. arXiv preprint arXiv:2304.07193.</a>

<a id="internlm2-2024" class="bib-item">Cai Zheng, Cao Maosong, Chen Haojiong, Chen Kai, Chen Keyu, Chen Xin, Chen Xun, Chen Zehui, Chen Zhi, Chu Pei, Dong Xiaoyi, Duan Haodong, Fan Qi, Fei Zhaoye, Gao Yang, Ge Jiaye, Gu Chenya, Gu Yuzhe, Gui Tao, Guo Aijia, Guo Qipeng, He Conghui, Hu Yingfan, Huang Ting, Jiang Tao, Jiao Penglong, Jin Zhenjiang, Lei Zhikai, Li Jiaxing, Li Jingwen, Li Linyang, Li Shuaibin, Li Wei, Li Yining, Liu Hongwei, Liu Jiawei, Liu Kaiwen, Liu Kuikun, Liu Xiaoran, Lv Chengqi, Lv Haijun, Lv Kai, Ma Li, Ma Runyuan, Ma Zerun, Ning Wenchang, Ouyang Linke, Qiu Jiantao, Qu Yuan, Shang Fukai, Shao Yunfan, Song Demin, Song Zifan, Sui Zhihao, Sun Peng, Sun Yu, Tang Huanze, Wang Bin, Wang Guoteng, Wang Jiaqi, Wang Jiayu, Wang Rui, Wang Yudong, Wang Ziyi, Wei Xingjian, Weng Qizhen, Wu Fan, Xiong Yingtong, Xu Chao, Xu Ruiliang, Yan Hang, Yan Yirong, Yang Xiaogui, Ye Haochen, Ying Huaiyuan, Yu Jia, Yu Jing, Zang Yuhang, Zhang Chuyu, Zhang Li, Zhang Pan, Zhang Peng, Zhang Ruijie, Zhang Shuo, Zhang Songyang, Zhang Wenjian, Zhang Wenwei, Zhang Xingcheng, Zhang Xinyue, Zhao Hui, Zhao Qian, Zhao Xiaomeng, Zhao Fengzhe, Zhou Zaida, Zhou Jingming, Zhuo Jingming, Zou Yicheng, Qiu Xipeng, Qiao Yu, Lin Dahua. (2024). InternLM2 Technical Report. arXiv preprint arXiv:2403.17297.</a>

<a id="omg-seg-arxiv-2024" class="bib-item">Li Xiangtai, Yuan Haobo, Li Wei, Ding Henghui, Wu Size, Zhang Wenwei, Li Yining, Chen Kai, Loy Chen Change. (2024). OMG-Seg: Is One Model Good Enough for All Segmentation? arXiv preprint arXiv:2401.10229.</a>

<a id="seem-2023" class="bib-item">Zou Xueyan, Yang Jianwei, Zhang Hao, Li Feng, Li Linjie, Wang Jianfeng, Wang Lijuan, Gao Jianfeng, Lee Yong Jae. (2023). Segment Everything Everywhere All at Once. arXiv preprint arXiv:2304.06718.</a>

<a id="siglip-2024" class="bib-item">Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, Lucas Beyer. Sigmoid Loss for Language Image Pre-Training, 2024. URL https://arxiv.org/abs/2303.15343.</a>

<a id="fastvlms-2025" class="bib-item">Vasu, Pavan Kumar Anasosalu, Fartash Faghri, Chun-Liang Li, Cem Koc, Nate True, Albert Antony, Gokula Santhanam et al. "Fastvlm: Efficient vision encoding for vision language models." In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 19769-19780. 2025.</a>

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