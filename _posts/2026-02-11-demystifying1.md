---
layout: post
title: "De-mystifying Multimodal Learning: The Hidden Inefficiency in Vision Language Modelling"
date: 2026-03-04 14:14:00
description: A blogpost series on the nuts and bolts of Multimodal Learning
tags: Multimodal-Learning Vision-Language-Modelling
# thumbnail: assets/img/mllms_visual_tokens_wide.png
thumbnail: https://cdn-uploads.huggingface.co/production/uploads/661d4e74b8f13412f6d48a50/Swoem2o9lRJO2Bs9G6e5Y.png
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
###### 🤗 [Comunity Article](https://huggingface.co/blog/MatteoNulli/de-mystifying-multimodal-learning-hidden-ineff/), 📝 [Blogpost](https://matteonulli.github.io/blog/2026/demystifying1/)
<br>



## Introduction
In the shift from text-only models to Vision Language Models (VLMs), we often talk about "parameters" and "emergent reasoning." However, there is a hidden currency that governs the performance, cost, and feasibility of these systems: **Visual Tokens (VT)**.

<a id="figure-1"></a>
<figure style="width: 70%; margin: auto; text-align: center;">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/661d4e74b8f13412f6d48a50/Swoem2o9lRJO2Bs9G6e5Y.png" 
       alt="token comparison" 
       style="width: 100%;">
  <figcaption style="margin-top: 10px; font-style: italic; color: #555;">
    Figure 1: <b>VLMs Token count vs Image Resolution.</b> We report the Visual Token count as a function of image resolution (in pixel count (px)) over four models LLaVA-1.5, LLaVA-OneVision, Qwen3VL, Gemma3. The estimated numbers are calculated based on Spatial Merge Size of 2, Any-Resolution with 3x3 windows and Spatial Average Pooling of size 4. For P&S method we assume the number of crops increase as a step function and start from same Any-Resolution configurations \( ^{*} \).
  </figcaption>
</figure>

<br>

In the [our previous blogpost](https://huggingface.co/blog/MatteoNulli/de-mystifying-multimodal-learning-enabiling-vision), we explored the architectural anatomy of VLMs and how images are converted into language-compatible vectors. In this second installment of `De-mystifying Multimodal Learning` we focus on the mathematics and operational impact of that conversion. Specifically, we will look at how to [Calculate Visual Tokens](#calculating-visual-tokens). Presenting a practical guide to estimating token counts across different SOTA strategies—from Qwen’s dynamic merging ([QwenTeam, 2025](#qwen3-vl-2025)), LLaVA’s Any-Res grids ([Li et al., 2024b](#llavonevision-2024)) and Gemma3 Pan&Scan ([Gemma-Team, 2025](#gemma-3-2025))—without running a single line of inference.

Understanding the computational overhead of these tokens is no longer just an academic exercise—it is a production necessity.


<small> \\( ^* \\) Please note that for LLaVa-OneVision and Gemma P&S the total number of visual tokens depends highliy on the amount of crops. </small>



## Calculating #Visual Tokens

As discussed in [our previous blogpost](https://huggingface.co/blog/MatteoNulli/de-mystifying-multimodal-learning-enabiling-vision), VTs are the fundamental units that allow LLMs to perceive visual data. Now that we understand the "what", we must address the "how much", answering the following:

<p align="center"><code>How many Visual Tokens do VLMs produce given an image input size?</code></p>

#### Original Recipe
Within the first VLM architectures like LLaVa-1.5 ([Liu et al., 2023](#visual-instruction-tuning-2023)) this estimation is straightforward. 
First generation VLMs relied on Vision Encoder with fixed input resolution and a patch size ( \\( PS \\) ). 
Mathematically, assume \\( H \\) and \\( W \\) to be the original image's heigth and width, and \\( X \\) and \\( Y \\) the resized dimension of the Vision Processor. 
In LLaVA-1.5, whatever the image resolution, the picture is always re-scaled to \\( X \times Y \\). Meaning the final number of VTs, i.e. dimension \\( V \\) from [our previous blogpost](https://huggingface.co/blog/MatteoNulli/de-mystifying-multimodal-learning-enabiling-vision) is:

$$ V_{\text{LLaVA-1.5}} = \frac{X}{PS} \times \frac{Y}{PS} $$

#### The Resolution Trap
This comes with several problems: 
1. Images resolutions are completely disregarded. Having the same amount of tokens for images of size 336^2 and 1024^2 does not make any sense.
2. Other than not making sense, it also does not work. For OCR, visual compositional reasoning and small object detection tasks, tasks accuracy is particularly low ([Yuksekgonul et al., 2023](#bags-of-words-vlms-2023), [Nulli et al., 2024](#icl-compositional-vlms-2024), [Tong et al., 2024](#eyes-wide-shut-2024), [Nulli et al., 2025](#nulliobjectguided-2025)). 

The simple solution of building vision encoders with higher resolution support is also not feasible. 
The fundamental issue stems from the fact that *Tripling* the resizing dimension of the processor <br> \\( X \times Y \leftrightarrow 336\times336 \rightarrow  1024\times1024 \\), would results in almost *10x* the amount of VT, making this prohibitive.

#### Modern Approaches 
Lets now look at newer approaches from Qwen2.5/3/3.5-VL, LLaVa-OneVision and Gemma3, overcoming these issues.

**Strategy A: The Dynamic Merger**<br>
We have to start with the game-changers: the Qwen2.5/3/3.5-VL series ([Bai et al. 2025](#qwen2-5-vl-2025), [QwenTeam, 2025](#qwen3-vl-2025)). These models ditched the "fixed resolution" rule entirely. Instead of squashing every image into a square, they process images at their native resolution. This sounds great, but it complicates our math: if the image size varies, so does the token count. To calculate it, we need a specific value from the model's `config.json` called the Spatial Merge Size ( \\( SMS \\) ). Think of \\( SMS \\) as a compression factor—it tells the model how many raw image patches to pool together into one VT. 
With this in mind, our formula becomes a bit more dynamic:

$$ V_{\text{Qwen3}} = \frac{H}{(PS \cdot SMS)} \times \frac{W}{(PS \cdot SMS)} $$

<u>Upside</u>: **perfect aspect ratios without distortion**.<br> 
<u>Downside</u>: large images (or several of them) can silently eat up your context window much faster than you expect.

**Strategy B: The Multi-Grid / AnyRes**<br> 
Around the same period LLaVA-Next/OneVision ([Liu et al., 2024](#llava-next-2024), [Li et al., 2024b](#llavonevision-2024)) came up with a clever, yet expensive encoding technique called "Dynamic-High Resolution"/"Any Resolution". Depicted in [Figure 2](#figure-2), it consists of splitting the image into \\(k\times k\\) grids, with \\(k \in \{1, 9\}\\) before the vision encoding.
This means repeating the encoding process \\((k \times k) + 1\\) times, with the 1 being the picture in its entirety. 

<a id="figure-2"></a>
<figure style="width: 70%; margin: auto; text-align: center;">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/661d4e74b8f13412f6d48a50/3-oqr7EOxK9iPxXqWQOuN.png" 
       alt="High res" 
       style="width: 100%;">
  <figcaption style="margin-top: 10px; font-style: italic; color: #555;">
    Figure 2: <b>Illustration of Dynamic High Resolution</b> on 2x2 grid from <a href="#llava-next-2024">LLaVA-NeXT</a> paper.
  </figcaption>
</figure>

Although this results higher detail understanding given the entire focus of the encoder on smaller portions of the image, it also most crucially implies an enormous increase in Visual Token count. Given the calculations in the [original recipe](#original-recipe), we have 

$$ V_{\text{LLaVA-OneVision}} = V_{\text{LLaVA-1.5}} \times [(k \times k) + 1] $$ 

<u>Upside</u>: **good quality for high-resolution inputs**.<br> 
<u>Downside</u>: massive increase in token count. Prohibitive for very high-resolution multi-image settings.

**Strategy C: Pan&Scan and Fixed Downsampler**<br>
Gemma3 ([Gemma-Team, 2025](#gemma-3-2025)) family of models, the most recent open source VLM from GDM also employes a fixed input sized Vision Encoder SigLIP ([Zhai et al., 2024](#siglip-2024)). Refer to [this blogpost](https://namangoyal.com/blog/2025/gemma3/) for a nice architectural overview of Gemma3.

To handle higher resolution images without using prohibitive amounts of VT, Gemma3 increases \\( X=Y \rightarrow 896 \\) while appling a spatial average pooling. This helps reducing the total number of visual tokens, while allowing the vision encoder to operate at higher resolution scale and only later heavily compressing the information. Thanks to the pooling, this yields a fixed amount of visual tokens which corresponds to 

$$ V_{\text{Gemma3}} = \frac{X}{PS*\text{pooling}} \times \frac{Y}{PS*\text{pooling}} = \frac{896}{14* 4} \times \frac{896}{14* 4} = 256 $$

with the pooling being applied within the modality connector.

A built-in alternative high-quality handling strategy is [Pan&Scan (P&S)](https://developers.googleblog.com/gemma-explained-whats-new-in-gemma-3/#:~:text=The%20vision%20encoder,overhead%20during%20inference.). Similar to [Strategy B](#Strategy-b-The-Multi-Grid-AnyRes) it adaptively segments the image into different parts and encodes them separately. 
The main differences between [Strategy B](#Strategy-b-The-Multi-Grid-AnyRes) and P&S are: <br>
(a.) the latter can be "turned on/off" by the user at inference and is more customisable (see [code](code-1) below),<br>
(b.) P&S can have overlapping crops, with \\( p = \text{number of crops}\\) \\( ^{**} \\).

$$ V_{\text{Gemma3-P&S}} = V_{\text{Gemma3}} \times [p + 1] $$


<a id="code-1"></a>
<figure style="width: 70%; margin: auto" markdown="1">

{% highlight python %}
class Gemma3ProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: Gemma3ImagesKwargs
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
        "images_kwargs": {
            "do_pan_and_scan": False,
            "pan_and_scan_min_crop_size": 256,
            "pan_and_scan_max_num_crops": 4,
            "pan_and_scan_min_ratio_to_activate": 1.2,
        },
    }
{% endhighlight %}

<figcaption style="margin-top: 10px; font-style: italic; color: #555; text-align: center;">
Code 1: <b>Overview of Gemma3 Pan&Scan</b> Implementation from <a href="https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/gemma3/processing_gemma3.py#L42C10-L47">source code</a>.
</figcaption>
</figure>    


<br>
<u>Upside</u>: **adaptive handling of high resolution input with lower token count**.<br> 
<u>Downside</u>: inconvenient for evaluation settings with alternating high-low input resolutions.

<small> \\( ^{**} \\) See [this blogpost](https://namangoyal.com/blog/2025/gemma3/) for more on Pan&Scan. </small>


A small sidenote: the total visual token number should also take into account the special tokens. Used across all strategies, these are simply signaling the beginning and end of visual content and are used for every picture. 


## Conclusions & Key Takeaways

<div style="margin-top: 2rem; margin-bottom: 2rem; font-family: sans-serif; width: 100%;">
    <table style="margin-left: auto; margin-right: auto; border-collapse: collapse; border: 1px solid #dee2e6; box-shadow: 0 2px 5px rgba(0,0,0,0.05); min-width: 60%;">
        <thead>
            <tr style="background-color: #f8f9fa; border-bottom: 2px solid #dee2e6;">
                <th style="padding: 12px 20px; text-align: left; border: 1px solid #dee2e6;">Representative Models</th>
                <th style="padding: 12px 20px; text-align: left; border: 1px solid #dee2e6;">Strategy</th>
                <th style="padding: 12px 20px; text-align: left; border: 1px solid #dee2e6;">Resolution Logic</th>
                <th style="padding: 12px 20px; text-align: center; border: 1px solid #dee2e6;">Token Efficiency</th>
            </tr>
        </thead>
        <tbody>
            <tr style="border-bottom: 1px solid #dee2e6;">
                <td style="padding: 12px 20px; border: 1px solid #dee2e6; font-weight: bold;">LLaVA-1.5</td>
                <td style="padding: 12px 20px; border: 1px solid #dee2e6;">Standard Resize</td>
                <td style="padding: 12px 20px; border: 1px solid #dee2e6;">Squash to fixed \( H \times W \)</td>
                <td style="padding: 12px 20px; border: 1px solid #dee2e6; text-align: center;">
                    <span style="background-color: #95a5a6; color: white; padding: 4px 10px; border-radius: 4px; font-size: 0.85em; font-weight: bold; white-space: nowrap; display: inline-block;">Fixed Count</span>
                </td>
            </tr>
            <tr style="border-bottom: 1px solid #dee2e6;">
                <td style="padding: 12px 20px; border: 1px solid #dee2e6; font-weight: bold;">Qwen3-VL</td>
                <td style="padding: 12px 20px; border: 1px solid #dee2e6;">Dynamic Merger</td>
                <td style="padding: 12px 20px; border: 1px solid #dee2e6;">Native (Preserves Aspect Ratio)</td>
                <td style="padding: 12px 20px; border: 1px solid #dee2e6; text-align: center;">
                    <span style="background-color: #3498db; color: white; padding: 4px 10px; border-radius: 4px; font-size: 0.85em; font-weight: bold; white-space: nowrap; display: inline-block;">Quadratic Growth</span>
                </td>
            </tr>
            <tr style="border-bottom: 1px solid #dee2e6;">
                <td style="padding: 12px 20px; border: 1px solid #dee2e6; font-weight: bold;">LLaVA-OneVision</td>
                <td style="padding: 12px 20px; border: 1px solid #dee2e6;">AnyRes / Multi-Grid</td>
                <td style="padding: 12px 20px; border: 1px solid #dee2e6;">Grid Split ( \( k \times k \) ) + Overview</td>
                <td style="padding: 12px 20px; border: 1px solid #dee2e6; text-align: center;">
                    <span style="background-color: #e74c3c; color: white; padding: 4px 10px; border-radius: 4px; font-size: 0.85em; font-weight: bold; white-space: nowrap; display: inline-block;">Massive Cost</span>
                </td>
            </tr>
            <tr style="border-bottom: 1px solid #dee2e6;">
                <td rowspan="2" style="padding: 12px 20px; border: 1px solid #dee2e6; font-weight: bold; vertical-align: middle; text-align: left; background-color: white;">
                    Gemma3
                </td>
                <td style="padding: 12px 20px; border: 1px solid #dee2e6;">Fixed Downsample</td>
                <td style="padding: 12px 20px; border: 1px solid #dee2e6;">Resize + Spatial Pooling</td>
                <td style="padding: 12px 20px; border: 1px solid #dee2e6; text-align: center;">
                    <span style="background-color: #2ecc71; color: white; padding: 4px 10px; border-radius: 4px; font-size: 0.85em; font-weight: bold; white-space: nowrap; display: inline-block;">Highly Compact</span>
                </td>
            </tr>
            <tr style="border-bottom: 1px solid #dee2e6;">
                <td style="padding: 12px 20px; border: 1px solid #dee2e6;">Pan and Scan</td>
                <td style="padding: 12px 20px; border: 1px solid #dee2e6;">Adaptive Grid Split + Downsample</td>
                <td style="padding: 12px 20px; border: 1px solid #dee2e6; text-align: center;">
                    <span style="background-color: #e6a23c; color: white; padding: 4px 10px; border-radius: 4px; font-size: 0.85em; font-weight: bold; white-space: nowrap; display: inline-block;">Adaptively Expensive</span>
                </td>
            </tr>
        </tbody>
    </table>
    <p style="text-align: center; margin-top: 15px; font-size: 0.9em; color: #555; font-style: italic;">
        Table 1: <b>Comparison of Visual Token calculation strategies</b> across modern SOTA VLM architectures.
    </p>
</div>

Visual Tokens are the bridge between the image and language world, but they are also the primary bottleneck in VLM deployment. As we have seen, moving from a fixed-resolution model (LLaVA1.5) to a dynamic one (Qwen3-VL or LLaVA-OneVision) and hybrid ones (Gemma3) can considerbly increase your input size ([Figure 1](#figure-1)). 

Here are some **key takeaways** to keep in mind when building multimodal systems:

- *Calculate, Don't Guess:* Use the formulas provided [above](#calculating-visual-tokens) to pre-calculate token counts. This allows you to dynamically resize images and/or adjust batch sizes to prevent OOM errors in production (more on this in our next blogpost). 
- *Tokens \\( \neq \\) Pixels:* High resolution doesn't always mean high cost. It depends entirely on the architecture (e.g., Fixed Downsampler vs. Multi-Grid).

Multimodal learning is evolving rapidly, but compute is finite. Mastering the math of Visual Tokens is the first step toward correctly exploiting VLM efficiency.

In the next blogpost (coming soon), we will dive deep into the impact of visual token counts on Context Windows, Latency, VRAM.


## Citation

If you use this work, please cite:

{% highlight bibtex %}
@misc{nulli2026thehidden,
  title={De-mystifying Multimodal Learning: The Hidden Inefficiency in Vision Language Modelling},
  author={Nulli, Matteo},
  year={2026},
  url={https://huggingface.co/blog/MatteoNulli/de-mystifying-multimodal-learning-hidden-ineff},
  howpublished={Available at \url{https://matteonulli.github.io/blog/2026/demystifying1/} and \url{https://huggingface.co/blog/MatteoNulli/de-mystifying-multimodal-learning-hidden-ineff}},
  note={Hugging Face Blog}
}
{% endhighlight %}




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

<a id="flash-attn-2022" class="bib-item"> Dao, Tri, et al. "Flashattention: Fast and memory-efficient exact attention with io-awareness." Advances in neural information processing systems 35 (2022): 16344-16359. </a>

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