---
layout: post
title: "De-mystifying Multimodal Learning<br><small>Enabiling Vision in Language Models"
date: 2026-02-29 14:14:00
description: A blogpost series on the nuts and bolts of Multimodal Learning
tags: Multimodal-Learning Vision-Language-Modelling
# thumbnail: assets/img/mllms_visual_tokens_wide.png
thumbnail: assets/img/demistfying_vlms_arch.png
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

In this first installment of our series, `De-mystifying Multimodal Learning`, we break down the mechanics of how images become language-compatible vectors. To truly understand how an LLM "sees," we must look at the mathematics defining the problem, the training objectives that align vision and text, and the specific architectural steps that process raw pixels. We will therefore cover:<br>

[Problem Setting](#problem-setting): The mathematical foundation and formal definitions of Vision Language Models.

[Contrastive Learning](): How models learn to map images and text to the same space.

[How do LLMs see?](#how-do-llms-see): An architectural deep dive into the birth of the Visual Token.
<a id="overview"></a>
<div class="row mt-3">
  <div class="col-12">
    <div class="mx-auto text-center" style="width: 80%;">
      {% include figure.liquid loading="eager" path="./assets/img/demistfying_vlms_arch.png" class="img-fluid rounded z-depth-1" %}
      <div class="caption text-center mt-2">
        Figure 1: .
      </div>
    </div>
  </div>
</div>

<br>

## Problem Setting

To understand Vision-Language Models (VLMs) / Multimodal Large Language Models (MLLMs), we first need to define the notation and the transformation pipeline formally.

Let $\mathbf{X} \in \mathbb{R}^{C \times H\times W}$ be an image and $t \in \Sigma$ be a language instruction input, where $\Sigma$ is the input space of character sequences. Let $s_{\theta, \gamma, \phi}$ be an MLLM parametrized by $\theta, \gamma, \phi$. We define $f_{v\theta}$ as a contrastively pre-trained Vision Encoder model:
<p align="center"> \[f_{v\theta}: \mathbb{R}^{C \times H \times W} \rightarrow \mathbb{R}^{V \times F},\] </p>

where $V$ is the number of visual tokens and $F$ is their hidden size. $f_{t\theta'}$ represents the corresponding Text Encoder used during the pre-training phase.

To bridge the gap between vision and language, we use a connector $m_\gamma: \mathbb{R}^{V \times F} \rightarrow \mathbb{R}^{V \times D}$, typically a Multi-Layer Perceptron (MLP). The token vocabulary for the model is defined as:
$$\mathcal{V}\;=\;\mathcal{V}_{\text{vision}}\;\cup\;\mathcal{V}_{\text{text}}$$

The Large Language Model itself is defined as:
<p align="center"> \[ g_{\phi}\;:=\;\mathcal{D}_d\;\circ\;\operatorname{softmax}\;\circ\;F_{\phi'}\;\;:\;\mathbb{R}^{J\times D}\;\longrightarrow\;\mathcal{V}^{J},
\qquad \phi=\bigl(\phi',d\bigr),\]</p>

where $F_{\phi'}$ is the transformer that produces logits, and $$\mathcal{D}_d$$ is a decoding operator (such as greedy, top-$k$, or nucleus sampling) with hyper-parameters $d$. Thus, $g_{\phi}$ maps an embedded input token sequence to an output token sequence.


## Vision Enocoder Architectural Breakdown
Now that we have established the mathematical setting, let's look at the architectural implementation of the Vision Encoder $f_{v\theta}$. 
Practically, the processing flow of $f_{v\theta}$ is broken down into the following steps:

#### 1. Patch Partitioning
The first step is breaking the high-resolution image $\mathbf{X}$ into a grid of fixed-size patches.Assuming our image has $336 \times 336$ pixels and we use a patch size of $P=14$, standard vision encoders divide the image into $24 \times 24 = 576$ distinct squares. Mathematically, the image is reshaped from $\mathbf{X} \in \mathbb{R}^{C \times H \times W}$ into a sequence of flattened 2D patches $\mathbf{x}_p \in \mathbb{R}^{N \times (P^2 \cdot C)}$, where $N$ is the total number of patches.

#### 2. Linear Projection and Position Embeddings
These patches are simply raw pixel values. To convert them into vectors, $f_{v\theta}$ projects each flattened patch into a latent representation through a linear layer.Given the lack of spatial priors in Vision Transformers (ViT), these vectors are equipped with learnable positional encodings, injecting "GPS-like" coordinates so the model knows where each patch belongs in the original image.

<a id="vit"></a>
<div class="row mt-3">
  <div class="col-12">
    <div class="mx-auto text-center" style="width: 80%;">
      {% include figure.liquid loading="eager" path="./assets/img/vit.png" class="img-fluid rounded z-depth-1" %}
      <div class="caption text-center mt-2">
        Figure 2: .
      </div>
    </div>
  </div>
</div>


#### 3. Transformer Layers
The resulting vectors are passed through several Transformer Layers consisting of Multi-Head Self-Attention and MLPs. The output is a sequence of vectors where each vector represents a patch within the context of the whole image. This full process produces the representations $\mathbf{X'} = f_{v\theta}(\mathbf{X}) \in \mathbb{R}^{V\times F}$.

## Contrastive Learning
Before the Vision Encoder $f_{v\theta}$ can be used in the MLLM pipeline, it must learn to extract features that are semantically aligned with text. 
This is achieved through Contrastive Learning, (extra sources [here](https://arxiv.org/pdf/2103.00020), [here](https://medium.com/rectlabs/clip-contrastive-language-image-pre-training-dce66ae18fe1) and [here](https://github.com/openai/CLIP)) a learning process through which Vision Encoders learn to be powerful feature extractors, compressing visual information into vectors (tokens) semantically aligned with language.  
Mathematically, during this pre-training phase, each encoder ($f_{v\theta}$, $$f_{t\theta'}$$) extracts feature representations for a batch of image-text pairs. Let $$t' = f_{t\theta'}(t)$$ be the text features and $$\mathbf{X}' = f_{v\theta}(\mathbf{X})$$ be the image features.
These are normalized as follows 
<p align="center"> \[ \mathbf{X}'_{e} = \frac{\mathbf{X}'}{\|\mathbf{X}'\|_2}, \quad t'_{e} = \frac{t'}{\|t'\|_2} \] </p>
These normalized features are used to compute the pairwise cosine similarities:
$$\textit{logits} = (\mathbf{X}_e' \cdot t_e'^T ) \cdot e^{\tau}$$where $$ t_e'^{T} $$ is the transpose of $$t_e'$$, and $\tau$ is a learnable temperature parameter.These logits are finally used to compute the joint loss function using cross-entropy (CE). The model attempts to maximize the similarity of correct image-text pairs (the diagonal of the matrix) while minimizing others:
<p align="center"> \[\begin{aligned}
\mathcal{L}_{\mathbf X} &= \operatorname{CE}(\textit{logits}, \textit{labels}, \text{axis}=0), \\[4pt]
\mathcal{L}_{t}         &= \operatorname{CE}(\textit{logits}, \textit{labels}, \text{axis}=1), \\[4pt]
\mathcal{L}             &= \tfrac{1}{2}\,\bigl(\mathcal{L}_{\mathbf X} + \mathcal{L}_{t}\bigr).
\end{aligned}\] </p>
Here, *labels* are the ground truths for that sample, and $\text{axis}=i, \text{with } i \in \{0,1\}$ represents the dimension along which the loss is computed.

## VLM Architecture and Flow
Once the Vision Encoder is pre-trained, we can assemble the full model. Architecturally, Vision Language Models are constituted by three major components:

- Vision Encoders ($f_{v\theta}$), usually a CLIP-like image encoder ([Dosovitskiy et al., 2021](#vit-2021),[Radford et al., 2021](#clip-2021), [Zhai et al., 2023](#sigmoid-loss-2023), [Bai et al., 2025](#qwen2-5-vl-2025)), but it can vary in architecture and training style. See [this](https://jina.ai/vision-encoder-survey.pdf) extensive survey for more information . 
- Modality Connectors ($m_\gamma$), often simple Multi-Layer Perceptron, with some architectures employing attention blocks ([Li et al., 2023](#blip2-2023)) and other alternatives ([Tong et al., 2024](#cambrian-1-2024), [Nulli et al,. 2025](#nulliobjectguided-2025)).  
- Large Language Models ($g_\phi$) like Qwen3 [Yang An, et al. 2025](#qwen3-2025), Llama3 [Abhimanyu, et al. 2024](#llama-3-herd-2024), Vicuna [Wei-Lin, et al. 2023](#vicuna-2023).

### Vision-Language Modeling Pipeline
Finally, we put all the pieces together. We describe the standard pipeline of MLLMs during inference, assuming a batch size of 1.

Vision Encoders $f_{v\theta}$ are used to encode an image $\mathbf{X}$ into a representation:
<p align="center"> \[\mathbf{X}' = f_{v\theta}(\mathbf{X}) \in \mathbb{R}^{V \times F}\]</p>
Here, $F$ is the feature dimension and $V$ is the vision encoder hidden dimension, calculated as $V = (\frac{\textit{image resolution}}{\textit{patch size}})^2$.
Subsequently, $\mathbf{X}'$ is transformed through the connector $m_\gamma$ into Visual Tokens ($\mathbf{VT}$):
<p align="center"> \[\mathbf{VT} = m_\gamma(\mathbf{X}') \in \mathbb{R}^{V \times D} \]</p>
Crucially, these tokens now exist in the input embedding space of the Large Language Model. 
In parallel, a Tokenizer $\mathcal{T}: \Sigma \rightarrow \mathcal{V}^{J}$ and a learned embedding $E:\mathcal{V}^{J}\;\longrightarrow\;\mathbb{R}^{D}$ turn the text input $t$ into textual tokens: $$\mathit{TT} = E^{\otimes}(\mathcal{T}(t)) \in \mathbb{R}^{J \times D},$$where $E^{\otimes}$ is the sequence-wise lifting of operator $E$.
Lastly, the visual tokens $\mathbf{VT}$ are concatenated with the textual tokens $\mathit{TT}$ and provided as input to the LLM $g_\phi$ to obtain the output tokens $\mathbf{T}_a$:
<p align="center"> \[\mathbf{T}_a = g_{\phi}(\mathbf{VT} \oplus \mathit{TT}) \in \mathcal{V}^{J}.\] </p>



<p align="center"><code>What is a Visual Token?</code></p>

<br>



## Citation

If you use this work, please cite:

```bibtex
@misc{nulli2026demistifying,
title={De-mystifying Multimodal Learning: Enabiling Vision in Language Models},
author={Matteo Nulli},
year={2026},
url={https://matteonulli.github.io/blog/2026/demystifying0/}}
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

<a id="qwen3-2025" class="bib-item">Yang An, et al. (2025). Qwen3 Technical Report. arXiv preprint arXiv:2505.09388.</a>

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

<a id="vicuna-2023" class="bib-item">Chiang Wei-Lin, Li Zhuohan, Lin Zi, Sheng Ying, Wu Zhanghao, Zhang Hao, Zheng Lianmin, Zhuang Siyuan, Zhuang Yonghao, Gonzalez Joseph E., Stoica Ion, Xing Eric P. (2023). Vicuna: An Open-Source Chatbot Impressing GPT-4 with 90%* ChatGPT Quality. LMSYS Org Blog. https://lmsys.org/blog/2023-03-30-vicuna/</a>

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