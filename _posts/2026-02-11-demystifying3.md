---
layout: post
title: "Demystifying Multimodal Learning: Speculative Decoding in Multimodal Architectures"
date: 2026-06-30 14:14:00
description: A blogpost series on the nuts and bolts of Multimodal Learning
tags: Multimodal-Learning Inference-Optimization
# thumbnail: assets/img/mllms_visual_tokens_wide.png
thumbnail: https://cdn-uploads.huggingface.co/production/uploads/661d4e74b8f13412f6d48a50/SfOleyYtgr6UtQ4lT8jv7.png

community_article_url: https://huggingface.co/blog/MatteoNulli/de-mystifying-multimodal-learning-impact-vt-laten
blogpost_url: https://matteonulli.github.io/blog/2026/demystifying2/
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
###### <a href="https://huggingface.co/blog/MatteoNulli/de-mystifying-multimodal-learning-impact-vt-laten" title="Community Article"><i class="fa-brands fa-hugging-face" style="font-size: 1.75em;"></i></a> <a href="https://matteonulli.github.io/blog/2025/demystifying2/" title="Blogpost"><i class="fa-regular fa-newspaper" style="font-size: 1.75em;"></i></a>
<br>

## Introduction

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