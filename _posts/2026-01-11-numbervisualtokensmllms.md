---
layout: post
title: "De-mystifying Visual Tokens: The Hidden Cost of Multimodal Large Language Models."
date: 2026-01-30 14:14:00
description: E-commerce adaptation of SOTA VLMs preserving general capabilities
tags: Multimodal-Learning E-commerce Model-Adaptation VLMs Benchmarking
# categories: Multimodal-Learning
thumbnail: assets/img/ecomvlm4.png
mathjax: true
math: true
---


##### <b>Matteo Nulli</b>
###### eBay, University of Amsterdam 
###### <img src="https://upload.wikimedia.org/wikipedia/commons/1/1b/EBay_logo.svg" alt="eBay" height="24"/> &nbsp; <img src="https://upload.wikimedia.org/wikipedia/commons/1/17/Uva%C2%AEmerken_ENG.png" alt="University of Amsterdam" height="24"/> &nbsp;
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

## How do LLMs actually see an image? 

Definition: Briefly define a "Visual Token" (a vector representation of an image patch).

The Stake: Why does token count matter? (Context window limits, inference cost/latency, VRAM usage).

## The "Standard" Recipe

Explain the standard Vision Transformer (ViT) approach.
The concept of breaking an image into patches.
The Formula: $N = (H/P)^2 + \text{special\_tokens}$

## The Resolution Trap

Why we need higher resolution (OCR, small object detection).
The math of explosion: Going from $336\times336$ to $1024\times1024$ doesn't triple the tokens; it squares them.

## Modern Architectures: Three Strategies

Strategy A: The Fixed Downsampler (e.g., Gemma/PaliGemma)

Explanation: Fixed input size + Pooling.

Your math on the "Spatial Merge Size" of 4.

Strategy B: The Dynamic Merger (e.g., Qwen-VL)

For Models like Qwen whose input size is not fixed, i.e. there is no resizing to allow for multi-resolution optimal processing by the ViT, the process is variable and depends on H and W. So for Qwen3VL with patch size 14 and  spatial_merge_size 2 you have this formula:

$ Visual Tokens = (H/ 14*2) * (W/ 14*2) +#special_tokens $

In summary, Preserving aspect ratio without resizing with Dynamic pooling.

Your math on variable H and W.

Strategy C: The Multi-Grid / AnyRes (e.g., LLaVA-OneVision/Cambrian)

Explanation: Cropping high-res images into local patches + 1 global patch.

The trade-off: Massive detail vs. Massive token count.

## Projection step

Briefly mention that the "Spatial Merge" often happens in the Connector (Projector) layer (C-Abstractor, MLP, etc.).

## Conclusions & Key Takeaways

Summary table comparing the models
Advice for developers: Which model to choose based on compute constraints vs. detail requirements.




## Additional Questions to Answer


To make this blog post truly comprehensive, consider adding answers to these questions. They bridge the gap between "How to calculate" and "How to use."

A. What happens to the Aspect Ratio?

Context: When using models like Gemma (fixed size) vs. Qwen (variable), what happens to the image shape?

Question: "Does tokenization squish the image or pad it? How does padding affect the token count (wasted tokens)?"

B. What are the "Special Tokens" actually doing?

Context: You mention + #special_tokens.

Question: "Why do we need <image_start> and <image_end>? Do the attention heads treat these differently than the pixel patches?" (Answer: They act as delimiters so the LLM knows where the text stops and the image begins).

C. What is the role of the "Connector" vs. the "Vision Encoder"?

Context: You mention pooling and spatial merging.

Question: "Where does the downsampling happen? Inside the ViT (Vision Encoder) or in the Adapter/Projector?"

Insight: Usually, the ViT outputs the standard patch count, and the Connector (like a C-Abstractor or a strided convolution) performs the spatial merge to reduce the count before it hits the LLM.

D. What is the cost trade-off?

Context: LLaVA-OneVision uses many more tokens.

Question: "If Model A uses 256 tokens and Model B uses 2,800 tokens for the same image, what is the impact on 'Time to First Token' (latency) and VRAM?"

<a id="mainfigure"></a>
<div class="row mt-3">
  <div class="col-12">
    <div class="mx-auto text-center" style="width: 80%;">
      {% include figure.liquid loading="eager" path="./assets/img/ecomvlm.png" class="img-fluid rounded z-depth-1" %}
      <div class="caption text-center mt-2">
        Figure 1: <b>Output of our E-commerce Adapted VLMs</b> compared against same size LLaVA-OneVision.
        We show our models ability to more faithfully extract attributes from e-commerce items. In red, we highlight wrong
        model predictions that are neither tied to the image nor valid item attributes.
      </div>
    </div>
  </div>
</div>


## Methodology



## Citation

If you use this work, please cite:

```bibtex
@misc{nulli2026demistifying,
title={De-mystifying Visual Tokens: The Hidden Cost of Multimodal Large Language Models.},
author={Matteo Nulli},
year={2026},
url={}
}
```

**References**

<a id="conme-2024">Huang Irene, Lin Wei, Mirza M. Jehanzeb, Hansen Jacob A., Doveh Sivan, Butoi Victor Ion, Herzig Roei, Arbelle Assaf, Kuehne Hilde, Darrell Trevor, Gan Chuang, Oliva Aude, Feris Rogerio, Karlinsky Leonid. (2024). Conme: Rethinking Evaluation of Compositional Reasoning for Modern VLMs. arXiv preprint arXiv:2406.08164.</a>

<a id="eyes-wide-shut-2024">Tong Shengbang, Liu Zhuang, Zhai Yuexiang, Ma Yi, LeCun Yann, Xie Saining. (2024). Eyes Wide Shut? Exploring the Visual Shortcomings of Multimodal LLMs. arXiv preprint arXiv:2401.06209.</a>

<a id="visual-instruction-tuning-2023">Liu Haotian, Li Chunyuan, Wu Qingyang, Lee Yong Jae. (2023). Visual Instruction Tuning. arXiv preprint arXiv:2304.08485.</a>

<a id="llavonevision-2024">Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. 2024a. Llava-onevision: Easy visual task transfer. Preprint,
arXiv:2408.03326.</a>


<a id="qwen2-5-vl-2025">Bai Shuai, Chen Keqin, Liu Xuejing, Wang Jialin, Ge Wenbin, Song Sibo, Dang Kai, Wang Peng, Wang Shijie, Tang Jun, Zhong Humen, Zhu Yuanzhi, Yang Mingkun, Li Zhaohai, Wan Jianqiang, Wang Pengfei, Ding Wei, Fu Zheren, Xu Yiheng, Ye Jiabo, Zhang Xi, Xie Tianbao, Cheng Zesen, Zhang Hang, Yang Zhibo, Xu Haiyang, Lin Junyang. (2025). Qwen2.5-VL Technical Report. arXiv preprint arXiv:2502.13923.</a>

<a id="internvl2-2024">OpenGVLab-Team. (2024). InternVL2: Better Than the Best—Expanding Performance Boundaries of Open-Source Multimodal Models with the Progressive Scaling Strategy. Blog post. URL https://internvl.github.io/blog/2024-07-02-InternVL-2.0/.</a>

<a id="gemma-3-2025">Gemma-Team. (2025). Gemma 3 Technical Report. arXiv preprint arXiv:2503.19786.</a>

<a id="bags-of-words-vlms-2023">Yuksekgonul Mert, Bianchi Federico, Kalluri Pratyusha, Jurafsky Dan, Zou James. (2023). When and Why Vision-Language Models Behave Like Bags-of-Words, and What to Do About It? arXiv preprint arXiv:2210.01936.</a>

<a id="icl-compositional-vlms-2024">Nulli Matteo, Ibrahimi Anesa, Pal Avik, Lee Hoshe, Najdenkoska Ivona. (2024). In-Context Learning Improves Compositional Understanding of Vision-Language Models. In ICML 2024 Workshop on Foundation Models in the Wild. arXiv preprint arXiv:2407.15487.</a>

<a id="vismin-2025">Awal Rabiul, Ahmadi Saba, Zhang Le, Agrawal Aishwarya. (2025). Vismin: Visual Minimal-Change Understanding. arXiv preprint arXiv:2407.16772.</a>

<a id="cambrian-1-2024">Tong Shengbang, Brown Ellis, Wu Penghao, Woo Sanghyun, Middepogu Manoj, Akula Sai Charitha, Yang Jihan, Yang Shusheng, Iyer Adithya, Pan Xichen, Wang Austin, Fergus Rob, LeCun Yann, Xie Saining. (2024). Cambrian-1: A Fully Open, Vision-Centric Exploration of Multimodal LLMs. arXiv preprint arXiv:2406.16860.</a>

<a id="llava-next-2024">Liu Haotian, Li Chunyuan, Li Yuheng, Li Bo, Zhang Yuanhan, Shen Sheng, Lee Yong Jae. (2024). LLaVA-NeXT: Improved Reasoning, OCR, and World Knowledge. Blog post (January 2024). URL https://llava-vl.github.io/blog/2024-01-30-llava-next/.</a>

<a id="sam-2-2024">Ravi Nikhila, Gabeur Valentin, Hu Yuan-Ting, Hu Ronghang, Ryali Chaitanya, Ma Tengyu, Khedr Haitham, Rädle Roman, Rolland Chloe, Gustafson Laura, Mintun Eric, Pan Junting, Alwala Kalyan Vasudev, Carion Nicolas, Wu Chao-Yuan, Girshick Ross, Dollár Piotr, Feichtenhofer Christoph. (2024). SAM 2: Segment Anything in Images and Videos. arXiv preprint arXiv:2408.00714.</a>

<a id="omg-seg-cvpr-2024">Li Xiangtai, Yuan Haobo, Li Wei, Ding Henghui, Wu Size, Zhang Wenwei, Li Yining, Chen Kai, Loy Chen Change. (2024). OMG-Seg: Is One Model Good Enough for All Segmentation? In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 27948–27959.</a>

<a id="eagle-2-5-2025">Chen Guo, Li Zhiqi, Wang Shihao, Jiang Jindong, Liu Yicheng, Lu Lidong, Huang De-An, Byeon Wonmin, Le Matthieu, Rintamaki Tuomas, Poon Tyler, Ehrlich Max, Lu Tong, Wang Limin, Catanzaro Bryan, Kautz Jan, Tao Andrew, Yu Zhiding, Liu Guilin. (2025). EAGLE 2.5: Boosting Long-Context Post-Training for Frontier Vision-Language Models. arXiv preprint arXiv:2504.15271.</a>

<a id="omg-llava-2024">Zhang Tao, Li Xiangtai, Fei Hao, Yuan Haobo, Wu Shengqiong, Ji Shunping, Loy Chen Change, Yan Shuicheng. (2024). OMG-LLaVA: Bridging Image-Level, Object-Level, Pixel-Level Reasoning and Understanding. arXiv preprint arXiv:2406.19389.</a>

<a id="sa2va-2025">Yuan Haobo, Li Xiangtai, Zhang Tao, Huang Zilong, Xu Shilin, Ji Shunping, Tong Yunhai, Qi Lu, Feng Jiashi, Yang Ming-Hsuan. (2025). SA2VA: Marrying SAM2 with LLaVA for Dense Grounded Understanding of Images and Videos. arXiv preprint arXiv:2501.04001.</a>

<a id="clip-2021">Radford Alec, Kim Jong Wook, Hallacy Chris, Ramesh Aditya, Goh Gabriel, Agarwal Sandhini, Sastry Girish, Askell Amanda, Mishkin Pamela, Clark Jack, Krueger Gretchen, Sutskever Ilya. (2021). Learning Transferable Visual Models from Natural Language Supervision. arXiv preprint arXiv:2103.00020.</a>

<a id="improved-vit-baselines-2024">Liu Haotian, Li Chunyuan, Li Yuheng, Lee Yong Jae. (2024). Improved Baselines with Visual Instruction Tuning. arXiv preprint arXiv:2310.03744.</a>

<a id="vit-2021">Dosovitskiy Alexey, Beyer Lucas, Kolesnikov Alexander, Weissenborn Dirk, Zhai Xiaohua, Unterthiner Thomas, Dehghani Mostafa, Minderer Matthias, Heigold Georg, Gelly Sylvain, Uszkoreit Jakob, Houlsby Neil. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. arXiv preprint arXiv:2010.11929.</a>

<a id="llama-2-2023">Touvron Hugo, Martin Louis, Stone Kevin, Albert Peter, Almahairi Amjad, Babaei Yasmine, Bashlykov Nikolay, Batra Soumya, Bhargava Prajjwal, Bhosale Shruti, Bikel Dan, Blecher Lukas, Canton Ferrer Cristian, Chen Moya, Cucurull Guillem, Esiobu David, Fernandes Jude, Fu Jeremy, Fu Wenyin, Fuller Brian, Gao Cynthia, Goswami Vedanuj, Goyal Naman, Hartshorn Anthony, Hosseini Saghar, Hou Rui, Inan Hakan, Kardas Marcin, Kerkez Viktor, Khabsa Madian, Kloumann Isabel, Korenev Artem, Koura Punit Singh, Lachaux Marie-Anne, Lavril Thibaut, Lee Jenya, Liskovich Diana, Lu Yinghai, Mao Yuning, Martinet Xavier, Mihaylov Todor, Mishra Pushkar, Molybog Igor, Nie Yixin, Poulton Andrew, Reizenstein Jeremy, Rungta Rashi, Saladi Kalyan, Schelten Alan, Silva Ruan, Smith Eric Michael, Subramanian Ranjan, Tan Xiao-qing Ellen, Tang Binh, Taylor Ross, Williams Adina, Kuan Jian Xiang, Xu Puxin, Yan Zheng, Zarov Iliyan, Zhang Yuchen, Fan Angela, Kambadur Melanie, Narang Sharan, Rodriguez Aurelien, Stojnic Robert, Edunov Sergey, Scialom Thomas. (2023). Llama 2: Open Foundation and Fine-Tuned Chat Models.</a>

<a id="llama-3-2-2024">Meta. (2024). Llama 3.2: Revolutionizing Edge AI and Vision with Open, Customizable Models. Blog post. URL https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/.</a>

<a id="lora-2021">Hu Edward J., Shen Yelong, Wallis Phillip, Allen-Zhu Zeyuan, Li Yuanzhi, Wang Shean, Wang Lu, Chen Weizhu. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv preprint arXiv:2106.09685.</a>

<a id="coco-2014">Lin Tsung-Yi, Maire Michael, Belongie Serge, Hays James, Perona Pietro, Ramanan Deva, Dollár Piotr, Zitnick C. Lawrence. (2014). Microsoft COCO: Common Objects in Context. In Computer Vision – ECCV 2014, pages 740–755. Springer.</a>

<a id="image-descriptions-2014">Young Peter, Lai Alice, Hodosh Micah, Hockenmaier Julia. (2014). From Image Descriptions to Visual Denotations: New Similarity Metrics for Semantic Inference Over Event Descriptions. Transactions of the Association for Computational Linguistics, 2:67–78.</a>

<a id="visual-genome-2017">Krishna Ranjay, Zhu Yuke, Groth Oliver, Johnson Justin, Hata Kenji, Kravitz Joshua, Chen Stephanie, Kalantidis Yannis, Li Li-Jia, Shamma David A., et al. (2017). Visual Genome: Connecting Language and Vision Using Crowdsourced Dense Image Annotations. International Journal of Computer Vision, 123:32–73.</a>

<a id="gqa-2019">Hudson Drew A., Manning Christopher D. (2019). GQA: A New Dataset for Real-World Visual Reasoning and Compositional Question Answering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 6700–6709.</a>

<a id="sugar-crepe-2023">Hsieh Cheng-Yu, Zhang Jieyu, Ma Zixian, Kembhavi Aniruddha, Krishna Ranjay. (2023). SugarCrepe: Fixing Hackable Benchmarks for Vision-Language Compositionality. Advances in Neural Information Processing Systems, 36:31096–31116.</a>

<a id="gpt-4-technical-report-2024">OpenAI. (2024). GPT-4 Technical Report. arXiv preprint arXiv:2303.08774.</a>

<a id="scaling-instruction-finetuned-2024">Chung Hyung Won, Hou Le, Longpre Shayne, Zoph Barret, Tay Yi, Fedus William, Li Yunxuan, Wang Xuezhi, Dehghani Mostafa, Brahma Siddhartha, et al. (2024). Scaling Instruction-Finetuned Language Models. Journal of Machine Learning Research, 25(70):1–53.</a>

<a id="diagram-2016">Kembhavi Aniruddha, Salvato Mike, Kolve Eric, Seo Minjoon, Hajishirzi Hannaneh, Farhadi Ali. (2016). A Diagram is Worth a Dozen Images. arXiv preprint arXiv:1603.07396.</a>

<a id="mme-2024">Fu Chaoyou, Chen Peixian, Shen Yunhang, Qin Yulei, Zhang Mengdan, Lin Xu, Yang Jinrui, Zheng Xiawu, Li Ke, Sun Xing, Wu Yunsheng, Ji Rongrong. (2024). MME: A Comprehensive Evaluation Benchmark for Multimodal Large Language Models. arXiv preprint arXiv:2306.13394.</a>

<a id="evaluating-vlms-right-way-2024">Chen Lin, Li Jinsong, Dong Xiaoyi, Zhang Pan, Zang Yuhang, Chen Zehui, Duan Haodong, Wang Jiaqi, Qiao Yu, Lin Dahua, Zhao Feng. (2024). Are We on the Right Way for Evaluating Large Vision-Language Models? arXiv preprint arXiv:2403.20330.</a>

<a id="mmbench-2024">Liu Yuan, Duan Haodong, Zhang Yuanhan, Li Bo, Zhang Songyang, Zhao Wangbo, Yuan Yike, Wang Jiaqi, He Conghui, Liu Ziwei, Chen Kai, Lin Dahua. (2024). MMBench: Is Your Multi-Modal Model an All-Around Player? arXiv preprint arXiv:2307.06281.</a>

<a id="subobject-tokenization-2025">Chen Delong, Cahyawijaya Samuel, Liu Jianfeng, Wang Baoyuan, Fung Pascale. (2025). Subobject-Level Image Tokenization. arXiv preprint arXiv:2402.14327.</a>

<a id="deepspeed-2020">Rasley Jeff, Rajbhandari Samyam, Ruwase Olatunji, He Yuxiong. (2020). DeepSpeed: System Optimizations Enable Training Deep Learning Models with Over 100 Billion Parameters. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD ’20), pages 3505–3506. doi:10.1145/3394486.3406703.</a>

<a id="zero-2020">Rajbhandari Samyam, Rasley Jeff, Ruwase Olatunji, He Yuxiong. (2020). ZeRO: Memory Optimizations Toward Training Trillion Parameter Models. In SC20: International Conference for High Performance Computing, Networking, Storage and Analysis, pages 1–16. doi:10.1109/SC41405.2020.00024.</a>

<a id="adam-2017">Kingma Diederik P., Ba Jimmy. (2017). Adam: A Method for Stochastic Optimization. arXiv preprint arXiv:1412.6980.</a>

<a id="adamw-2019">Loshchilov Ilya, Hutter Frank. (2019). Decoupled Weight Decay Regularization. arXiv preprint arXiv:1711.05101.</a>

<a id="bert-2019">Devlin Jacob, Chang Ming-Wei, Lee Kenton, Toutanova Kristina. (2019). BERT: Pre-Training of Deep Bidirectional Transformers for Language Understanding. In Proceedings of NAACL-HLT 2019, pages 4171–4186.</a>

<a id="attention-is-all-you-need-2017">Vaswani Ashish, Shazeer Noam, Parmar Niki, Uszkoreit Jakob, Jones Llion, Gomez Aidan N., Kaiser Łukasz, Polosukhin Illia. (2017). Attention Is All You Need. Advances in Neural Information Processing Systems, 30.</a>

<a id="pixtral-12b-2024">Agrawal Pravesh, Antoniak Szymon, Bou Hanna Emma, Bout Baptiste, Chaplot Devendra, Chudnovsky Jessica, Costa Diogo, De Monicault Baudouin, Garg Saurabh, Gervet Theophile, Ghosh Soham, Héliou Amélie, Jacob Paul, Jiang Albert Q., Khandelwal Kartik, Lacroix Timothée, Lample Guillaume, Las Casas Diego, Lavril Thibaut, Le Scao Teven, Lo Andy, Marshall Louis, Martin Arthur, Mensch Arthur, Muddireddy Pavankumar, Nemychnikova Valera, Pellat Marie, Von Platen Patrick, Raghuraman Nikhil, Bout Rozière Baptiste, Sablayrolles Alexandre, Saulnier Lucile, Sauvestre Romain, Rozière Baptiste, Shang Wendy, Soletskyi Roman, Stewart Lawrence, Stock Pierre, Studnia Joachim, Subramanian Sandeep, Vaze Sagar, Wang Thomas, Yang Sophia. (2024). Pixtral 12B. arXiv preprint arXiv:2410.07073.</a>

<a id="roformer-2023">Su Jianlin, Lu Yu, Pan Shengfeng, Murtadha Ahmed, Wen Bo, Liu Yunfeng. (2023). RoFormer: Enhanced Transformer with Rotary Position Embedding. arXiv preprint arXiv:2104.09864.</a>

<a id="llama-3-herd-2024">Dubey Abhimanyu, et al. (2024). The Llama 3 Herd of Models. arXiv preprint arXiv:2407.21783.</a>

<a id="reproducible-scaling-laws-2023">Cherti Mehdi, Beaumont Romain, Wightman Ross, Wortsman Mitchell, Ilharco Gabriel, Gordon Cade, Schuhmann Christoph, Schmidt Ludwig, Jitsev Jenia. (2023). Reproducible Scaling Laws for Contrastive Language-Image Learning. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 2818–2829. doi:10.1109/CVPR52729.2023.00276.</a>

<a id="sigmoid-loss-2023">Zhai Xiaohua, Mustafa Basil, Kolesnikov Alexander, Beyer Lucas. (2023). Sigmoid Loss for Language Image Pre-Training. arXiv preprint arXiv:2303.15343.</a>

<a id="dinov2-2024">Oquab Maxime, Darcet Timothée, Moutakanni Théo, Vo Huy, Szafraniec Marc, Khalidov Vasil, Fernandez Pierre, Haziza Daniel, Massa Francisco, El-Nouby Alaaeldin, Assran Mahmoud, Ballas Nicolas, Galuba Wojciech, Misra Ishan, Rabbat Michael, Sharma Vasu, Synnaeve Gabriel, Xu Hu, Jegou Hervé, Mairal Julien, Labatut Patrick, Joulin Armand, Bojanowski Piotr. (2024). DINOv2: Learning Robust Visual Features Without Supervision. arXiv preprint arXiv:2304.07193.</a>

<a id="internlm2-2024">Cai Zheng, Cao Maosong, Chen Haojiong, Chen Kai, Chen Keyu, Chen Xin, Chen Xun, Chen Zehui, Chen Zhi, Chu Pei, Dong Xiaoyi, Duan Haodong, Fan Qi, Fei Zhaoye, Gao Yang, Ge Jiaye, Gu Chenya, Gu Yuzhe, Gui Tao, Guo Aijia, Guo Qipeng, He Conghui, Hu Yingfan, Huang Ting, Jiang Tao, Jiao Penglong, Jin Zhenjiang, Lei Zhikai, Li Jiaxing, Li Jingwen, Li Linyang, Li Shuaibin, Li Wei, Li Yining, Liu Hongwei, Liu Jiawei, Liu Kaiwen, Liu Kuikun, Liu Xiaoran, Lv Chengqi, Lv Haijun, Lv Kai, Ma Li, Ma Runyuan, Ma Zerun, Ning Wenchang, Ouyang Linke, Qiu Jiantao, Qu Yuan, Shang Fukai, Shao Yunfan, Song Demin, Song Zifan, Sui Zhihao, Sun Peng, Sun Yu, Tang Huanze, Wang Bin, Wang Guoteng, Wang Jiaqi, Wang Jiayu, Wang Rui, Wang Yudong, Wang Ziyi, Wei Xingjian, Weng Qizhen, Wu Fan, Xiong Yingtong, Xu Chao, Xu Ruiliang, Yan Hang, Yan Yirong, Yang Xiaogui, Ye Haochen, Ying Huaiyuan, Yu Jia, Yu Jing, Zang Yuhang, Zhang Chuyu, Zhang Li, Zhang Pan, Zhang Peng, Zhang Ruijie, Zhang Shuo, Zhang Songyang, Zhang Wenjian, Zhang Wenwei, Zhang Xingcheng, Zhang Xinyue, Zhao Hui, Zhao Qian, Zhao Xiaomeng, Zhao Fengzhe, Zhou Zaida, Zhou Jingming, Zhuo Jingming, Zou Yicheng, Qiu Xipeng, Qiao Yu, Lin Dahua. (2024). InternLM2 Technical Report. arXiv preprint arXiv:2403.17297.</a>

<a id="omg-seg-arxiv-2024">Li Xiangtai, Yuan Haobo, Li Wei, Ding Henghui, Wu Size, Zhang Wenwei, Li Yining, Chen Kai, Loy Chen Change. (2024). OMG-Seg: Is One Model Good Enough for All Segmentation? arXiv preprint arXiv:2401.10229.</a>

<a id="seem-2023">Zou Xueyan, Yang Jianwei, Zhang Hao, Li Feng, Li Linjie, Wang Jianfeng, Wang Lijuan, Gao Jianfeng, Lee Yong Jae. (2023). Segment Everything Everywhere All at Once. arXiv preprint arXiv:2304.06718.</a>
