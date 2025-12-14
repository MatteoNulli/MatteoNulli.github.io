---
layout: page
title: ELLIS Honours Student 2025
description: Part of ELLIS Honours Students for the Master of AI
img: assets/img/ellis_cees.png
importance: 1
category: research
related_publications: false
---


Below is a summary of my experience as [ELLIS Honours Student](https://ivi.fnwi.uva.nl/ellis/news-/recognizing-young-ai-talent-excellence-11-ellis-msc-honours-students-graduated-cum-laude-under-joint-supervision-of-ellis-members-across-europe/#:~:text=(6)%20Matteo%20Nulli,rigorous%20and%20rewarding.), you can also find more information [here](https://ivi.fnwi.uva.nl/ellis/news-/recognizing-young-ai-talent-excellence-11-ellis-msc-honours-students-graduated-cum-laude-under-joint-supervision-of-ellis-members-across-europe/#:~:text=(6)%20Matteo%20Nulli,rigorous%20and%20rewarding.). 



<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/ellis_cees.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
  <b>Figure 1:</b> Cees Snoek & Matteo Nulli at the ELLIS Honors Presentations.
</div>


Title: <b>Object-Guided Visual Tokens: Eliciting Compositional Reasoning in Multimodal Language Models</b>

Links: 📄 [Paper](https://openreview.net/forum?id=yvY1T3hHEQ) | 📜 [Full Thesis](https://github.com/MatteoNulli/og_llava/blob/main/paper/LongPaper.pdf) | 📝 [Blogpost](https://matteonulli.github.io/blog/2025/ogllava/) | 🧑‍💻 [Code](https://github.com/MatteoNulli/og_llava/tree/main)



Co-supervisors: Ivona Najdenkoska (ELLIS Postdoc-Amsterdam), Yuki M. Asano (ELLIS Member-Nuremberg), Marcel Worring (ELLIS Fellow-Amsterdam), Vladimir Orshoulevich (eBay Foundation Models Team)

Research Summary:

Standard Multimodal Large Language Models (MLLMs) employ contrastive pre-trained vision encoders whose performance, while undoubtedly good in a good range of tasks, falls short in Compositional Understanding and Reasoning on the visual input. This is mostly due their pre-training objective aimed at retrieval between similar image/captions rather than in-depth understanding of all components of an image. Moreover, while state-of-the-art image encoding methods yield strong performance, they inflate the number of visual input tokens by roughly two to three times, thereby significantly lengthening both training and inference times.

To alleviate these issues, we present OG-LLaVA (Object-Guided LLaVA), a novel multimodal architecture which, through a novel connector design (OG-Fusion), enhances the model’s ability to understand and reason about visual content without substantially increasing the number of tokens or unfreezing the Vision Encoder. A core element of OG-Fusion is the combination of CLIP output representations with segmentation masks. By leveraging the descriptive power of advanced segmentation models, OG-LLaVA attains superior performance at tasks which require a deeper understanding of object relationships and spatial arrangements and, more broadly, within the domains of compositional reasoning and visual grounding.

MSc Honours experience:

The MSc Honours Programme has been a real game‑changer for my research. While working on my thesis, I was able to dig deep into multimodal learning exploring the subtle challenges of compositional reasoning and visual understanding within vision‑language models. The best part was spending a month in Nuremberg: teaming up with Prof. Yuki Asano at the new Foundational AI Lab (University of Technology Nuremberg) let me sharpen my experiments, try out fresh ideas, and see how blue‑sky theory can quickly turn practical. This experience—made possible by the ELLIS HonoursProgramme—allowed me to learn directly from Europe’s leading AI researchers, while PhD Ivona Nadjenkoska, the VISLab team at the University of Amsterdam, and my supervisors at eBay ensured that each step stayed both rigorous and rewarding.