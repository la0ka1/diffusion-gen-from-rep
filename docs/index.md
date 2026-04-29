---
title: "Generalization of Diffusion Models Arises with a Balanced Representation Space (ICLR '26)"
permalink: /
layout: single
classes: wide
---

<p class="home-heading"><a href="https://la0ka1.github.io/" aria-label="Back to homepage"><span aria-hidden="true">&larr;</span> Back to homepage</a></p>

<p class="button-row">
<a class="btn btn--success" href="{{ site.github.repository_url }}"><i class="fab fa-github" aria-hidden="true"></i> Code</a>
<a class="btn btn--arxiv" href="https://arxiv.org/abs/2512.20963"><i class="fas fa-file-alt" aria-hidden="true"></i> arXiv</a>
<a class="btn btn--alpha" href="https://alphaxiv.org/abs/2512.20963"><i class="fas fa-comments" aria-hidden="true"></i> alphaXiv</a>
<a class="btn btn--openreview" href="https://openreview.net/forum?id=57THeGgNAN"><i class="fas fa-book-open" aria-hidden="true"></i> OpenReview</a>
<a class="btn btn--warning" href="https://drive.google.com/file/d/12A0cRa1vq_kCqEHYl_2rMLuMIZv64RmV/view?usp=drive_link"><i class="fas fa-chalkboard" aria-hidden="true"></i> Slides</a>
</p>

<p class="author-row">
<a class="author-link" href="https://la0ka1.github.io/"><strong>Zekai Zhang</strong></a><sup>1,*</sup>, Xiao Li<sup>1,*</sup>, Xiang Li<sup>1</sup>, Lianghe Shi<sup>1</sup>, Meng Wu<sup>1</sup>, Molei Tao<sup>2</sup>, and Qing Qu<sup>1</sup>
</p>
<p class="affiliation-row">
<sup>1</sup>University of Michigan &nbsp;&middot;&nbsp; <sup>2</sup>Georgia Institute of Technology
&nbsp;&middot;&nbsp; <sup>*</sup>Equal contribution
</p>


<p class="tldr-box"><strong>TL;DR.</strong> Diffusion models' generalization (ability to generate novel samples) should be studied together with their representation learning (ability to perceive and understand samples).</p>

---

<p class="lead-italic"><em>Generalization is a strong bias/capability of neural networks</em></p>
<p>A generalizing model learns beyond the finite training set to approximate the <em>underlying distribution</em> <span class="math-inline">\(p_{\mathrm{gt}}\)</span> (often human-defined or perceived).</p>

<img class="feature-figure figure--narrow" src="{{ '/assets/figures/network_learns.png' | relative_url }}" alt="Diagram of a model learning beyond finite training samples toward the underlying distribution." width="55%" style="display:block;margin:auto;" />
<p class="figure-caption"><strong>Generalization of diffusion models.</strong> Learning <span class="math-inline">\(p_{\mathrm{gt}}\)</span> allows the model to generate novel and realistic samples.</p>

<p>For diffusion models, this means generating realistic (in-distribution) images not present in the training set, and this is done by <button class="inline-note-trigger" type="button" aria-expanded="false" aria-controls="note-learning" data-note-target="note-learning">learning a denoising network</button> from training samples <span class="math-inline">\(\bm{x}_{i=1\dots n}\sim p_{\mathrm{gt}}\)</span>.</p>
<div id="note-learning" class="inline-note-body" hidden>

<p>The denoising network is trained to recover clean samples from noisy inputs across noise levels:</p>
$$
\frac{1}{T}\sum_{t=0}^{T}
\mathbb{E}_{\bm{x}\sim p_{\mathrm{gt}},\,\bm{\epsilon}\sim\mathcal{N}(\bm{0},\bm{I})}
\!\left[\big\|\bm{f}_{\bm{\theta}}(\bm{x}+\sigma_t \bm{\epsilon},t)-\bm{x}\big\|^2\right].
$$

</div>

<p>After training, we learn a rich <button class="inline-note-trigger inline-note-trigger--math" type="button" aria-expanded="false" aria-controls="note-fgt" data-note-target="note-fgt">\(\bm{f}_{\bm{\theta}}(\bm{y}, t)\approx\bm{f}_{\mathrm{gt}}(\bm{y}, t)\)</button> that removes noise with respect to <span class="math-inline">\(p_{\mathrm{gt}}\)</span>. Sampling then starts from noise and iteratively denoises into meaningful images, i.e., <em>generalizes</em>.</p>
<div id="note-fgt" class="inline-note-body" hidden>

$$
\bm{f}_{\bm{\theta}}(\bm{y}, t)\approx\bm{f}_{\mathrm{gt}}(\bm{y}, t)=\mathbb{E}\!\left[\bm{x} \mid \bm{x} + \sigma_t \bm{\epsilon} = \bm{y};\, \bm{x} \sim p_{\mathrm{gt}}\right].
$$

</div>
<p>However, such success is <em>not</em> guaranteed by neural networks' ability to approximate any function. Otherwise, they would overfit to an empirical solution <button class="inline-note-trigger inline-note-trigger--math" type="button" aria-expanded="false" aria-controls="note-femp" data-note-target="note-femp">\(\bm{f}_{\mathrm{emp}}(\bm{y}, t)\)</button> that denoises inputs toward training samples and effectively <em>memorizes</em> them. So what bias of networks allows diffusion models to generalize? We connect it to another crucial aspect: their learned internal representations.</p>
<div id="note-femp" class="inline-note-body" hidden>
$$
\bm{f}_{\mathrm{emp}}(\bm{y}, t)
= \mathbb{E}\!\left[\bm{x}\mid \bm{x}+\sigma_t\bm{\epsilon}=\bm{y};\,\bm{x}\sim p_{\mathrm{emp}}\right]
= \frac{\sum_{i=1}^n \mathcal{N}(\bm{y};\bm{x}_i,\sigma_t^2\bm{I})\,\bm{x}_i}
{\sum_{i=1}^n \mathcal{N}(\bm{y};\bm{x}_i,\sigma_t^2\bm{I})}\neq \bm{f}_{\mathrm{gt}}(\bm{y}, t).
$$

</div>

---
<p class="lead-italic"><em>Looking into networks.</em></p>
<p>We study training of parameterized diffusion models as a two-layer ReLU network, under a single noise level. Since it is also a <button class="inline-note-trigger" type="button" aria-expanded="false" aria-controls="note-dae" data-note-target="note-dae">denoising autoencoder</button>, we call it <strong>ReLU-DAE</strong>. This is a minimal nonlinear model for studying representation learning and denoising.</p>
<div id="note-dae" class="inline-note-body" hidden>
  <p><strong>Lineage:</strong> Pascal Vincent, "<a href="https://direct.mit.edu/neco/article-abstract/23/7/1661/7677">A Connection Between Score Matching and Denoising Autoencoders</a>," <em>Neural Computation</em>, 2011; and Yoshua Bengio, Li Yao, Guillaume Alain, and Pascal Vincent, "<a href="https://proceedings.neurips.cc/paper/2013/hash/559cb990c9dffd8675f6bc2186971dc2-Abstract.html">Generalized Denoising Auto-Encoders as Generative Models</a>," NeurIPS 2013.</p>
</div>

$$
\bm{f}_{\bm{W}_2,\bm{W}_1}(\bm{x})
= \bm{W}_2\bm{h}(\bm{x})
= \bm{W}_2\,[\bm{W}_1^\top \bm{x}]_+.
$$

<p>We prove that under the diffusion loss:<br>
(i) <em>memorization</em> corresponds to <span class="math-inline">\(\bm{W}_1, \bm{W}_2\)</span> storing raw samples in the weights, approximating <span class="math-inline">\(\bm{f}_{\mathrm{emp}}\)</span>;<br>
(ii) <em>generalization</em> corresponds to <span class="math-inline">\(\bm{W}_1, \bm{W}_2\)</span> learning local data statistics, efficiently approximating <span class="math-inline">\(\bm{f}_{\mathrm{gt}}\)</span>;<br>
(iii) a <em>hybrid regime</em> due to data imbalance.</p>

<img class="feature-figure" src="{{ '/assets/figures/teaser.png' | relative_url }}" alt="Illustration of memorization, hybrid behavior, and generalization regimes in ReLU-DAE learning." width="80%" style="display:block;margin:auto;" />
<p class="figure-caption"><strong>Three regimes in ReLU-DAE learning.</strong> Memorization (left), hybrid (center), and generalization (right).</p>

---
<p class="lead-italic"><em>Representation learning in real models:</em></p>
Memorized samples align perfectly with stored structures and produce *spiky* representations: a strong single-neuron stimulation or retrieval of a specific training example.  
Generalized samples align with a broader set of structures, yielding *balanced* representations that compose across neurons and reflect the underlying distribution, as coordinates for the image manifold.

<div class="figure-grid figure-grid--triple">
  <div style="flex:1; min-width:240px;">
    <img src="{{ '/assets/figures/celeba_rep.png' | relative_url }}" alt="Balanced and spiky representation examples on CelebA." style="width:100%;" />
  </div>
  <div style="flex:1; min-width:240px;">
    <img src="{{ '/assets/figures/imagenet_rep.png' | relative_url }}" alt="Balanced and spiky representation examples on ImageNet." style="width:100%;" />
  </div>
  <div style="flex:1; min-width:240px;">
    <img src="{{ '/assets/figures/LAION_rep.png' | relative_url }}" alt="Balanced and spiky representation examples on LAION." style="width:100%;" />
  </div>
</div>

<p class="figure-caption"><strong>Same signature in real diffusion models.</strong> The spiky-vs-balanced separation persists in large models.</p>

---

<p>Generalized reps can also be <button class="inline-note-trigger" type="button" aria-expanded="false" aria-controls="note-manipulated" data-note-target="note-manipulated">manipulated</button> to change the final output, whereas memorized ones cannot.</p>
<div id="note-manipulated" class="inline-note-body" hidden>
  <p>Lvmin Zhang, Anyi Rao, and Maneesh Agrawala, "<a href="https://openaccess.thecvf.com/content/ICCV2023/html/Zhang_Adding_Conditional_Control_to_Text-to-Image_Diffusion_Models_ICCV_2023_paper.html">Adding Conditional Control to Text-to-Image Diffusion Models</a>," ICCV 2023.</p>
</div>
<div class="figure-grid figure-grid--dual figure-grid--center">
  <div style="flex:1; min-width:260px; max-width:360px;">
    <img src="{{ '/assets/figures/man_age.png' | relative_url }}" alt="Representation steering successfully aging a generalized sample." style="width:100%;" />
    <p class="mini-caption"><strong>+Old (Gen.)</strong></p>
  </div>
  <div style="flex:1; min-width:260px; max-width:360px;">
    <img src="{{ '/assets/figures/mem_dt_age.png' | relative_url }}" alt="Representation steering failing to age a memorized sample." style="width:100%;" />
    <p class="mini-caption"><strong>+Old (Mem.)</strong></p>
  </div>
</div>

<p class="figure-caption"><strong>Image editing via representation steering.</strong> Works for generalized samples, not for memorized samples.</p>

---
<p class="lead-italic"><em>Our theory starts from a simple two-layer network, but</em></p>
we believe it reflects a fundamental mechanism in deep models: they project noisy inputs onto learned low-dimensional structure, arranging visually similar inputs into similar and meaningful activations (via ReLU gating in our theory).

<p>This smart arrangement underlies their <button class="inline-note-trigger" type="button" aria-expanded="false" aria-controls="note-compressing-denoising" data-note-target="note-compressing-denoising">compressing and denoising</button> behavior and shares an intuitive similarity with human perception. Internally, this appears as <em>representation learning</em>.</p>
<div id="note-compressing-denoising" class="inline-note-body" hidden>
  <p><strong>Compressing:</strong> Sam Buchanan, Druv Pai, Peng Wang, and Yi Ma, <a href="https://people.eecs.berkeley.edu/~yima/Publication.html"><em>Principles and Practice of Deep Representation Learning</em></a>, online book, 2026. <strong>Denoising:</strong> Dmitry Ulyanov, Andrea Vedaldi, and Victor Lempitsky, "<a href="https://openaccess.thecvf.com/content_cvpr_2018/html/Ulyanov_Deep_Image_Prior_CVPR_2018_paper.html">Deep Image Prior</a>," CVPR 2018.</p>
</div>

<img class="feature-figure" src="{{ '/assets/figures/network_learns_rep.png' | relative_url }}" alt="Diagram showing how learned representations organize inputs for denoising and generation." width="75%" style="display:block;margin:auto;" />
