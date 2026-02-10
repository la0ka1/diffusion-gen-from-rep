# Generalization of Diffusion Models Arises with a Balanced Representation Space

<p align="center">
  <a href="https://la0ka1.github.io/Diffusion_Rep_Gen_test/"><img alt="Blog" src="https://img.shields.io/badge/Blog-GitHub%20Pages-2ea44f.svg"></a>
  <a href="https://arxiv.org/abs/2512.20963"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2512.20963-b31b1b.svg"></a>
  <a href="https://alphaxiv.org/abs/2512.20963"><img alt="alphaXiv" src="https://img.shields.io/badge/alphaXiv-2512.20963-ff6b6b.svg"></a>
  <a href="https://openreview.net/forum?id=57THeGgNAN"><img alt="OpenReview" src="https://img.shields.io/badge/OpenReview-ICLR%202026-0b7fd1.svg"></a>
</p>

Code and figures for the ICLR 2026 paper.
[**Generalization of Diffusion Models Arises with a Balanced Representation Space**](https://www.alphaxiv.org/abs/2512.20963).

<p align="center">
  <img src="Figs/teaser.png" alt="Teaser" width="80%">
</p>

## At a glance
- `1-Theory/` (CPU): ReLU-DAE analysis + diffusion extension.
- `2-Application/` (GPU recommended): SD v1.4 + LAION representations and steering.
- `Figs/`: plots are saved automatically when you run the notebooks and are shown in this preview.

## Quickstart
```bash
pip install numpy torch torchvision diffusers transformers accelerate datasets scikit-learn pillow tqdm matplotlib seaborn
```


## 1-Theory (CPU)
Notebooks: [`ReLU_DAE.ipynb`](1-Theory/ReLU_DAE.ipynb), [`ReLU_Diffusion.ipynb`](1-Theory/ReLU_Diffusion.ipynb)

ReLU-DAE on CelebA: representations and weights for memorization vs. generalization (Figures 4-5).

<p align="center">
  <a href="Figs/celeba_rep.png"><img src="Figs/celeba_rep.png" alt="Figure 5: CelebA representations" width="48%"></a>
  <a href="Figs/celeba_weights.png"><img src="Figs/celeba_weights.png" alt="Figure 4: CelebA DAE weights" width="48%"></a>
</p>

A time-conditioned diffusion extension in the same toy setup.
<p align="center">
  <a href="Figs/celeba_sampling_mem.png"><img src="Figs/celeba_sampling_mem.png" alt="CelebA Mem sampling" width="48%"></a>
  <a href="Figs/celeba_sampling_gen.png"><img src="Figs/celeba_sampling_gen.png" alt="CelebA Gen sampling" width="48%"></a>
</p>

## 2-Application (GPU recommended)
Notebooks: [`SD_compare_reps.ipynb`](2-Application/SD_compare_reps.ipynb), [`SD_steering.ipynb`](2-Application/SD_steering.ipynb)

Stable Diffusion v1.4 + LAION representation structure and separation (Figures 6a/6b).

<p align="center">
  <a href="Figs/laion_reps.png"><img src="Figs/laion_reps.png" alt="Figure 6a: LAION reps" width="48%"></a>
  <a href="Figs/laion_separation.png"><img src="Figs/laion_separation.png" alt="Figure 6b: LAION separation" width="48%"></a>
</p>

Representation steering: generalization vs. memorization (Figure 8).

<p align="center">
  <a href="Figs/laion_steering_gen.png"><img src="Figs/laion_steering_gen.png" alt="Figure 8: Steering generalization" width="48%"></a>
  <a href="Figs/laion_steering_mem.png"><img src="Figs/laion_steering_mem.png" alt="Figure 8: Steering memorization" width="48%"></a>
</p>

## Additional figure
Steering trajectory in representation space, showing *separation between concepts/styles* and how steering transfers across them.
<p align="center">
  <a href="Figs/laion_rep_traj.png"><img src="Figs/laion_rep_traj.png" alt="Steering trajectory" width="45%"></a>
</p>

## Citation
```bibtex
@inproceedings{zhang2026balanceddiffusion,
  title={Generalization of Diffusion Models Arises with a Balanced Representation Space},
  author={Zhang, Zekai and Li, Xiao and Li, Xiang and Shi, Lianghe and Wu, Meng and Tao, Molei and Qu, Qing},
  booktitle={ICLR},
  year={2026}
}
```
