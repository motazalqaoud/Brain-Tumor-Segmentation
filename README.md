---
title: Brain Tumor Segmentation 3D
emoji: 🧠
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# brain-tumor-segmentation-3d

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org)
[![MONAI](https://img.shields.io/badge/MONAI-1.0%2B-green)](https://monai.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-motazalqaoud-black)](https://github.com/motazalqaoud)

> Real 3D volumetric brain tumor segmentation: full NIfTI volumes in, full 3D
> segmentation masks out. No 2D slice shortcut anywhere in the pipeline.

## Why this exists

The [Brain-Tumor-Segmentation](https://github.com/motazalqaoud/Brain-Tumor-Segmentation)
repo in this portfolio is called "3D Attention U-Net," but look closely at its
architecture notes: it operates on "pseudo-3D inputs" with a depth of 2-8
frames, and pooling only touches height/width. That's 2D images stacked with
an artificial depth dimension — not a real volumetric scan.

This project is the honest version: a genuine 3D Attention U-Net operating on
full (C, H, W, D) MRI volumes, trained on real multi-modal BraTS data, with
sliding-window inference over entire scans at evaluation time.

## The dataset

**Medical Segmentation Decathlon, Task01_BrainTumour** — 750 4D MRI volumes
(484 train / 266 test), sourced from the real BraTS 2016/2017 challenge.
Four co-registered modalities per subject: FLAIR, T1w, T1gd (contrast-enhanced),
T2w. Publicly downloadable with no registration wall (unlike raw BraTS),
via MONAI's built-in downloader — see Setup.

| Region | Raw labels combined | Meaning |
|---|---|---|
| TC (Tumor Core) | 1 + 4 | Necrotic core + enhancing tumor |
| WT (Whole Tumor) | 1 + 2 + 4 | Everything abnormal: core + edema |
| ET (Enhancing Tumor) | 4 | Actively enhancing tissue only |

These three regions are **nested, not mutually exclusive** (ET ⊂ TC ⊂ WT),
which is why this is a 3-channel multi-label (independent sigmoid per
channel) problem rather than a 4-class softmax problem — this is also exactly
how the official BraTS leaderboard evaluates submissions.

Citation:
> Simpson, A. L. et al. A large annotated medical image dataset for the
> development and evaluation of segmentation algorithms. *Medical Segmentation
> Decathlon* (2019). http://medicaldecathlon.com/

**The dataset is not bundled in this repo** (~7GB compressed). See Setup.

## Clinical context

| Common tutorial | This repo |
|---|---|
| 2D slices, sometimes dressed up as "3D" with a fake depth axis | Genuine (H, W, D) volumetric input and output, no shortcut |
| Random image-level train/test split | Split by **subject** (each 4D file is one patient — no leakage) |
| Patch-only evaluation | **Sliding-window inference over the full volume** at eval/inference time, since a model that only ever sees training patches still needs to work on a whole clinical scan |
| Dice score only | Dice **and** 95th-percentile Hausdorff distance (both official BraTS metrics) |
| One-size-fits-all training config | GPU-memory-aware patch/batch sizing — doesn't assume you know your node's exact GPU in advance |

## Architecture

**3D Attention U-Net** (MONAI `AttentionUnet`, `spatial_dims=3`), 4 input
channels (the 4 MRI modalities) → 3 output channels (TC/WT/ET), with
attention gates on each skip connection. This keeps the same architecture
family already used in this portfolio's other segmentation project, now
correctly applied to real volumetric data instead of 2D slices.

MONAI's `SegResNet` — the architecture that actually won BraTS 2018 ("3D MRI
Brain Tumor Segmentation Using Autoencoder Regularization," Myronenko 2018)
— is a well-documented, strong alternative worth trying on this exact
dataset as a comparison; swapping `src/model.py`'s `build_model()` to use it
is a small change if you want to benchmark both.

## Repository structure

```
brain-tumor-segmentation-3d/
├── app.py                    # Gradio demo: NIfTI upload -> slice-overlay visualization
├── data_prep.py               # Downloads Task01_BrainTumour via MONAI's built-in downloader
├── requirements.txt
├── slurm/
│   └── train_job.slurm        # SLURM batch script for university/HPC clusters
├── src/
│   ├── dataset.py             # MONAI transforms: load, resample, patch-crop, BraTS multi-label conversion
│   ├── model.py                # 3D Attention U-Net, GPU-memory-aware sizing
│   ├── train.py                 # Patch-based training + full-volume sliding-window validation
│   ├── evaluate.py              # Per-region Dice + Hausdorff distance (HD95) on full volumes
│   └── predict.py               # Single-subject inference -> segmentation NIfTI output
└── checkpoints/                 # best_model.pth lands here after training (gitignored)
```

## Setup

```bash
git clone https://github.com/motazalqaoud/brain-tumor-segmentation-3d
cd brain-tumor-segmentation-3d
pip install -r requirements.txt
```

Download the dataset (MONAI handles this automatically — ~7GB, needs internet access):

```bash
python data_prep.py --root_dir ./data
```

This creates `./data/Task01_BrainTumour/` with `imagesTr/` (484 4D volumes) and
`labelsTr/` (matching masks).

## Training

**On a SLURM/HPC cluster** (recommended — see why below):
```bash
sbatch slurm/train_job.slurm
```
See that file for one-time environment setup notes (modules, conda env, dataset path).

**Locally, if you have a GPU:**
```bash
python src/train.py --data_dir ./data/Task01_BrainTumour --epochs 100
```

Patch size (128³ or 96³) and batch size are **auto-detected from available GPU
memory** (`src/dataset.py:recommend_patch_and_batch_size`) — this was written
without knowing in advance whether a SLURM job would land on a 16GB or 80GB
node, so it sizes itself down rather than risking an out-of-memory crash
partway through a multi-hour job. Override with `--patch_size`/`--batch_size`
once you know your node's GPU and want to tune for throughput instead of
safety.

**Why GPU/HPC, not a laptop CPU:** 3D convolutions over full medical volumes
are memory- and compute-heavy in a way 2D image classifiers aren't. A CPU can
verify the pipeline runs (tiny patch size, a couple of epochs), but a real
100-epoch run needs a GPU — hours instead of potentially days.

Training uses mixed precision (AMP) by default, Dice loss (sigmoid, per
nested region), cosine LR annealing, and checkpoints the best model by mean
validation Dice across all three regions. Full-volume validation via
sliding-window inference runs every `--val_interval` epochs (default 5).

## Evaluation

```bash
python src/evaluate.py --data_dir ./data/Task01_BrainTumour --checkpoint checkpoints/best_model.pth
```

Reports, on full held-out volumes (sliding-window inference, not patches):
- Per-region Dice (TC, WT, ET) and mean
- Per-region 95th-percentile Hausdorff distance (HD95, in mm)

These are the two standard BraTS leaderboard metrics.

## Inference on a new scan

```bash
python src/predict.py --image path/to/subject_4mod.nii.gz \
    --checkpoint checkpoints/best_model.pth --output segmentation.nii.gz
```

Produces a segmentation NIfTI you can load in 3D Slicer, ITK-SNAP, or any
NIfTI viewer alongside the original scan. Expects a single 4D NIfTI (H, W, D,
4) in the MSD layout; if your modalities are four separate files, stack them
first (see the docstring in `predict.py`).

## Running the demo locally

```bash
python app.py
```

**If `checkpoints/best_model.pth` doesn't exist yet, the app runs in a
clearly-labeled demo mode**: the full 3D pipeline (NIfTI loading, resampling,
sliding-window inference over the entire volume) runs for real, but the
network is untrained, so the segmentation shown isn't meaningful. Train with
`src/train.py` or `sbatch slurm/train_job.slurm` and the app will
automatically detect the checkpoint.

Note: CPU-tier Hugging Face Spaces will be slow for 3D sliding-window
inference (potentially 1-2 minutes per volume). A GPU-tier Space, or running
locally/on the HPC, is recommended for anything beyond a quick demo.

## Results

**No checkpoint is bundled yet** — training requires the ~7GB dataset (not
included, see Setup) and GPU compute time. Once trained, run `src/evaluate.py`
and paste the output here:

| Region | Dice | HD95 (mm) |
|---|---|---|
| TC | *(run evaluate.py)* | *(run evaluate.py)* |
| WT | *(run evaluate.py)* | *(run evaluate.py)* |
| ET | *(run evaluate.py)* | *(run evaluate.py)* |

For context, published results on this architecture family and dataset
report mean Dice scores in the 0.80-0.90 range for WT, somewhat lower (0.70-0.85)
for TC and ET, which are smaller and harder sub-regions — these are literature
ranges, not results from this specific checkpoint.

## Limitations and disclaimer

**This is a research and portfolio project, not a medical device.** Not
validated prospectively, not reviewed by a regulatory body, must never be
used to make or defer an actual diagnosis. BraTS-derived data comes from a
specific set of institutions and scanner protocols; a model trained only on
this data should not be assumed to generalize to arbitrary clinical scanners
without further validation. Any actual diagnosis requires a radiologist and
the full clinical picture.

## Tech stack

| Tool | Purpose |
|---|---|
| `MONAI` | Medical imaging transforms, 3D networks, sliding-window inference |
| `PyTorch` | Deep learning |
| `nibabel` | NIfTI I/O |
| `Gradio` | Interactive demo |
| `matplotlib` | Slice-overlay visualization |

## About the author

**Motaz Alqaoud, PhD**
- PhD in Biomedical Engineering with focus on medical image analysis and deep learning
- Senior AI/ML Engineer specializing in medical imaging, segmentation models, and clinical AI systems
- GitHub: [@motazalqaoud](https://github.com/motazalqaoud)
- LinkedIn: [linkedin.com/in/motazalqaoud](https://linkedin.com/in/motazalqaoud)

## License

MIT — see [LICENSE](LICENSE).
