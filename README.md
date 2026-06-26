# Brain Tumor Segmentation — 3D Attention U-Net

> Multi-class segmentation of glioma, meningioma, and pituitary tumors from MRI using 3D Attention U-Net with WHO-grade classification.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-motazalqaoud-black)](https://github.com/motazalqaoud)

---

## Who is this for?

This repo is for:
- Engineers building **brain tumor detection systems** with production-ready code
- Researchers training **segmentation models on real MRI data** (12K+ images from Kaggle)
- Medical professionals developing **diagnostic support tools** for neuroradiology
- Anyone deploying AI for **surgical planning and intraoperative guidance**

This is **not** a toy tutorial. Every component reflects real clinical workflows from neurosurgery and radiology departments.

---

## Repository Structure

```
medical-imaging-ai-basics/
│
├── notebooks/
│   ├── 01_load_visualize_medical_images.ipynb   # DICOM & NIfTI loading, 3D viz
│   ├── 02_preprocessing_pipeline.ipynb          # Normalization, augmentation, resampling
│   └── 03_tumor_segmentation_unet.ipynb         # U-Net from scratch for lesion segmentation
│
├── src/
│   ├── preprocessing/
│   │   ├── dicom_loader.py          # DICOM stack → numpy/tensor
│   │   ├── nifti_loader.py          # NIfTI loader with metadata & affine
│   │   ├── transforms.py            # Anatomy-aware augmentations
│   │   └── brain_tumor_loader.py    # Kaggle brain tumor dataset loader
│   ├── segmentation/
│   │   ├── unet.py                  # 2D U-Net architecture
│   │   ├── losses.py                # Dice, BCE+Dice, Focal losses
│   │   ├── unet3d.py                # 3D Attention U-Net (channel + spatial attention)
│   │   └── losses_advanced.py       # Weighted Dice, Focal, Boundary, Hybrid loss
│   └── visualization/
│       ├── viewer.py                # 3-plane viewer (axial/sagittal/coronal)
│       └── visualizer3d.py          # Segmentation comparison & training curve plots
│
├── scripts/
│   ├── generate_sample_data.py      # Synthetic NIfTI volumes (no download needed)
│   ├── train.py                     # Train 2D U-Net on synthetic data
│   ├── train3d.py                   # Train 3D Attention U-Net on Kaggle dataset
│   ├── test_model.py                # End-to-end test (dataset → model → viz)
│   └── predict.py                   # Run a trained checkpoint on an image
│
├── data/
│   ├── README.md                    # How to get the datasets
│   └── samples/                     # Synthetic NIfTI pairs (Option A)
│
├── docs/
│   └── clinical_context.md          # Why these choices matter in real hospitals
│
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
git clone https://github.com/motazalqaoud/AI-Basics_Medical-Imaging.git
cd AI-Basics_Medical-Imaging
pip install -r requirements.txt
```

### Option A — Run with synthetic data (no download)

```bash
# Generate 20 synthetic MRI volumes
python scripts/generate_sample_data.py --n 20 --size 128

# Train 2D U-Net on synthetic slices
python scripts/train.py --epochs 30

# Predict with the trained checkpoint
python scripts/predict.py --checkpoint checkpoints/best_unet.pt
```

### Option B — Train on Kaggle brain tumor dataset (12K images)

```bash
# Download via Kaggle API (see data/README.md)
kaggle datasets download -d fernando2rad/brain-tumor-12k-mri-images-w-masks-meta-and-bbox
unzip *.zip -d data/raw/

# Verify setup (dataset loading, model forward pass, visualization)
python scripts/test_model.py --data-root data/raw/Images_

# Train 3D Attention U-Net
python scripts/train3d.py --epochs 50 --batch 4 --data-root data/raw/Images_/Images_

# Resume from checkpoint
python scripts/train3d.py --resume checkpoints/best_model_dice_*.pt --epochs 100
```

---

## Models

### 2D U-Net (`src/segmentation/unet.py`)

Classic U-Net for slice-level binary tumor segmentation.

```python
from src.segmentation import UNet

model = UNet(in_channels=1, n_classes=1, base_filters=32, depth=4)
# Input: (B, 1, H, W)  →  Output: (B, 1, H, W) logits
```

### 3D Attention U-Net (`src/segmentation/unet3d.py`)

Volumetric model with channel attention (SE blocks) and spatial attention gates.
Supports multi-class output for WHO tumor classification.

```python
from src.segmentation import AttentionUNet3D

model = AttentionUNet3D(in_channels=1, num_classes=4, base_filters=32, depth=4)
# Input: (B, 1, D, H, W)  →  Output: (B, 4, D, H, W) logits
# Classes: 0=background, 1=glioma, 2=meningioma, 3=pituitary
```

---

## Loss Functions

| Loss | Module | Use case |
|---|---|---|
| `DiceLoss` | `losses.py` | Binary segmentation |
| `BCEDiceLoss` | `losses.py` | Binary, faster convergence |
| `FocalLoss` | `losses.py` | Very small lesions |
| `WeightedDiceLoss` | `losses_advanced.py` | Multi-class with imbalance |
| `HybridLoss` | `losses_advanced.py` | Multi-class (Dice + Focal + Boundary) |

---

## Notebooks

### 1. Load & Visualize Brain MRI
`notebooks/01_load_visualize_medical_images.ipynb`

- Load Kaggle brain tumor MRI dataset (12K+ images)
- Visualize T1/T2 weighted scans with tumor overlays
- Extract and inspect bounding boxes and segmentation masks

### 2. Preprocessing Pipeline for Brain MRI
`notebooks/02_preprocessing_pipeline.ipynb`

- Intensity normalization for T1/T2 weighted images
- Skull stripping and registration
- Anatomy-aware augmentation (small rotations, no random flips)

### 3. Brain Tumor Segmentation with U-Net
`notebooks/03_tumor_segmentation_unet.ipynb`

- Train 2D U-Net on brain MRI images with ground truth masks
- Multi-class segmentation (glioma, meningioma, pituitary)
- Evaluate with Dice, Hausdorff distance, and volumetric metrics

---

## Clinical Context

> Most medical AI tutorials miss the clinical reality. Here's what's different about this repo:

| Common Tutorial | This Repo |
|---|---|
| Random image flipping | Anatomy-aware augmentation (no random flips) |
| RGB normalization only | T1/T2 weighted intensity handling |
| Pixel accuracy only | Dice + Hausdorff + volumetric metrics |
| 2D slices only | 3D volumetric model with attention gates |
| Binary classifier | Multi-class segmentation (WHO tumor types) |
| Generic datasets | Brain tumor MRI (12K+ clinical images) |

See `docs/clinical_context.md` for the full explanation.

---

## Tech Stack

| Tool | Purpose |
|---|---|
| `pydicom` | DICOM file loading |
| `nibabel` | NIfTI file loading |
| `SimpleITK` | Resampling, registration |
| `PyTorch` | Deep learning (U-Net, 3D Attention U-Net) |
| `MONAI` | Medical imaging transforms |
| `matplotlib` | Visualization |
| `numpy` | Array operations |

---

## About the Author

**Motaz Alqaoud, PhD**

- PhD in Biomedical Engineering — dissertation on real-time breast cancer surgery navigation using AI + FEA
- Experience in surgical navigation, medical device ML, MRI/US registration
- GitHub: [@motazalqaoud](https://github.com/motazalqaoud)
- LinkedIn: [linkedin.com/in/motazalqaoud](https://linkedin.com/in/motazalqaoud)

---

## License

MIT License — use freely, attribution appreciated.

---

*Connect on [LinkedIn](https://linkedin.com/in/motazalqaoud) or open an issue for questions and collaboration.*
