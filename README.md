# 🧠 Medical Imaging AI Basics

> A practical, clinical-grade toolkit for AI in medical imaging — built by a biomedical engineer who has actually worked in the OR.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-motazalqaoud-black)](https://github.com/motazalqaoud)

---

## 👋 Who is this for?

This repo is for:
- Engineers entering **medical AI** who want real, working code (not toy examples)
- Researchers looking for **clinical-context preprocessing pipelines**
- Anyone building AI tools for **surgical navigation, diagnostics, or medical devices**

This is **not** another generic deep learning tutorial. Every notebook here reflects real challenges from medical imaging in clinical settings.

---

## 📁 Repository Structure

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
│   │   ├── dicom_loader.py                      # DICOM stack → numpy/tensor
│   │   ├── nifti_loader.py                      # NIfTI loader with metadata
│   │   └── transforms.py                        # Medical-specific augmentations
│   ├── segmentation/
│   │   ├── unet.py                              # U-Net architecture
│   │   └── losses.py                            # Dice loss, BCE+Dice combo
│   └── visualization/
│       └── viewer.py                            # 3-plane viewer (axial/sagittal/coronal)
│
├── data/
│   └── samples/                                 # Sample slices for testing
│
├── docs/
│   └── clinical_context.md                      # Why these choices matter in real hospitals
│
├── requirements.txt
└── README.md
```

---

## 🚀 Quickstart

```bash
git clone https://github.com/motazalqaoud/medical-imaging-ai-basics.git
cd medical-imaging-ai-basics
pip install -r requirements.txt
jupyter notebook notebooks/
```

---

## 📓 Notebooks

### 1. Load & Visualize Medical Images
`notebooks/01_load_visualize_medical_images.ipynb`

- Load DICOM series and NIfTI files
- Visualize axial, sagittal, and coronal planes
- Understand voxel spacing, orientation, and metadata
- Why this matters: wrong spacing = wrong measurements in the OR

### 2. Preprocessing Pipeline
`notebooks/02_preprocessing_pipeline.ipynb`

- Intensity normalization (Z-score, min-max, percentile clipping)
- Resampling to isotropic voxel spacing
- Data augmentation for medical images (no random flipping of anatomy!)
- Why this matters: preprocessing failures are the #1 reason medical AI breaks in production

### 3. Tumor Segmentation with U-Net
`notebooks/03_tumor_segmentation_unet.ipynb`

- Build U-Net from scratch in PyTorch
- Train on sample breast MRI slices
- Dice loss + evaluation metrics
- Why this matters: segmentation is the foundation of surgical navigation

---

## 🔬 Clinical Context

> Most medical AI tutorials miss the clinical reality. Here's what's different about this repo:

| Common Tutorial | This Repo |
|---|---|
| Random image flipping | Anatomy-aware augmentation |
| Pixel normalization only | Voxel spacing + orientation handling |
| Accuracy metric | Dice score + Hausdorff distance |
| 2D slices only | 3D volume awareness |
| Generic datasets | Breast MRI / surgical navigation context |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| `pydicom` | DICOM file loading |
| `nibabel` | NIfTI file loading |
| `SimpleITK` | Resampling, registration |
| `PyTorch` | Deep learning (U-Net) |
| `MONAI` | Medical imaging transforms |
| `matplotlib` | Visualization |
| `numpy` | Array operations |

---

## 🧑‍💻 About the Author

**Motaz Alqaoud, PhD**
Senior AI/ML Engineer @ Abbott | Biomedical Engineer

- PhD in Biomedical Engineering — dissertation on real-time breast cancer surgery navigation using AI + FEA
- Experience in surgical navigation, medical device ML, MRI/US registration
- GitHub: [@motazalqaoud](https://github.com/motazalqaoud)
- LinkedIn: [linkedin.com/in/motazalqaoud](https://linkedin.com/in/motazalqaoud)

---

## 🗺️ Roadmap

- [x] DICOM/NIfTI loading
- [x] Preprocessing pipeline
- [x] U-Net segmentation
- [ ] MRI → Ultrasound registration
- [ ] Biomechanical deformation modeling (FEA-based)
- [ ] Real-time inference pipeline
- [ ] Hugging Face Space demo

---

## 📄 License

MIT License — use freely, attribution appreciated.

---

*If this helped you, ⭐ the repo and connect on LinkedIn. Let's build better AI for healthcare.*
