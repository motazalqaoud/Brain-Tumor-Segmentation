# Why Medical Imaging Breaks Standard ML Assumptions

> Motaz Alqaoud, PhD — Biomedical Engineer

If you've built image classifiers or segmentation models on natural images, medical imaging will break most of your default assumptions. This document explains which ones, and what we do differently in this repo as a result.

---

## 1. Images Have Physical Units — Pixels Don't

In natural images, a 256×256 JPEG is just a grid of values. In MRI, every voxel represents a physical volume in millimeters. Two brain MRI scans at 256×256 resolution can have completely different field-of-view sizes depending on scanner settings — one voxel might be 1mm × 1mm, another might be 0.9mm × 1.2mm.

If you feed both to a model without normalizing for this, the model sees the same tumor at different physical scales and learns inconsistent features. It also won't generalize across scanner types.

The fix is resampling to **isotropic spacing** before training — every image gets rescaled so each voxel represents the same physical size. This happens in `nifti_loader.py` before any normalization or augmentation.

---

## 2. Standard Augmentation Will Hurt You

ImageNet training taught us to augment aggressively: random flips, 90° rotations, color jitter. None of these transfer directly to brain MRI.

**Horizontal flip:** The brain has a left side and a right side, and in this dataset tumor location carries spatial meaning. Randomly mirroring images teaches the model that left and right are interchangeable — they're not.

**90° rotation:** MRI is always acquired in a defined orientation (axial = top-down, sagittal = side view, coronal = front view). A 90° rotation doesn't produce a valid MRI from a different angle — it produces something that doesn't exist in the real data distribution. A model trained on these will see something it's never encountered at inference time.

**Intensity jitter:** MRI intensity values encode tissue properties. Unlike RGB images where color jitter is purely visual, aggressive intensity shifts in MRI can make white matter look like gray matter.

Safe augmentation here: rotations within ±15°, small translations, mild intensity scaling. See `transforms.py`.

---

## 3. Cross-Entropy Will Train a Model That Does Nothing

Tumor pixels are roughly 0.5–3% of each image. The rest is background and normal brain.

Run cross-entropy on this and watch what happens: the model learns to predict background everywhere and gets 97%+ accuracy. Technically correct, completely useless. This isn't a bug — it's cross-entropy doing exactly what it's supposed to do, optimizing for average pixel accuracy across a heavily imbalanced dataset.

**Dice loss** sidesteps this by measuring overlap between the predicted mask and the ground truth mask directly. It doesn't reward correct background predictions — it only rewards finding the tumor. For multi-class segmentation, per-class Dice ensures the model can't hide a collapsing class behind a good average.

The hybrid loss in this repo (`HybridLoss`) combines:
- **Dice** — class imbalance at the segment level
- **Focal** — harder weight on misclassified pixels
- **Boundary** — extra penalty on blurry tumor edges

---

## 4. Why 4 Output Classes, Not 1

The simplest version of this problem is binary: tumor vs background. That's what most tutorials do.

This repo outputs 4 classes: background, glioma, meningioma, pituitary. The reason is that the three tumor types look different on MRI and have different spatial patterns — a model forced to predict a single "tumor" label has to ignore those differences. A model with 4 classes can learn them.

It also makes the evaluation more honest. A binary model can score well on Dice while consistently misidentifying tumor type. Per-class Dice exposes that.

---

## 5. A 2D Slice Is an Incomplete View

Brain tumors are 3D structures. A 2D axial slice shows the tumor at one cross-section — the same tumor looks different 5mm higher or lower. Two-dimensional models trained on individual slices learn features from one viewing plane.

The 3D Attention U-Net in this repo processes a volume, not a slice, so it can use spatial context across the depth dimension. The pseudo-3D approach (stacking repeated slices) bridges the gap when true volumetric data isn't available.

The 3-plane viewer in `src/visualization/viewer.py` lets you inspect predictions from all three orientations — axial, sagittal, coronal — which is useful for catching failures that only show up in one view.

---

## Further Reading

- [The Medical Segmentation Decathlon](http://medicaldecathlon.com) — benchmark datasets for segmentation tasks
- [BraTS Challenge](https://www.synapse.org/brats) — the standard brain tumor segmentation benchmark
- Ronneberger et al. (2015) — [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- Lin et al. (2017) — [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)

---

*Questions — open a GitHub issue or reach out on [LinkedIn](https://linkedin.com/in/motazalqaoud).*
