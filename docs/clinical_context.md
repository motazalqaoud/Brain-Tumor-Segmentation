# Clinical Context: Why These Choices Matter

> Written by Motaz Alqaoud, PhD — Biomedical Engineer & Senior AI/ML Engineer at Abbott

Most medical AI tutorials are written by software engineers who've never been inside an OR.
This document explains the *why* behind the engineering decisions in this repo.

---

## 1. Voxel Spacing is Not Optional

Every medical image has a **physical spacing** — the real-world size (in mm) of each voxel.

A 256×256 image from one MRI scanner might represent a 25cm × 25cm field of view.
The same resolution from another scanner might be 18cm × 18cm.

**If you ignore spacing:**
- A tumor you measure as 2cm is actually 1.4cm
- Your model trained on scanner A fails completely on scanner B
- Your surgical margin calculation is wrong

**This is why** all preprocessing in this repo resamples to isotropic spacing before any analysis.

---

## 2. Anatomy-Aware Augmentation

Standard computer vision augmentation tutorials tell you to:
- Randomly flip horizontally ✓ (fine for dogs vs cats)
- Randomly rotate 90° ✓ (fine for texture classification)
- Aggressive color jitter ✓ (fine for natural images)

In medical imaging:
- **Horizontal flip** of a chest X-ray swaps the heart to the wrong side
- **90° rotation** of a brain MRI produces an anatomically impossible orientation — your model has never seen this and will fail
- **Aggressive intensity changes** destroy the clinical meaning of signal (T1 vs T2 MRI have specific intensity ranges)

**This is why** this repo uses small rotations (≤15°), controlled intensity shifts, and no left-right flips on asymmetric anatomy.

---

## 3. Dice Loss, Not Cross-Entropy

In a typical breast MRI slice, a tumor might occupy **0.5% of all pixels**.

If you train with cross-entropy:
- Predicting "background" for every pixel gets you **99.5% accuracy**
- Your model never learns to find tumors
- You get a useless model that looks great on metrics

**Dice loss** directly optimizes overlap between prediction and ground truth.
It doesn't care about the 99.5% background — it cares about whether you found the lesion.

**This is why** clinical AI almost always uses Dice or a Dice+BCE combination.

---

## 4. The Affine Matrix in NIfTI

NIfTI files contain a 4×4 **affine matrix** that maps from voxel indices to real-world coordinates.

In surgical navigation, you need to know: *"This tumor centroid is at voxel (128, 95, 67). Where is that in the OR?"*

The affine matrix answers that question. Without it, you cannot:
- Register pre-op MRI to intra-op ultrasound
- Guide a surgical tool to the correct location
- Compute real-world margins

**This is why** the NIfTI loader in this repo always preserves and exposes the affine.

---

## 5. The 3-Plane Rule

Radiologists and surgeons never look at a single 2D slice.

A finding that is ambiguous in the axial plane may be obvious in the sagittal or coronal plane.
A fake "lesion" (artifact) in one plane often disappears in the others.

**This is why** the visualization tools always show all 3 anatomical planes.

---

## Further Reading

- [MONAI: Medical Open Network for AI](https://monai.io)
- [3D Slicer](https://www.slicer.org) — open source surgical navigation platform
- [The Medical Segmentation Decathlon](http://medicaldecathlon.com) — benchmark datasets
- Ronneberger et al. (2015) — U-Net paper

---

*Questions? Connect with me on [LinkedIn](https://linkedin.com/in/motazalqaoud) or open a GitHub issue.*
