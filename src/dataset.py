"""
Real 3D volumetric data pipeline for brain tumor segmentation -- full NIfTI
volumes in, full NIfTI segmentation masks out.

Dataset: Medical Segmentation Decathlon, Task01_BrainTumour -- 750 4D MRI
volumes (484 train / 266 test), sourced from the BraTS 2016/2017 challenge.
Four co-registered modalities per subject: FLAIR, T1w, T1gd, T2w.

Label convention (raw, single-channel):
0 = background
1 = necrotic / non-enhancing tumor core
2 = peritumoral edema
4 = GD-enhancing tumor (label 3 is unused, per the BraTS convention)

We convert these into the three standard BraTS evaluation regions via the
built-in ConvertToMultiChannelBasedOnBratsClassesd transform:
TC (Tumor Core) = labels 1, 4
WT (Whole Tumor) = labels 1, 2, 4
ET (Enhancing Tumor) = label 4

These regions overlap (ET inside TC inside WT), which is why this is a
3-channel multi-label (independent sigmoid per channel) problem.
"""

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped,
    ConvertToMultiChannelBasedOnBratsClassesd, Orientationd, Spacingd,
    RandCropByLabelClassesd, MapLabelValued, CenterSpatialCropd, RandFlipd, RandRotate90d,
    NormalizeIntensityd, RandScaleIntensityd, RandShiftIntensityd,
)

REGION_NAMES = ["TC", "WT", "ET"]  # channel order from the Brats-classes transform
MODALITIES = ["FLAIR", "T1w", "T1gd (T1ce)", "T2w"]  # input channel order

def train_transforms(patch_size=(128, 128, 128)):
    """
    Training pipeline: load -> orient/resample -> crop a patch biased toward
    tumor sub-regions (especially rare Enhancing Tumor) -> convert to the
    3-channel TC/WT/ET encoding -> augment -> normalize.

    IMPORTANT: the crop happens on the RAW single-channel label, before the
    TC/WT/ET one-hot conversion, and is biased toward foreground. Enhancing
    Tumor (raw label 4) is often just a few percent of an already small tumor
    region, so a plain uniform random crop over the whole brain volume very
    rarely lands on any ET voxels at all. Train that way long enough and the
    model learns the loss-optimal strategy for what it actually sees: always
    predict no ET -- exactly the collapse this pipeline previously produced
    (ET Dice stuck at 0.0000 for the entire run). RandCropByLabelClassesd
    guarantees a configurable mix of patches centered on each raw label
    value, so ET actually appears in training often enough to learn from.
    """
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys="image"),
        EnsureTyped(keys=["image", "label"]),
        EnsureChannelFirstd(keys="label"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0),
                 mode=("bilinear", "nearest")),
        RandCropByLabelClassesd(
            keys=["image", "label"],
            label_key="label",
            spatial_size=patch_size,
            num_classes=5,  # raw label values 0..4 (3 is unused in BraTS)
            ratios=[1, 2, 2, 0, 4],  # bg, necrotic, edema, unused, ET -- ET oversampled
            num_samples=1,
        ),
        MapLabelValued(keys="label", orig_labels=[3], target_labels=[4]),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
    ])

def eval_transforms():
    """Validation/test pipeline: full volumes, sliding-window inference at eval time."""
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys="image"),
        EnsureTyped(keys=["image", "label"]),
        MapLabelValued(keys="label", orig_labels=[3], target_labels=[4]),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0),
                 mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ])

def inference_transforms():
    """Same as eval_transforms but no label key, for predict.py on new scans."""
    return Compose([
        LoadImaged(keys="image"),
        EnsureChannelFirstd(keys="image"),
        EnsureTyped(keys="image"),
        Orientationd(keys="image", axcodes="RAS"),
        Spacingd(keys="image", pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ])

def recommend_patch_and_batch_size(device_index: int = 0):
    """Pick patch/batch size from available GPU memory (V100 vs A100 etc)."""
    import torch
    if not torch.cuda.is_available():
        return (64, 64, 64), 1, "no GPU -> tiny patch/batch (CPU, testing only)"

    total_gb = torch.cuda.get_device_properties(device_index).total_memory / (1024 ** 3)
    if total_gb < 20:
        return (96, 96, 96), 1, f"{total_gb:.0f}GB GPU -> small patch/batch"
    elif total_gb < 48:
        return (128, 128, 128), 2, f"{total_gb:.0f}GB GPU -> standard patch/batch"
    else:
        return (128, 128, 128), 4, f"{total_gb:.0f}GB GPU -> larger batch"
