"""
Real 3D volumetric data pipeline for brain tumor segmentation -- full NIfTI
volumes in, full NIfTI segmentation masks out. No 2D slice extraction anywhere
in this pipeline; every transform below operates on the (C, H, W, D) volume.

Dataset: Medical Segmentation Decathlon, Task01_BrainTumour -- 750 4D MRI
volumes (484 train / 266 test), sourced from the BraTS 2016/2017 challenge.
Four co-registered modalities per subject: FLAIR, T1w, T1gd (T1 contrast-
enhanced), T2w. Publicly downloadable with no registration wall, via MONAI's
built-in DecathlonDataset (see data_prep.py).

Label convention (raw, single-channel):
    0 = background
    1 = necrotic / non-enhancing tumor core
    2 = peritumoral edema
    4 = GD-enhancing tumor        (label 3 is unused, per the BraTS convention)

We convert these into the three standard, clinically-nested BraTS evaluation
regions (this is what MONAI's own official BraTS tutorial does, via the
built-in ConvertToMultiChannelBasedOnBratsClassesd transform):
    TC (Tumor Core)      = labels {1, 4}
    WT (Whole Tumor)      = labels {1, 2, 4}
    ET (Enhancing Tumor)  = label  {4}

These regions overlap (ET is inside TC is inside WT), which is why this is a
3-channel multi-label (independent sigmoid per channel) problem, not a
4-class softmax problem.
"""

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped,
    ConvertToMultiChannelBasedOnBratsClassesd, Orientationd, Spacingd,
    RandSpatialCropd, CenterSpatialCropd, RandFlipd, RandRotate90d,
    NormalizeIntensityd, RandScaleIntensityd, RandShiftIntensityd,
)

REGION_NAMES = ["TC", "WT", "ET"]  # channel order produced by the Brats-classes transform
MODALITIES = ["FLAIR", "T1w", "T1gd (T1ce)", "T2w"]  # input channel order in the raw data


def train_transforms(patch_size=(128, 128, 128)):
    """
    Training pipeline: load -> orient/resample to a consistent frame -> random
    patch crop (full volumes don't fit in GPU memory with 3D convs at any
    reasonable batch size) -> augmentation -> intensity normalization.
    """
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys="image"),
        EnsureTyped(keys=["image", "label"]),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0),
                  mode=("bilinear", "nearest")),
        RandSpatialCropd(keys=["image", "label"], roi_size=patch_size, random_size=False),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
    ])


def eval_transforms():
    """
    Validation/test pipeline: same geometric normalization as training, but no
    random cropping or augmentation -- full volumes are kept whole and handled
    at inference time via sliding-window inference (see evaluate.py/predict.py),
    since that is what gives a realistic, deployable Dice score (a model that
    only ever sees fixed-size training patches still needs to work on a full,
    arbitrarily-shaped clinical volume at test time).
    """
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys="image"),
        EnsureTyped(keys=["image", "label"]),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0),
                  mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ])


def inference_transforms():
    """Same as eval_transforms but with no label key, for predict.py on new scans."""
    return Compose([
        LoadImaged(keys="image"),
        EnsureChannelFirstd(keys="image"),
        EnsureTyped(keys="image"),
        Orientationd(keys="image", axcodes="RAS"),
        Spacingd(keys="image", pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ])


def recommend_patch_and_batch_size(device_index: int = 0):
    """
    Pick a patch size and batch size based on available GPU memory, since we
    don't know in advance whether a job lands on a V100 (16/32GB) or an A100
    (40/80GB) node on the HPC cluster. Conservative, not tuned for max
    throughput -- the goal is "doesn't OOM on whatever node SLURM gives us."
    """
    import torch
    if not torch.cuda.is_available():
        return (64, 64, 64), 1, "no GPU detected -> tiny patch/batch for CPU (very slow, testing only)"

    total_gb = torch.cuda.get_device_properties(device_index).total_memory / (1024 ** 3)
    if total_gb < 20:
        return (96, 96, 96), 1, f"{total_gb:.0f}GB GPU -> small patch/batch"
    elif total_gb < 48:
        return (128, 128, 128), 2, f"{total_gb:.0f}GB GPU -> standard patch/batch"
    else:
        return (128, 128, 128), 4, f"{total_gb:.0f}GB GPU -> larger batch"
