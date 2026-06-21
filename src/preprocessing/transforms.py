"""
transforms.py
-------------
Medical-imaging-specific preprocessing transforms.

Author: Motaz Alqaoud, PhD
GitHub: https://github.com/motazalqaoud

Clinical note:
    Standard computer vision augmentations (random horizontal flip, color jitter)
    are DANGEROUS in medical imaging. Flipping a chest X-ray swaps left/right.
    Rotating a brain MRI 90° makes it unrecognizable to the model.
    These transforms are anatomy-aware.
"""

import numpy as np
import torch
from typing import Tuple, Optional


# ─── Intensity Normalization ──────────────────────────────────────────────────

def zscore_normalize(volume: np.ndarray,
                     mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Z-score normalization. 
    
    If a mask is provided (e.g. brain mask), statistics are computed 
    only within the masked region — much more stable than whole-volume stats.

    Args:
        volume: 3D numpy array
        mask: optional binary mask (same shape as volume)

    Returns:
        Normalized volume with zero mean and unit std
    """
    if mask is not None:
        roi = volume[mask > 0]
    else:
        roi = volume.flatten()

    mean = roi.mean()
    std = roi.std()

    if std < 1e-8:
        return volume - mean

    return (volume - mean) / std


def percentile_clip(volume: np.ndarray,
                    lower: float = 1.0,
                    upper: float = 99.0) -> np.ndarray:
    """
    Clip volume intensities to percentile range, then normalize to [0, 1].
    
    More robust than min-max for MRI (handles bright artifacts).

    Args:
        volume: 3D numpy array
        lower: lower percentile (default 1%)
        upper: upper percentile (default 99%)

    Returns:
        Volume normalized to [0, 1]
    """
    p_low = np.percentile(volume, lower)
    p_high = np.percentile(volume, upper)
    clipped = np.clip(volume, p_low, p_high)
    return (clipped - p_low) / (p_high - p_low + 1e-8)


def minmax_normalize(volume: np.ndarray) -> np.ndarray:
    """Normalize volume to [0, 1] range."""
    vmin, vmax = volume.min(), volume.max()
    if vmax - vmin < 1e-8:
        return np.zeros_like(volume)
    return (volume - vmin) / (vmax - vmin)


# ─── Resampling ───────────────────────────────────────────────────────────────

def resample_to_spacing(volume: np.ndarray,
                        current_spacing: Tuple[float, float, float],
                        target_spacing: Tuple[float, float, float],
                        order: int = 1) -> np.ndarray:
    """
    Resample a volume to isotropic or target voxel spacing.

    Critical for: comparing scans from different scanners/protocols.
    All spatial measurements (tumor size, margin) depend on correct spacing.

    Args:
        volume: 3D numpy array
        current_spacing: (z, y, x) spacing in mm
        target_spacing: desired (z, y, x) spacing in mm
        order: interpolation order (0=nearest, 1=linear, 3=cubic)

    Returns:
        Resampled volume
    """
    from scipy.ndimage import zoom

    zoom_factors = tuple(
        c / t for c, t in zip(current_spacing, target_spacing)
    )
    resampled = zoom(volume, zoom_factors, order=order)
    return resampled.astype(np.float32)


# ─── Safe Medical Augmentations ───────────────────────────────────────────────

class MedicalAugmentation:
    """
    Anatomy-aware augmentation for medical images.

    Rules:
    - Small rotations only (±15°) — not 90° flips
    - No horizontal flip for asymmetric anatomy (brain, heart)
    - Intensity augmentation must stay within physically meaningful range
    - Elastic deformation allowed but must preserve topology
    """

    def __init__(self,
                 rotation_range: float = 15.0,
                 flip_axes: Optional[list] = None,
                 intensity_shift: float = 0.1,
                 intensity_scale: float = 0.1,
                 gaussian_noise_std: float = 0.01):
        """
        Args:
            rotation_range: max rotation in degrees (keep small for anatomy)
            flip_axes: list of axes to allow flipping. None = no flipping.
                       For breast MRI: [0] (axial flip ok, left-right NOT ok)
            intensity_shift: random additive intensity shift range
            intensity_scale: random multiplicative intensity scale range
            gaussian_noise_std: std of Gaussian noise to add
        """
        self.rotation_range = rotation_range
        self.flip_axes = flip_axes or []
        self.intensity_shift = intensity_shift
        self.intensity_scale = intensity_scale
        self.gaussian_noise_std = gaussian_noise_std

    def __call__(self, volume: np.ndarray,
                 mask: Optional[np.ndarray] = None):
        """
        Apply augmentations to volume (and optionally mask).

        Args:
            volume: 3D numpy array, normalized
            mask: optional segmentation mask (same augmentations applied)

        Returns:
            aug_volume, aug_mask (or just aug_volume if mask is None)
        """
        from scipy.ndimage import rotate

        aug_vol = volume.copy()
        aug_mask = mask.copy() if mask is not None else None

        # Small rotation (2D rotation on axial plane)
        if self.rotation_range > 0:
            angle = np.random.uniform(-self.rotation_range, self.rotation_range)
            aug_vol = rotate(aug_vol, angle, axes=(1, 2), reshape=False, order=1)
            if aug_mask is not None:
                aug_mask = rotate(aug_mask, angle, axes=(1, 2), reshape=False, order=0)

        # Axis-safe flipping
        for axis in self.flip_axes:
            if np.random.rand() > 0.5:
                aug_vol = np.flip(aug_vol, axis=axis).copy()
                if aug_mask is not None:
                    aug_mask = np.flip(aug_mask, axis=axis).copy()

        # Intensity augmentation (volume only, not mask)
        shift = np.random.uniform(-self.intensity_shift, self.intensity_shift)
        scale = np.random.uniform(1 - self.intensity_scale, 1 + self.intensity_scale)
        aug_vol = aug_vol * scale + shift

        # Gaussian noise
        if self.gaussian_noise_std > 0:
            noise = np.random.normal(0, self.gaussian_noise_std, aug_vol.shape)
            aug_vol = aug_vol + noise

        if mask is not None:
            return aug_vol.astype(np.float32), aug_mask.astype(np.float32)
        return aug_vol.astype(np.float32)


# ─── Patch Extraction ─────────────────────────────────────────────────────────

def extract_patches_3d(volume: np.ndarray,
                       patch_size: Tuple[int, int, int],
                       stride: Tuple[int, int, int]) -> np.ndarray:
    """
    Extract overlapping 3D patches from a volume.
    Used for training on large volumes that don't fit in GPU memory.

    Args:
        volume: 3D numpy array
        patch_size: (D, H, W) patch dimensions
        stride: (D, H, W) stride

    Returns:
        patches: array of shape (N, D, H, W)
    """
    D, H, W = volume.shape
    pd, ph, pw = patch_size
    sd, sh, sw = stride

    patches = []
    for d in range(0, D - pd + 1, sd):
        for h in range(0, H - ph + 1, sh):
            for w in range(0, W - pw + 1, sw):
                patch = volume[d:d+pd, h:h+ph, w:w+pw]
                patches.append(patch)

    return np.array(patches)
