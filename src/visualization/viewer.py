"""
viewer.py
---------
3-plane medical image viewer (axial, sagittal, coronal).

Author: Motaz Alqaoud, PhD
GitHub: https://github.com/motazalqaoud

Clinical note:
    Radiologists and surgeons always review images in 3 planes.
    A tumor visible in one plane may look very different in another.
    Always visualize all 3 planes before drawing conclusions.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Optional, Tuple


def show_3planes(volume: np.ndarray,
                 mask: Optional[np.ndarray] = None,
                 title: str = "Medical Image — 3 Planes",
                 cmap: str = "gray",
                 mask_alpha: float = 0.4,
                 mask_color: str = "red",
                 figsize: Tuple[int, int] = (14, 5)) -> plt.Figure:
    """
    Display axial, sagittal, and coronal center slices.

    Args:
        volume: 3D numpy array (Z, Y, X)
        mask:   optional binary mask overlay (same shape)
        title:  figure title
        cmap:   colormap for image
        mask_alpha: transparency for mask overlay
        mask_color: color for mask overlay
        figsize: figure size

    Returns:
        matplotlib Figure object
    """
    z, y, x = volume.shape
    cz, cy, cx = z // 2, y // 2, x // 2

    slices = {
        "Axial (Z)":    volume[cz, :, :],
        "Sagittal (X)": volume[:, :, cx],
        "Coronal (Y)":  volume[:, cy, :],
    }

    mask_slices = {}
    if mask is not None:
        mask_slices = {
            "Axial (Z)":    mask[cz, :, :],
            "Sagittal (X)": mask[:, :, cx],
            "Coronal (Y)":  mask[:, cy, :],
        }

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)

    for ax, (plane_name, img) in zip(axes, slices.items()):
        ax.imshow(img, cmap=cmap, aspect='equal')

        if mask is not None and plane_name in mask_slices:
            m = mask_slices[plane_name]
            colored_mask = np.zeros((*m.shape, 4))
            colored_mask[m > 0] = [1, 0, 0, mask_alpha]
            ax.imshow(colored_mask, aspect='equal')

        ax.set_title(plane_name, fontsize=11)
        ax.axis('off')

    plt.tight_layout()
    return fig


def show_slice_sequence(volume: np.ndarray,
                        mask: Optional[np.ndarray] = None,
                        axis: int = 0,
                        n_slices: int = 9,
                        cmap: str = "gray",
                        figsize: Tuple[int, int] = (15, 15)) -> plt.Figure:
    """
    Display a grid of evenly-spaced slices along one axis.

    Args:
        volume:   3D numpy array
        mask:     optional binary mask
        axis:     axis along which to slice (0=axial, 1=coronal, 2=sagittal)
        n_slices: number of slices to show
        cmap:     colormap
        figsize:  figure size

    Returns:
        matplotlib Figure object
    """
    n_total = volume.shape[axis]
    indices = np.linspace(0, n_total - 1, n_slices, dtype=int)

    cols = 3
    rows = (n_slices + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()

    for i, idx in enumerate(indices):
        if axis == 0:
            img = volume[idx, :, :]
            m = mask[idx, :, :] if mask is not None else None
        elif axis == 1:
            img = volume[:, idx, :]
            m = mask[:, idx, :] if mask is not None else None
        else:
            img = volume[:, :, idx]
            m = mask[:, :, idx] if mask is not None else None

        axes[i].imshow(img, cmap=cmap)
        if m is not None:
            overlay = np.zeros((*m.shape, 4))
            overlay[m > 0] = [1, 0, 0, 0.4]
            axes[i].imshow(overlay)
        axes[i].set_title(f"Slice {idx}", fontsize=8)
        axes[i].axis('off')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    return fig


def show_segmentation_comparison(image: np.ndarray,
                                  gt_mask: np.ndarray,
                                  pred_mask: np.ndarray,
                                  slice_idx: Optional[int] = None,
                                  figsize: Tuple[int, int] = (15, 5)) -> plt.Figure:
    """
    Side-by-side: original | ground truth | prediction | overlay.
    Standard figure for segmentation papers and model evaluation.

    Args:
        image:     2D or 3D array (if 3D, uses center or slice_idx)
        gt_mask:   ground truth binary mask
        pred_mask: predicted binary mask
        slice_idx: which slice to show (if 3D)
        figsize:   figure size

    Returns:
        matplotlib Figure
    """
    if image.ndim == 3:
        idx = slice_idx if slice_idx is not None else image.shape[0] // 2
        image = image[idx]
        gt_mask = gt_mask[idx]
        pred_mask = pred_mask[idx]

    fig, axes = plt.subplots(1, 4, figsize=figsize)
    titles = ["Image", "Ground Truth", "Prediction", "Overlay (GT=green, Pred=red)"]

    axes[0].imshow(image, cmap='gray')

    axes[1].imshow(image, cmap='gray')
    gt_overlay = np.zeros((*gt_mask.shape, 4))
    gt_overlay[gt_mask > 0] = [0, 1, 0, 0.5]
    axes[1].imshow(gt_overlay)

    axes[2].imshow(image, cmap='gray')
    pred_overlay = np.zeros((*pred_mask.shape, 4))
    pred_overlay[pred_mask > 0] = [1, 0, 0, 0.5]
    axes[2].imshow(pred_overlay)

    axes[3].imshow(image, cmap='gray')
    tp = ((gt_mask > 0) & (pred_mask > 0))
    fp = ((gt_mask == 0) & (pred_mask > 0))
    fn = ((gt_mask > 0) & (pred_mask == 0))

    combo = np.zeros((*image.shape, 4))
    combo[tp] = [0, 1, 0, 0.5]   # True positive: green
    combo[fp] = [1, 0, 0, 0.5]   # False positive: red
    combo[fn] = [0, 0, 1, 0.5]   # False negative: blue
    axes[3].imshow(combo)

    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    return fig
