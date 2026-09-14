"""
predict3d.py
------------
Run a trained 3D Attention U-Net checkpoint on a brain tumor MRI image.

Reads model architecture from the checkpoint — no need to specify flags
that were used during training.

Examples:
    # Run on a real image from the Kaggle dataset:
    python scripts/predict3d.py \
        --checkpoint checkpoints/best_model_dice_0.6446.pt \
        --image data/raw/Images_/Glioma/T1C+/Gliomas T1/image.jpg

    # Run on a real image with its ground truth mask:
    python scripts/predict3d.py \
        --checkpoint checkpoints/best_model_dice_0.6446.pt \
        --image path/to/image.jpg \
        --mask  path/to/image_mask_consensus.png

    # Run on a synthetic sample (no dataset needed):
    python scripts/predict3d.py \
        --checkpoint checkpoints/best_model_dice_0.6446.pt

    # Force CPU even if CUDA is available:
    python scripts/predict3d.py \
        --checkpoint checkpoints/best_model_dice_0.6446.pt \
        --device cpu

Author: Motaz Alqaoud, PhD
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from segmentation.unet3d import AttentionUNet3D

# Class label colours (RGBA) — 8-class: background + 7 WHO tumor categories
CLASS_COLORS = {
    0: None,                    # background      — not shown
    1: [1.0, 0.2, 0.2, 0.6],   # glioma          — red
    2: [0.2, 1.0, 0.2, 0.6],   # meningioma       — green
    3: [0.2, 0.6, 1.0, 0.6],   # nerve sheath     — blue
    4: [1.0, 1.0, 0.2, 0.6],   # embryonic        — yellow
    5: [1.0, 0.2, 1.0, 0.6],   # mixed neuronal   — magenta
    6: [0.2, 1.0, 1.0, 0.6],   # mesenchymal      — cyan
    7: [1.0, 0.6, 0.2, 0.6],   # germ cell        — orange
}
CLASS_NAMES = {
    1: 'Glioma', 2: 'Meningioma', 3: 'Nerve Sheath',
    4: 'Embryonic', 5: 'Mixed Neuronal', 6: 'Mesenchymal', 7: 'Germ Cell',
}


def load_model(checkpoint_path: str, device: torch.device):
    """Load model and config from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device)

    cfg = ckpt.get('model_config', {})
    base_filters = cfg.get('base_filters', 16)
    depth        = cfg.get('depth', 2)
    num_classes  = cfg.get('num_classes', 4)
    image_size   = cfg.get('image_size', 64)
    d_frames     = cfg.get('d_frames', 2)

    model = AttentionUNet3D(
        in_channels=1,
        num_classes=num_classes,
        base_filters=base_filters,
        depth=depth,
    ).to(device)

    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    epoch    = ckpt.get('epoch', '?')
    val_dice = ckpt.get('val_dice', '?')
    val_dice_str = f"{val_dice:.4f}" if isinstance(val_dice, float) else str(val_dice)
    print(f"Loaded checkpoint  epoch={epoch}  val_dice={val_dice_str}")
    print(f"Model config: base_filters={base_filters}, depth={depth}, "
          f"num_classes={num_classes}, image_size={image_size}, d_frames={d_frames}")

    return model, image_size, d_frames, num_classes


def load_image(path: str, image_size: int) -> np.ndarray:
    """Load and resize a JPG/PNG MRI slice to (image_size, image_size), normalised [0,1]."""
    img = np.array(Image.open(path).convert('L'), dtype=np.float32) / 255.0
    img = np.array(
        Image.fromarray((img * 255).astype(np.uint8)).resize(
            (image_size, image_size), Image.BILINEAR
        ), dtype=np.float32
    ) / 255.0
    return img


def make_synthetic(image_size: int):
    """Generate a synthetic brain slice for demo purposes."""
    rng = np.random.default_rng(42)
    img = rng.normal(0.15, 0.04, (image_size, image_size)).astype(np.float32)
    yg, xg = np.mgrid[0:image_size, 0:image_size]
    brain = ((yg - image_size/2)**2 / (image_size/2.3)**2 +
             (xg - image_size/2)**2 / (image_size/2.3)**2) < 1
    img[brain] += rng.normal(0.3, 0.05, img[brain].shape)
    cy, cx = image_size//2 + 10, image_size//2 - 8
    ry, rx = image_size//8, image_size//10
    lesion = ((yg - cy)**2/ry**2 + (xg - cx)**2/rx**2) < 1
    img[lesion] += 0.35
    img = np.clip(img, 0, 1)
    return img, None   # no ground truth for synthetic


def overlay_mask(ax, mask: np.ndarray, num_classes: int):
    """Draw coloured class overlays on a matplotlib axis."""
    for cls in range(1, num_classes):
        color = CLASS_COLORS.get(cls)
        if color is None:
            continue
        rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
        rgba[mask == cls] = color
        ax.imshow(rgba)


def run(args):
    device = torch.device(args.device)
    model, image_size, d_frames, num_classes = load_model(args.checkpoint, device)

    # Load input
    if args.image:
        img = load_image(args.image, image_size)
        gt = load_image(args.mask, image_size) if args.mask else None
        if gt is not None:
            gt = (gt > 0.5).astype(np.uint8)
    else:
        print("No --image provided — using synthetic data for demo.")
        img, gt = make_synthetic(image_size)

    # Normalise
    img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)

    # Inference
    x = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,H,W)
    x3d = x.unsqueeze(2).repeat(1, 1, d_frames, 1, 1)                    # (1,1,D,H,W)

    with torch.no_grad():
        logits = model(x3d)                        # (1, C, D, H, W)
        preds = torch.argmax(logits, dim=1)        # (1, D, H, W)

    mid = d_frames // 2
    pred_np = preds[0, mid].cpu().numpy()          # (H, W)

    # Confidence map for the predicted class at the middle slice
    probs = torch.softmax(logits[0, :, mid], dim=0).cpu().numpy()  # (C, H, W)
    confidence = probs.max(axis=0)                                   # (H, W)

    # Plot
    has_gt = gt is not None
    n_cols = 4 if has_gt else 3
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))

    # 1. Input
    axes[0].imshow(img_norm, cmap='gray')
    axes[0].set_title('Input MRI', fontweight='bold')
    axes[0].axis('off')

    # 2. Ground truth (if provided)
    col = 1
    if has_gt:
        axes[col].imshow(img_norm, cmap='gray')
        overlay_mask(axes[col], gt, num_classes)
        axes[col].set_title('Ground Truth', fontweight='bold')
        axes[col].axis('off')
        col += 1

    # 3. Prediction overlay
    axes[col].imshow(img_norm, cmap='gray')
    overlay_mask(axes[col], pred_np, num_classes)
    present_classes = [c for c in range(1, num_classes) if np.any(pred_np == c)]
    patches = [mpatches.Patch(color=CLASS_COLORS[c][:3], label=CLASS_NAMES[c])
               for c in present_classes if c in CLASS_COLORS]
    if patches:
        axes[col].legend(handles=patches, loc='lower right', fontsize=8)
    axes[col].set_title('Prediction', fontweight='bold')
    axes[col].axis('off')
    col += 1

    # 4. Confidence map
    cm = axes[col].imshow(confidence, cmap='viridis', vmin=0, vmax=1)
    fig.colorbar(cm, ax=axes[col], fraction=0.046, pad=0.04)
    axes[col].set_title('Confidence', fontweight='bold')
    axes[col].axis('off')

    val_dice = torch.load(args.checkpoint, map_location='cpu').get('val_dice', None)
    title = f'3D Attention U-Net Inference'
    if val_dice is not None:
        title += f'  |  val Dice = {val_dice:.4f}'
    fig.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches='tight')
    print(f"Saved → {args.out}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description='Run inference with a trained 3D Attention U-Net checkpoint'
    )
    ap.add_argument('--checkpoint', required=True,
                    help='Path to .pt checkpoint (e.g. checkpoints/best_model_dice_0.6446.pt)')
    ap.add_argument('--image', default=None,
                    help='Path to input MRI image (JPG/PNG). Omit to use synthetic demo data.')
    ap.add_argument('--mask', default=None,
                    help='Path to ground truth mask PNG (optional, for side-by-side comparison)')
    ap.add_argument('--out', default='prediction.png',
                    help='Output file path for the saved figure (default: prediction.png)')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                    help='Device: "cuda" or "cpu" (auto-detected by default)')
    run(ap.parse_args())


if __name__ == '__main__':
    main()
