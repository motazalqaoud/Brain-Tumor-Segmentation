"""
predict.py
----------
Run a trained 2D U-Net checkpoint on a brain tumor MRI image.

For the 3D Attention U-Net use predict3d.py instead.

Examples:
    # Auto-finds a real image from the dataset:
    python scripts/predict.py --checkpoint checkpoints/best_unet.pt \
                               --data-root data/raw/Images_

    # Specify your own image + ground truth mask:
    python scripts/predict.py --checkpoint checkpoints/best_unet.pt \
                               --image path/to/image.jpg \
                               --mask  path/to/image_mask_consensus.png

Author: Motaz Alqaoud, PhD
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from segmentation.unet import UNet


def load_model(checkpoint_path: str, device: torch.device) -> tuple:
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = UNet(
        in_channels=1,
        n_classes=1,
        base_filters=ckpt.get("base_filters", 32)
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    image_size = ckpt.get("image_size", 128)
    val_dice   = ckpt.get("val_dice", "?")
    print(f"Loaded checkpoint  val_dice={val_dice:.4f if isinstance(val_dice, float) else val_dice}")
    return model, image_size


def load_image(path: str, size: int) -> np.ndarray:
    img = np.array(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    img = np.array(
        Image.fromarray((img * 255).astype(np.uint8)).resize((size, size), Image.BILINEAR),
        dtype=np.float32
    ) / 255.0
    return img


def find_sample_image(data_root: str, tumor_type: str = "Glioma"):
    """Find one image + mask pair from the dataset."""
    root = Path(data_root)
    tumor_dir = root / tumor_type
    if not tumor_dir.exists():
        return None, None
    for mod_dir in tumor_dir.iterdir():
        if not mod_dir.is_dir():
            continue
        for sub_dir in mod_dir.iterdir():
            if not sub_dir.is_dir():
                continue
            for f in sub_dir.iterdir():
                if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    if "_mask" in f.name or "_bbox" in f.name:
                        continue
                    mask = f.parent / f"{f.stem}_mask_consensus.png"
                    if mask.exists():
                        return f, mask
    return None, None


def main():
    ap = argparse.ArgumentParser(
        description="Run inference with a trained 2D U-Net checkpoint"
    )
    ap.add_argument("--checkpoint", default="checkpoints/best_unet.pt",
                    help="Path to .pt checkpoint")
    ap.add_argument("--image", default=None,
                    help="Path to input MRI image (JPG/PNG). Auto-detected if omitted.")
    ap.add_argument("--mask", default=None,
                    help="Path to ground truth mask PNG (optional)")
    ap.add_argument("--data-root", default="data/raw/Images_",
                    help="Dataset root — used to auto-find a sample when --image is omitted")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Sigmoid threshold for binary prediction (default: 0.5)")
    ap.add_argument("--out", default="prediction.png",
                    help="Output figure path (default: prediction.png)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    model, image_size = load_model(args.checkpoint, device)

    # Load image
    img_path = args.image
    mask_path = args.mask

    if img_path is None:
        img_path, mask_path = find_sample_image(args.data_root)
        if img_path is None:
            print("No dataset found at", args.data_root)
            print("Run: python run.py setup   to download the dataset first.")
            sys.exit(1)
        print(f"Using: {img_path.name}")

    img = load_image(str(img_path), image_size)
    img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)

    gt = None
    if mask_path and Path(str(mask_path)).exists():
        gt_raw = load_image(str(mask_path), image_size)
        gt = (gt_raw > 0.5).astype(np.float32)

    # Inference
    x = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    pred = (prob > args.threshold).astype(np.float32)

    # Plot
    n_cols = 4 if gt is not None else 3
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))

    axes[0].imshow(img_norm, cmap="gray")
    axes[0].set_title("Input MRI", fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(prob, cmap="magma", vmin=0, vmax=1)
    axes[1].set_title("Tumor Probability", fontweight="bold")
    axes[1].axis("off")
    fig.colorbar(plt.cm.ScalarMappable(cmap="magma"), ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(img_norm, cmap="gray")
    overlay = np.zeros((*pred.shape, 4))
    overlay[pred > 0] = [1.0, 0.2, 0.2, 0.6]
    axes[2].imshow(overlay)
    axes[2].set_title("Prediction", fontweight="bold")
    axes[2].axis("off")

    if gt is not None:
        axes[3].imshow(img_norm, cmap="gray")
        gt_overlay = np.zeros((*gt.shape, 4))
        gt_overlay[gt > 0] = [0.2, 1.0, 0.2, 0.6]
        axes[3].imshow(gt_overlay)
        axes[3].set_title("Ground Truth", fontweight="bold")
        axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved → {args.out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
