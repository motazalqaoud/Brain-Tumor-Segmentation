"""
predict.py
----------
Run a trained U-Net checkpoint on an image and save a visualization.

Examples:
    # Predict on a generated synthetic slice (no input file needed):
    python scripts/predict.py --checkpoint checkpoints/best_unet.pt

    # Predict on a real 2D slice stored as .npy (shape HxW, normalized 0..1):
    python scripts/predict.py --checkpoint checkpoints/best_unet.pt --input slice.npy
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from segmentation.unet import UNet


def _make_slice(size=128, seed=None):
    rng = np.random.default_rng(seed)
    img = rng.normal(0.15, 0.04, (size, size)).astype(np.float32)
    mask = np.zeros((size, size), dtype=np.float32)
    y_g, x_g = np.mgrid[0:size, 0:size]

    tissue = ((y_g - size // 2) ** 2 / (size / 2.3) ** 2 +
              (x_g - size // 2) ** 2 / (size / 2.3) ** 2) < 1
    img[tissue] += rng.normal(0.3, 0.05, img[tissue].shape).astype(np.float32)

    cy = rng.integers(size // 4, 3 * size // 4)
    cx = rng.integers(size // 4, 3 * size // 4)
    ry, rx = rng.integers(8, 20), rng.integers(8, 20)
    lesion = ((y_g - cy) ** 2 / ry ** 2 + (x_g - cx) ** 2 / rx ** 2) < 1
    img[lesion] += rng.uniform(0.25, 0.45)
    mask[lesion] = 1.0

    img = np.clip(img + rng.normal(0, 0.02, img.shape), 0, 1).astype(np.float32)
    return img, (mask > 0.5).astype(np.float32)


def load_model(checkpoint, device):
    ckpt = torch.load(checkpoint, map_location=device)
    model = UNet(in_channels=1, n_classes=1,
                 base_filters=ckpt.get("base_filters", 32)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded {checkpoint}  (val_dice={ckpt.get('val_dice', '?')})")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/best_unet.pt")
    ap.add_argument("--input", default=None, help="path to a 2D .npy slice (optional)")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", default="prediction.png")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)

    gt = None
    if args.input:
        img = np.load(args.input).astype(np.float32)
    else:
        img, gt = _make_slice(seed=12345)

    x = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    pred = (prob > args.threshold).astype(np.float32)

    n = 3 if gt is None else 4
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    axes[0].imshow(img, cmap="gray"); axes[0].set_title("Input")
    axes[1].imshow(prob, cmap="magma"); axes[1].set_title("Probability")
    axes[2].imshow(img, cmap="gray")
    ov = np.zeros((*pred.shape, 4)); ov[pred > 0] = [1, 0.2, 0.2, 0.5]
    axes[2].imshow(ov); axes[2].set_title("Prediction")
    if gt is not None:
        axes[3].imshow(img, cmap="gray")
        ov2 = np.zeros((*gt.shape, 4)); ov2[gt > 0] = [0, 0.9, 0.2, 0.5]
        axes[3].imshow(ov2); axes[3].set_title("Ground Truth")
    for a in axes:
        a.axis("off")
    plt.tight_layout()
    plt.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
