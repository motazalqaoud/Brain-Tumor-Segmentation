"""
predict.py
----------
Run a trained U-Net checkpoint on an image and save a visualization.

Examples:
    # Predict on a generated synthetic slice (no input file needed):
    python predict.py --checkpoint checkpoints/best_unet.pt

    # Predict on a real 2D slice stored as .npy (shape HxW, normalized 0..1):
    python predict.py --checkpoint checkpoints/best_unet.pt --input slice.npy
"""

import argparse
import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from segmentation.unet import UNet                  # noqa: E402
from train import _make_slice                       # noqa: E402  (reuse synthetic gen)


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
        img, gt = _make_slice(seed=12345)   # demo slice with known ground truth

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
