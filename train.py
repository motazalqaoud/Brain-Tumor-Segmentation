"""
train.py
--------
Train the U-Net tumor segmenter from the command line.

By default it trains on synthetic breast-MRI-like 2D slices (no data download
needed), using the real model/loss code in src/. The best checkpoint is saved
to checkpoints/best_unet.pt.

Examples:
    python train.py                      # quick default run
    python train.py --epochs 50 --batch 16
    python train.py --device cuda
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from segmentation.unet import UNet                 # noqa: E402
from segmentation.losses import BCEDiceLoss, dice_coefficient  # noqa: E402


# ─── Synthetic dataset (same generative idea as notebook 03) ──────────────────

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


class SyntheticMRIDataset(Dataset):
    def __init__(self, n_samples=300, size=128, seed=0):
        self.n, self.size, self.seed = n_samples, size, seed

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        img, mask = _make_slice(self.size, seed=self.seed + idx)
        img = torch.from_numpy(img).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)
        return img, mask


# ─── Training loop ────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, device, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    total_loss = total_dice = 0.0

    with torch.set_grad_enabled(train):
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            loss = criterion(preds, masks)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            total_dice += dice_coefficient(preds, masks)

    n = len(loader)
    return total_loss / n, total_dice / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--train-n", type=int, default=300)
    ap.add_argument("--val-n", type=int, default=60)
    ap.add_argument("--base-filters", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="checkpoints/best_unet.pt")
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")

    train_ds = SyntheticMRIDataset(args.train_n, args.size, seed=0)
    val_ds = SyntheticMRIDataset(args.val_n, args.size, seed=5000)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False)

    model = UNet(in_channels=1, n_classes=1, base_filters=args.base_filters).to(device)
    print(f"U-Net parameters: {model.count_parameters():,}")

    criterion = BCEDiceLoss(alpha=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    best_dice = 0.0

    print(f"\n{'Epoch':>6}{'TrLoss':>10}{'VlLoss':>10}{'TrDice':>10}{'VlDice':>10}")
    print("-" * 46)
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_dice = run_epoch(model, train_loader, criterion, device, optimizer)
        vl_loss, vl_dice = run_epoch(model, val_loader, criterion, device)
        scheduler.step(vl_loss)
        print(f"{epoch:>6}{tr_loss:>10.4f}{vl_loss:>10.4f}{tr_dice:>10.4f}{vl_dice:>10.4f}")

        if vl_dice > best_dice:
            best_dice = vl_dice
            torch.save({"model_state": model.state_dict(),
                        "base_filters": args.base_filters,
                        "size": args.size,
                        "val_dice": best_dice,
                        "epoch": epoch}, args.out)

    print(f"\nBest Val Dice: {best_dice:.4f}  ->  saved to {args.out}")


if __name__ == "__main__":
    main()
