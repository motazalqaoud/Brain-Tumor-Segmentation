"""
train.py
--------
Train the 2D U-Net on the Kaggle Brain Tumor MRI dataset.

A simpler entry point than train3d.py — binary segmentation (tumor vs
background) on 2D slices. Good starting point before moving to the full
3D Attention U-Net pipeline.

Examples:
    python scripts/train.py --data-root data/raw/Images_
    python scripts/train.py --data-root data/raw/Images_ --epochs 50 --device cuda

Author: Motaz Alqaoud, PhD
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from segmentation.unet import UNet
from segmentation.losses import BCEDiceLoss, dice_coefficient
from preprocessing.brain_tumor_loader import BrainTumorDataset


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss = total_dice = 0.0

    with torch.set_grad_enabled(training):
        for batch in loader:
            imgs  = batch['image'].to(device)           # (B, 1, H, W)
            masks = batch['mask'].float().unsqueeze(1).to(device)  # (B, 1, H, W)

            preds = model(imgs)
            loss  = criterion(preds, masks)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_dice += dice_coefficient(preds, masks)

    n = len(loader)
    return total_loss / n, total_dice / n


def main():
    ap = argparse.ArgumentParser(description="Train 2D U-Net on Brain Tumor MRI dataset")
    ap.add_argument("--data-root", default="data/raw/Images_",
                    help="Path to the Kaggle dataset Images_ directory")
    ap.add_argument("--epochs",       type=int,   default=30)
    ap.add_argument("--batch",        type=int,   default=16)
    ap.add_argument("--lr",           type=float, default=1e-3)
    ap.add_argument("--image-size",   type=int,   default=128)
    ap.add_argument("--base-filters", type=int,   default=32)
    ap.add_argument("--num-workers",  type=int,   default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="checkpoints/best_unet.pt")
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"Device : {device}")
    print(f"Dataset: {args.data_root}")

    # Dataset
    train_ds = BrainTumorDataset(
        root_dir=args.data_root, modality="T1c",
        split="train", volume_size=args.image_size
    )
    val_ds = BrainTumorDataset(
        root_dir=args.data_root, modality="T1c",
        split="val", volume_size=args.image_size
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch,
                              shuffle=True,  num_workers=args.num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch,
                              shuffle=False, num_workers=args.num_workers)

    print(f"Train : {len(train_ds)} samples")
    print(f"Val   : {len(val_ds)} samples")

    # Model
    model = UNet(in_channels=1, n_classes=1, base_filters=args.base_filters).to(device)
    print(f"U-Net parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = BCEDiceLoss(alpha=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    best_dice = 0.0

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Val Loss':>10}  "
          f"{'Train Dice':>10}  {'Val Dice':>10}")
    print("-" * 56)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_dice = run_epoch(model, train_loader, criterion, device, optimizer)
        vl_loss, vl_dice = run_epoch(model, val_loader,   criterion, device)
        scheduler.step(vl_loss)

        print(f"{epoch:>6}  {tr_loss:>10.4f}  {vl_loss:>10.4f}  "
              f"{tr_dice:>10.4f}  {vl_dice:>10.4f}")

        if vl_dice > best_dice:
            best_dice = vl_dice
            torch.save({
                "model_state":  model.state_dict(),
                "base_filters": args.base_filters,
                "image_size":   args.image_size,
                "val_dice":     best_dice,
                "epoch":        epoch,
            }, args.out)

    print(f"\nBest Val Dice: {best_dice:.4f}  →  saved to {args.out}")
    print("Next step: python scripts/predict.py --checkpoint", args.out)


if __name__ == "__main__":
    main()
