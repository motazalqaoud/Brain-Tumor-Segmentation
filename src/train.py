"""
Train the 3D Attention U-Net on real BraTS-format volumetric data.

Usage (on the HPC, inside a SLURM job -- see slurm/train_job.slurm):
    python src/train.py --data_dir /path/to/Task01_BrainTumour --epochs 100

Patch size and batch size are auto-selected based on detected GPU memory
(see dataset.recommend_patch_and_batch_size), since the exact GPU type on the
cluster wasn't known in advance -- this is designed to not OOM whether it
lands on a 16GB V100 or an 80GB A100 node. Override either with --patch_size
/ --batch_size if you want to tune manually once you know your node's GPU.

Uses mixed precision (torch.cuda.amp) since 3D convolutions are memory-heavy;
this alone roughly doubles the patch/batch size you can fit versus fp32.
"""

import argparse
import csv
import glob
import json
import os
import time

import torch
from monai.data import Dataset, DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import Activations, AsDiscrete, Compose

from dataset import train_transforms, eval_transforms, REGION_NAMES, recommend_patch_and_batch_size
from model import build_model


def discover_datalist(data_dir: str, val_frac: float = 0.2, seed: int = 42):
    """
    MSD Task01_BrainTumour layout: imagesTr/*.nii.gz + labelsTr/*.nii.gz,
    matched by filename. Splits by subject (each file = one subject, so no
    lesion-level leakage concern here, unlike HAM10000).
    """
    images = sorted(glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
    labels = sorted(glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))
    if not images:
        raise FileNotFoundError(
            f"No images found under {data_dir}/imagesTr/. "
            "Run data_prep.py first to download/verify the dataset."
        )
    assert len(images) == len(labels), \
        f"Mismatched counts: {len(images)} images vs {len(labels)} labels"

    datalist = [{"image": img, "label": lbl} for img, lbl in zip(images, labels)]

    import random
    rng = random.Random(seed)
    rng.shuffle(datalist)
    n_val = max(1, int(len(datalist) * val_frac))
    return datalist[n_val:], datalist[:n_val]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True,
                         help="Directory containing imagesTr/ and labelsTr/ (Task01_BrainTumour layout)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=None, help="Override auto-detected batch size")
    parser.add_argument("--patch_size", type=int, default=None, help="Override auto-detected cubic patch size")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output_dir", default="checkpoints")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_interval", type=int, default=5, help="Run full-volume validation every N epochs")
    parser.add_argument("--amp", action="store_true", default=True, help="Use mixed precision (default on)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    auto_patch, auto_batch, note = recommend_patch_and_batch_size()
    patch = (args.patch_size,) * 3 if args.patch_size else auto_patch
    batch_size = args.batch_size or auto_batch
    print(f"GPU sizing: {note}")
    print(f"Using patch_size={patch}, batch_size={batch_size}")

    train_files, val_files = discover_datalist(args.data_dir)
    print(f"Split sizes -> train: {len(train_files)}  val: {len(val_files)}")

    train_ds = Dataset(data=train_files, transform=train_transforms(patch_size=patch))
    val_ds = Dataset(data=val_files, transform=eval_transforms())

    # drop_last=True: with a small patch size relative to network depth, the deepest
    # bottleneck can collapse to a 1x1x1 spatial feature map; combined with a final
    # undersized batch (batch_size doesn't evenly divide the training set size, which
    # is the common case), BatchNorm crashes with "Expected more than 1 value per
    # channel." Dropping the incomplete final batch avoids this reliably.
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
                               drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=args.num_workers)

    model = build_model(in_channels=4, out_channels=3).to(device)
    loss_fn = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and torch.cuda.is_available())

    post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    dice_metric = DiceMetric(include_background=True, reduction="mean_batch")

    log_path = os.path.join(args.output_dir, "training_log.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_dice_TC", "val_dice_WT", "val_dice_ET",
                                 "val_dice_mean", "lr", "seconds"])

    best_mean_dice = -1.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            inputs, labels = batch["image"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=args.amp and torch.cuda.is_available()):
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
        epoch_loss /= max(1, len(train_loader))
        scheduler.step()

        val_dice_scores = [float("nan")] * 3
        if epoch % args.val_interval == 0 or epoch == args.epochs:
            model.eval()
            dice_metric.reset()
            with torch.no_grad():
                for val_batch in val_loader:
                    val_inputs = val_batch["image"].to(device)
                    val_labels = val_batch["label"].to(device)
                    # Full volumes at eval time -> sliding-window inference,
                    # since the model only ever saw fixed-size patches in training.
                    val_outputs = sliding_window_inference(
                        val_inputs, roi_size=patch, sw_batch_size=1, predictor=model, overlap=0.5
                    )
                    val_outputs = [post_pred(x) for x in decollate_batch(val_outputs)]
                    val_labels_list = decollate_batch(val_labels)
                    dice_metric(y_pred=val_outputs, y=val_labels_list)
            val_dice_scores = dice_metric.aggregate().cpu().numpy().tolist()

        mean_dice = sum(v for v in val_dice_scores if v == v) / max(1, sum(1 for v in val_dice_scores if v == v)) \
            if any(v == v for v in val_dice_scores) else float("nan")
        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        dice_str = "  ".join(f"{r}={d:.4f}" for r, d in zip(REGION_NAMES, val_dice_scores)) if val_dice_scores[0] == val_dice_scores[0] else "n/a"
        print(f"Epoch {epoch:3d}/{args.epochs} | train_loss {epoch_loss:.4f} | val_dice [{dice_str}] | "
              f"lr {current_lr:.2e} | {elapsed:.1f}s")

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, epoch_loss, *val_dice_scores, mean_dice, current_lr, elapsed])

        if mean_dice == mean_dice and mean_dice > best_mean_dice:  # mean_dice==mean_dice excludes NaN
            best_mean_dice = mean_dice
            torch.save(
                {"model_state_dict": model.state_dict(), "epoch": epoch, "val_dice_mean": mean_dice,
                 "val_dice_per_region": dict(zip(REGION_NAMES, val_dice_scores))},
                os.path.join(args.output_dir, "best_model.pth"),
            )
            print(f"  -> saved new best checkpoint (mean_dice={mean_dice:.4f})")

    print(f"\nTraining complete. Best mean Dice: {best_mean_dice:.4f}")
    print(f"Checkpoint: {os.path.join(args.output_dir, 'best_model.pth')}")


if __name__ == "__main__":
    main()
