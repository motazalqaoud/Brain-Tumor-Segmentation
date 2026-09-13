"""
Evaluate a trained checkpoint on held-out volumes using full-volume
sliding-window inference (not patches) -- the standard BraTS evaluation
protocol: per-region Dice (TC, WT, ET) and 95th-percentile Hausdorff distance.

Usage:
    python src/evaluate.py --data_dir /path/to/Task01_BrainTumour --checkpoint checkpoints/best_model.pth
"""

import argparse

import torch
from monai.data import Dataset, DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import Activations, AsDiscrete, Compose

from dataset import eval_transforms, REGION_NAMES
from model import build_model
from train import discover_datalist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--roi_size", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _train_files, val_files = discover_datalist(args.data_dir)
    val_ds = Dataset(data=val_files, transform=eval_transforms())
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

    model = build_model(in_channels=4, out_channels=3).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
    model.eval()

    post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    dice_metric = DiceMetric(include_background=True, reduction="mean_batch")
    hd_metric = HausdorffDistanceMetric(include_background=True, percentile=95, reduction="mean_batch")

    roi = (args.roi_size,) * 3
    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = batch["image"].to(device), batch["label"].to(device)
            outputs = sliding_window_inference(inputs, roi_size=roi, sw_batch_size=1, predictor=model, overlap=0.5)
            outputs = [post_pred(x) for x in decollate_batch(outputs)]
            labels_list = decollate_batch(labels)
            dice_metric(y_pred=outputs, y=labels_list)
            try:
                hd_metric(y_pred=outputs, y=labels_list)
            except Exception:
                pass  # HD is undefined when a region is absent in both pred and label; skip that case

    dice_scores = dice_metric.aggregate().cpu().numpy()
    print("=" * 60)
    print(f"{'Region':<10}{'Dice':>10}")
    print("=" * 60)
    for region, dice in zip(REGION_NAMES, dice_scores):
        print(f"{region:<10}{dice:>10.4f}")
    print(f"{'Mean':<10}{dice_scores.mean():>10.4f}")

    try:
        hd_scores = hd_metric.aggregate().cpu().numpy()
        print()
        print(f"{'Region':<10}{'HD95 (mm)':>10}")
        for region, hd in zip(REGION_NAMES, hd_scores):
            print(f"{region:<10}{hd:>10.2f}")
    except Exception as e:
        print(f"\n(Hausdorff distance not computed: {e})")

    print("\nDice is bounded [0,1], higher is better. HD95 is in mm, lower is better.")
    print("These are the two standard BraTS leaderboard metrics.")


if __name__ == "__main__":
    main()
