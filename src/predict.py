"""
Run inference on a single subject's 4-modality NIfTI volume, producing a
segmentation mask as a NIfTI file you can load in any viewer (3D Slicer, ITK-
SNAP, etc.) alongside the original scan.

Usage:
    python src/predict.py --image path/to/subject_4mod.nii.gz \
        --checkpoint checkpoints/best_model.pth --output segmentation.nii.gz

Expects --image to be a single 4D NIfTI (H, W, D, 4) in the MSD
Task01_BrainTumour layout (FLAIR, T1w, T1gd, T2w stacked as the 4th
dimension). If your scans are four separate 3D files (one per modality,
common with raw BraTS/clinical exports), stack them into one 4D NIfTI first
--nib.concat_images() or a short numpy stack + nib.save() will do it.
"""

import argparse

import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.transforms import Activations, AsDiscrete, Compose

from dataset import inference_transforms, REGION_NAMES
from model import load_model


def predict_volume(image_path: str, checkpoint_path: str, roi_size=(128, 128, 128), device: str = "cpu"):
    model, weights_loaded = load_model(checkpoint_path=checkpoint_path, device=device)
    if not weights_loaded:
        raise FileNotFoundError(
            f"No checkpoint found at {checkpoint_path}. Train a model first with src/train.py."
        )

    transform = inference_transforms()
    data = transform({"image": image_path})
    input_tensor = data["image"].unsqueeze(0).to(device)  # add batch dim

    post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    with torch.no_grad():
        output = sliding_window_inference(input_tensor, roi_size=roi_size, sw_batch_size=1,
                                           predictor=model, overlap=0.5)
        output = post_pred(output.squeeze(0))  # (3, H, W, D) binary multi-label

    return output.cpu().numpy(), data["image"].meta.get("affine", np.eye(4))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument("--output", default="segmentation.nii.gz")
    parser.add_argument("--roi_size", type=int, default=128)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    roi = (args.roi_size,) * 3
    seg, affine = predict_volume(args.image, args.checkpoint, roi_size=roi, device=device)

    # Collapse the 3 nested regions (TC, WT, ET) into a single label map for
    # easy viewing: 0=background, 1=edema-only (part of WT, not TC), 2=TC
    # non-enhancing, 4=ET -- matches the original BraTS label convention.
    tc, wt, et = seg[0], seg[1], seg[2]
    label_map = np.zeros_like(tc, dtype=np.uint8)
    label_map[(wt > 0) & (tc == 0)] = 2       # edema (whole tumor minus core)
    label_map[(tc > 0) & (et == 0)] = 1       # necrotic / non-enhancing core
    label_map[et > 0] = 4                      # enhancing tumor

    affine_np = np.asarray(affine) if not isinstance(affine, np.ndarray) else affine
    nib.save(nib.Nifti1Image(label_map, affine_np), args.output)

    print(f"Saved segmentation to {args.output}")
    print()
    print("Voxel counts by region:")
    for name, mask in zip(REGION_NAMES, [tc, wt, et]):
        print(f"  {name}: {int(mask.sum())} voxels")
    print()
    print("This is a research tool, not a diagnosis. Any clinical decision "
          "must involve a radiologist and the full clinical picture.")


if __name__ == "__main__":
    main()
