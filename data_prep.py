"""
Download and verify the Medical Segmentation Decathlon Task01_BrainTumour
dataset -- 750 real 4D MRI volumes (484 train / 266 test) sourced from the
BraTS 2016/2017 challenge, publicly downloadable with no registration wall.

This uses MONAI's built-in DecathlonDataset downloader, which handles the
~7GB download, integrity check (md5), and extraction automatically.

Usage:
    python data_prep.py --root_dir ./data

After this completes, --root_dir will contain:
    Task01_BrainTumour/
      imagesTr/   (484 4D NIfTI volumes: FLAIR, T1w, T1gd, T2w)
      labelsTr/   (484 matching segmentation masks)
      imagesTs/   (266 test volumes, unlabeled -- held out by the Decathlon organizers)

Point src/train.py --data_dir at the Task01_BrainTumour folder this creates.
"""

import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", default="./data",
                         help="Where to download/extract the dataset (needs ~10GB free: 7GB archive + extracted files)")
    args = parser.parse_args()

    os.makedirs(args.root_dir, exist_ok=True)

    # Imported here so `python data_prep.py --help` doesn't require monai to be
    # importable just to print usage.
    from monai.apps import DecathlonDataset
    from monai.transforms import Compose

    print(f"Downloading Task01_BrainTumour to {args.root_dir} (this is a ~7GB download; "
          f"do this on the HPC's data/scratch storage, not your home directory quota)...")

    # section="training" downloads+indexes imagesTr/labelsTr; we don't need
    # DecathlonDataset's own transform/caching machinery here, just the download,
    # so an empty Compose (not a lambda -- lambdas aren't picklable and will
    # break under num_workers>0 multiprocessing) keeps this fast and simple.
    DecathlonDataset(
        root_dir=args.root_dir,
        task="Task01_BrainTumour",
        section="training",
        download=True,
        cache_rate=0.0,
        num_workers=4,
        transform=Compose([]),
    )

    data_dir = os.path.join(args.root_dir, "Task01_BrainTumour")
    images_dir = os.path.join(data_dir, "imagesTr")
    labels_dir = os.path.join(data_dir, "labelsTr")

    n_images = len([f for f in os.listdir(images_dir) if f.endswith(".nii.gz")])
    n_labels = len([f for f in os.listdir(labels_dir) if f.endswith(".nii.gz")])

    print(f"\nDownload complete.")
    print(f"  Images: {n_images} volumes in {images_dir}")
    print(f"  Labels: {n_labels} volumes in {labels_dir}")

    if n_images != n_labels:
        print(f"WARNING: image/label count mismatch ({n_images} vs {n_labels}). "
              f"Re-run this script; the download may have been interrupted.")
    else:
        print(f"\nOK. Next step:")
        print(f"  python src/train.py --data_dir {data_dir}")


if __name__ == "__main__":
    main()
