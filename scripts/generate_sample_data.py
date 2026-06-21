"""
generate_sample_data.py
-----------------------
Generate synthetic breast-MRI-like volumes + lesion masks and save them as
NIfTI files into data/samples/.

This lets the whole pipeline (loading -> preprocessing -> training) run
end-to-end WITHOUT needing access to real patient data (which is protected
health information and cannot be redistributed).

Usage:
    python scripts/generate_sample_data.py --n 20 --size 128
"""

import argparse
from pathlib import Path

import numpy as np
import nibabel as nib


def make_volume(size: int = 128, n_slices: int = 32, n_lesions: int = 1,
                rng: np.random.Generator = None):
    """Create one synthetic 3D MRI volume (Z, Y, X) and its binary lesion mask."""
    if rng is None:
        rng = np.random.default_rng()

    vol = rng.normal(0.15, 0.04, (n_slices, size, size)).astype(np.float32)
    mask = np.zeros_like(vol)

    z_g, y_g, x_g = np.mgrid[0:n_slices, 0:size, 0:size]

    # Background "tissue" ellipsoid
    tissue = (((y_g - size // 2) ** 2) / (size / 2.3) ** 2 +
              ((x_g - size // 2) ** 2) / (size / 2.3) ** 2) < 1
    vol[tissue] += rng.normal(0.3, 0.05, vol[tissue].shape).astype(np.float32)

    # Spherical lesions (bright, small -> realistic class imbalance)
    for _ in range(n_lesions):
        cz = rng.integers(n_slices // 4, 3 * n_slices // 4)
        cy = rng.integers(size // 4, 3 * size // 4)
        cx = rng.integers(size // 4, 3 * size // 4)
        r = rng.integers(6, 14)
        lesion = ((z_g - cz) ** 2 + (y_g - cy) ** 2 + (x_g - cx) ** 2) < r ** 2
        vol[lesion] += rng.uniform(0.25, 0.45)
        mask[lesion] = 1.0

    vol = np.clip(vol, 0, 1).astype(np.float32)
    mask = (mask > 0.5).astype(np.float32)
    return vol, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="number of volumes")
    ap.add_argument("--size", type=int, default=128, help="in-plane size")
    ap.add_argument("--slices", type=int, default=32, help="slices per volume")
    ap.add_argument("--out", type=str, default="data/samples", help="output dir")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # Isotropic 1mm spacing encoded in the affine (so spacing-aware code works)
    affine = np.diag([1.0, 1.0, 1.0, 1.0])

    for i in range(args.n):
        vol, mask = make_volume(args.size, args.slices, n_lesions=1, rng=rng)
        nib.save(nib.Nifti1Image(vol, affine), out / f"case_{i:03d}_image.nii.gz")
        nib.save(nib.Nifti1Image(mask, affine), out / f"case_{i:03d}_mask.nii.gz")
        print(f"  wrote case_{i:03d}  (lesion = {mask.mean()*100:.2f}% of voxels)")

    print(f"\nDone. {args.n} image/mask pairs in {out.resolve()}")


if __name__ == "__main__":
    main()
