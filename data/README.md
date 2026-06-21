# Data

Real patient images are **protected health information** and cannot be shipped
with this repo. You have two options.

## Option A — Synthetic data (works out of the box, no download)

```bash
python scripts/generate_sample_data.py --n 20 --size 128
```

This writes `case_XXX_image.nii.gz` / `case_XXX_mask.nii.gz` pairs into
`data/samples/`. Training (`python train.py`) also generates synthetic slices
on the fly, so you can train without any files at all.

## Option B — Real public datasets

Download into `data/raw/` (kept out of git via `.gitignore`):

| Dataset | What | Link |
|---|---|---|
| Medical Segmentation Decathlon | 10 organ/tumor tasks (brain, liver, lung…), NIfTI | http://medicaldecathlon.com |
| BraTS | Brain tumor MRI segmentation | https://www.synapse.org/brats |
| Breast MRI (Duke / TCIA) | Breast DCE-MRI + annotations | https://www.cancerimagingarchive.net |
| LIDC-IDRI | Lung CT nodules (DICOM) | https://www.cancerimagingarchive.net |

Most are free but require registration / a data-use agreement. After
downloading, load them with the helpers in `src/preprocessing/`:

```python
from src.preprocessing import load_nifti, load_dicom_series
volume, affine, meta = load_nifti("data/raw/BRATS_001.nii.gz")
volume, meta = load_dicom_series("data/raw/patient_01/")
```

## Folders

```
data/
├── samples/   # generated synthetic NIfTI pairs (Option A)
└── raw/       # real datasets you download (gitignored)
```
