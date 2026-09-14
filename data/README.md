# Data

## Kaggle Brain Tumor Dataset (12K+ images)

Download via the Kaggle API:

```bash
pip install kaggle
kaggle datasets download -d fernando2rad/brain-tumor-12k-mri-images-w-masks-meta-and-bbox
unzip brain-tumor-12k-mri-images-w-masks-meta-and-bbox.zip -d data/raw/
```

You need a Kaggle account and an API key at `~/.kaggle/kaggle.json`.
See the [Kaggle API docs](https://www.kaggle.com/docs/api) for setup instructions.

Dataset contents:
- 12,000+ T1/T2-weighted MRI brain slices
- Tumor segmentation masks (consensus ground truth)
- Bounding boxes and per-image JSON metadata
- Tumor types: Gliomas, Meningothelial Tumors, Germ Cell Tumors, and more

After unzipping, the folder layout should look like:

```
data/raw/Images_/Images_/
├── Gliomas/
│   ├── Gliomas T1C+/
│   │   └── [subtype]/
│   │       ├── image.jpg
│   │       ├── image_mask_consensus.png
│   │       └── image_meta.json
│   ├── Gliomas T1/
│   └── Gliomas T2/
├── Meningothelial Tumors/
├── Germ Cell Tumors/
├── Embryonic Tumors/
├── Nerve Sheath Tumors/
└── Normal/
```

Verify the setup end-to-end before training:

```bash
python scripts/test_model.py --data-root data/raw/Images_
```

Or use the automated setup which handles everything:

```bash
python run.py setup
```

## Additional public datasets

| Dataset | What | Link |
|---|---|---|
| BraTS | Multi-modal brain tumor MRI (larger, multi-site) | https://www.synapse.org/brats |
| Medical Segmentation Decathlon | Brain + other organs | http://medicaldecathlon.com |

Both require registration and a data-use agreement. After downloading, load with:

```python
from src.preprocessing import load_nifti, load_dicom_series
volume, affine, meta = load_nifti("data/raw/BRATS_001.nii.gz")
volume, meta = load_dicom_series("data/raw/patient_01/")
```

## Folder layout

```
data/
└── raw/       # real datasets — gitignored
```
