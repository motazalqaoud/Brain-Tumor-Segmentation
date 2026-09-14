# Contributing to Brain-Tumor-Segmentation

## Getting Started

```bash
git clone https://github.com/motazalqaoud/Brain-Tumor-Segmentation.git
cd Brain-Tumor-Segmentation
pip install -r requirements.txt
```

Verify the setup runs end-to-end before making changes:

```bash
python scripts/generate_sample_data.py --n 10 --size 64
python scripts/train.py --epochs 2
```

## Project Structure

| Directory | What lives here |
|---|---|
| `src/preprocessing/` | Dataset loaders, transforms, normalization |
| `src/segmentation/` | Model architectures and loss functions |
| `src/visualization/` | Plotting utilities |
| `scripts/` | Training, inference, and test entry points |
| `notebooks/` | Educational walkthroughs (synthetic data only) |

## Making Changes

- **Bug fix**: open an issue first describing the bug and how you reproduced it
- **New feature**: open a Discussion under the Ideas category before writing code
- **Notebook change**: keep synthetic data — do not commit real patient images

## Code Style

- Python 3.8+, type hints on all public functions
- No magic numbers — use named constants or argparse arguments
- No hardcoded absolute paths — use `Path(__file__).parent` relative to the file
- Test your change with `python scripts/test_model.py --data-root data/raw/Images_` if touching the data pipeline

## Submitting a Pull Request

1. Fork the repo and create a branch: `git checkout -b fix/your-description`
2. Make your change and verify it runs
3. Open a PR against `main` — fill in the PR template
4. Reference the issue number in the PR description

## Questions

Open a [Discussion](https://github.com/motazalqaoud/Brain-Tumor-Segmentation/discussions) — bug reports go in Issues.
