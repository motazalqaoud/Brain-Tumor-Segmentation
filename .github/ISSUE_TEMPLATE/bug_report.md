---
name: Bug report
about: Something is broken or producing wrong results
title: "[BUG] "
labels: bug
assignees: motazalqaoud
---

## What happened?

<!-- Describe what went wrong. Include error messages verbatim. -->

## Steps to reproduce

```bash
# Paste the exact command(s) that trigger the bug
```

## Expected behavior

<!-- What should have happened instead? -->

## Environment

- OS:
- Python version (`python --version`):
- PyTorch version (`python -c "import torch; print(torch.__version__)"`):
- GPU / CUDA (if relevant):

## Which component is affected?

- [ ] Data loading (`src/preprocessing/` or `scripts/test_model.py`)
- [ ] Training (`scripts/train.py` or `scripts/train3d.py`)
- [ ] Model architecture (`src/segmentation/`)
- [ ] Visualization (`src/visualization/`)
- [ ] Inference (`scripts/predict.py`)
- [ ] Notebooks
- [ ] Other

## Additional context

<!-- Attach logs, screenshots, or sample data if helpful. Never attach real patient data. -->
