# Dataset

Drone vs Bird images from [Kaggle](https://www.kaggle.com/datasets/muhammadsaoodsarwar/drone-vs-bird).

The actual image files are not in git (too large). They're stored in Google Cloud Storage and managed by DVC.

- **GCP Bucket**: `gs://drone-detection-mlops-data/dvc-store/`
- **DVC tracking files**: `bird.dvc` and `drone.dvc` (these are in git)

## Get the data

```bash
# First time: authenticate with GCP
gcloud auth application-default login

# Pull the data
uv run dvc pull
```

Images will appear in `data/drone/` and `data/bird/`.

## Updating data

```bash
# After adding/modifying images
uv run dvc add data/drone data/bird
git add data/drone.dvc data/bird.dvc
git commit -m "Update dataset"
uv run dvc push
git push
```

## Overview

The dataset has 828 images across two classes: drone (51.7%) and bird (48.3%).

- **Format**: All JPEG
- **Total images**: 828 (428 drone, 400 bird)
- **Splits**: Pre-defined stratified train/val/test (70/15/15) in `data/splits/` for reproducibility
  - Train: 580 images
  - Val: 123 images
  - Test: 125 images

The dataset uses stratified splitting to maintain class balance across all splits. Images require resizing when loading (typically 224x224) - this is handled via transforms in the dataloader.
