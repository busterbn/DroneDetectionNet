# Model Training

## Local Training

```bash
# Train with default config
uv run python -m drone_detector_mlops.workflows.train

# Override hyperparameters
uv run python -m drone_detector_mlops.workflows.train \
  epochs=20 batch_size=64 learning_rate=0.001
```

## Cloud Training (Vertex AI)

Set storage mode to `"cloud"` in `src/drone_detector_mlops/utils/settings.py`.

```bash
# Single training run
invoke cloud-train

# With custom hyperparameters
invoke cloud-train --epochs=50 --batch-size=128 --lr=0.001

# Hyperparameter sweep (20 Optuna trials)
uv run -m scripts.submit_training --sweep --yes
```

Training runs on `n1-standard-4` + NVIDIA T4 GPU in `europe-west4`.

## Configuration

Configs are in `configs/` using Hydra:

```yaml title="configs/config.yaml"
defaults:
  - params: param_1

batch_size: 32
epochs: 10
learning_rate: 0.001
```

## Experiment Tracking

All runs log to Weights & Biases:

- Training/validation loss & accuracy
- Model checkpoints
- Hyperparameters
- GPU utilization

View at: [wandb.ai](https://wandb.ai)

## Model Architecture

ResNet18 from TIMM with custom classifier:

```python
model = timm.create_model('resnet18', pretrained=True, num_classes=2)
```

## Data Loading

- Dataset: Drone vs Bird images from Kaggle
- Storage: GCS bucket (`gs://drone-detection-mlops-data/structured`)
- Splits: 70/15/15 train/val/test
- Augmentations: Random flip, rotation, color jitter

## Output

Models saved to:

- Local: `models/model-lr{lr}-bs{bs}-e{epochs}-{date}.pth`
- Cloud: `gs://drone-detection-mlops-models/`
- W&B: Auto-uploaded as artifacts
