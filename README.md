# Drone Detector MLOps

A drone detection framework with extra MLOps sauce

## Cool features

- [x] Automatically delete branches after PR is merged
- [x] `make pr` uses Claude Code with the gh-pull-requests skill to create a PR
- [x] `make test` runs tests with coverage
- [x] Use the `uv run train` command to run the training script
- [x] Set up pre-commit hooks for linting and formatting
- [x] Github Actions check to make sure PR branch is up to date with main before running other workflows to minimize runner usage
- [x] Load environment variables from Github Secrets

## Cloud Infrastructure

### Region Architecture

We initially started only on `europe-north2`, but found GPU computing to be an issue here, so we expanded to `europe-west4`, thus giving us the setup:

- **Storage (GCS) & Registry:** `europe-north2` (Finland) - Minimal latency for local uploads.
- **Compute (Vertex AI):** `europe-west4` (Netherlands) - Selected for **NVIDIA T4** availability.

### Storage Context Manager

We use a storage abstraction layer ([`storage.py`](src/drone_detector_mlops/utils/storage.py)) that handles local vs cloud paths transparently. Set `MODE=local` or `MODE=cloud` in [`settings`](src/drone_detector_mlops/utils/settings.py) - the same code works in both environments.

### GCS Buckets

- **`gs://drone-detection-mlops-data/`** - Training data
  - `/structured/` - Direct training access (drone/, bird/, splits/)
  - `/dvc-store/` - DVC versioning history (not used during training)
- **`gs://drone-detection-mlops-models/`** - Trained models
  - `/checkpoints/` - Production model storage

#### Update structured data - for example if we get new data

```bash
gsutil -m rsync -r -d data/drone gs://drone-detection-mlops-data/structured/drone

gsutil -m rsync -r -d data/bird gs://drone-detection-mlops-data/structured/bird

gsutil -m rsync -r -d data/splits gs://drone-detection-mlops-data/structured/splits
```

### Cloud Training Pipeline

#### 1. Build the Container

We use Google Cloud Build to offload the construction of the PyTorch/CUDA environment.

```bash
invoke cloud-build-train
```

**Artifact Registry:** `europe-north2-docker.pkg.dev/drone-detection-mlops/ml-containers/train.`

Every build updates the :latest tag used for production runs.

#### 2. Submit to Vertex AI

Training jobs run on dedicated NVIDIA T4 GPUs.

```bash
# Default
invoke cloud-train --epochs 50

# Custom configuration
invoke cloud-train --epochs 50 --batch-size 64
```

### API Deployment

The API uses **ONNX Runtime** for optimized CPU inference (much smaller container than PyTorch).

#### Build and Deploy

```bash
# Build API container
uv run invoke cloud-build-api

# Deploy to Cloud Run
uv run invoke deploy-api
```

**Live endpoint:** `https://drone-detector-api-66108710596.europe-north2.run.app`

Models are automatically loaded from GCS (`model-latest.onnx`).
