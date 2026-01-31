# Getting Started

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- Google Cloud CLI (for cloud features)

## Installation

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone repo
git clone https://github.com/dtu-mlops/drone-detection-mlops.git
cd drone-detector-mlops

# Install dependencies
uv sync

# Set up pre-commit hooks
uv run pre-commit install
```

## Environment Setup

Create `.env` file:

```bash
WANDB_API_KEY="your-key"
```

Authenticate with GCP:

```bash
gcloud auth login
gcloud auth application-default login
```

## Quick Test

```bash
# Pull data
dvc pull

# Run tests
uv run pytest

# Train locally
uv run python -m drone_detector_mlops.workflows.train
```

## Next Steps

- [Training](training.md) - Train models locally or on Vertex AI
- [Deployment](deployment.md) - Deploy API to Cloud Run
- [Monitoring](monitoring.md) - Set up drift detection
