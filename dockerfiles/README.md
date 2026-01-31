# Docker Containers

## Training Container

### Purpose

Containerized training environment for the drone detector model using multi-stage build with `uv` for dependency management. Exports both PyTorch (.pth) and ONNX (.onnx) models.

### Build Process

1. **Stage 1**: Extract `uv` binary from official image
2. **Stage 2**:
   - Python 3.12 slim base based on Debian
   - Install system dependencies (`libgl1`, `libglib2.0-0`)
   - Install Python packages via `uv`
   - Copy source code and data splits

### Usage

```bash
# Build
docker-compose build train

# Run with defaults
docker-compose up train

# Custom parameters
docker-compose run train --epochs 20 --batch-size 64 --lr 0.0001
```

### Container Structure

```plain
/app/
├── src/                   # Application code
├── data/splits/           # Train/val/test splits (built-in)
├── data/                  # Raw data (volume mounted)
└── models/                # Output (volume mounted)
    ├── model-latest.pth   # PyTorch checkpoint
    └── model-latest.onnx  # ONNX export (for API)
```

## API Container

### Purpose

FastAPI inference server using **ONNX Runtime** for optimized CPU inference.

### Build Process

1. **Stage 1**: Extract `uv` binary
2. **Stage 2**:
   - Python 3.12 slim base
   - Install system dependencies
   - Install Python packages (FastAPI, ONNX Runtime, Pillow, etc.)
   - Copy source code

### Usage

```bash
# Build
docker-compose build api

# Run locally
docker-compose up api

# Cloud deployment
uv run invoke cloud-build-api   # Build on Cloud Build
uv run invoke deploy-api  # Deploy to Cloud Run
```

The API automatically loads `model-latest.onnx` from local storage or GCS depending on `MODE` setting.
