# Deployment

## API Overview

FastAPI service with ONNX Runtime for inference, deployed on Cloud Run.

**Live API**: <https://drone-detector-api-66108710596.europe-north2.run.app>

**Endpoints**:

- `GET /health` - Health check
- `GET /v1/info` - API metadata
- `POST /v1/predict` - Image classification
- `GET /v1/monitoring/*` - Drift detection endpoints
- `GET /metrics` - Prometheus metrics

## Local Development

```bash
# Run API locally
uvicorn drone_detector_mlops.api.main:app --reload

# Test prediction
curl -X POST http://localhost:8000/v1/predict \
  -F "file=@image.jpg"
```

## Docker Build

```bash
# Build API container
invoke docker-build-api

# Build frontend
invoke docker-build-frontend

# Run locally
docker run -p 8000:8000 api:latest
```

## Cloud Deployment

```bash
# Build and push to Artifact Registry
invoke cloud-build-api

# Deploy to Cloud Run
invoke deploy-api
```

Deployment config: `cloud/cloudrun-api.yaml`

**Auto-deployment**: Pushes to `main` trigger deployment via GitHub Actions (`.github/workflows/deploy.yaml`)

## Configuration

Cloud Run settings:

- Region: `europe-north2`
- Memory: 4GB
- CPU: 2 vCPUs
- Min instances: 1
- Max instances: 10
- Port: 8000

## ONNX Conversion

Convert PyTorch model to ONNX for faster CPU inference:

```bash
python scripts/convert_to_onnx.py \
  --input models/model.pth \
  --output models/model.onnx
```

Performance improvement: ~25% faster inference vs PyTorch.

## Frontend

SvelteKit UI deployed separately on Cloud Run.

**Live**: <https://drone-detector-frontend-66108710596.europe-north2.run.app>

Features:

- Image upload
- Real-time predictions
- Confidence scores
- Class visualization

## Monitoring

API includes Prometheus metrics:

- Request counts
- Prediction latency
- Error rates
- Model status

Metrics scraped by GCP Managed Prometheus sidecar.
