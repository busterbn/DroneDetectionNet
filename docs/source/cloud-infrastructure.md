# Cloud Infrastructure

## GCP Services Used

**Storage & Data**:

- Cloud Storage - Data, models, inference results
- DVC - Dataset version control

**Compute**:

- Vertex AI - GPU training (n1-standard-4 + T4)
- Cloud Run - Serverless API and frontend

**Build & Deploy**:

- Cloud Build - Docker image builds
- Artifact Registry - Container storage

**Monitoring**:

- Cloud Logging - Centralized logs
- GCP Managed Prometheus - Metrics collection

## Multi-Region Setup

**europe-north2 (Finland)**:

- GCS buckets (data, models)
- Artifact Registry
- Cloud Run (API, frontend)
- Cloud Build

**europe-west4 (Netherlands)**:

- Vertex AI training
- T4 GPU availability

Multi-region setup was a pragmatic choice driven by GPU availability rather than architectural design.

## Storage Buckets

```plaintext
gs://drone-detection-mlops-data/
  ├── structured/           # Training data (DVC)
  ├── inference/           # Production predictions
  └── splits/              # Train/val/test splits

gs://drone-detection-mlops-models/
  ├── *.pth               # PyTorch checkpoints
  ├── *.onnx              # ONNX models
  └── profiling/          # Training profiles
```

## Compute Resources

**Vertex AI Training**:

- Machine: n1-standard-4 (4 vCPU, 15GB RAM)
- GPU: NVIDIA Tesla T4
- Region: europe-west4

**Cloud Run API**:

- Memory: 4GB
- CPU: 2 vCPUs
- Instances: 1-10 (autoscaling)
- Region: europe-north2

## CI/CD Workflows

Located in `.github/workflows/`:

- `tests.yaml` - Run tests on PR
- `linting.yaml` - Code quality checks
- `build.yaml` - Build Docker images
- `deploy.yaml` - Deploy to Cloud Run (auto on push to main)
- `train.yaml` - Submit Vertex AI training jobs (manual)
- `drift-monitoring.yaml` - Daily drift checks

## IAM & Permissions

Team members need:

- Storage Object Viewer/Creator (GCS access)
- Vertex AI User (training jobs)
- Cloud Run Developer (deployments)
- Artifact Registry Writer (image push)

Add email to `src/drone_detector_mlops/permissions/cloud_members.txt` and run permission setup scripts.

## Cost Optimization

- Use Cloud Run (pay per request, auto-scaling to zero)
- ONNX models for faster/cheaper CPU inference
- Vertex AI jobs tear down automatically after training
- Artifact Registry lifecycle policies for old images
