# Monitoring & Drift Detection

## Overview

Monitoring tracks model performance and detects data drift in production.

## Prometheus Metrics

API exposes metrics at `/metrics`:

- `prediction_requests_total` - Total predictions
- `predictions_by_class_total` - Per-class counts
- `prediction_latency_seconds` - Inference time histogram
- `request_size_bytes` - Upload size
- `http_errors_total` - Error counts
- `model_loaded_info` - Model status

Metrics scraped by GCP Managed Prometheus sidecar.

## Drift Detection

Three levels of drift monitoring using Evidently:

**1. Prediction-level**: Confidence scores, class distribution
**2. Image-level**: Brightness, contrast, RGB statistics
**3. Embedding-level**: ResNet feature similarity

### API Endpoints

```bash
# Get drift summary
curl https://drone-detector-api.../v1/monitoring/drift-summary

# Run drift tests
curl https://drone-detector-api.../v1/monitoring/drift-tests

# Generate HTML report
curl https://drone-detector-api.../v1/monitoring/drift-report

# Comprehensive analysis
curl https://drone-detector-api.../v1/monitoring/comprehensive-drift
```

### Automated Monitoring

GitHub Actions workflow runs daily drift checks (`.github/workflows/drift-monitoring.yaml`):

```yaml
schedule:
  - cron: '0 0 * * *'  # Daily at midnight
```

Reports uploaded as artifacts when drift detected.

## Data Collection

Predictions automatically saved to GCS:

```
gs://drone-detection-mlops-data/inference/
  prediction_20250123_120000_123456.json
```

Each file contains:
- Prediction results
- Image features
- Model version
- Timestamp

## Local Testing

```bash
# Run drift analysis locally
uv run python -m drone_detector_mlops.monitoring.drift_detection

# Generate report
python scripts/generate_drift_report.py
```

## Alerts

Configure GCP Monitoring alerts for:
- High error rates
- Latency spikes
- Low confidence predictions
- Drift detection failures
