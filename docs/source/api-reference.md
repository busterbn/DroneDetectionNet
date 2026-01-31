# API Reference

## Base URL

**Production**: https://drone-detector-api-66108710596.europe-north2.run.app

**Local**: http://localhost:8000

**Docs**: https://drone-detector-api-66108710596.europe-north2.run.app/docs

## Endpoints

### Health Check

```bash
GET /health
```

**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### API Info

```bash
GET /v1/info
```

**Response**:
```json
{
  "model_version": "model-lr0.001-bs32-e50-20250121",
  "uptime_seconds": 12345.67
}
```

### Predict

```bash
POST /v1/predict
```

**Request**: Multipart form with image file

**Example**:
```bash
curl -X POST https://drone-detector-api.../v1/predict \
  -F "file=@image.jpg"
```

**Response**:
```json
{
  "prediction": {
    "class_name": "drone",
    "confidence": 0.95,
    "scores": {
      "drone": 0.95,
      "bird": 0.05
    }
  },
  "metadata": {
    "inference_time_ms": 45.2,
    "model_version": "model-lr0.001-bs32-e50-20250121"
  }
}
```

### Drift Monitoring

**Drift Summary**:
```bash
GET /v1/monitoring/drift-summary?max_predictions=100
```

**Drift Tests**:
```bash
GET /v1/monitoring/drift-tests?max_predictions=100
```

**HTML Report**:
```bash
GET /v1/monitoring/drift-report?max_predictions=100
```

**Comprehensive Analysis**:
```bash
GET /v1/monitoring/comprehensive-drift?max_predictions=100
```

### Prometheus Metrics

```bash
GET /metrics
```

Returns metrics in Prometheus format.

## Error Responses

```json
{
  "detail": "Error message"
}
```

Common status codes:
- `400` - Invalid input
- `413` - File too large
- `500` - Server error

## Rate Limits

No rate limits currently enforced. Cloud Run autoscales from 1-10 instances.

## Authentication

Currently public. Add authentication for production use.
