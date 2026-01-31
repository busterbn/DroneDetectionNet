# Load Testing Guide

## Run the Test

```bash
uv run locust -f tests/load/locustfile.py --host https://drone-detector-api-66108710596.europe-north2.run.app
```

Open <http://localhost:8089>, configure:

- **Users:** 10
- **Spawn rate:** 2
- **Run time:** 10m
- Click **START**

## Endpoints Tested

| Endpoint      | Method | Weight | Description             |
|---------------|--------|--------|-------------------------|
| `/healtht`    | GET    | 1      | Health check            |
| `/`           | GET    | 1      | Root endpoint           |
| `/v1/info`    | GET    | 2      | API info                |
| `/v1/predict` | POST   | 10     | Standard image (224x224) |
| `/v1/predict` | POST   | 3      | Large image (1920x1080) |
| `/docs`       | GET    | 1      | OpenAPI documentation   |

## Read Results

Go to **Statistics** tab, look at **POST /v1/predict**:

- **Median:** Typical response time
- **95th percentile:** What most users see
- **99th percentile:** Cold starts
- **Failures:** Should be 0%

Download CSV from **Download Data** tab for comparison.
