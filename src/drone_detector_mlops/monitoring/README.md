# Comprehensive Drift Monitoring

**Multi-level drift detection system for drone vs bird classification.**

This monitoring system tracks drift across three levels, from most to least relevant for the ResNet18 model:

1. **Prediction-level** ✅ (Most Important) - Confidence scores, class distribution
2. **Image-level** ⚠️ (Supplementary) - Brightness, colors, contrast
3. **Embedding-level** 🚀 (Advanced) - Model's learned representations

## Why Multiple Levels?

The ResNet18 model doesn't directly use brightness or colors - it learns abstract features through deep convolutional layers. Monitoring only image statistics (brightness, colors) is a **proxy** that may not reflect actual model performance. That's why we monitor at multiple levels.

## Quick Start

### Prerequisites
- Trained model available (locally or in GCS)
- GCP credentials configured (`gcloud auth login`)
- Data splits generated (`data/train_split.txt` exists)

### Setup

```bash
# 1. Install dependencies
uv sync

# 2. Configure environment
# For local development without GCS:
echo "MODE=local" >> .env

# For production with GCS (requires billing enabled):
echo "MODE=cloud" >> .env

# 3. Generate reference dataset from training images
# This extracts features from the training data to use as baseline
python -m drone_detector_mlops.monitoring.generate_reference_dataset

# You should see: "Reference dataset saved to data/reference_dataset.csv"
```

### Running Drift Monitoring

```bash
# 4. Start API (automatically extracts features from predictions)
uvicorn drone_detector_mlops.api.main:app --reload

# 5. Make some predictions (features saved to GCS automatically)
# Use actual images from dataset
curl -X POST http://localhost:8000/v1/predict -F "file=@data/drone/101264.jpg"
curl -X POST http://localhost:8000/v1/predict -F "file=@data/bird/69.jpg"
# Make several more...

# 6. RECOMMENDED: Run comprehensive drift analysis (all levels)
curl http://localhost:8000/v1/monitoring/comprehensive-drift | jq

# This returns:
# - Prediction drift (confidence drops, class distribution shifts)
# - Image feature drift (brightness, colors)
# - Overall assessment with severity and recommended actions

# 7. Or view detailed HTML report
curl http://localhost:8000/v1/monitoring/drift-report > report.html
open report.html
```

## Monitoring Endpoints

### 🎯 Comprehensive Drift Analysis (RECOMMENDED)
```bash
GET /v1/monitoring/comprehensive-drift?max_predictions=100
```
Analyzes all monitoring levels and provides actionable insights:
```json
{
  "timestamp": "2026-01-21T12:00:00Z",
  "reference_samples": 1000,
  "current_samples": 100,
  "drift_levels": {
    "prediction": {
      "confidence": {
        "reference_mean_confidence": 0.95,
        "current_mean_confidence": 0.68,
        "confidence_drop": 0.27,
        "significant_drop": true
      },
      "class_distribution": {
        "reference_drone_rate": 0.5,
        "current_drone_rate": 0.3,
        "distribution_shift": 0.2
      }
    },
    "image_features": {
      "all_passed": false,
      "summary": {...}
    }
  },
  "overall_assessment": {
    "severity": "high",
    "requires_action": true,
    "alerts": [
      "Model confidence dropped by 27%",
      "Class distribution shifted significantly"
    ],
    "recommended_actions": [
      "Review drift report to identify drifted features",
      "Correlate with model performance metrics in W&B",
      "Consider retraining model with recent production data"
    ]
  }
}
```

### HTML Report
```bash
GET /v1/monitoring/drift-report?max_predictions=100
```
Interactive HTML report with visualizations for all features.

### Programmatic Tests
```bash
GET /v1/monitoring/drift-tests?max_predictions=100
```
Returns JSON with `all_passed` boolean for alerting:
```json
{"all_passed": false, "summary": {"failed_tests": 3}}
```

### Drift Summary
```bash
GET /v1/monitoring/drift-summary?max_predictions=100
```
Compact JSON metrics for dashboards.

## Python API

```python
from drone_detector_mlops.monitoring import DriftDetector

detector = DriftDetector()

# Generate HTML report
detector.generate_drift_report(max_predictions=100, output_path="report.html")

# Run tests for alerting
results = detector.run_drift_tests(max_predictions=100)
if not results["all_passed"]:
    send_alert("Drift detected!")

# Fetch data for custom analysis
data = detector.fetch_recent_predictions(max_predictions=100)
```

## How It Works: Three Monitoring Levels

### 1. Prediction-Level Monitoring ✅ (Most Relevant)

**What it monitors:** Model outputs (confidence scores, predicted classes)

**Why it matters:** This directly measures what the model produces. If confidence drops or class distribution shifts dramatically, the model might be struggling.

**Metrics tracked:**
- `confidence`: Average prediction confidence (should be > 0.7)
- `class_distribution`: Ratio of drone vs bird predictions
- `low_confidence_rate`: Percentage of predictions with confidence < 0.7

**Example drift scenario:**
- Training: 95% average confidence, 50/50 drone/bird split
- Production: 65% average confidence, 80% bird predictions → **HIGH ALERT**

This indicates the model is uncertain and biased - likely due to data it hasn't seen before.

### 2. Image-Level Monitoring ⚠️ (Supplementary)

**What it monitors:** Hand-crafted image statistics (brightness, colors, contrast)

**Why it matters:** Detects environment changes (lighting, camera, weather) that *might* affect the model.

**Metrics tracked:**
- `brightness_mean`: Average brightness (0-255)
- `contrast`, `sharpness`: Image quality metrics
- `r_mean`, `g_mean`, `b_mean`: Color channel averages
- `saturation_mean`, `edge_density`: Visual characteristics

**Important:** These are **proxy metrics**. The ResNet18 model doesn't directly use brightness values - it learns abstract features. Image-level drift doesn't guarantee model degradation, but it's a warning sign.

**Example drift scenario:**
- Training: Daytime images (brightness=145, contrast=42)
- Production: Nighttime images (brightness=68, contrast=28) → **INVESTIGATE**

The model *might* still work fine if it learned robust features, but you should check actual performance.

### 3. Embedding-Level Monitoring 🚀 (Advanced)

**What it monitors:** ResNet18's internal representations (512-dim vectors from the last layer before classification)

**Why it matters:** This is what the model **actually sees** - the learned features it uses to make decisions.

**How it works:**
- Extracts 512-dimensional embeddings from ResNet18's final pooling layer
- Compares embedding distributions using statistical tests
- Calculates cosine similarity and L2 distance between reference and production embeddings

**When to use:** If prediction-level metrics look fine but you still suspect drift, or for deep analysis of model behavior changes.

**Note:** Requires loading PyTorch model (not just ONNX), so it's not run by default on every prediction.

## Architecture: What the Model Actually Uses

```
The ResNet18 Model:
Input Image (RGB)
    ↓
Conv Layer 1 → Learns edges, textures
    ↓
Conv Layer 2 → Learns shapes, patterns
    ↓
Conv Layer 3 → Learns object parts (wings, rotors)
    ↓
Conv Layer 4 → Learns complex features (bird body, drone structure)
    ↓
Global Avg Pool → 512-dim embedding ← [EMBEDDING MONITORING]
    ↓
Fully Connected → 2 classes (drone/bird)
    ↓
Softmax → Confidence scores ← [PREDICTION MONITORING]

[IMAGE MONITORING] ← Hand-crafted features (brightness, colors)
                     NOT what the model uses!
```

## How Drift Occurs in Practice

### Scenario 1: Nighttime Deployment
- **Training:** Daytime images
- **Production:** Nighttime images
- **Image drift:** ✓ Detected (brightness drops)
- **Prediction drift:** ✓ Detected (confidence drops to 60%)
- **Action:** Model struggles in low light → Retrain with nighttime data

### Scenario 2: Camera Change
- **Training:** DSLR camera
- **Production:** Security camera (lower quality)
- **Image drift:** ✓ Detected (sharpness, resolution)
- **Prediction drift:** ✗ Not detected (confidence still 90%)
- **Action:** Model is robust to quality changes → No action needed

### Scenario 3: Seasonal Change
- **Training:** Summer (green backgrounds)
- **Production:** Winter (snow, gray sky)
- **Image drift:** ✓ Detected (colors, saturation)
- **Prediction drift:** ? Check it!
- **Embedding drift:** ✓ Detected (learned features shifted)
- **Action:** Model sees different patterns → Monitor closely, consider retraining

## What to Do When Drift Detected

### Severity: HIGH (Prediction-level drift with confidence drop > 15%)

**Immediate Actions:**
1. **Check W&B metrics**: Correlate drift timing with accuracy drops
2. **Review drift report**: Understand which patterns changed
3. **Collect ground truth**: Label recent predictions to measure actual accuracy
4. **Plan retraining**: Gather recent production data for model update

**Example:** Confidence dropped from 95% to 65%, predictions biased toward "bird" → Model is struggling, likely needs retraining

### Severity: MEDIUM (Class distribution shift or moderate confidence drop)

**Actions:**
1. **Monitor closely**: Check drift daily instead of weekly
2. **Investigate causes**: Did deployment environment change?
3. **Verify accuracy**: Sample predictions manually or collect labels
4. **Prepare data**: Start collecting production data for potential retraining

**Example:** Confidence dropped from 95% to 82%, class ratio shifted → Model might be okay, but watch it

### Severity: LOW (Only image-level drift, predictions stable)

**Actions:**
1. **No immediate action**: Image features changed but model still performs well
2. **Continue monitoring**: Keep tracking prediction-level metrics
3. **Document changes**: Note what changed in deployment (new camera, lighting, etc.)

**Example:** Brightness/contrast drifted but confidence still 95% → Model is robust, no action needed

### Severity: NONE

**Actions:**
1. **Keep monitoring**: Continue scheduled drift checks
2. **Celebrate**: The model is performing consistently! 🎉

## Configuration

### Environment Variables
- `MODE`: Set to `"local"` or `"cloud"` (default: `"cloud"`)
- `GCS_MODELS_BUCKET`: GCS bucket for models (default: `"gs://drone-detection-mlops-models"`)

### DriftDetector Parameters
```python
from drone_detector_mlops.monitoring import DriftDetector

detector = DriftDetector(
    reference_dataset_path="data/reference_dataset.csv",  # Local reference dataset
    inference_bucket="drone-detection-mlops-data",        # GCS bucket with predictions
    inference_prefix="inference/prediction_"              # Prefix for prediction files
)
```

### Customizing Reference Dataset Generation
```bash
# Use validation split instead of training
python -m drone_detector_mlops.monitoring.generate_reference_dataset --split val

# Limit number of samples for faster generation
python -m drone_detector_mlops.monitoring.generate_reference_dataset --max-samples 500

# Custom paths
python -m drone_detector_mlops.monitoring.generate_reference_dataset \
  --data-dir /custom/path \
  --output-path /custom/reference.csv
```

## When to Regenerate Reference Dataset

You should regenerate the reference dataset when:
1. **After retraining** the model with new data
2. **Data collection changes** (new camera, different conditions become the norm)
3. **Drift becomes permanent** (production conditions are the new baseline)

## Troubleshooting

### "Reference dataset not found"
Run: `python -m drone_detector_mlops.monitoring.generate_reference_dataset`

### "No current data available"
Make predictions first via the API. The drift detector needs production data to compare against the reference.

### "Forbidden: The billing account is disabled"
Your GCP project billing is disabled. Either:
- Enable billing in GCP Console
- Set `MODE=local` in `.env` to use local model files

### Authentication Errors
Ensure you're authenticated: `gcloud auth login` and `gcloud auth application-default login`

## Files

- `feature_extraction.py` - Extract numerical features from images
- `drift_detection.py` - Evidently integration and drift analysis
- `generate_reference_dataset.py` - Create baseline from training data

## CI/CD Integration

### Automatic Reference Dataset Updates
After each training run, the reference dataset is automatically regenerated (`.github/workflows/train.yaml`). This ensures drift detection baseline stays aligned with the latest model.

### Scheduled Comprehensive Drift Monitoring
A GitHub Actions workflow (`.github/workflows/drift-monitoring.yaml`) runs daily to check for drift:
- **Multi-level analysis**: Prediction, image, and embedding drift
- **Smart alerting**: Only fails if severity is HIGH or MEDIUM
- **Actionable reports**: Provides specific recommended actions
- **Artifact generation**: Saves HTML reports for investigation
- **Manual trigger**: Can be run on-demand via GitHub Actions UI

**Workflow behavior:**
- ✅ **Severity: NONE/LOW** → Workflow passes, logs summary
- ⚠️ **Severity: MEDIUM** → Workflow fails, generates report, alerts team
- 🚨 **Severity: HIGH** → Workflow fails, urgent alert, recommends retraining

To disable scheduled checks, comment out the `schedule` section in the workflow file.

## Best Practices

### 1. Prioritize Prediction-Level Monitoring
- **Always check confidence scores first** before investigating image-level drift
- Image features may drift without affecting model performance
- Confidence drops are the most direct signal of model degradation

### 2. Correlate with W&B Metrics
- Drift detection tells you **data changed**, not that **model failed**
- Always cross-reference with W&B accuracy, loss, and F1 scores
- Example: Drift detected + accuracy dropped 10% → Retrain needed

### 3. Set Appropriate Thresholds
- **Confidence drop > 15%**: High severity, investigate immediately
- **Confidence drop 5-15%**: Medium severity, monitor closely
- **Confidence drop < 5%**: Low severity, likely normal variation

### 4. Don't Over-React to Image-Level Drift
- The ResNet18 model doesn't use brightness/colors directly
- Image drift alone is not a reason to retrain
- Only act if prediction-level metrics also show issues

### 5. Regenerate Reference Dataset After Major Changes
- After retraining with new data
- When deployment environment permanently changes (new camera, new location)
- When class distribution shifts permanently (more drones in production)

## FAQ

**Q: I see image-level drift but predictions look fine. Should I worry?**
A: No. The model learned robust features that aren't affected by brightness/color changes. Keep monitoring prediction-level metrics.

**Q: How often should I check for drift?**
A:
- Automated: Daily via GitHub Actions
- Manual: After any deployment environment changes
- Emergency: If users report accuracy issues

**Q: What's the minimum number of predictions needed for drift detection?**
A: At least 30-50 predictions for statistical validity, but 100+ is recommended.

**Q: Should I retrain every time drift is detected?**
A: No! Only retrain if:
- Prediction-level drift is HIGH (confidence drop > 15%)
- W&B metrics show actual accuracy degradation
- Manual review confirms model is struggling

**Q: Can I monitor embeddings on every prediction?**
A: Not recommended - it requires loading the PyTorch model which is slower. Use it for deep investigation when prediction metrics are ambiguous.

## Examples

See `examples/drift_monitoring_example.py` for complete usage examples:
- Basic drift monitoring workflow
- Continuous monitoring loop
- API endpoint usage
