# Drone Detector MLOps

<div class="hero" markdown>

**A production-ready drone detection framework with modern MLOps practices**

Detect drones in images using deep learning, powered by a fully automated ML pipeline with cloud training, model versioning, and scalable API deployment.

[:material-github: View on GitHub](https://github.com/dtu-mlops/drone-detection-mlops/){ .md-button .md-button--primary }
[:material-rocket-launch: Live API](https://drone-detector-api-66108710596.europe-north2.run.app/docs){ .md-button }
[:material-web: Try Frontend](https://drone-detector-frontend-66108710596.europe-north2.run.app/){ .md-button }

</div>

---

## :material-star-shooting: Feature Highlights

<div class="grid cards" markdown>

-   :material-cloud-upload:{ .lg .middle } **Cloud Training on Vertex AI**

    ---

    Train models at scale on Google Cloud with **NVIDIA T4 GPUs**. Automatic container builds and job submission with a single command.

    ```bash
    invoke cloud-train
    ```

-   :material-api:{ .lg .middle } **Serverless API Deployment**

    ---

    Deploy the trained models to **Cloud Run** with ONNX Runtime for optimized CPU inference. Auto-scaling and zero cold starts.

    ```bash
    invoke deploy-api
    ```

-   :material-database-sync:{ .lg .middle } **Data Versioning with DVC**

    ---

    Track and version the datasets with **DVC** backed by Google Cloud Storage. Reproducible experiments every time.

    ```bash
    uv run dvc pull
    ```

-   :material-github:{ .lg .middle } **GitHub Actions CI/CD**

    ---

    Automated pipelines for testing, linting, building, and deployment. Branch protection ensures quality code reaches production.

</div>

---

## :material-lightbulb-on: Why This Project?

!!! success "Production-Ready MLOps"

    This project demonstrates how to build a **complete ML system** from data to deployment, following industry best practices:

    - **Reproducibility**: Every experiment can be reproduced with versioned data, code, and configurations
    - **Automation**: From training to deployment, everything is automated via CI/CD
    - **Scalability**: Train on cloud GPUs, serve on auto-scaling infrastructure
    - **Observability**: Monitor model performance and detect data drift in production

<div class="grid" markdown>

!!! example "For ML Engineers"

    Learn how to structure ML projects for production, implement proper data versioning, and deploy models at scale.

!!! example "For Students"

    A comprehensive example of MLOps practices taught in university courses, with real-world cloud infrastructure.

</div>

---

## :material-chart-timeline-variant: Architecture Overview

![Architecture Diagram](images/diagram_architecture.png)

!!! info "Multi-Region Setup - A Pragmatic Choice"

    We use a multi-region setup across two GCP regions:

    - **Storage & Registry**: `europe-north2` (Finland) - Minimal latency for uploads
    - **Compute (Vertex AI)**: `europe-west4` (Netherlands) - NVIDIA T4 GPU availability

    This wasn't a planned architecture decision but a pragmatic response to GPU availability constraints. T4 GPUs were often unavailable in our primary region, so we deployed compute resources where capacity was available while keeping storage close to our team.

---

## :material-rocket-launch-outline: Quick Start

### Prerequisites

!!! note "Requirements"

    - Python 3.12+
    - [uv](https://docs.astral.sh/uv/) package manager
    - Google Cloud SDK (for cloud features)
    - Docker (for containerized deployment)

!!! warning "GCP Access Required"

    To use any cloud features (training, deployment, data access), you must be granted permissions to the GCP project first.

    New team members should:

    1. Add your email to `src/drone_detector_mlops/permissions/cloud_members.txt`
    2. Request a project admin to run the permission setup scripts
    3. Authenticate with `gcloud auth login`

    Without proper IAM roles, you won't be able to access GCS buckets, submit Vertex AI jobs, or deploy to Cloud Run.

### Installation

```bash
# Clone the repository
git clone https://github.com/dtu-mlops/drone-detection-mlops.git
cd drone_detector_mlops

# Install dependencies with uv
uv sync

# Set up pre-commit hooks
uv run pre-commit install

# Pull data with DVC
dvc pull
```

### Local Training

```bash
# Train a model locally
uv run train --epochs 10 --batch-size 32
```

### Cloud Training

```bash
# Build the training container
invoke cloud-build-train

# Submit a training job to Vertex AI
invoke cloud-train --epochs 50
```

### API Deployment

```bash
# Build the API container
uv run invoke cloud-build-api

# Deploy to Cloud Run
uv run invoke deploy-api
```

---

## :material-folder-open: Project Structure

```
drone-detector-mlops/
├── src/drone_detector_mlops/      # Main package
│   ├── data/                      # Dataloader & augmentations
│   ├── workflows/                 # Training & testing
│   ├── api/                       # FastAPI service
│   │   ├── main.py                # API endpoints & drift monitoring
│   │   ├── inference.py           # ONNX model inference
│   │   └── schemas.py             # Request/response models
│   ├── monitoring/                # Drift detection & feature extraction
│   ├── utils/                     # Storage, settings, and logging
│   ├── permissions/               # Access definitions for members
│   └── model.py                   # ResNet18 model
├── tests/                         # Unit & load tests
├── configs/                       # Hydra configurations
├── data/                          # Local dataset (drone/, bird/, splits/)
├── models/                        # Trained model (.pth)
├── cloud/                         # Cloud Build & deployment configs
├── dockerfiles/                   # Training, API, frontend containers
├── frontend/                      # SvelteKit web UI
├── scripts/                       # Executable scripts
├── reports/                       # Project report & figures
├── .github/workflows/             # CI/CD pipelines
├── docs/                          # Documentation (MkDocs)
├── tasks.py                       # Invoke automation tasks
└── pyproject.toml                 # Dependencies
```

---

## :material-link-variant: Quick Links

<div class="grid cards" markdown>

-   :material-book-open-page-variant:{ .lg .middle } **API Reference**

    ---

    Auto-generated documentation from docstrings.

    [:octicons-arrow-right-24: View API docs](api-reference.md)

-   :material-cloud:{ .lg .middle } **Cloud Training**

    ---

    Train models on Vertex AI with GPU acceleration.

    [:octicons-arrow-right-24: Training guide](training.md)

-   :material-rocket-launch:{ .lg .middle } **Deployment**

    ---

    Deploy your API to Cloud Run.

    [:octicons-arrow-right-24: Deployment guide](deployment.md)

-   :material-chart-line:{ .lg .middle } **Monitoring**

    ---

    Monitor model performance and detect drift.

    [:octicons-arrow-right-24: Monitoring guide](monitoring.md)

</div>

---

## :material-scale-balance: License

This project is developed as part of the [DTU MLOps](https://skaftenicki.github.io/dtu_mlops/) course.

---

<div class="footer-note" markdown>

Built with :material-heart: using [MkDocs Material](https://squidfunk.github.io/mkdocs-material/)

</div>
