from pydantic import BaseModel
from datetime import datetime


class PredictionScores(BaseModel):
    """Class probability scores."""

    drone: float
    bird: float


class Prediction(BaseModel):
    """Prediction result."""

    class_name: str  # "drone" or "bird"
    confidence: float
    scores: PredictionScores


class PredictionMetadata(BaseModel):
    """Metadata about the prediction."""

    model_version: str
    inference_time_ms: float
    timestamp: datetime


class PredictionResponse(BaseModel):
    """Complete prediction response."""

    prediction: Prediction
    metadata: PredictionMetadata


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool


class InfoResponse(BaseModel):
    """API info response."""

    model_version: str
    uptime_seconds: float


class DriftTestResponse(BaseModel):
    """Drift test results response."""

    all_passed: bool
    summary: dict
    timestamp: datetime


class DriftSummaryResponse(BaseModel):
    """Drift summary response."""

    timestamp: str
    reference_samples: int
    current_samples: int
    metrics: list


class ComprehensiveDriftResponse(BaseModel):
    """Comprehensive drift analysis response across all monitoring levels."""

    timestamp: str
    reference_samples: int
    current_samples: int
    drift_levels: dict  # Contains prediction, image_features, embeddings
    overall_assessment: dict  # severity, alerts, requires_action, recommended_actions
