"""Comprehensive drift monitoring for drone detection model.

This package provides multi-level drift detection:
- Image-level: Brightness, contrast, colors (proxy metrics)
- Prediction-level: Confidence scores, class distribution (model outputs)
- Embedding-level: Model's learned representations (what the model sees)
"""

from drone_detector_mlops.monitoring.drift_detection import DriftDetector
from drone_detector_mlops.monitoring.feature_extraction import ImageFeatureExtractor
from drone_detector_mlops.monitoring.prediction_drift import PredictionDriftMonitor
from drone_detector_mlops.monitoring.embedding_drift import (
    EmbeddingExtractor,
    EmbeddingDriftMonitor,
)

__all__ = [
    "DriftDetector",
    "ImageFeatureExtractor",
    "PredictionDriftMonitor",
    "EmbeddingExtractor",
    "EmbeddingDriftMonitor",
]
