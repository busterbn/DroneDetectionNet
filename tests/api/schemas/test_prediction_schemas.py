from datetime import datetime, timezone
import pytest
from pydantic import ValidationError

from drone_detector_mlops.api.schemas import (
    PredictionScores,
    Prediction,
    PredictionMetadata,
    PredictionResponse,
)


class TestPredictionScores:
    """Tests for PredictionScores schema."""

    def test_valid_scores(self):
        """Test creating valid prediction scores."""
        scores = PredictionScores(drone=0.75, bird=0.25)
        assert scores.drone == 0.75
        assert scores.bird == 0.25

    def test_scores_as_floats(self):
        """Test that scores are properly converted to floats."""
        scores = PredictionScores(drone=1, bird=0)
        assert isinstance(scores.drone, float)
        assert isinstance(scores.bird, float)

    def test_missing_fields_raises_error(self):
        """Test that missing required fields raise validation error."""
        with pytest.raises(ValidationError):
            PredictionScores(drone=0.5)


class TestPrediction:
    """Tests for Prediction schema."""

    def test_valid_prediction(self):
        """Test creating valid prediction."""
        prediction = Prediction(
            class_name="drone",
            confidence=0.95,
            scores=PredictionScores(drone=0.95, bird=0.05),
        )
        assert prediction.class_name == "drone"
        assert prediction.confidence == 0.95
        assert prediction.scores.drone == 0.95
        assert prediction.scores.bird == 0.05

    def test_prediction_with_bird_class(self):
        """Test prediction for bird class."""
        prediction = Prediction(
            class_name="bird",
            confidence=0.88,
            scores=PredictionScores(drone=0.12, bird=0.88),
        )
        assert prediction.class_name == "bird"
        assert prediction.confidence == 0.88

    def test_missing_fields_raises_error(self):
        """Test that missing required fields raise validation error."""
        with pytest.raises(ValidationError):
            Prediction(class_name="drone", confidence=0.95)


class TestPredictionMetadata:
    """Tests for PredictionMetadata schema."""

    def test_valid_metadata(self):
        """Test creating valid prediction metadata."""
        timestamp = datetime.now(timezone.utc)
        metadata = PredictionMetadata(
            model_version="model_v1.onnx",
            inference_time_ms=45.5,
            timestamp=timestamp,
        )
        assert metadata.model_version == "model_v1.onnx"
        assert metadata.inference_time_ms == 45.5
        assert metadata.timestamp == timestamp

    def test_inference_time_as_float(self):
        """Test that inference time is properly converted to float."""
        metadata = PredictionMetadata(
            model_version="model_v1.onnx",
            inference_time_ms=50,
            timestamp=datetime.now(timezone.utc),
        )
        assert isinstance(metadata.inference_time_ms, float)


class TestPredictionResponse:
    """Tests for PredictionResponse schema."""

    def test_valid_prediction_response(self):
        """Test creating valid prediction response."""
        timestamp = datetime.now(timezone.utc)
        response = PredictionResponse(
            prediction=Prediction(
                class_name="drone",
                confidence=0.95,
                scores=PredictionScores(drone=0.95, bird=0.05),
            ),
            metadata=PredictionMetadata(
                model_version="model_v1.onnx",
                inference_time_ms=45.5,
                timestamp=timestamp,
            ),
        )
        assert response.prediction.class_name == "drone"
        assert response.metadata.model_version == "model_v1.onnx"

    def test_prediction_response_serialization(self):
        """Test that prediction response can be serialized to dict."""
        timestamp = datetime.now(timezone.utc)
        response = PredictionResponse(
            prediction=Prediction(
                class_name="bird",
                confidence=0.88,
                scores=PredictionScores(drone=0.12, bird=0.88),
            ),
            metadata=PredictionMetadata(
                model_version="model_v1.onnx",
                inference_time_ms=45.5,
                timestamp=timestamp,
            ),
        )
        data = response.model_dump()
        assert data["prediction"]["class_name"] == "bird"
        assert data["metadata"]["model_version"] == "model_v1.onnx"
