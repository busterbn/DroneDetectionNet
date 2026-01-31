from datetime import datetime, timezone

from drone_detector_mlops.api.schemas import (
    DriftTestResponse,
    DriftSummaryResponse,
    ComprehensiveDriftResponse,
)


class TestDriftTestResponse:
    """Tests for DriftTestResponse schema."""

    def test_valid_drift_test_response(self):
        """Test creating valid drift test response."""
        timestamp = datetime.now(timezone.utc)
        response = DriftTestResponse(
            all_passed=True,
            summary={"total_tests": 5, "passed": 5, "failed": 0},
            timestamp=timestamp,
        )
        assert response.all_passed is True
        assert response.summary["total_tests"] == 5
        assert response.timestamp == timestamp

    def test_failed_drift_tests(self):
        """Test drift test response with failures."""
        timestamp = datetime.now(timezone.utc)
        response = DriftTestResponse(
            all_passed=False,
            summary={"total_tests": 5, "passed": 3, "failed": 2},
            timestamp=timestamp,
        )
        assert response.all_passed is False
        assert response.summary["failed"] == 2


class TestDriftSummaryResponse:
    """Tests for DriftSummaryResponse schema."""

    def test_valid_drift_summary(self):
        """Test creating valid drift summary response."""
        response = DriftSummaryResponse(
            timestamp="2026-01-23T10:30:00Z",
            reference_samples=1000,
            current_samples=100,
            metrics=["confidence", "brightness", "contrast"],
        )
        assert response.timestamp == "2026-01-23T10:30:00Z"
        assert response.reference_samples == 1000
        assert response.current_samples == 100
        assert len(response.metrics) == 3


class TestComprehensiveDriftResponse:
    """Tests for ComprehensiveDriftResponse schema."""

    def test_valid_comprehensive_drift_response(self):
        """Test creating valid comprehensive drift response."""
        response = ComprehensiveDriftResponse(
            timestamp="2026-01-23T10:30:00Z",
            reference_samples=1000,
            current_samples=100,
            drift_levels={
                "prediction": {"drift_detected": False},
                "image_features": {"drift_detected": True},
                "embeddings": {"drift_detected": False},
            },
            overall_assessment={
                "severity": "low",
                "alerts": ["Image brightness has drifted"],
                "requires_action": False,
                "recommended_actions": [],
            },
        )
        assert response.timestamp == "2026-01-23T10:30:00Z"
        assert response.reference_samples == 1000
        assert response.drift_levels["prediction"]["drift_detected"] is False
        assert response.overall_assessment["severity"] == "low"

    def test_high_severity_drift(self):
        """Test comprehensive drift response with high severity."""
        response = ComprehensiveDriftResponse(
            timestamp="2026-01-23T10:30:00Z",
            reference_samples=1000,
            current_samples=100,
            drift_levels={
                "prediction": {"drift_detected": True},
                "image_features": {"drift_detected": True},
                "embeddings": {"drift_detected": True},
            },
            overall_assessment={
                "severity": "high",
                "alerts": ["Multiple drift indicators detected"],
                "requires_action": True,
                "recommended_actions": ["Retrain model", "Investigate data quality"],
            },
        )
        assert response.overall_assessment["severity"] == "high"
        assert response.overall_assessment["requires_action"] is True
        assert len(response.overall_assessment["recommended_actions"]) == 2
