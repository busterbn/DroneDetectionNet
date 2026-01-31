"""Tests for FastAPI main endpoints."""

import io
import json
from datetime import datetime, timezone
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from drone_detector_mlops.api.main import app
from drone_detector_mlops.api.schemas import Prediction, PredictionScores, PredictionMetadata


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def sample_image_bytes():
    """Create sample image bytes."""
    img = Image.new("RGB", (224, 224), color=(100, 150, 200))
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes.seek(0)
    return img_bytes.getvalue()


@pytest.fixture
def mock_prediction():
    """Create a mock prediction."""
    return Prediction(
        class_name="drone",
        confidence=0.95,
        scores=PredictionScores(drone=0.95, bird=0.05),
    )


@pytest.fixture
def mock_metadata():
    """Create mock prediction metadata."""
    return PredictionMetadata(
        inference_time_ms=25.5,
        model_version="test-model",
        timestamp=datetime.now(timezone.utc),
    )


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_check_returns_200(self, client):
        """Test that health check returns 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_check_returns_json(self, client):
        """Test that health check returns JSON."""
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data

    def test_health_check_status_healthy(self, client):
        """Test that status is healthy."""
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"

    def test_health_check_model_loaded_is_boolean(self, client):
        """Test that model_loaded is a boolean."""
        response = client.get("/health")
        data = response.json()
        assert isinstance(data["model_loaded"], bool)


class TestInfoEndpoint:
    """Tests for /v1/info endpoint."""

    @patch("drone_detector_mlops.api.main.startup_time", datetime.now(timezone.utc))
    def test_info_returns_200(self, client):
        """Test that info endpoint returns 200."""
        response = client.get("/v1/info")
        assert response.status_code == 200

    @patch("drone_detector_mlops.api.main.startup_time", datetime.now(timezone.utc))
    def test_info_returns_json(self, client):
        """Test that info returns JSON with required fields."""
        response = client.get("/v1/info")
        data = response.json()
        assert "model_version" in data
        assert "uptime_seconds" in data

    @patch("drone_detector_mlops.api.main.startup_time", datetime.now(timezone.utc))
    def test_info_uptime_is_positive(self, client):
        """Test that uptime is positive."""
        response = client.get("/v1/info")
        data = response.json()
        assert data["uptime_seconds"] >= 0


class TestPredictEndpoint:
    """Tests for /v1/predict endpoint."""

    @patch("drone_detector_mlops.api.main.predict_image")
    @patch("drone_detector_mlops.api.main.save_prediction_to_gcp")
    def test_predict_returns_200(
        self, mock_save_gcp, mock_predict, client, sample_image_bytes, mock_prediction, mock_metadata
    ):
        """Test that predict endpoint returns 200."""
        mock_predict.return_value = (mock_prediction, mock_metadata)

        response = client.post("/v1/predict", files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")})

        assert response.status_code == 200

    @patch("drone_detector_mlops.api.main.predict_image")
    @patch("drone_detector_mlops.api.main.save_prediction_to_gcp")
    def test_predict_returns_prediction_response(
        self, mock_save_gcp, mock_predict, client, sample_image_bytes, mock_prediction, mock_metadata
    ):
        """Test that predict returns PredictionResponse."""
        mock_predict.return_value = (mock_prediction, mock_metadata)

        response = client.post("/v1/predict", files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")})

        data = response.json()
        assert "prediction" in data
        assert "metadata" in data
        assert data["prediction"]["class_name"] == "drone"
        assert data["prediction"]["confidence"] == 0.95

    @patch("drone_detector_mlops.api.main.predict_image")
    @patch("drone_detector_mlops.api.main.save_prediction_to_gcp")
    def test_predict_includes_scores(
        self, mock_save_gcp, mock_predict, client, sample_image_bytes, mock_prediction, mock_metadata
    ):
        """Test that prediction includes scores."""
        mock_predict.return_value = (mock_prediction, mock_metadata)

        response = client.post("/v1/predict", files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")})

        data = response.json()
        assert "scores" in data["prediction"]
        assert "drone" in data["prediction"]["scores"]
        assert "bird" in data["prediction"]["scores"]

    def test_predict_rejects_non_image(self, client):
        """Test that non-image files are rejected."""
        text_file = io.BytesIO(b"This is not an image")

        response = client.post("/v1/predict", files={"file": ("test.txt", text_file, "text/plain")})

        assert response.status_code == 400
        assert "must be an image" in response.json()["detail"].lower()

    @patch("drone_detector_mlops.api.main.settings.MAX_UPLOAD_SIZE_MB", 0.001)  # 1KB limit
    def test_predict_rejects_large_files(self, client):
        """Test that files over size limit are rejected."""
        # Create a 2KB image
        img = Image.new("RGB", (1000, 1000))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="JPEG")
        img_bytes.seek(0)

        response = client.post("/v1/predict", files={"file": ("large.jpg", img_bytes, "image/jpeg")})

        assert response.status_code == 413
        assert "too large" in response.json()["detail"].lower()

    def test_predict_rejects_invalid_image(self, client):
        """Test that invalid image data is rejected."""
        invalid_bytes = io.BytesIO(b"not an image")

        response = client.post("/v1/predict", files={"file": ("bad.jpg", invalid_bytes, "image/jpeg")})

        assert response.status_code == 400
        assert "invalid image" in response.json()["detail"].lower()

    @patch("drone_detector_mlops.api.main.predict_image")
    @patch("drone_detector_mlops.api.main.save_prediction_to_gcp")
    def test_predict_handles_rgba_images(self, mock_save_gcp, mock_predict, client, mock_prediction, mock_metadata):
        """Test that RGBA images are converted to RGB."""
        mock_predict.return_value = (mock_prediction, mock_metadata)

        # Create RGBA image
        img = Image.new("RGBA", (224, 224), color=(100, 150, 200, 255))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        response = client.post("/v1/predict", files={"file": ("test.png", img_bytes, "image/png")})

        assert response.status_code == 200

    @patch("drone_detector_mlops.api.main.predict_image")
    def test_predict_handles_prediction_error(self, mock_predict, client, sample_image_bytes):
        """Test that prediction errors are handled gracefully."""
        mock_predict.side_effect = Exception("Model error")

        response = client.post("/v1/predict", files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")})

        assert response.status_code == 500
        assert "prediction failed" in response.json()["detail"].lower()


class TestSavePredictionToGCP:
    """Tests for save_prediction_to_gcp function."""

    @patch("drone_detector_mlops.api.main.storage.Client")
    def test_saves_to_gcs(self, mock_storage_client, sample_image_bytes):
        """Test that prediction is saved to GCS."""
        from drone_detector_mlops.api.main import save_prediction_to_gcp

        mock_bucket = Mock()
        mock_blob = Mock()
        mock_bucket.blob.return_value = mock_blob
        mock_storage_client.return_value.bucket.return_value = mock_bucket

        image = Image.new("RGB", (224, 224))
        scores = PredictionScores(drone=0.9, bird=0.1)
        metadata = {"inference_time_ms": 25.0, "model_version": "v1"}

        save_prediction_to_gcp("drone", 0.9, scores, metadata, sample_image_bytes, image)

        mock_blob.upload_from_string.assert_called_once()
        # Check that the uploaded data is valid JSON
        uploaded_data = mock_blob.upload_from_string.call_args[0][0]
        parsed_data = json.loads(uploaded_data)
        assert parsed_data["class_name"] == "drone"
        assert parsed_data["confidence"] == 0.9
        assert "image_base64" in parsed_data
        assert "features" in parsed_data


class TestDriftReportEndpoint:
    """Tests for /v1/monitoring/drift-report endpoint."""

    @patch("drone_detector_mlops.api.main.DriftDetector")
    def test_drift_report_returns_html(self, mock_detector_class, client, tmp_path):
        """Test that drift report returns HTML."""
        mock_detector = Mock()
        mock_detector_class.return_value = mock_detector

        # Mock the report file
        report_content = "<html><body>Test Report</body></html>"

        def mock_generate_report(*args, **kwargs):
            output_path = kwargs.get("output_path", "/tmp/drift_report.html")
            with open(output_path, "w") as f:
                f.write(report_content)

        mock_detector.generate_drift_report.side_effect = mock_generate_report

        response = client.get("/v1/monitoring/drift-report")

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Test Report" in response.text

    @patch("drone_detector_mlops.api.main.DriftDetector")
    def test_drift_report_respects_max_predictions(self, mock_detector_class, client):
        """Test that max_predictions parameter is passed."""
        mock_detector = Mock()
        mock_detector_class.return_value = mock_detector

        def mock_generate_report(*args, **kwargs):
            output_path = kwargs.get("output_path", "/tmp/drift_report.html")
            with open(output_path, "w") as f:
                f.write("<html>Report</html>")

        mock_detector.generate_drift_report.side_effect = mock_generate_report

        client.get("/v1/monitoring/drift-report?max_predictions=50")

        mock_detector.generate_drift_report.assert_called_once()
        call_kwargs = mock_detector.generate_drift_report.call_args[1]
        assert call_kwargs["max_predictions"] == 50

    @patch("drone_detector_mlops.api.main.DriftDetector")
    def test_drift_report_handles_errors(self, mock_detector_class, client):
        """Test that drift report errors are handled."""
        mock_detector = Mock()
        mock_detector_class.return_value = mock_detector
        mock_detector.generate_drift_report.side_effect = Exception("Report generation failed")

        response = client.get("/v1/monitoring/drift-report")

        assert response.status_code == 500
        assert "failed" in response.json()["detail"].lower()


class TestDriftTestsEndpoint:
    """Tests for /v1/monitoring/drift-tests endpoint."""

    @patch("drone_detector_mlops.api.main.DriftDetector")
    def test_drift_tests_returns_200(self, mock_detector_class, client):
        """Test that drift tests returns 200."""
        mock_detector = Mock()
        mock_detector_class.return_value = mock_detector
        mock_detector.run_drift_tests.return_value = {
            "all_passed": True,
            "summary": {},
        }

        response = client.get("/v1/monitoring/drift-tests")

        assert response.status_code == 200

    @patch("drone_detector_mlops.api.main.DriftDetector")
    def test_drift_tests_returns_test_results(self, mock_detector_class, client):
        """Test that drift tests returns results."""
        mock_detector = Mock()
        mock_detector_class.return_value = mock_detector
        mock_detector.run_drift_tests.return_value = {
            "all_passed": False,
            "summary": {"total_tests": 10, "failed_tests": 2},
        }

        response = client.get("/v1/monitoring/drift-tests")

        data = response.json()
        assert "all_passed" in data
        assert data["all_passed"] is False
        assert "summary" in data

    @patch("drone_detector_mlops.api.main.DriftDetector")
    def test_drift_tests_handles_errors(self, mock_detector_class, client):
        """Test that drift tests handles errors."""
        mock_detector = Mock()
        mock_detector_class.return_value = mock_detector
        mock_detector.run_drift_tests.side_effect = Exception("Tests failed")

        response = client.get("/v1/monitoring/drift-tests")

        assert response.status_code == 500


class TestDriftSummaryEndpoint:
    """Tests for /v1/monitoring/drift-summary endpoint."""

    @patch("drone_detector_mlops.api.main.DriftDetector")
    def test_drift_summary_returns_200(self, mock_detector_class, client):
        """Test that drift summary returns 200."""
        mock_detector = Mock()
        mock_detector_class.return_value = mock_detector
        mock_detector.get_drift_summary.return_value = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "reference_samples": 100,
            "current_samples": 50,
            "metrics": [],
        }

        response = client.get("/v1/monitoring/drift-summary")

        assert response.status_code == 200

    @patch("drone_detector_mlops.api.main.DriftDetector")
    def test_drift_summary_returns_summary_data(self, mock_detector_class, client):
        """Test that drift summary returns summary data."""
        mock_detector = Mock()
        mock_detector_class.return_value = mock_detector
        mock_detector.get_drift_summary.return_value = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "reference_samples": 100,
            "current_samples": 50,
            "metrics": [{"name": "test_metric", "value": 0.5}],
        }

        response = client.get("/v1/monitoring/drift-summary")

        data = response.json()
        assert "reference_samples" in data
        assert data["reference_samples"] == 100
        assert "current_samples" in data
        assert data["current_samples"] == 50


class TestComprehensiveDriftEndpoint:
    """Tests for /v1/monitoring/comprehensive-drift endpoint."""

    @patch("drone_detector_mlops.api.main.DriftDetector")
    def test_comprehensive_drift_returns_200(self, mock_detector_class, client):
        """Test that comprehensive drift returns 200."""
        mock_detector = Mock()
        mock_detector_class.return_value = mock_detector
        mock_detector.run_comprehensive_drift_analysis.return_value = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "reference_samples": 100,
            "current_samples": 50,
            "drift_levels": {},
            "overall_assessment": {
                "severity": "none",
                "alerts": [],
                "requires_action": False,
                "recommended_actions": [],
            },
        }

        response = client.get("/v1/monitoring/comprehensive-drift")

        assert response.status_code == 200

    @patch("drone_detector_mlops.api.main.DriftDetector")
    def test_comprehensive_drift_returns_analysis(self, mock_detector_class, client):
        """Test that comprehensive drift returns full analysis."""
        mock_detector = Mock()
        mock_detector_class.return_value = mock_detector
        mock_detector.run_comprehensive_drift_analysis.return_value = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "reference_samples": 100,
            "current_samples": 50,
            "drift_levels": {
                "prediction": {"available": True, "alerts": []},
                "image_features": {"available": True, "all_passed": True},
                "embeddings": {"available": False},
            },
            "overall_assessment": {
                "severity": "low",
                "alerts": ["Some drift detected"],
                "requires_action": False,
                "recommended_actions": [],
            },
        }

        response = client.get("/v1/monitoring/comprehensive-drift")

        data = response.json()
        assert "drift_levels" in data
        assert "overall_assessment" in data
        assert data["overall_assessment"]["severity"] == "low"

    @patch("drone_detector_mlops.api.main.DriftDetector")
    def test_comprehensive_drift_handles_errors(self, mock_detector_class, client):
        """Test that comprehensive drift handles errors."""
        mock_detector = Mock()
        mock_detector_class.return_value = mock_detector
        mock_detector.run_comprehensive_drift_analysis.side_effect = Exception("Analysis failed")

        response = client.get("/v1/monitoring/comprehensive-drift")

        assert response.status_code == 500


class TestMetricsEndpoint:
    """Tests for /metrics endpoint (Prometheus)."""

    def test_metrics_endpoint_exists(self, client):
        """Test that /metrics endpoint exists."""
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_returns_prometheus_format(self, client):
        """Test that metrics are in Prometheus format."""
        response = client.get("/metrics")
        # Prometheus metrics should be plain text
        assert "text/plain" in response.headers.get("content-type", "")
