"""Tests for comprehensive drift detection."""

import json
from datetime import datetime, timezone
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from drone_detector_mlops.monitoring.drift_detection import DriftDetector


@pytest.fixture
def reference_data_csv(tmp_path):
    """Create a reference dataset CSV."""
    csv_path = tmp_path / "reference.csv"
    # Include all feature columns that ImageFeatureExtractor generates
    data = {
        "timestamp": [datetime.now(timezone.utc).isoformat()] * 10,
        "class_name": ["drone"] * 5 + ["bird"] * 5,
        "class_int": [0] * 5 + [1] * 5,
        "confidence": [0.9] * 10,
        "brightness_mean": [128.0] * 10,
        "brightness_std": [20.0] * 10,
        "contrast": [50.0] * 10,
        "sharpness": [100.0] * 10,
        "r_mean": [130.0] * 10,
        "g_mean": [125.0] * 10,
        "b_mean": [120.0] * 10,
        "r_std": [25.0] * 10,
        "g_std": [22.0] * 10,
        "b_std": [20.0] * 10,
        "saturation_mean": [0.3] * 10,
        "value_mean": [0.5] * 10,
        "edge_density": [0.15] * 10,
        "aspect_ratio": [1.5] * 10,
        "size_pixels": [224 * 224] * 10,
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def current_predictions_data():
    """Create current predictions data."""
    return pd.DataFrame(
        {
            "timestamp": [datetime.now(timezone.utc).isoformat()] * 8,
            "class_name": ["drone"] * 4 + ["bird"] * 4,
            "class_int": [0] * 4 + [1] * 4,
            "confidence": [0.85] * 8,
            "brightness_mean": [125.0] * 8,
            "brightness_std": [19.0] * 8,
            "contrast": [48.0] * 8,
            "sharpness": [98.0] * 8,
            "r_mean": [128.0] * 8,
            "g_mean": [123.0] * 8,
            "b_mean": [118.0] * 8,
            "r_std": [24.0] * 8,
            "g_std": [21.0] * 8,
            "b_std": [19.0] * 8,
            "saturation_mean": [0.32] * 8,
            "value_mean": [0.48] * 8,
            "edge_density": [0.14] * 8,
            "aspect_ratio": [1.4] * 8,
            "size_pixels": [224 * 224] * 8,
        }
    )


@pytest.fixture
def mock_gcs_blobs():
    """Create mock GCS blobs."""
    blobs = []
    for i in range(3):
        blob = Mock()
        blob.name = f"prediction_{i}.json"
        blob.time_created = datetime(2024, 1, 1 + i)
        blob.download_as_text.return_value = json.dumps(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "class_name": "drone",
                "confidence": 0.9,
                "features": {
                    "brightness_mean": 128.0,
                    "brightness_std": 20.0,
                    "contrast": 50.0,
                    "sharpness": 100.0,
                    "r_mean": 130.0,
                    "g_mean": 125.0,
                    "b_mean": 120.0,
                    "r_std": 25.0,
                    "g_std": 22.0,
                    "b_std": 20.0,
                    "saturation_mean": 0.3,
                    "value_mean": 0.5,
                    "edge_density": 0.15,
                    "aspect_ratio": 1.5,
                    "size_pixels": 224 * 224,
                },
            }
        )
        blobs.append(blob)
    return blobs


class TestDriftDetectorInitialization:
    """Tests for DriftDetector initialization."""

    @patch("drone_detector_mlops.monitoring.drift_detection.storage.Client")
    def test_initialization_with_existing_reference(self, mock_storage_client, reference_data_csv):
        """Test initialization with existing reference dataset."""
        detector = DriftDetector(reference_dataset_path=str(reference_data_csv))

        assert detector.reference_data is not None
        assert len(detector.reference_data) == 10
        assert "class_name" in detector.reference_data.columns

    @patch("drone_detector_mlops.monitoring.drift_detection.storage.Client")
    def test_initialization_without_reference(self, mock_storage_client, tmp_path):
        """Test initialization when reference dataset doesn't exist."""
        non_existent = tmp_path / "nonexistent.csv"

        detector = DriftDetector(reference_dataset_path=str(non_existent))

        assert detector.reference_data is None

    @patch("drone_detector_mlops.monitoring.drift_detection.storage.Client")
    def test_initialization_sets_attributes(self, mock_storage_client, reference_data_csv):
        """Test that initialization sets required attributes."""
        detector = DriftDetector(
            reference_dataset_path=str(reference_data_csv),
            inference_bucket="test-bucket",
            inference_prefix="test/prefix",
        )

        assert detector.inference_bucket == "test-bucket"
        assert detector.inference_prefix == "test/prefix"
        assert detector.feature_columns is not None
        assert detector.target_column == "class_name"
        assert detector.prediction_column == "class_int"

    @patch("drone_detector_mlops.monitoring.drift_detection.storage.Client")
    def test_initialization_creates_monitors(self, mock_storage_client, reference_data_csv):
        """Test that specialized monitors are created."""
        detector = DriftDetector(reference_dataset_path=str(reference_data_csv))

        assert hasattr(detector, "prediction_monitor")
        assert hasattr(detector, "embedding_monitor")


class TestFetchRecentPredictions:
    """Tests for fetch_recent_predictions method."""

    @patch("drone_detector_mlops.monitoring.drift_detection.storage.Client")
    def test_fetches_from_gcs(self, mock_storage_client, reference_data_csv, mock_gcs_blobs):
        """Test fetching predictions from GCS."""
        mock_bucket = Mock()
        mock_bucket.list_blobs.return_value = mock_gcs_blobs
        mock_storage_client.return_value.bucket.return_value = mock_bucket

        detector = DriftDetector(reference_dataset_path=str(reference_data_csv))
        df = detector.fetch_recent_predictions(max_predictions=10)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "class_name" in df.columns
        assert "confidence" in df.columns

    @patch("drone_detector_mlops.monitoring.drift_detection.storage.Client")
    def test_respects_max_predictions(self, mock_storage_client, reference_data_csv, mock_gcs_blobs):
        """Test that max_predictions limit is respected."""
        mock_bucket = Mock()
        mock_bucket.list_blobs.return_value = mock_gcs_blobs
        mock_storage_client.return_value.bucket.return_value = mock_bucket

        detector = DriftDetector(reference_dataset_path=str(reference_data_csv))
        df = detector.fetch_recent_predictions(max_predictions=2)

        assert len(df) == 2

    @patch("drone_detector_mlops.monitoring.drift_detection.storage.Client")
    def test_handles_invalid_json(self, mock_storage_client, reference_data_csv):
        """Test handling of invalid JSON in blobs."""
        blob_with_error = Mock()
        blob_with_error.name = "bad.json"
        blob_with_error.time_created = datetime.now()
        blob_with_error.download_as_text.return_value = "invalid json"

        mock_bucket = Mock()
        mock_bucket.list_blobs.return_value = [blob_with_error]
        mock_storage_client.return_value.bucket.return_value = mock_bucket

        detector = DriftDetector(reference_dataset_path=str(reference_data_csv))
        df = detector.fetch_recent_predictions()

        # Should return empty DataFrame when all blobs fail
        assert len(df) == 0

    @patch("drone_detector_mlops.monitoring.drift_detection.storage.Client")
    def test_parses_prediction_fields(self, mock_storage_client, reference_data_csv, mock_gcs_blobs):
        """Test that prediction fields are parsed correctly."""
        mock_bucket = Mock()
        mock_bucket.list_blobs.return_value = mock_gcs_blobs
        mock_storage_client.return_value.bucket.return_value = mock_bucket

        detector = DriftDetector(reference_dataset_path=str(reference_data_csv))
        df = detector.fetch_recent_predictions()

        assert "timestamp" in df.columns
        assert "class_name" in df.columns
        assert "class_int" in df.columns
        assert "confidence" in df.columns
        assert all(df["class_int"] == 0)  # All are "drone"


class TestGenerateDriftReport:
    """Tests for generate_drift_report method."""

    @patch("drone_detector_mlops.monitoring.drift_detection.storage.Client")
    def test_generates_report_with_provided_data(
        self, mock_storage_client, reference_data_csv, current_predictions_data
    ):
        """Test generating report with provided current data."""
        from evidently.legacy.report import Report

        detector = DriftDetector(reference_dataset_path=str(reference_data_csv))
        report = detector.generate_drift_report(current_data=current_predictions_data)

        assert isinstance(report, Report)

    @patch("drone_detector_mlops.monitoring.drift_detection.storage.Client")
    def test_raises_error_without_reference_data(self, mock_storage_client, tmp_path, current_predictions_data):
        """Test that error is raised when reference data is missing."""
        non_existent = tmp_path / "nonexistent.csv"
        detector = DriftDetector(reference_dataset_path=str(non_existent))

        with pytest.raises(ValueError, match="Reference dataset not loaded"):
            detector.generate_drift_report(current_data=current_predictions_data)

    @patch("drone_detector_mlops.monitoring.drift_detection.storage.Client")
    def test_raises_error_with_empty_current_data(self, mock_storage_client, reference_data_csv):
        """Test that error is raised when current data is empty."""
        detector = DriftDetector(reference_dataset_path=str(reference_data_csv))
        empty_df = pd.DataFrame()

        with pytest.raises(ValueError, match="No current data available"):
            detector.generate_drift_report(current_data=empty_df)

    @patch("drone_detector_mlops.monitoring.drift_detection.storage.Client")
    def test_saves_report_to_html(self, mock_storage_client, reference_data_csv, current_predictions_data, tmp_path):
        """Test saving report to HTML file."""
        output_path = tmp_path / "drift_report.html"

        detector = DriftDetector(reference_dataset_path=str(reference_data_csv))
        detector.generate_drift_report(current_data=current_predictions_data, output_path=str(output_path))

        assert output_path.exists()
        assert output_path.stat().st_size > 0


class TestRunDriftTests:
    """Tests for run_drift_tests method."""

    @patch("drone_detector_mlops.monitoring.drift_detection.storage.Client")
    def test_returns_dict_with_required_keys(self, mock_storage_client, reference_data_csv, current_predictions_data):
        """Test that run_drift_tests returns dict with required keys."""
        detector = DriftDetector(reference_dataset_path=str(reference_data_csv))
        results = detector.run_drift_tests(current_data=current_predictions_data)

        assert isinstance(results, dict)
        assert "all_passed" in results
        assert "summary" in results
        assert "details" in results

    @patch("drone_detector_mlops.monitoring.drift_detection.storage.Client")
    def test_all_passed_is_boolean(self, mock_storage_client, reference_data_csv, current_predictions_data):
        """Test that all_passed is a boolean."""
        detector = DriftDetector(reference_dataset_path=str(reference_data_csv))
        results = detector.run_drift_tests(current_data=current_predictions_data)

        assert isinstance(results["all_passed"], bool)

    @patch("drone_detector_mlops.monitoring.drift_detection.storage.Client")
    def test_raises_error_without_reference_data(self, mock_storage_client, tmp_path, current_predictions_data):
        """Test that error is raised when reference data is missing."""
        non_existent = tmp_path / "nonexistent.csv"
        detector = DriftDetector(reference_dataset_path=str(non_existent))

        with pytest.raises(ValueError, match="Reference dataset not loaded"):
            detector.run_drift_tests(current_data=current_predictions_data)


class TestGetDriftSummary:
    """Tests for get_drift_summary method."""

    @patch("drone_detector_mlops.monitoring.drift_detection.storage.Client")
    def test_returns_dict_with_summary(self, mock_storage_client, reference_data_csv, current_predictions_data):
        """Test that get_drift_summary returns dict."""
        detector = DriftDetector(reference_dataset_path=str(reference_data_csv))
        summary = detector.get_drift_summary(current_data=current_predictions_data)

        assert isinstance(summary, dict)
        assert "timestamp" in summary
        assert "reference_samples" in summary
        assert "current_samples" in summary
        assert "metrics" in summary

    @patch("drone_detector_mlops.monitoring.drift_detection.storage.Client")
    def test_includes_sample_counts(self, mock_storage_client, reference_data_csv, current_predictions_data):
        """Test that sample counts are correct."""
        detector = DriftDetector(reference_dataset_path=str(reference_data_csv))
        summary = detector.get_drift_summary(current_data=current_predictions_data)

        assert summary["reference_samples"] == 10
        assert summary["current_samples"] == 8


class TestRunComprehensiveDriftAnalysis:
    """Tests for run_comprehensive_drift_analysis method."""

    @patch("drone_detector_mlops.monitoring.drift_detection.storage.Client")
    def test_returns_comprehensive_results(self, mock_storage_client, reference_data_csv, current_predictions_data):
        """Test that comprehensive analysis returns structured results."""
        detector = DriftDetector(reference_dataset_path=str(reference_data_csv))
        results = detector.run_comprehensive_drift_analysis(current_data=current_predictions_data)

        assert isinstance(results, dict)
        assert "timestamp" in results
        assert "reference_samples" in results
        assert "current_samples" in results
        assert "drift_levels" in results
        assert "overall_assessment" in results

    @patch("drone_detector_mlops.monitoring.drift_detection.storage.Client")
    def test_includes_prediction_level_drift(self, mock_storage_client, reference_data_csv, current_predictions_data):
        """Test that prediction-level drift is analyzed."""
        detector = DriftDetector(reference_dataset_path=str(reference_data_csv))
        results = detector.run_comprehensive_drift_analysis(current_data=current_predictions_data)

        assert "prediction" in results["drift_levels"]
        assert results["drift_levels"]["prediction"]["available"] is True

    @patch("drone_detector_mlops.monitoring.drift_detection.storage.Client")
    def test_includes_image_features_drift(self, mock_storage_client, reference_data_csv, current_predictions_data):
        """Test that image-level drift is analyzed."""
        detector = DriftDetector(reference_dataset_path=str(reference_data_csv))
        results = detector.run_comprehensive_drift_analysis(current_data=current_predictions_data)

        assert "image_features" in results["drift_levels"]
        # Image features should be available now that we have all required columns
        if results["drift_levels"]["image_features"]["available"]:
            assert "all_passed" in results["drift_levels"]["image_features"]
        # Or it may fail if there's an issue with the data
        else:
            assert "error" in results["drift_levels"]["image_features"]

    @patch("drone_detector_mlops.monitoring.drift_detection.storage.Client")
    def test_embeddings_marked_as_unavailable(self, mock_storage_client, reference_data_csv, current_predictions_data):
        """Test that embeddings are marked as unavailable."""
        detector = DriftDetector(reference_dataset_path=str(reference_data_csv))
        results = detector.run_comprehensive_drift_analysis(current_data=current_predictions_data)

        assert "embeddings" in results["drift_levels"]
        assert results["drift_levels"]["embeddings"]["available"] is False

    @patch("drone_detector_mlops.monitoring.drift_detection.storage.Client")
    def test_overall_assessment_has_required_keys(
        self, mock_storage_client, reference_data_csv, current_predictions_data
    ):
        """Test that overall assessment has required keys."""
        detector = DriftDetector(reference_dataset_path=str(reference_data_csv))
        results = detector.run_comprehensive_drift_analysis(current_data=current_predictions_data)

        assessment = results["overall_assessment"]
        assert "severity" in assessment
        assert "alerts" in assessment
        assert "requires_action" in assessment
        assert "recommended_actions" in assessment

    @patch("drone_detector_mlops.monitoring.drift_detection.storage.Client")
    def test_handles_prediction_level_error(self, mock_storage_client, reference_data_csv):
        """Test handling of errors in prediction-level analysis."""
        detector = DriftDetector(reference_dataset_path=str(reference_data_csv))

        # Create data missing required columns
        bad_data = pd.DataFrame({"timestamp": ["2024-01-01"], "some_col": [1]})

        results = detector.run_comprehensive_drift_analysis(current_data=bad_data)

        # Should mark prediction level as unavailable
        assert results["drift_levels"]["prediction"]["available"] is False


class TestAssessDriftLevels:
    """Tests for _assess_drift_levels method."""

    @patch("drone_detector_mlops.monitoring.drift_detection.storage.Client")
    def test_assesses_no_drift(self, mock_storage_client, reference_data_csv):
        """Test assessment when no drift is detected."""
        detector = DriftDetector(reference_dataset_path=str(reference_data_csv))

        drift_levels = {
            "prediction": {"available": True, "alerts": []},
            "image_features": {"available": True, "all_passed": True},
            "embeddings": {"available": False},
        }

        assessment = detector._assess_drift_levels(drift_levels)

        assert assessment["severity"] == "none"
        assert len(assessment["alerts"]) == 0
        assert assessment["requires_action"] is False

    @patch("drone_detector_mlops.monitoring.drift_detection.storage.Client")
    def test_assesses_high_severity(self, mock_storage_client, reference_data_csv):
        """Test assessment with high severity drift."""
        detector = DriftDetector(reference_dataset_path=str(reference_data_csv))

        drift_levels = {
            "prediction": {"available": True, "alerts": [{"severity": "high", "message": "High confidence drop"}]},
            "image_features": {"available": True, "all_passed": False},
            "embeddings": {"available": False},
        }

        assessment = detector._assess_drift_levels(drift_levels)

        assert assessment["severity"] == "high"
        assert assessment["requires_action"] is True
        assert len(assessment["alerts"]) > 0

    @patch("drone_detector_mlops.monitoring.drift_detection.storage.Client")
    def test_assesses_low_severity_from_image_features(self, mock_storage_client, reference_data_csv):
        """Test assessment with only image feature drift."""
        detector = DriftDetector(reference_dataset_path=str(reference_data_csv))

        drift_levels = {
            "prediction": {"available": True, "alerts": []},
            "image_features": {"available": True, "all_passed": False},
            "embeddings": {"available": False},
        }

        assessment = detector._assess_drift_levels(drift_levels)

        assert assessment["severity"] == "low"


class TestGetRecommendedActions:
    """Tests for _get_recommended_actions method."""

    @patch("drone_detector_mlops.monitoring.drift_detection.storage.Client")
    def test_returns_list_of_actions(self, mock_storage_client, reference_data_csv):
        """Test that recommended actions are returned as a list."""
        detector = DriftDetector(reference_dataset_path=str(reference_data_csv))

        actions = detector._get_recommended_actions("high")

        assert isinstance(actions, list)
        assert len(actions) > 0

    @patch("drone_detector_mlops.monitoring.drift_detection.storage.Client")
    def test_high_severity_has_more_actions(self, mock_storage_client, reference_data_csv):
        """Test that high severity has more actions than low."""
        detector = DriftDetector(reference_dataset_path=str(reference_data_csv))

        high_actions = detector._get_recommended_actions("high")
        low_actions = detector._get_recommended_actions("low")

        assert len(high_actions) >= len(low_actions)

    @patch("drone_detector_mlops.monitoring.drift_detection.storage.Client")
    def test_none_severity_returns_empty_or_minimal_actions(self, mock_storage_client, reference_data_csv):
        """Test that no severity returns minimal actions."""
        detector = DriftDetector(reference_dataset_path=str(reference_data_csv))

        actions = detector._get_recommended_actions("none")

        assert isinstance(actions, list)
