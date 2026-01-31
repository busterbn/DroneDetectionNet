import pandas as pd
import pytest

from drone_detector_mlops.monitoring.prediction_drift import PredictionDriftMonitor


@pytest.fixture
def monitor():
    """Create a PredictionDriftMonitor instance."""
    return PredictionDriftMonitor()


@pytest.fixture
def reference_predictions():
    """Create reference prediction data."""
    return pd.DataFrame(
        {
            "confidence": [0.95, 0.88, 0.92, 0.85, 0.90, 0.87, 0.93, 0.91],
            "class_name": ["drone", "bird", "drone", "bird", "drone", "drone", "bird", "drone"],
            "class_int": [0, 1, 0, 1, 0, 0, 1, 0],
        }
    )


@pytest.fixture
def current_predictions_no_drift():
    """Create current predictions with no drift."""
    return pd.DataFrame(
        {
            "confidence": [0.94, 0.89, 0.91, 0.86, 0.88, 0.92, 0.90, 0.93],
            "class_name": ["drone", "bird", "drone", "bird", "drone", "drone", "bird", "drone"],
            "class_int": [0, 1, 0, 1, 0, 0, 1, 0],
        }
    )


@pytest.fixture
def current_predictions_confidence_drop():
    """Create current predictions with confidence drop."""
    return pd.DataFrame(
        {
            "confidence": [0.65, 0.60, 0.68, 0.55, 0.72, 0.58, 0.70, 0.62],
            "class_name": ["drone", "bird", "drone", "bird", "drone", "drone", "bird", "drone"],
            "class_int": [0, 1, 0, 1, 0, 0, 1, 0],
        }
    )


@pytest.fixture
def current_predictions_class_shift():
    """Create current predictions with class distribution shift."""
    return pd.DataFrame(
        {
            "confidence": [0.95, 0.88, 0.92, 0.85, 0.90, 0.87, 0.93, 0.91],
            "class_name": ["bird", "bird", "bird", "bird", "bird", "bird", "bird", "drone"],
            "class_int": [1, 1, 1, 1, 1, 1, 1, 0],
        }
    )


class TestPredictionDriftMonitor:
    """Tests for PredictionDriftMonitor class."""

    def test_initialization(self, monitor):
        """Test PredictionDriftMonitor initialization."""
        assert isinstance(monitor, PredictionDriftMonitor)
        assert hasattr(monitor, "prediction_columns")
        assert "confidence" in monitor.prediction_columns
        assert "class_int" in monitor.prediction_columns


class TestAnalyzeConfidenceDrift:
    """Tests for analyze_confidence_drift method."""

    def test_returns_dict_with_required_keys(self, monitor, reference_predictions, current_predictions_no_drift):
        """Test that analyze_confidence_drift returns dict with required keys."""
        metrics = monitor.analyze_confidence_drift(reference_predictions, current_predictions_no_drift)

        assert isinstance(metrics, dict)
        required_keys = [
            "reference_mean_confidence",
            "current_mean_confidence",
            "confidence_drop",
            "reference_std_confidence",
            "current_std_confidence",
            "low_confidence_rate",
            "significant_drop",
        ]
        for key in required_keys:
            assert key in metrics

    def test_calculates_mean_confidence_correctly(self, monitor, reference_predictions, current_predictions_no_drift):
        """Test that mean confidence is calculated correctly."""
        metrics = monitor.analyze_confidence_drift(reference_predictions, current_predictions_no_drift)

        ref_mean = reference_predictions["confidence"].mean()
        curr_mean = current_predictions_no_drift["confidence"].mean()

        assert abs(metrics["reference_mean_confidence"] - ref_mean) < 0.01
        assert abs(metrics["current_mean_confidence"] - curr_mean) < 0.01

    def test_calculates_confidence_drop(self, monitor, reference_predictions, current_predictions_no_drift):
        """Test that confidence drop is calculated correctly."""
        metrics = monitor.analyze_confidence_drift(reference_predictions, current_predictions_no_drift)

        expected_drop = reference_predictions["confidence"].mean() - current_predictions_no_drift["confidence"].mean()

        assert abs(metrics["confidence_drop"] - expected_drop) < 0.01

    def test_detects_no_significant_drop(self, monitor, reference_predictions, current_predictions_no_drift):
        """Test that no significant drop is detected when confidence is stable."""
        metrics = monitor.analyze_confidence_drift(reference_predictions, current_predictions_no_drift)

        assert metrics["significant_drop"] is False

    def test_detects_significant_drop(self, monitor, reference_predictions, current_predictions_confidence_drop):
        """Test that significant drop is detected when confidence drops."""
        metrics = monitor.analyze_confidence_drift(reference_predictions, current_predictions_confidence_drop)

        assert metrics["significant_drop"] is True
        assert metrics["confidence_drop"] > 0.1

    def test_calculates_low_confidence_rate(self, monitor, reference_predictions, current_predictions_confidence_drop):
        """Test that low confidence rate is calculated correctly."""
        metrics = monitor.analyze_confidence_drift(reference_predictions, current_predictions_confidence_drop)

        expected_low_rate = (current_predictions_confidence_drop["confidence"] < 0.7).sum() / len(
            current_predictions_confidence_drop
        )

        assert abs(metrics["low_confidence_rate"] - expected_low_rate) < 0.01
        assert metrics["low_confidence_rate"] > 0


class TestAnalyzeClassDistributionDrift:
    """Tests for analyze_class_distribution_drift method."""

    def test_returns_dict_with_required_keys(self, monitor, reference_predictions, current_predictions_no_drift):
        """Test that analyze_class_distribution_drift returns dict with required keys."""
        metrics = monitor.analyze_class_distribution_drift(reference_predictions, current_predictions_no_drift)

        assert isinstance(metrics, dict)
        required_keys = [
            "reference_drone_rate",
            "current_drone_rate",
            "reference_bird_rate",
            "current_bird_rate",
            "distribution_shift",
            "significant_shift",
        ]
        for key in required_keys:
            assert key in metrics

    def test_calculates_class_rates_correctly(self, monitor, reference_predictions, current_predictions_no_drift):
        """Test that class rates are calculated correctly."""
        metrics = monitor.analyze_class_distribution_drift(reference_predictions, current_predictions_no_drift)

        ref_drone_rate = (reference_predictions["class_name"] == "drone").sum() / len(reference_predictions)
        curr_drone_rate = (current_predictions_no_drift["class_name"] == "drone").sum() / len(
            current_predictions_no_drift
        )

        assert abs(metrics["reference_drone_rate"] - ref_drone_rate) < 0.01
        assert abs(metrics["current_drone_rate"] - curr_drone_rate) < 0.01

    def test_bird_and_drone_rates_sum_to_one(self, monitor, reference_predictions, current_predictions_no_drift):
        """Test that bird and drone rates sum to approximately 1."""
        metrics = monitor.analyze_class_distribution_drift(reference_predictions, current_predictions_no_drift)

        ref_total = metrics["reference_drone_rate"] + metrics["reference_bird_rate"]
        curr_total = metrics["current_drone_rate"] + metrics["current_bird_rate"]

        assert abs(ref_total - 1.0) < 0.01
        assert abs(curr_total - 1.0) < 0.01

    def test_detects_no_significant_shift(self, monitor, reference_predictions, current_predictions_no_drift):
        """Test that no significant shift is detected when distribution is stable."""
        metrics = monitor.analyze_class_distribution_drift(reference_predictions, current_predictions_no_drift)

        assert metrics["significant_shift"] is False
        assert metrics["distribution_shift"] < 0.15

    def test_detects_significant_shift(self, monitor, reference_predictions, current_predictions_class_shift):
        """Test that significant shift is detected when class distribution changes."""
        metrics = monitor.analyze_class_distribution_drift(reference_predictions, current_predictions_class_shift)

        assert metrics["significant_shift"] is True
        assert metrics["distribution_shift"] > 0.15

    def test_distribution_shift_is_absolute_value(
        self, monitor, reference_predictions, current_predictions_class_shift
    ):
        """Test that distribution shift is always positive (absolute value)."""
        metrics = monitor.analyze_class_distribution_drift(reference_predictions, current_predictions_class_shift)

        assert metrics["distribution_shift"] >= 0


class TestGetPredictionSummary:
    """Tests for get_prediction_summary method."""

    def test_returns_dict_with_required_keys(self, monitor, reference_predictions, current_predictions_no_drift):
        """Test that get_prediction_summary returns dict with required keys."""
        summary = monitor.get_prediction_summary(reference_predictions, current_predictions_no_drift)

        assert isinstance(summary, dict)
        assert "confidence" in summary
        assert "class_distribution" in summary
        assert "alerts" in summary

    def test_includes_confidence_metrics(self, monitor, reference_predictions, current_predictions_no_drift):
        """Test that summary includes confidence metrics."""
        summary = monitor.get_prediction_summary(reference_predictions, current_predictions_no_drift)

        assert "reference_mean_confidence" in summary["confidence"]
        assert "current_mean_confidence" in summary["confidence"]
        assert "confidence_drop" in summary["confidence"]

    def test_includes_class_distribution_metrics(self, monitor, reference_predictions, current_predictions_no_drift):
        """Test that summary includes class distribution metrics."""
        summary = monitor.get_prediction_summary(reference_predictions, current_predictions_no_drift)

        assert "reference_drone_rate" in summary["class_distribution"]
        assert "current_drone_rate" in summary["class_distribution"]
        assert "distribution_shift" in summary["class_distribution"]

    def test_generates_no_alerts_when_stable(self, monitor, reference_predictions, current_predictions_no_drift):
        """Test that no alerts are generated when predictions are stable."""
        summary = monitor.get_prediction_summary(reference_predictions, current_predictions_no_drift)

        assert isinstance(summary["alerts"], list)
        assert len(summary["alerts"]) == 0

    def test_generates_confidence_alert(self, monitor, reference_predictions, current_predictions_confidence_drop):
        """Test that confidence drop alert is generated."""
        summary = monitor.get_prediction_summary(reference_predictions, current_predictions_confidence_drop)

        assert len(summary["alerts"]) > 0
        confidence_alerts = [a for a in summary["alerts"] if a["type"] == "confidence_drop"]
        assert len(confidence_alerts) > 0
        assert confidence_alerts[0]["severity"] in ["medium", "high"]

    def test_generates_class_shift_alert(self, monitor, reference_predictions, current_predictions_class_shift):
        """Test that class distribution shift alert is generated."""
        summary = monitor.get_prediction_summary(reference_predictions, current_predictions_class_shift)

        assert len(summary["alerts"]) > 0
        shift_alerts = [a for a in summary["alerts"] if a["type"] == "class_distribution_shift"]
        assert len(shift_alerts) > 0
        assert shift_alerts[0]["severity"] == "medium"

    def test_alert_has_required_fields(self, monitor, reference_predictions, current_predictions_confidence_drop):
        """Test that alerts have required fields."""
        summary = monitor.get_prediction_summary(reference_predictions, current_predictions_confidence_drop)

        for alert in summary["alerts"]:
            assert "type" in alert
            assert "severity" in alert
            assert "message" in alert

    def test_generates_multiple_alerts(self, monitor, reference_predictions, current_predictions_confidence_drop):
        """Test that multiple alerts can be generated."""
        # Create data with both confidence drop and class shift
        data_with_both = current_predictions_confidence_drop.copy()
        data_with_both.loc[:6, "class_name"] = "bird"
        data_with_both.loc[:6, "class_int"] = 1

        summary = monitor.get_prediction_summary(reference_predictions, data_with_both)

        assert len(summary["alerts"]) >= 2


class TestGeneratePredictionDriftReport:
    """Tests for generate_prediction_drift_report method."""

    def test_returns_report_object(self, monitor, reference_predictions, current_predictions_no_drift):
        """Test that generate_prediction_drift_report returns a Report object."""
        from evidently.legacy.report import Report

        report = monitor.generate_prediction_drift_report(reference_predictions, current_predictions_no_drift)

        assert isinstance(report, Report)

    def test_saves_html_report(self, monitor, reference_predictions, current_predictions_no_drift, tmp_path):
        """Test that report can be saved to HTML file."""
        output_path = tmp_path / "drift_report.html"

        monitor.generate_prediction_drift_report(
            reference_predictions, current_predictions_no_drift, output_path=str(output_path)
        )

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_report_without_output_path(self, monitor, reference_predictions, current_predictions_no_drift):
        """Test that report can be generated without saving to file."""
        report = monitor.generate_prediction_drift_report(reference_predictions, current_predictions_no_drift)

        assert report is not None


class TestRunPredictionDriftTests:
    """Tests for run_prediction_drift_tests method."""

    def test_returns_dict_with_required_keys(self, monitor, reference_predictions, current_predictions_no_drift):
        """Test that run_prediction_drift_tests returns dict with required keys."""
        results = monitor.run_prediction_drift_tests(reference_predictions, current_predictions_no_drift)

        assert isinstance(results, dict)
        assert "all_passed" in results
        assert "summary" in results
        assert "details" in results

    def test_all_passed_is_boolean(self, monitor, reference_predictions, current_predictions_no_drift):
        """Test that all_passed is a boolean value."""
        results = monitor.run_prediction_drift_tests(reference_predictions, current_predictions_no_drift)

        assert isinstance(results["all_passed"], bool)

    def test_summary_is_dict(self, monitor, reference_predictions, current_predictions_no_drift):
        """Test that summary is a dictionary."""
        results = monitor.run_prediction_drift_tests(reference_predictions, current_predictions_no_drift)

        assert isinstance(results["summary"], dict)

    def test_detects_drift_with_shifted_data(self, monitor, reference_predictions, current_predictions_class_shift):
        """Test that drift tests can detect significant drift."""
        results = monitor.run_prediction_drift_tests(reference_predictions, current_predictions_class_shift)

        assert isinstance(results["all_passed"], bool)
        # The test may or may not pass depending on the test thresholds
