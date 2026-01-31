"""Prediction-level drift detection.

This module monitors drift in model predictions (confidence scores, class distribution)
which is more directly relevant to model performance than input image statistics.
"""

import pandas as pd
from evidently.legacy.metrics import DatasetDriftMetric
from evidently.legacy.report import Report
from evidently.legacy.test_suite import TestSuite
from evidently.legacy.test_preset import DataDriftTestPreset
from evidently.legacy.pipeline.column_mapping import ColumnMapping

from drone_detector_mlops.utils.logger import get_logger

logger = get_logger(__name__)


class PredictionDriftMonitor:
    """Monitor drift in model predictions and outputs."""

    def __init__(self):
        """Initialize prediction drift monitor."""
        self.prediction_columns = ["confidence", "class_int"]

    def analyze_confidence_drift(self, reference_data: pd.DataFrame, current_data: pd.DataFrame) -> dict:
        """Analyze drift in prediction confidence scores.

        Args:
            reference_data: Reference predictions with 'confidence' column
            current_data: Current predictions with 'confidence' column

        Returns:
            Dictionary with confidence drift metrics
        """
        ref_conf = reference_data["confidence"]
        curr_conf = current_data["confidence"]

        # Calculate statistics
        metrics = {
            "reference_mean_confidence": float(ref_conf.mean()),
            "current_mean_confidence": float(curr_conf.mean()),
            "confidence_drop": float(ref_conf.mean() - curr_conf.mean()),
            "reference_std_confidence": float(ref_conf.std()),
            "current_std_confidence": float(curr_conf.std()),
            "low_confidence_rate": float((curr_conf < 0.7).sum() / len(curr_conf)),
        }

        # Determine if significant drop
        metrics["significant_drop"] = metrics["confidence_drop"] > 0.1

        if metrics["significant_drop"]:
            logger.warning(f"Significant confidence drop detected: {metrics['confidence_drop']:.3f}")

        return metrics

    def analyze_class_distribution_drift(self, reference_data: pd.DataFrame, current_data: pd.DataFrame) -> dict:
        """Analyze drift in predicted class distribution.

        Args:
            reference_data: Reference predictions with 'class_name' column
            current_data: Current predictions with 'class_name' column

        Returns:
            Dictionary with class distribution drift metrics
        """
        ref_dist = reference_data["class_name"].value_counts(normalize=True)
        curr_dist = current_data["class_name"].value_counts(normalize=True)

        # Calculate metrics
        metrics = {
            "reference_drone_rate": float(ref_dist.get("drone", 0)),
            "current_drone_rate": float(curr_dist.get("drone", 0)),
            "reference_bird_rate": float(ref_dist.get("bird", 0)),
            "current_bird_rate": float(curr_dist.get("bird", 0)),
        }

        # Calculate distribution shift
        metrics["distribution_shift"] = abs(metrics["reference_drone_rate"] - metrics["current_drone_rate"])
        metrics["significant_shift"] = metrics["distribution_shift"] > 0.15

        if metrics["significant_shift"]:
            logger.warning(f"Significant class distribution shift: {metrics['distribution_shift']:.3f}")

        return metrics

    def generate_prediction_drift_report(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        output_path: str | None = None,
    ) -> Report:
        """Generate Evidently report for prediction-level drift.

        Args:
            reference_data: Reference predictions
            current_data: Current predictions
            output_path: Optional path to save HTML report

        Returns:
            Evidently Report object
        """
        # Use only prediction-relevant columns
        pred_cols = ["confidence", "class_int", "class_name"]
        ref_df = reference_data[pred_cols].copy()
        curr_df = current_data[pred_cols].copy()

        # Configure column mapping for predictions
        column_mapping = ColumnMapping(
            target="class_name",
            prediction="class_int",
            numerical_features=["confidence"],
        )

        # Create report focusing on predictions
        report = Report(metrics=[DatasetDriftMetric()])

        report.run(reference_data=ref_df, current_data=curr_df, column_mapping=column_mapping)

        if output_path:
            report.save_html(output_path)
            logger.success(f"Prediction drift report saved to {output_path}")

        return report

    def run_prediction_drift_tests(self, reference_data: pd.DataFrame, current_data: pd.DataFrame) -> dict:
        """Run statistical tests on prediction drift.

        Args:
            reference_data: Reference predictions
            current_data: Current predictions

        Returns:
            Dictionary with test results
        """
        pred_cols = ["confidence", "class_int", "class_name"]
        ref_df = reference_data[pred_cols].copy()
        curr_df = current_data[pred_cols].copy()

        column_mapping = ColumnMapping(
            target="class_name",
            prediction="class_int",
            numerical_features=["confidence"],
        )

        test_suite = TestSuite(tests=[DataDriftTestPreset()])
        test_suite.run(reference_data=ref_df, current_data=curr_df, column_mapping=column_mapping)

        results = test_suite.as_dict()
        summary = results.get("summary", {})

        return {
            "all_passed": summary.get("all_passed", False),
            "summary": summary,
            "details": results,
        }

    def get_prediction_summary(self, reference_data: pd.DataFrame, current_data: pd.DataFrame) -> dict:
        """Get comprehensive summary of prediction-level metrics.

        Args:
            reference_data: Reference predictions
            current_data: Current predictions

        Returns:
            Dictionary with all prediction metrics
        """
        confidence_metrics = self.analyze_confidence_drift(reference_data, current_data)
        class_metrics = self.analyze_class_distribution_drift(reference_data, current_data)

        # Combine all metrics
        summary = {
            "confidence": confidence_metrics,
            "class_distribution": class_metrics,
            "alerts": [],
        }

        # Generate alerts
        if confidence_metrics["significant_drop"]:
            summary["alerts"].append(
                {
                    "type": "confidence_drop",
                    "severity": "high" if confidence_metrics["confidence_drop"] > 0.15 else "medium",
                    "message": f"Model confidence dropped by {confidence_metrics['confidence_drop']:.1%}",
                }
            )

        if class_metrics["significant_shift"]:
            summary["alerts"].append(
                {
                    "type": "class_distribution_shift",
                    "severity": "medium",
                    "message": f"Class distribution shifted by {class_metrics['distribution_shift']:.1%}",
                }
            )

        logger.info(f"Generated prediction summary with {len(summary['alerts'])} alerts")

        return summary
