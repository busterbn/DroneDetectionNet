"""Comprehensive drift detection using Evidently AI.

This module provides multi-level drift detection:
1. Image-level: Brightness, contrast, colors (proxy metrics)
2. Prediction-level: Confidence scores, class distribution (model outputs)
3. Embedding-level: Model's learned representations (what the model actually sees)
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from evidently.legacy.pipeline.column_mapping import ColumnMapping
from evidently.legacy.metrics import DatasetDriftMetric, DatasetMissingValuesMetric
from evidently.legacy.report import Report
from evidently.legacy.test_suite import TestSuite
from evidently.legacy.test_preset import DataDriftTestPreset, DataQualityTestPreset
from evidently.legacy.metric_preset import DataDriftPreset, DataQualityPreset, TargetDriftPreset
from google.cloud import storage

from drone_detector_mlops.monitoring.feature_extraction import ImageFeatureExtractor
from drone_detector_mlops.monitoring.prediction_drift import PredictionDriftMonitor
from drone_detector_mlops.monitoring.embedding_drift import EmbeddingDriftMonitor
from drone_detector_mlops.utils.logger import get_logger

logger = get_logger(__name__)


class DriftDetector:
    """Comprehensive drift detection using multiple monitoring levels."""

    def __init__(
        self,
        reference_dataset_path: str = "data/reference_dataset.csv",
        inference_bucket: str = "drone-detection-mlops-data",
        inference_prefix: str = "inference/prediction_",
        model_path: str | None = None,
    ):
        """Initialize comprehensive drift detector.

        Args:
            reference_dataset_path: Path to reference dataset CSV
            inference_bucket: GCS bucket containing predictions
            inference_prefix: Prefix for prediction files in bucket
            model_path: Optional path to PyTorch model for embedding extraction
        """
        self.reference_dataset_path = Path(reference_dataset_path)
        self.inference_bucket = inference_bucket
        self.inference_prefix = inference_prefix
        self.storage_client = storage.Client()

        # Load reference dataset
        if self.reference_dataset_path.exists():
            self.reference_data = pd.read_csv(self.reference_dataset_path)
            logger.info(f"Loaded reference dataset: {len(self.reference_data)} samples")
        else:
            logger.warning(f"Reference dataset not found: {self.reference_dataset_path}")
            self.reference_data = None

        # Define feature columns
        self.feature_columns = ImageFeatureExtractor.get_feature_names()
        self.target_column = "class_name"
        self.prediction_column = "class_int"

        # Initialize specialized monitors
        self.prediction_monitor = PredictionDriftMonitor()
        self.embedding_monitor = EmbeddingDriftMonitor(model_path)

    def fetch_recent_predictions(self, max_predictions: int = 100) -> pd.DataFrame:
        """Fetch recent predictions from GCS.

        Args:
            max_predictions: Maximum number of recent predictions to fetch

        Returns:
            DataFrame with prediction data and features
        """
        bucket = self.storage_client.bucket(self.inference_bucket)
        blobs = list(bucket.list_blobs(prefix=self.inference_prefix))

        # Sort by time created (most recent first)
        blobs = sorted(blobs, key=lambda b: b.time_created, reverse=True)
        blobs = blobs[:max_predictions]

        logger.info(f"Fetching {len(blobs)} recent predictions from GCS")

        records = []
        for blob in blobs:
            try:
                data = json.loads(blob.download_as_text())

                # Extract features and metadata
                record = {
                    "timestamp": data.get("timestamp"),
                    "class_name": data.get("class_name"),
                    "class_int": 0 if data.get("class_name") == "drone" else 1,
                    "confidence": data.get("confidence"),
                    **data.get("features", {}),
                }

                records.append(record)

            except Exception as e:
                logger.error(f"Failed to parse prediction {blob.name}: {e}")
                continue

        df = pd.DataFrame(records)
        logger.info(f"Loaded {len(df)} predictions with features")

        return df

    def generate_drift_report(
        self,
        current_data: pd.DataFrame | None = None,
        max_predictions: int = 100,
        output_path: str | None = None,
    ) -> Report:
        """Generate Evidently drift report.

        Args:
            current_data: Current/production data. If None, fetches from GCS
            max_predictions: Max predictions to fetch if current_data is None
            output_path: Path to save HTML report (optional)

        Returns:
            Evidently Report object
        """
        if self.reference_data is None:
            raise ValueError("Reference dataset not loaded")

        # Fetch current data if not provided
        if current_data is None:
            current_data = self.fetch_recent_predictions(max_predictions)

        if len(current_data) == 0:
            raise ValueError("No current data available")

        # Prepare datasets - select only common columns
        common_cols = ["timestamp", "class_name", "class_int"] + self.feature_columns
        reference_df = self.reference_data[common_cols].copy()
        current_df = current_data[common_cols].copy()

        logger.info(f"Comparing reference ({len(reference_df)}) vs current ({len(current_df)})")

        # Configure column mapping
        column_mapping = ColumnMapping(
            target="class_name",
            prediction="class_int",
            numerical_features=self.feature_columns,
        )

        # Create report with multiple presets
        report = Report(
            metrics=[
                DataDriftPreset(),
                DataQualityPreset(),
                TargetDriftPreset(),
            ]
        )

        # Run report
        report.run(reference_data=reference_df, current_data=current_df, column_mapping=column_mapping)

        logger.success("Drift report generated successfully")

        # Save to HTML if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            report.save_html(str(output_path))
            logger.success(f"Report saved to {output_path}")

        return report

    def run_drift_tests(
        self, current_data: pd.DataFrame | None = None, max_predictions: int = 100
    ) -> dict[str, bool | dict]:
        """Run programmatic drift tests using Evidently TestSuite.

        Args:
            current_data: Current/production data. If None, fetches from GCS
            max_predictions: Max predictions to fetch if current_data is None

        Returns:
            Dictionary with test results:
            - all_passed: bool indicating if all tests passed
            - summary: dict with test summary
            - details: full test results
        """
        if self.reference_data is None:
            raise ValueError("Reference dataset not loaded")

        # Fetch current data if not provided
        if current_data is None:
            current_data = self.fetch_recent_predictions(max_predictions)

        if len(current_data) == 0:
            raise ValueError("No current data available")

        # Prepare datasets
        common_cols = ["timestamp", "class_name", "class_int"] + self.feature_columns
        reference_df = self.reference_data[common_cols].copy()
        current_df = current_data[common_cols].copy()

        # Configure column mapping
        column_mapping = ColumnMapping(
            target="class_name",
            prediction="class_int",
            numerical_features=self.feature_columns,
        )

        # Create test suite
        test_suite = TestSuite(tests=[DataDriftTestPreset(), DataQualityTestPreset()])

        # Run tests
        test_suite.run(reference_data=reference_df, current_data=current_df, column_mapping=column_mapping)

        # Get results
        results = test_suite.as_dict()

        # Extract summary
        summary = results.get("summary", {})
        all_passed = summary.get("all_passed", False)

        logger.info(f"Drift tests completed: {'PASSED' if all_passed else 'FAILED'}")

        return {"all_passed": all_passed, "summary": summary, "details": results}

    def get_drift_summary(self, current_data: pd.DataFrame | None = None, max_predictions: int = 100) -> dict:
        """Get a summary of drift metrics.

        Args:
            current_data: Current/production data. If None, fetches from GCS
            max_predictions: Max predictions to fetch if current_data is None

        Returns:
            Dictionary with drift summary statistics
        """
        if self.reference_data is None:
            raise ValueError("Reference dataset not loaded")

        # Fetch current data if not provided
        if current_data is None:
            current_data = self.fetch_recent_predictions(max_predictions)

        if len(current_data) == 0:
            raise ValueError("No current data available")

        # Prepare datasets
        common_cols = ["timestamp", "class_name", "class_int"] + self.feature_columns
        reference_df = self.reference_data[common_cols].copy()
        current_df = current_data[common_cols].copy()

        # Configure column mapping
        column_mapping = ColumnMapping(
            target="class_name",
            prediction="class_int",
            numerical_features=self.feature_columns,
        )

        # Create report with specific metrics
        report = Report(metrics=[DatasetDriftMetric(), DatasetMissingValuesMetric()])

        # Run report
        report.run(reference_data=reference_df, current_data=current_df, column_mapping=column_mapping)

        # Extract metrics
        results = report.as_dict()

        summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "reference_samples": len(reference_df),
            "current_samples": len(current_df),
            "metrics": results.get("metrics", []),
        }

        return summary

    def run_comprehensive_drift_analysis(
        self, current_data: pd.DataFrame | None = None, max_predictions: int = 100
    ) -> dict:
        """Run all three levels of drift detection.

        This is the recommended method for production monitoring as it analyzes:
        1. Image-level features (brightness, colors)
        2. Prediction-level metrics (confidence, class distribution)
        3. Model embeddings (what the model actually sees) - if available

        Args:
            current_data: Current/production data. If None, fetches from GCS
            max_predictions: Max predictions to fetch if current_data is None

        Returns:
            Dictionary with comprehensive drift analysis across all levels
        """
        if self.reference_data is None:
            raise ValueError("Reference dataset not loaded")

        # Fetch current data if not provided
        if current_data is None:
            current_data = self.fetch_recent_predictions(max_predictions)

        if len(current_data) == 0:
            raise ValueError("No current data available")

        logger.info("Running comprehensive drift analysis across all monitoring levels")

        results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "reference_samples": len(self.reference_data),
            "current_samples": len(current_data),
            "drift_levels": {},
        }

        # 1. PREDICTION-LEVEL DRIFT (Most Important)
        logger.info("Analyzing prediction-level drift...")
        try:
            prediction_summary = self.prediction_monitor.get_prediction_summary(self.reference_data, current_data)
            results["drift_levels"]["prediction"] = prediction_summary
            results["drift_levels"]["prediction"]["available"] = True
        except Exception as e:
            logger.error(f"Prediction-level drift analysis failed: {e}")
            results["drift_levels"]["prediction"] = {"available": False, "error": str(e)}

        # 2. IMAGE-LEVEL DRIFT (Supplementary)
        logger.info("Analyzing image-level feature drift...")
        try:
            # Run statistical tests on image features
            common_cols = ["timestamp", "class_name", "class_int"] + self.feature_columns
            ref_df = self.reference_data[common_cols].copy()
            curr_df = current_data[common_cols].copy()

            column_mapping = ColumnMapping(
                target="class_name",
                prediction="class_int",
                numerical_features=self.feature_columns,
            )

            test_suite = TestSuite(tests=[DataDriftTestPreset()])
            test_suite.run(reference_data=ref_df, current_data=curr_df, column_mapping=column_mapping)
            test_results = test_suite.as_dict()

            results["drift_levels"]["image_features"] = {
                "available": True,
                "all_passed": test_results.get("summary", {}).get("all_passed", False),
                "summary": test_results.get("summary", {}),
            }
        except Exception as e:
            logger.error(f"Image-level drift analysis failed: {e}")
            results["drift_levels"]["image_features"] = {"available": False, "error": str(e)}

        # 3. EMBEDDING DRIFT (Advanced - may not always be available)
        logger.info("Embedding drift analysis skipped (requires image data)")
        results["drift_levels"]["embeddings"] = {
            "available": False,
            "note": "Embedding drift requires re-processing images with PyTorch model",
        }

        # Generate overall assessment
        results["overall_assessment"] = self._assess_drift_levels(results["drift_levels"])

        logger.success("Comprehensive drift analysis completed")
        return results

    def _assess_drift_levels(self, drift_levels: dict) -> dict:
        """Assess overall drift severity across all levels.

        Args:
            drift_levels: Results from all drift monitoring levels

        Returns:
            Dictionary with overall assessment
        """
        alerts = []
        severity = "none"

        # Check prediction-level drift (highest priority)
        if drift_levels.get("prediction", {}).get("available"):
            pred_alerts = drift_levels["prediction"].get("alerts", [])
            if any(a["severity"] == "high" for a in pred_alerts):
                severity = "high"
                alerts.append("High confidence drop detected in predictions")
            elif any(a["severity"] == "medium" for a in pred_alerts):
                severity = max(severity, "medium", key=lambda x: ["none", "low", "medium", "high"].index(x))
                alerts.extend([a["message"] for a in pred_alerts])

        # Check image-level drift (lower priority)
        if drift_levels.get("image_features", {}).get("available"):
            if not drift_levels["image_features"].get("all_passed"):
                if severity == "none":
                    severity = "low"
                alerts.append("Image feature drift detected (may not affect model performance)")

        # Check embedding drift
        if drift_levels.get("embeddings", {}).get("available"):
            if drift_levels["embeddings"].get("significant_drift"):
                severity = "high"
                alerts.append("Model embedding drift detected - model sees different patterns")

        return {
            "severity": severity,
            "alerts": alerts,
            "requires_action": severity in ["high", "medium"],
            "recommended_actions": self._get_recommended_actions(severity),
        }

    def _get_recommended_actions(self, severity: str) -> list[str]:
        """Get recommended actions based on drift severity.

        Args:
            severity: Drift severity level

        Returns:
            List of recommended actions
        """
        if severity == "high":
            return [
                "Review drift report to identify drifted features",
                "Correlate with model performance metrics in W&B",
                "Consider retraining model with recent production data",
                "If permanent shift, regenerate reference dataset",
            ]
        elif severity == "medium":
            return [
                "Monitor model performance metrics closely",
                "Review drift report for specific drifted features",
                "Plan data collection for potential retraining",
            ]
        elif severity == "low":
            return [
                "Continue monitoring - drift may not impact performance",
                "Verify model accuracy with ground truth if available",
            ]
        else:
            return ["No action required - no significant drift detected"]
