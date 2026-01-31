"""Model embedding drift detection.

This module monitors drift in the model's learned representations (embeddings)
which directly reflects what the model sees, rather than input image statistics.
"""

import numpy as np
import pandas as pd
import torch
from PIL import Image
from evidently.legacy.metrics import DatasetDriftMetric
from evidently.legacy.report import Report
from evidently.legacy.test_suite import TestSuite
from evidently.legacy.test_preset import DataDriftTestPreset
from evidently.legacy.pipeline.column_mapping import ColumnMapping

from drone_detector_mlops.model import DroneDetectorModel
from drone_detector_mlops.data.transforms import test_transform
from drone_detector_mlops.utils.logger import get_logger
from drone_detector_mlops.utils.storage import get_storage

logger = get_logger(__name__)


class EmbeddingExtractor:
    """Extract embeddings from ResNet18 model."""

    def __init__(self, model_path: str | None = None):
        """Initialize embedding extractor.

        Args:
            model_path: Path to PyTorch model checkpoint (.pth file)
        """
        self.device = torch.device("cpu")  # Use CPU for embedding extraction
        self.model = None
        self.model_path = model_path

    def load_model(self):
        """Load PyTorch model for embedding extraction."""
        if self.model is not None:
            return

        logger.info("Loading PyTorch model for embedding extraction")

        # Create model
        self.model = DroneDetectorModel(num_classes=2, pretrained=False)

        # Load weights if path provided
        if self.model_path:
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device)
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    self.model.load_state_dict(checkpoint["model_state_dict"])
                else:
                    self.model.load_state_dict(checkpoint)
                logger.success("Loaded model weights from checkpoint")
            except Exception as e:
                logger.warning(f"Could not load checkpoint: {e}. Using random weights.")
        else:
            # Try to load from storage
            try:
                storage = get_storage()
                # Try to load the latest .pth model
                model_path = storage.models_dir / "model-latest.pth"
                if storage.mode == "local" and model_path.exists():
                    checkpoint = torch.load(model_path, map_location=self.device)
                    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                        self.model.load_state_dict(checkpoint["model_state_dict"])
                    else:
                        self.model.load_state_dict(checkpoint)
                    logger.success("Loaded model weights from storage")
                else:
                    logger.warning("No checkpoint found. Using pretrained ImageNet weights for embeddings.")
            except Exception as e:
                logger.warning(f"Could not load model from storage: {e}")

        self.model.eval()
        self.model.to(self.device)

    def extract_embedding(self, image: Image.Image) -> np.ndarray:
        """Extract embedding from a single image.

        Args:
            image: PIL Image

        Returns:
            512-dimensional embedding vector (ResNet18 final layer before classification)
        """
        self.load_model()

        # Transform image
        img_tensor = test_transform(image).unsqueeze(0).to(self.device)

        # Extract features before final classification layer
        with torch.no_grad():
            # Get features from ResNet18's avgpool layer (512 dimensions)
            # The model.model is the TIMM ResNet18
            features = self.model.model.forward_features(img_tensor)
            # Global average pooling if not already done
            if features.dim() == 4:  # [B, C, H, W]
                features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
                features = features.flatten(1)  # [B, 512]

        embedding = features.cpu().numpy()[0]  # Remove batch dimension
        return embedding

    def extract_embeddings_batch(self, images: list[Image.Image]) -> np.ndarray:
        """Extract embeddings from multiple images.

        Args:
            images: List of PIL Images

        Returns:
            Array of shape (N, 512) with embeddings
        """
        embeddings = []
        for img in images:
            emb = self.extract_embedding(img)
            embeddings.append(emb)

        return np.array(embeddings)


class EmbeddingDriftMonitor:
    """Monitor drift in model embeddings."""

    def __init__(self, model_path: str | None = None):
        """Initialize embedding drift monitor.

        Args:
            model_path: Path to PyTorch model checkpoint
        """
        self.extractor = EmbeddingExtractor(model_path)
        self.embedding_dim = 512  # ResNet18 embedding dimension

    def prepare_embedding_dataframe(self, embeddings: np.ndarray, metadata: pd.DataFrame = None) -> pd.DataFrame:
        """Convert embeddings to DataFrame format for Evidently.

        Args:
            embeddings: Array of shape (N, 512)
            metadata: Optional metadata to include

        Returns:
            DataFrame with embedding columns
        """
        # Create column names for embeddings
        embedding_cols = [f"emb_{i}" for i in range(self.embedding_dim)]

        df = pd.DataFrame(embeddings, columns=embedding_cols)

        # Add metadata if provided
        if metadata is not None:
            for col in metadata.columns:
                if col not in df.columns:
                    df[col] = metadata[col].values

        return df

    def generate_embedding_drift_report(
        self,
        reference_embeddings: np.ndarray,
        current_embeddings: np.ndarray,
        output_path: str | None = None,
    ) -> Report:
        """Generate Evidently report for embedding drift.

        Args:
            reference_embeddings: Reference embeddings (N, 512)
            current_embeddings: Current embeddings (M, 512)
            output_path: Optional path to save HTML report

        Returns:
            Evidently Report object
        """
        logger.info(
            f"Analyzing embedding drift: reference={len(reference_embeddings)}, current={len(current_embeddings)}"
        )

        # Convert to DataFrames
        ref_df = self.prepare_embedding_dataframe(reference_embeddings)
        curr_df = self.prepare_embedding_dataframe(current_embeddings)

        # Define all embedding columns as numerical features
        embedding_cols = [f"emb_{i}" for i in range(self.embedding_dim)]

        column_mapping = ColumnMapping(numerical_features=embedding_cols)

        # Create report
        report = Report(metrics=[DatasetDriftMetric()])

        report.run(reference_data=ref_df, current_data=curr_df, column_mapping=column_mapping)

        if output_path:
            report.save_html(output_path)
            logger.success(f"Embedding drift report saved to {output_path}")

        return report

    def run_embedding_drift_tests(self, reference_embeddings: np.ndarray, current_embeddings: np.ndarray) -> dict:
        """Run statistical tests on embedding drift.

        Args:
            reference_embeddings: Reference embeddings (N, 512)
            current_embeddings: Current embeddings (M, 512)

        Returns:
            Dictionary with test results
        """
        ref_df = self.prepare_embedding_dataframe(reference_embeddings)
        curr_df = self.prepare_embedding_dataframe(current_embeddings)

        embedding_cols = [f"emb_{i}" for i in range(self.embedding_dim)]
        column_mapping = ColumnMapping(numerical_features=embedding_cols)

        test_suite = TestSuite(tests=[DataDriftTestPreset()])
        test_suite.run(reference_data=ref_df, current_data=curr_df, column_mapping=column_mapping)

        results = test_suite.as_dict()
        summary = results.get("summary", {})

        return {
            "all_passed": summary.get("all_passed", False),
            "summary": summary,
            "details": results,
        }

    def calculate_embedding_statistics(self, reference_embeddings: np.ndarray, current_embeddings: np.ndarray) -> dict:
        """Calculate summary statistics for embedding drift.

        Args:
            reference_embeddings: Reference embeddings
            current_embeddings: Current embeddings

        Returns:
            Dictionary with statistics
        """
        # Calculate cosine similarity between mean embeddings
        ref_mean = reference_embeddings.mean(axis=0)
        curr_mean = current_embeddings.mean(axis=0)

        cosine_sim = np.dot(ref_mean, curr_mean) / (np.linalg.norm(ref_mean) * np.linalg.norm(curr_mean))

        # Calculate L2 distance between means
        l2_distance = float(np.linalg.norm(ref_mean - curr_mean))

        # Calculate variance change
        ref_var = float(reference_embeddings.var(axis=0).mean())
        curr_var = float(current_embeddings.var(axis=0).mean())

        stats = {
            "cosine_similarity": float(cosine_sim),
            "l2_distance": l2_distance,
            "reference_mean_variance": ref_var,
            "current_mean_variance": curr_var,
            "variance_ratio": curr_var / ref_var if ref_var > 0 else 0,
        }

        # Determine if significant drift
        stats["significant_drift"] = (
            cosine_sim < 0.95  # Mean embeddings diverged
            or l2_distance > 5.0  # Large distance
            or abs(1 - stats["variance_ratio"]) > 0.3  # Variance changed significantly
        )

        if stats["significant_drift"]:
            logger.warning(f"Significant embedding drift detected: {stats}")

        return stats
