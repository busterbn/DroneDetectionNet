"""Tests for embedding drift detection."""

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from unittest.mock import Mock, patch

from drone_detector_mlops.monitoring.embedding_drift import (
    EmbeddingExtractor,
    EmbeddingDriftMonitor,
)


@pytest.fixture
def sample_image():
    """Create a sample RGB image."""
    return Image.new("RGB", (224, 224), color=(128, 128, 128))


@pytest.fixture
def reference_embeddings():
    """Create reference embeddings."""
    np.random.seed(42)
    return np.random.randn(10, 512).astype(np.float32)


@pytest.fixture
def current_embeddings_no_drift():
    """Create current embeddings with no drift."""
    np.random.seed(43)
    return np.random.randn(10, 512).astype(np.float32)


@pytest.fixture
def current_embeddings_with_drift():
    """Create current embeddings with significant drift."""
    np.random.seed(42)
    # Create embeddings with different mean and variance
    return (np.random.randn(10, 512) * 2 + 5).astype(np.float32)


class TestEmbeddingExtractor:
    """Tests for EmbeddingExtractor class."""

    def test_initialization(self):
        """Test EmbeddingExtractor initialization."""
        extractor = EmbeddingExtractor()
        assert extractor.model is None
        assert extractor.device == torch.device("cpu")
        assert extractor.model_path is None

    def test_initialization_with_model_path(self):
        """Test initialization with model path."""
        extractor = EmbeddingExtractor(model_path="models/test.pth")
        assert extractor.model_path == "models/test.pth"

    @patch("drone_detector_mlops.monitoring.embedding_drift.DroneDetectorModel")
    def test_load_model_creates_model(self, mock_model_class):
        """Test that load_model creates a model instance."""
        mock_model = Mock()
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_model_class.return_value = mock_model

        extractor = EmbeddingExtractor()
        extractor.load_model()

        mock_model_class.assert_called_once_with(num_classes=2, pretrained=False)
        mock_model.eval.assert_called_once()
        mock_model.to.assert_called_once()

    @patch("drone_detector_mlops.monitoring.embedding_drift.DroneDetectorModel")
    def test_load_model_only_once(self, mock_model_class):
        """Test that model is only loaded once."""
        mock_model = Mock()
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_model_class.return_value = mock_model

        extractor = EmbeddingExtractor()
        extractor.load_model()
        extractor.load_model()  # Second call should not reload

        mock_model_class.assert_called_once()

    @patch("drone_detector_mlops.monitoring.embedding_drift.torch.load")
    @patch("drone_detector_mlops.monitoring.embedding_drift.DroneDetectorModel")
    def test_load_model_with_checkpoint_state_dict(self, mock_model_class, mock_torch_load, tmp_path):
        """Test loading model with checkpoint containing model_state_dict."""
        checkpoint_path = tmp_path / "checkpoint.pth"
        checkpoint_path.touch()

        mock_model = Mock()
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_model_class.return_value = mock_model

        mock_checkpoint = {"model_state_dict": {}}
        mock_torch_load.return_value = mock_checkpoint

        extractor = EmbeddingExtractor(model_path=str(checkpoint_path))
        extractor.load_model()

        mock_model.load_state_dict.assert_called_once_with({})

    @patch("drone_detector_mlops.monitoring.embedding_drift.torch.load")
    @patch("drone_detector_mlops.monitoring.embedding_drift.DroneDetectorModel")
    def test_load_model_with_direct_state_dict(self, mock_model_class, mock_torch_load, tmp_path):
        """Test loading model with direct state dict."""
        checkpoint_path = tmp_path / "checkpoint.pth"
        checkpoint_path.touch()

        mock_model = Mock()
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_model_class.return_value = mock_model

        mock_checkpoint = {"layer.weight": torch.randn(10, 10)}
        mock_torch_load.return_value = mock_checkpoint

        extractor = EmbeddingExtractor(model_path=str(checkpoint_path))
        extractor.load_model()

        mock_model.load_state_dict.assert_called_once()

    @patch("drone_detector_mlops.monitoring.embedding_drift.torch.load")
    @patch("drone_detector_mlops.monitoring.embedding_drift.DroneDetectorModel")
    def test_load_model_handles_checkpoint_error(self, mock_model_class, mock_torch_load, tmp_path):
        """Test that checkpoint loading errors are handled gracefully."""
        checkpoint_path = tmp_path / "checkpoint.pth"
        checkpoint_path.touch()

        mock_model = Mock()
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_model_class.return_value = mock_model

        mock_torch_load.side_effect = Exception("Load failed")

        extractor = EmbeddingExtractor(model_path=str(checkpoint_path))
        extractor.load_model()  # Should not raise

        # Model should still be created
        assert extractor.model is not None

    @patch("drone_detector_mlops.monitoring.embedding_drift.get_storage")
    @patch("drone_detector_mlops.monitoring.embedding_drift.torch.load")
    @patch("drone_detector_mlops.monitoring.embedding_drift.DroneDetectorModel")
    def test_load_model_from_storage_local(self, mock_model_class, mock_torch_load, mock_get_storage, tmp_path):
        """Test loading model from local storage."""
        mock_model = Mock()
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_model_class.return_value = mock_model

        mock_storage = Mock()
        mock_storage.mode = "local"
        model_path = tmp_path / "model-latest.pth"
        model_path.touch()
        mock_storage.models_dir = tmp_path
        mock_get_storage.return_value = mock_storage

        mock_checkpoint = {"model_state_dict": {}}
        mock_torch_load.return_value = mock_checkpoint

        extractor = EmbeddingExtractor()  # No model_path
        extractor.load_model()

        mock_torch_load.assert_called_once()

    @patch("drone_detector_mlops.monitoring.embedding_drift.DroneDetectorModel")
    def test_extract_embedding_shape(self, mock_model_class, sample_image):
        """Test that extract_embedding returns correct shape."""
        mock_model = Mock()
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model

        # Mock forward_features to return embeddings
        mock_features = torch.randn(1, 512)
        mock_model.model.forward_features.return_value = mock_features
        mock_model_class.return_value = mock_model

        extractor = EmbeddingExtractor()
        embedding = extractor.extract_embedding(sample_image)

        assert embedding.shape == (512,)
        assert isinstance(embedding, np.ndarray)

    @patch("drone_detector_mlops.monitoring.embedding_drift.DroneDetectorModel")
    def test_extract_embedding_with_4d_features(self, mock_model_class, sample_image):
        """Test extraction when features are 4D (need pooling)."""
        mock_model = Mock()
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model

        # Mock forward_features to return 4D tensor
        mock_features = torch.randn(1, 512, 7, 7)
        mock_model.model.forward_features.return_value = mock_features
        mock_model_class.return_value = mock_model

        extractor = EmbeddingExtractor()
        embedding = extractor.extract_embedding(sample_image)

        assert embedding.shape == (512,)

    @patch("drone_detector_mlops.monitoring.embedding_drift.DroneDetectorModel")
    def test_extract_embeddings_batch(self, mock_model_class):
        """Test batch embedding extraction."""
        mock_model = Mock()
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model

        mock_features = torch.randn(1, 512)
        mock_model.model.forward_features.return_value = mock_features
        mock_model_class.return_value = mock_model

        images = [Image.new("RGB", (224, 224)) for _ in range(3)]
        extractor = EmbeddingExtractor()
        embeddings = extractor.extract_embeddings_batch(images)

        assert embeddings.shape == (3, 512)
        assert isinstance(embeddings, np.ndarray)


class TestEmbeddingDriftMonitor:
    """Tests for EmbeddingDriftMonitor class."""

    def test_initialization(self):
        """Test EmbeddingDriftMonitor initialization."""
        monitor = EmbeddingDriftMonitor()
        assert monitor.embedding_dim == 512
        assert isinstance(monitor.extractor, EmbeddingExtractor)

    def test_initialization_with_model_path(self):
        """Test initialization with model path."""
        monitor = EmbeddingDriftMonitor(model_path="models/test.pth")
        assert monitor.extractor.model_path == "models/test.pth"

    def test_prepare_embedding_dataframe_shape(self, reference_embeddings):
        """Test that prepare_embedding_dataframe creates correct shape."""
        monitor = EmbeddingDriftMonitor()
        df = monitor.prepare_embedding_dataframe(reference_embeddings)

        assert df.shape == (10, 512)
        assert all(col.startswith("emb_") for col in df.columns)

    def test_prepare_embedding_dataframe_with_metadata(self, reference_embeddings):
        """Test adding metadata to embedding dataframe."""
        monitor = EmbeddingDriftMonitor()
        metadata = pd.DataFrame({"class_name": ["drone"] * 10, "confidence": [0.9] * 10})

        df = monitor.prepare_embedding_dataframe(reference_embeddings, metadata)

        assert "class_name" in df.columns
        assert "confidence" in df.columns
        assert df.shape[0] == 10

    def test_prepare_embedding_dataframe_avoids_duplicate_columns(self, reference_embeddings):
        """Test that duplicate columns in metadata are not added."""
        monitor = EmbeddingDriftMonitor()
        metadata = pd.DataFrame({"emb_0": [0.5] * 10, "class_name": ["drone"] * 10})

        df = monitor.prepare_embedding_dataframe(reference_embeddings, metadata)

        # Should use embedding value, not metadata value for emb_0
        assert "class_name" in df.columns
        # emb_0 should come from embeddings, not metadata
        assert df["emb_0"].iloc[0] == reference_embeddings[0, 0]

    def test_generate_embedding_drift_report_returns_report(self, reference_embeddings, current_embeddings_no_drift):
        """Test that generate_embedding_drift_report returns Report object."""
        from evidently.legacy.report import Report

        monitor = EmbeddingDriftMonitor()
        report = monitor.generate_embedding_drift_report(reference_embeddings, current_embeddings_no_drift)

        assert isinstance(report, Report)

    def test_generate_embedding_drift_report_saves_html(
        self, reference_embeddings, current_embeddings_no_drift, tmp_path
    ):
        """Test that report can be saved to HTML."""
        output_path = tmp_path / "embedding_drift.html"

        monitor = EmbeddingDriftMonitor()
        monitor.generate_embedding_drift_report(
            reference_embeddings, current_embeddings_no_drift, output_path=str(output_path)
        )

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_run_embedding_drift_tests_returns_dict(self, reference_embeddings, current_embeddings_no_drift):
        """Test that run_embedding_drift_tests returns dict with required keys."""
        monitor = EmbeddingDriftMonitor()
        results = monitor.run_embedding_drift_tests(reference_embeddings, current_embeddings_no_drift)

        assert isinstance(results, dict)
        assert "all_passed" in results
        assert "summary" in results
        assert "details" in results

    def test_run_embedding_drift_tests_all_passed_is_boolean(self, reference_embeddings, current_embeddings_no_drift):
        """Test that all_passed is a boolean."""
        monitor = EmbeddingDriftMonitor()
        results = monitor.run_embedding_drift_tests(reference_embeddings, current_embeddings_no_drift)

        assert isinstance(results["all_passed"], bool)

    def test_calculate_embedding_statistics_returns_dict(self, reference_embeddings, current_embeddings_no_drift):
        """Test that calculate_embedding_statistics returns dict."""
        monitor = EmbeddingDriftMonitor()
        stats = monitor.calculate_embedding_statistics(reference_embeddings, current_embeddings_no_drift)

        assert isinstance(stats, dict)
        required_keys = [
            "cosine_similarity",
            "l2_distance",
            "reference_mean_variance",
            "current_mean_variance",
            "variance_ratio",
            "significant_drift",
        ]
        for key in required_keys:
            assert key in stats

    def test_calculate_embedding_statistics_values_are_floats(self, reference_embeddings, current_embeddings_no_drift):
        """Test that statistics are float values."""
        monitor = EmbeddingDriftMonitor()
        stats = monitor.calculate_embedding_statistics(reference_embeddings, current_embeddings_no_drift)

        assert isinstance(stats["cosine_similarity"], float)
        assert isinstance(stats["l2_distance"], float)
        assert isinstance(stats["reference_mean_variance"], float)
        assert isinstance(stats["current_mean_variance"], float)
        assert isinstance(stats["variance_ratio"], float)

    def test_calculate_embedding_statistics_cosine_similarity_in_range(
        self, reference_embeddings, current_embeddings_no_drift
    ):
        """Test that cosine similarity is in valid range [-1, 1]."""
        monitor = EmbeddingDriftMonitor()
        stats = monitor.calculate_embedding_statistics(reference_embeddings, current_embeddings_no_drift)

        assert -1 <= stats["cosine_similarity"] <= 1

    def test_calculate_embedding_statistics_detects_no_drift(self, reference_embeddings):
        """Test that no drift is detected for identical embeddings."""
        monitor = EmbeddingDriftMonitor()
        # Use same embeddings
        stats = monitor.calculate_embedding_statistics(reference_embeddings, reference_embeddings.copy())

        assert stats["cosine_similarity"] > 0.99  # Should be close to 1
        assert stats["l2_distance"] < 0.01  # Should be close to 0
        assert not stats["significant_drift"]

    def test_calculate_embedding_statistics_detects_drift(self, reference_embeddings, current_embeddings_with_drift):
        """Test that significant drift is detected."""
        monitor = EmbeddingDriftMonitor()
        stats = monitor.calculate_embedding_statistics(reference_embeddings, current_embeddings_with_drift)

        # Should detect drift due to large difference
        assert stats["significant_drift"]

    def test_calculate_embedding_statistics_l2_distance_positive(
        self, reference_embeddings, current_embeddings_no_drift
    ):
        """Test that L2 distance is positive."""
        monitor = EmbeddingDriftMonitor()
        stats = monitor.calculate_embedding_statistics(reference_embeddings, current_embeddings_no_drift)

        assert stats["l2_distance"] >= 0

    def test_calculate_embedding_statistics_variance_ratio(self, reference_embeddings):
        """Test variance ratio calculation."""
        monitor = EmbeddingDriftMonitor()
        # Create embeddings with different variance
        high_var_embeddings = reference_embeddings * 2

        stats = monitor.calculate_embedding_statistics(reference_embeddings, high_var_embeddings)

        # Variance should be roughly 4x (since we multiplied by 2)
        assert stats["variance_ratio"] > 3.5

    def test_calculate_embedding_statistics_handles_zero_variance(self):
        """Test handling of zero variance in reference."""
        monitor = EmbeddingDriftMonitor()
        zero_var_ref = np.zeros((10, 512), dtype=np.float32)
        current = np.random.randn(10, 512).astype(np.float32)

        stats = monitor.calculate_embedding_statistics(zero_var_ref, current)

        # Should handle division by zero
        assert stats["variance_ratio"] == 0
