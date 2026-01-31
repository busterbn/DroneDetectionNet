import pytest
import torch
from unittest.mock import MagicMock, patch
from torch import nn

from drone_detector_mlops.utils.storage import StorageContext


@pytest.fixture
def simple_model():
    """Create a simple PyTorch model for testing."""
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 224 * 224, 2))
    return model


class TestLoadStateDict:
    """Tests for loading model state dicts."""

    def test_load_state_dict_local_mode(self, simple_model, tmp_path, monkeypatch):
        """Should load state dict from local filesystem."""
        monkeypatch.chdir(tmp_path)
        storage = StorageContext(mode="local")

        # Save model first
        storage.save_model(simple_model, "test-model")

        # Load state dict
        state_dict = storage.load_state_dict("test-model.pth")

        assert isinstance(state_dict, dict)
        assert len(state_dict) > 0

    @patch("drone_detector_mlops.utils.storage.gcs.Client")
    @patch("drone_detector_mlops.utils.storage.settings")
    def test_load_state_dict_cloud_mode(self, mock_settings, mock_gcs_client):
        """Should load state dict from GCS."""
        mock_settings.GCS_MODELS_BUCKET = "gs://test-bucket"

        # Mock GCS client, bucket, and blob
        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_blob = MagicMock()

        mock_gcs_client.return_value = mock_client
        mock_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob

        # Create a simple state dict
        state_dict = {"layer.weight": torch.randn(2, 10)}

        storage = StorageContext(mode="cloud")

        with patch("drone_detector_mlops.utils.storage.torch.load", return_value=state_dict):
            loaded = storage.load_state_dict("test-model.pth")

        assert loaded == state_dict
        mock_client.bucket.assert_called_once_with("test-bucket")
        mock_bucket.blob.assert_called_once_with("checkpoints/test-model.pth")
        mock_blob.download_to_filename.assert_called_once()


class TestLoadOnnxPath:
    """Tests for loading ONNX model paths."""

    def test_load_onnx_path_local_mode(self, simple_model, tmp_path, monkeypatch):
        """Should return local ONNX path."""
        monkeypatch.chdir(tmp_path)
        storage = StorageContext(mode="local")

        # Save model first to create ONNX file
        storage.save_model(simple_model, "test-model")

        # Load ONNX path
        onnx_path = storage.load_onnx_path("model-latest.onnx")

        assert onnx_path.name == "model-latest.onnx"
        assert "models" in str(onnx_path)
        assert onnx_path.exists()

    def test_load_onnx_path_local_file_not_found(self, tmp_path, monkeypatch):
        """Should raise FileNotFoundError if ONNX file doesn't exist."""
        monkeypatch.chdir(tmp_path)
        storage = StorageContext(mode="local")

        with pytest.raises(FileNotFoundError):
            storage.load_onnx_path("nonexistent.onnx")

    @patch("drone_detector_mlops.utils.storage.gcs.Client")
    @patch("drone_detector_mlops.utils.storage.settings")
    def test_load_onnx_path_cloud_mode(self, mock_settings, mock_gcs_client, tmp_path):
        """Should download ONNX from GCS to temp location."""
        mock_settings.GCS_MODELS_BUCKET = "gs://test-bucket"

        # Mock GCS client, bucket, and blob
        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_blob = MagicMock()

        mock_gcs_client.return_value = mock_client
        mock_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob

        storage = StorageContext(mode="cloud")

        with patch("drone_detector_mlops.utils.storage.tempfile.gettempdir", return_value=str(tmp_path)):
            onnx_path = storage.load_onnx_path("model-latest.onnx")

        # Should download to temp location
        assert onnx_path == tmp_path / "model-latest.onnx"
        mock_client.bucket.assert_called_once_with("test-bucket")
        mock_bucket.blob.assert_called_once_with("checkpoints/model-latest.onnx")
        mock_blob.download_to_filename.assert_called_once_with(str(tmp_path / "model-latest.onnx"))
