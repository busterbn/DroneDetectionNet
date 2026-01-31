import pytest
from unittest.mock import MagicMock, patch
from torch import nn

from drone_detector_mlops.utils.storage import StorageContext


@pytest.fixture
def simple_model():
    """Create a simple PyTorch model for testing."""
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 224 * 224, 2))
    return model


class TestSaveModelCloud:
    """Tests for saving models to GCS."""

    @patch("drone_detector_mlops.utils.storage.settings")
    @patch("drone_detector_mlops.utils.storage.gcs.Client")
    def test_save_cloud_creates_pytorch_blob(self, mock_gcs_client, mock_settings, simple_model):
        """Should upload PyTorch model to GCS."""
        mock_settings.GCS_MODELS_BUCKET = "gs://test-bucket/models"
        mock_settings.GCP_PROJECT = "test-project"
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_gcs_client.return_value.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        storage = StorageContext(mode="cloud")
        result = storage.save_model(simple_model, "test-model")
        mock_bucket.blob.assert_any_call("checkpoints/test-model.pth")
        assert result == "gs://test-bucket/checkpoints/test-model.pth"

    @patch("drone_detector_mlops.utils.storage.settings")
    @patch("drone_detector_mlops.utils.storage.gcs.Client")
    def test_save_cloud_creates_onnx_blob(self, mock_gcs_client, mock_settings, simple_model):
        """Should upload ONNX model to GCS."""
        mock_settings.GCS_MODELS_BUCKET = "gs://test-bucket/models"
        mock_settings.GCP_PROJECT = "test-project"
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_gcs_client.return_value.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        storage = StorageContext(mode="cloud")
        storage.save_model(simple_model, "test-model")
        mock_bucket.blob.assert_any_call("checkpoints/test-model.onnx")

    @patch("drone_detector_mlops.utils.storage.settings")
    @patch("drone_detector_mlops.utils.storage.gcs.Client")
    def test_save_cloud_creates_latest_files(self, mock_gcs_client, mock_settings, simple_model):
        """Should create latest PyTorch and ONNX files."""
        mock_settings.GCS_MODELS_BUCKET = "gs://test-bucket/models"
        mock_settings.GCP_PROJECT = "test-project"
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_gcs_client.return_value.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        storage = StorageContext(mode="cloud")
        storage.save_model(simple_model, "test-model")
        mock_bucket.blob.assert_any_call("checkpoints/model-latest.pth")
        mock_bucket.blob.assert_any_call("checkpoints/model-latest.onnx")


class TestConvertToOnnx:
    """Tests for ONNX conversion."""

    def test_convert_to_onnx_creates_file(self, simple_model, tmp_path):
        """Should create ONNX file."""
        storage = StorageContext(mode="local")
        output_path = tmp_path / "model.onnx"
        storage._convert_to_onnx(simple_model, output_path)
        assert output_path.exists()

    def test_convert_to_onnx_creates_valid_onnx(self, simple_model, tmp_path):
        """Should create valid ONNX model."""
        import onnx

        storage = StorageContext(mode="local")
        output_path = tmp_path / "model.onnx"
        storage._convert_to_onnx(simple_model, output_path)
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)

    def test_convert_to_onnx_sets_correct_input_names(self, simple_model, tmp_path):
        """Should set correct input names in ONNX model."""
        import onnx

        storage = StorageContext(mode="local")
        output_path = tmp_path / "model.onnx"
        storage._convert_to_onnx(simple_model, output_path)
        onnx_model = onnx.load(str(output_path))
        input_names = [input.name for input in onnx_model.graph.input]
        assert "image" in input_names

    def test_convert_to_onnx_sets_correct_output_names(self, simple_model, tmp_path):
        """Should set correct output names in ONNX model."""
        import onnx

        storage = StorageContext(mode="local")
        output_path = tmp_path / "model.onnx"
        storage._convert_to_onnx(simple_model, output_path)
        onnx_model = onnx.load(str(output_path))
        output_names = [output.name for output in onnx_model.graph.output]
        assert "logits" in output_names
