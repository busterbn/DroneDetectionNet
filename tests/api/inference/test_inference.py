from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np
import pytest
from PIL import Image

from drone_detector_mlops.api.inference import load_model_singleton, predict_image
from drone_detector_mlops.api.schemas import Prediction, PredictionMetadata


class TestLoadModelSingleton:
    """Tests for load_model_singleton function."""

    @pytest.fixture(autouse=True)
    def reset_cache(self):
        """Reset model cache before each test."""
        import drone_detector_mlops.api.inference as inf_module

        inf_module._model_cache = None
        inf_module._model_version = None
        yield
        inf_module._model_cache = None
        inf_module._model_version = None

    @patch("drone_detector_mlops.api.inference.get_storage")
    @patch("drone_detector_mlops.api.inference.ort.InferenceSession")
    def test_loads_model_on_first_call(self, mock_session, mock_get_storage):
        """Test that model is loaded on first call."""
        mock_storage = Mock()
        mock_storage.load_onnx_path.return_value = Path("/tmp/model.onnx")
        mock_get_storage.return_value = mock_storage
        mock_session.return_value = Mock()

        model = load_model_singleton()

        assert model is not None
        mock_storage.load_onnx_path.assert_called_once()
        mock_session.assert_called_once()

    @patch("drone_detector_mlops.api.inference.get_storage")
    @patch("drone_detector_mlops.api.inference.ort.InferenceSession")
    def test_returns_cached_model_on_subsequent_calls(self, mock_session, mock_get_storage):
        """Test that cached model is returned on subsequent calls."""
        mock_storage = Mock()
        mock_storage.load_onnx_path.return_value = Path("/tmp/model.onnx")
        mock_get_storage.return_value = mock_storage
        mock_session.return_value = Mock()

        model1 = load_model_singleton()
        model2 = load_model_singleton()

        assert model1 is model2
        mock_session.assert_called_once()

    @patch("drone_detector_mlops.api.inference.get_storage")
    @patch("drone_detector_mlops.api.inference.ort.InferenceSession")
    def test_sets_model_version(self, mock_session, mock_get_storage):
        """Test that model version is set correctly."""
        from drone_detector_mlops.utils.settings import settings

        mock_storage = Mock()
        mock_storage.load_onnx_path.return_value = Path("/tmp/model.onnx")
        mock_get_storage.return_value = mock_storage
        mock_session.return_value = Mock()

        load_model_singleton()

        import drone_detector_mlops.api.inference as inf_module

        assert inf_module._model_version == settings.MODEL_FILENAME


class TestPredictImage:
    """Tests for predict_image function."""

    @pytest.fixture(autouse=True)
    def reset_cache(self):
        """Reset model cache before each test."""
        import drone_detector_mlops.api.inference as inf_module

        inf_module._model_cache = None
        inf_module._model_version = "test_model.onnx"
        yield
        inf_module._model_cache = None
        inf_module._model_version = None

    @pytest.fixture
    def sample_image(self):
        """Create a sample test image."""
        return Image.new("RGB", (224, 224), color="red")

    @patch("drone_detector_mlops.api.inference.load_model_singleton")
    def test_returns_prediction_and_metadata(self, mock_load_model, sample_image):
        """Test that predict_image returns prediction and metadata."""
        mock_model = Mock()
        mock_input = Mock()
        mock_input.name = "input"
        mock_output = Mock()
        mock_output.name = "output"
        mock_model.get_inputs.return_value = [mock_input]
        mock_model.get_outputs.return_value = [mock_output]
        mock_model.run.return_value = [np.array([[2.0, -1.0]])]
        mock_load_model.return_value = mock_model

        prediction, metadata = predict_image(sample_image)

        assert isinstance(prediction, Prediction)
        assert isinstance(metadata, PredictionMetadata)

    @patch("drone_detector_mlops.api.inference.load_model_singleton")
    def test_predicts_drone_class(self, mock_load_model, sample_image):
        """Test prediction for drone class."""
        mock_model = Mock()
        mock_input = Mock()
        mock_input.name = "input"
        mock_output = Mock()
        mock_output.name = "output"
        mock_model.get_inputs.return_value = [mock_input]
        mock_model.get_outputs.return_value = [mock_output]
        mock_model.run.return_value = [np.array([[2.0, -1.0]])]
        mock_load_model.return_value = mock_model

        prediction, metadata = predict_image(sample_image)

        assert prediction.class_name == "drone"
        assert prediction.confidence > 0.5
        assert prediction.scores.drone > prediction.scores.bird

    @patch("drone_detector_mlops.api.inference.load_model_singleton")
    def test_predicts_bird_class(self, mock_load_model, sample_image):
        """Test prediction for bird class."""
        mock_model = Mock()
        mock_input = Mock()
        mock_input.name = "input"
        mock_output = Mock()
        mock_output.name = "output"
        mock_model.get_inputs.return_value = [mock_input]
        mock_model.get_outputs.return_value = [mock_output]
        mock_model.run.return_value = [np.array([[-1.0, 2.0]])]
        mock_load_model.return_value = mock_model

        prediction, metadata = predict_image(sample_image)

        assert prediction.class_name == "bird"
        assert prediction.confidence > 0.5
        assert prediction.scores.bird > prediction.scores.drone

    @patch("drone_detector_mlops.api.inference.load_model_singleton")
    def test_scores_sum_to_one(self, mock_load_model, sample_image):
        """Test that prediction scores sum to approximately 1."""
        mock_model = Mock()
        mock_input = Mock()
        mock_input.name = "input"
        mock_output = Mock()
        mock_output.name = "output"
        mock_model.get_inputs.return_value = [mock_input]
        mock_model.get_outputs.return_value = [mock_output]
        mock_model.run.return_value = [np.array([[1.5, 0.5]])]
        mock_load_model.return_value = mock_model

        prediction, metadata = predict_image(sample_image)

        total = prediction.scores.drone + prediction.scores.bird
        assert abs(total - 1.0) < 0.001

    @patch("drone_detector_mlops.api.inference.load_model_singleton")
    def test_metadata_contains_inference_time(self, mock_load_model, sample_image):
        """Test that metadata contains inference time."""
        mock_model = Mock()
        mock_input = Mock()
        mock_input.name = "input"
        mock_output = Mock()
        mock_output.name = "output"
        mock_model.get_inputs.return_value = [mock_input]
        mock_model.get_outputs.return_value = [mock_output]
        mock_model.run.return_value = [np.array([[2.0, -1.0]])]
        mock_load_model.return_value = mock_model

        prediction, metadata = predict_image(sample_image)

        assert metadata.inference_time_ms > 0
        assert isinstance(metadata.inference_time_ms, float)

    @patch("drone_detector_mlops.api.inference.load_model_singleton")
    def test_metadata_contains_timestamp(self, mock_load_model, sample_image):
        """Test that metadata contains timestamp."""
        mock_model = Mock()
        mock_input = Mock()
        mock_input.name = "input"
        mock_output = Mock()
        mock_output.name = "output"
        mock_model.get_inputs.return_value = [mock_input]
        mock_model.get_outputs.return_value = [mock_output]
        mock_model.run.return_value = [np.array([[2.0, -1.0]])]
        mock_load_model.return_value = mock_model

        prediction, metadata = predict_image(sample_image)

        assert isinstance(metadata.timestamp, datetime)

    @patch("drone_detector_mlops.api.inference.load_model_singleton")
    def test_handles_different_image_sizes(self, mock_load_model):
        """Test that different image sizes are handled correctly."""
        mock_model = Mock()
        mock_input = Mock()
        mock_input.name = "input"
        mock_output = Mock()
        mock_output.name = "output"
        mock_model.get_inputs.return_value = [mock_input]
        mock_model.get_outputs.return_value = [mock_output]
        mock_model.run.return_value = [np.array([[2.0, -1.0]])]
        mock_load_model.return_value = mock_model

        # Test with different image size
        large_image = Image.new("RGB", (800, 600), color="blue")
        prediction, metadata = predict_image(large_image)

        assert isinstance(prediction, Prediction)
        assert isinstance(metadata, PredictionMetadata)
