import pytest
import torch
from torch import nn

from drone_detector_mlops.utils.storage import StorageContext


@pytest.fixture
def simple_model():
    """Create a simple PyTorch model for testing."""
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 224 * 224, 2))
    return model


class TestSaveModelLocal:
    """Tests for saving models locally."""

    def test_save_local_creates_directory(self, simple_model, tmp_path, monkeypatch):
        """Should create models directory if it doesn't exist."""
        monkeypatch.chdir(tmp_path)
        storage = StorageContext(mode="local")
        storage.save_model(simple_model, "test-model")
        assert (tmp_path / "models").exists()

    def test_save_local_creates_pytorch_file(self, simple_model, tmp_path, monkeypatch):
        """Should create timestamped PyTorch file."""
        monkeypatch.chdir(tmp_path)
        storage = StorageContext(mode="local")
        result = storage.save_model(simple_model, "test-model")
        assert (tmp_path / "models" / "test-model.pth").exists()
        assert result.name == "test-model.pth"
        assert "models" in str(result)

    def test_save_local_creates_latest_pytorch_file(self, simple_model, tmp_path, monkeypatch):
        """Should create latest PyTorch file."""
        monkeypatch.chdir(tmp_path)
        storage = StorageContext(mode="local")
        storage.save_model(simple_model, "test-model")
        assert (tmp_path / "models" / "model-latest.pth").exists()

    def test_save_local_creates_onnx_file(self, simple_model, tmp_path, monkeypatch):
        """Should create timestamped ONNX file."""
        monkeypatch.chdir(tmp_path)
        storage = StorageContext(mode="local")
        storage.save_model(simple_model, "test-model")
        assert (tmp_path / "models" / "test-model.onnx").exists()

    def test_save_local_creates_latest_onnx_file(self, simple_model, tmp_path, monkeypatch):
        """Should create latest ONNX file."""
        monkeypatch.chdir(tmp_path)
        storage = StorageContext(mode="local")
        storage.save_model(simple_model, "test-model")
        assert (tmp_path / "models" / "model-latest.onnx").exists()

    def test_save_local_overwrites_latest_files(self, simple_model, tmp_path, monkeypatch):
        """Should overwrite latest files on subsequent saves."""
        monkeypatch.chdir(tmp_path)
        storage = StorageContext(mode="local")
        storage.save_model(simple_model, "model-v1")
        first_pth_mtime = (tmp_path / "models" / "model-latest.pth").stat().st_mtime
        storage.save_model(simple_model, "model-v2")
        second_pth_mtime = (tmp_path / "models" / "model-latest.pth").stat().st_mtime
        assert second_pth_mtime >= first_pth_mtime

    def test_saved_pytorch_model_can_be_loaded(self, simple_model, tmp_path, monkeypatch):
        """Should be able to load saved PyTorch model."""
        monkeypatch.chdir(tmp_path)
        storage = StorageContext(mode="local")
        storage.save_model(simple_model, "test-model")
        loaded_state = torch.load(tmp_path / "models" / "test-model.pth")
        assert isinstance(loaded_state, dict)
        assert len(loaded_state) > 0
