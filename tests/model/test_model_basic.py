import pytest
import torch
import torch.nn as nn

from drone_detector_mlops.model import DroneDetectorModel, get_model


def test_drone_detector_model_is_nn_module():
    """Test that DroneDetectorModel is a valid PyTorch nn.Module."""
    model = DroneDetectorModel(num_classes=2, pretrained=False)
    assert isinstance(model, nn.Module)


def test_drone_detector_model_forward_shape():
    """Test that model output has correct shape."""
    model = DroneDetectorModel(num_classes=2, pretrained=False)
    model.eval()
    x = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        output = model(x)
    assert output.shape == (4, 2)


def test_drone_detector_model_custom_num_classes():
    """Test that model respects num_classes parameter."""
    model = DroneDetectorModel(num_classes=5, pretrained=False)
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(x)
    assert output.shape == (1, 5)


def test_get_model_factory():
    """Test that get_model factory function returns DroneDetectorModel."""
    model = get_model(num_classes=2, pretrained=False)
    assert isinstance(model, DroneDetectorModel)


def test_get_model_default_args():
    """Test that get_model works with default arguments."""
    model = get_model(pretrained=False)
    assert isinstance(model, DroneDetectorModel)


def test_drone_detector_model_pretrained_parameter():
    """Test that pretrained parameter is respected."""
    model_pretrained = DroneDetectorModel(num_classes=2, pretrained=True)
    model_not_pretrained = DroneDetectorModel(num_classes=2, pretrained=False)
    assert isinstance(model_pretrained, DroneDetectorModel)
    assert isinstance(model_not_pretrained, DroneDetectorModel)


def test_drone_detector_model_output_is_logits():
    """Test that model output is raw logits (not probabilities)."""
    model = DroneDetectorModel(num_classes=2, pretrained=False)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = model(x)
    row_sums = output.sum(dim=1)
    assert not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_drone_detector_model_different_batch_sizes():
    """Test that model handles different batch sizes."""
    model = DroneDetectorModel(num_classes=2, pretrained=False)
    model.eval()
    batch_sizes = [1, 2, 8, 16, 32]
    for batch_size in batch_sizes:
        x = torch.randn(batch_size, 3, 224, 224)
        with torch.no_grad():
            output = model(x)
        assert output.shape == (batch_size, 2)


def test_drone_detector_model_output_dtype():
    """Test that model output is float tensor."""
    model = DroneDetectorModel(num_classes=2, pretrained=False)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = model(x)
    assert output.dtype == torch.float32


def test_drone_detector_model_wrong_input_shape_raises_error():
    """Test that model raises error for incorrect input shape."""
    model = DroneDetectorModel(num_classes=2, pretrained=False)
    model.eval()
    x_wrong_channels = torch.randn(1, 1, 224, 224)
    with pytest.raises(RuntimeError):
        with torch.no_grad():
            model(x_wrong_channels)


def test_get_model_creates_new_instance():
    """Test that get_model creates new independent instances."""
    model1 = get_model(num_classes=2, pretrained=False)
    model2 = get_model(num_classes=2, pretrained=False)
    assert model1 is not model2
    assert type(model1) is type(model2)


def test_get_model_with_pretrained_true():
    """Test that get_model can create pretrained model."""
    model = get_model(num_classes=2, pretrained=True)
    assert isinstance(model, DroneDetectorModel)
