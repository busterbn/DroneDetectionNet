import pytest
import torch
from torch import nn

from drone_detector_mlops.workflows.training import setup_training


@pytest.fixture
def device():
    """Device for testing."""
    return torch.device("cpu")


def test_setup_training_returns_three_objects(device):
    """Test that setup_training returns model, optimizer, and criterion."""
    model, optimizer, criterion = setup_training(device)
    assert model is not None
    assert optimizer is not None
    assert criterion is not None


def test_setup_training_model_is_nn_module(device):
    """Test that returned model is a PyTorch nn.Module."""
    model, _, _ = setup_training(device)
    assert isinstance(model, nn.Module)


def test_setup_training_optimizer_is_adam(device):
    """Test that optimizer is Adam."""
    _, optimizer, _ = setup_training(device)
    assert isinstance(optimizer, torch.optim.Adam)


def test_setup_training_criterion_is_crossentropy(device):
    """Test that criterion is CrossEntropyLoss."""
    _, _, criterion = setup_training(device)
    assert isinstance(criterion, nn.CrossEntropyLoss)


def test_setup_training_model_on_correct_device(device):
    """Test that model is on the correct device."""
    model, _, _ = setup_training(device)
    first_param = next(model.parameters())
    assert first_param.device.type == device.type


def test_setup_training_uses_learning_rate(device):
    """Test that learning rate is applied correctly."""
    _, optimizer, _ = setup_training(device, learning_rate=0.01)
    assert optimizer.param_groups[0]["lr"] == 0.01
