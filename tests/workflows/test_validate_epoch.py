import pytest
import torch
from torch import nn

from drone_detector_mlops.workflows.training import validate_epoch


@pytest.fixture
def device():
    """Device for testing."""
    return torch.device("cpu")


def test_validate_epoch_returns_correct_keys(simple_model, simple_dataloader, device):
    """Test that validate_epoch returns dict with correct keys."""
    criterion = nn.CrossEntropyLoss()
    metrics = validate_epoch(simple_model, simple_dataloader, criterion, device)
    assert "loss" in metrics
    assert "accuracy" in metrics


def test_validate_epoch_accuracy_in_range(simple_model, simple_dataloader, device):
    """Test that accuracy is between 0 and 1."""
    criterion = nn.CrossEntropyLoss()
    metrics = validate_epoch(simple_model, simple_dataloader, criterion, device)
    assert 0 <= metrics["accuracy"] <= 1


def test_validate_epoch_loss_is_positive(simple_model, simple_dataloader, device):
    """Test that loss is positive."""
    criterion = nn.CrossEntropyLoss()
    metrics = validate_epoch(simple_model, simple_dataloader, criterion, device)
    assert metrics["loss"] > 0


def test_validate_epoch_sets_model_to_eval_mode(simple_model, simple_dataloader, device):
    """Test that model is set to eval mode."""
    criterion = nn.CrossEntropyLoss()
    simple_model.train()
    assert simple_model.training
    validate_epoch(simple_model, simple_dataloader, criterion, device)
    assert not simple_model.training


def test_validate_epoch_does_not_update_weights(simple_model, simple_dataloader, device):
    """Test that model weights are NOT updated during validation."""
    criterion = nn.CrossEntropyLoss()
    initial_weights = next(simple_model.parameters()).clone()
    validate_epoch(simple_model, simple_dataloader, criterion, device)
    final_weights = next(simple_model.parameters())
    assert torch.equal(initial_weights, final_weights)


def test_validate_epoch_no_gradient_computation(simple_model, simple_dataloader, device):
    """Test that gradients are not computed during validation."""
    criterion = nn.CrossEntropyLoss()
    validate_epoch(simple_model, simple_dataloader, criterion, device)
    for param in simple_model.parameters():
        assert param.grad is None
