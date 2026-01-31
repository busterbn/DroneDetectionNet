import pytest
import torch
from torch import nn

from drone_detector_mlops.workflows.training import train_epoch


@pytest.fixture
def device():
    """Device for testing."""
    return torch.device("cpu")


def test_train_epoch_returns_correct_keys(simple_model, simple_dataloader, device):
    """Test that train_epoch returns dict with correct keys."""
    optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    metrics = train_epoch(simple_model, simple_dataloader, optimizer, criterion, device)
    assert "loss" in metrics
    assert "accuracy" in metrics


def test_train_epoch_accuracy_in_range(simple_model, simple_dataloader, device):
    """Test that accuracy is between 0 and 1."""
    optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    metrics = train_epoch(simple_model, simple_dataloader, optimizer, criterion, device)
    assert 0 <= metrics["accuracy"] <= 1


def test_train_epoch_loss_is_positive(simple_model, simple_dataloader, device):
    """Test that loss is positive."""
    optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    metrics = train_epoch(simple_model, simple_dataloader, optimizer, criterion, device)
    assert metrics["loss"] > 0


def test_train_epoch_sets_model_to_train_mode(simple_model, simple_dataloader, device):
    """Test that model is set to train mode."""
    optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    simple_model.eval()
    assert not simple_model.training
    train_epoch(simple_model, simple_dataloader, optimizer, criterion, device)
    assert simple_model.training


def test_train_epoch_updates_model_weights(simple_model, simple_dataloader, device):
    """Test that model weights are updated during training."""
    optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    initial_weights = next(simple_model.parameters()).clone()
    train_epoch(simple_model, simple_dataloader, optimizer, criterion, device)
    final_weights = next(simple_model.parameters())
    assert not torch.equal(initial_weights, final_weights)


def test_train_epoch_with_profiler(simple_model, simple_dataloader, device):
    """Test that train_epoch works with a profiler."""
    from unittest.mock import Mock

    optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    profiler = Mock()

    metrics = train_epoch(simple_model, simple_dataloader, optimizer, criterion, device, profiler=profiler)

    assert "loss" in metrics
    assert "accuracy" in metrics
    # Verify profiler.step() was called for each batch
    assert profiler.step.call_count > 0
