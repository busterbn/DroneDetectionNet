import pytest
import torch
from unittest.mock import patch, MagicMock
from torch import nn
from typer.testing import CliRunner

from drone_detector_mlops.workflows.test import main, app
from drone_detector_mlops.workflows.testing import evaluate_model


@pytest.fixture
def runner():
    """Typer CLI test runner."""
    return CliRunner()


@pytest.fixture
def mock_model():
    """Mock PyTorch model."""
    model = MagicMock(spec=nn.Module)
    model.to.return_value = model
    return model


@pytest.fixture
def mock_metrics():
    """Mock evaluation metrics."""
    return {
        "loss": 0.1234,
        "accuracy": 0.9500,
        "drone_accuracy": 0.9600,
        "bird_accuracy": 0.9400,
    }


def test_main_function_runs(mock_model, mock_metrics):
    """Test that main function executes without errors."""
    with (
        patch("drone_detector_mlops.workflows.test.torch.cuda.is_available", return_value=False),
        patch("drone_detector_mlops.workflows.test.torch.backends.mps.is_available", return_value=False),
        patch("drone_detector_mlops.workflows.test.get_model", return_value=mock_model),
        patch("drone_detector_mlops.workflows.test.torch.load"),
        patch("drone_detector_mlops.workflows.test.get_dataloaders") as mock_dataloaders,
        patch("drone_detector_mlops.workflows.test.evaluate_model", return_value=mock_metrics),
    ):
        mock_dataloaders.return_value = (None, None, MagicMock())
        main()
        mock_model.load_state_dict.assert_called_once()


def test_main_calls_evaluate_model(mock_model, mock_metrics):
    """Test that evaluate_model is called."""
    with (
        patch("drone_detector_mlops.workflows.test.torch.cuda.is_available", return_value=False),
        patch("drone_detector_mlops.workflows.test.torch.backends.mps.is_available", return_value=False),
        patch("drone_detector_mlops.workflows.test.get_model", return_value=mock_model),
        patch("drone_detector_mlops.workflows.test.torch.load"),
        patch("drone_detector_mlops.workflows.test.get_dataloaders") as mock_dataloaders,
        patch("drone_detector_mlops.workflows.test.evaluate_model") as mock_evaluate,
    ):
        mock_dataloaders.return_value = (None, None, MagicMock())
        mock_evaluate.return_value = mock_metrics
        main()
        mock_evaluate.assert_called_once()


def test_cli_runs_successfully(runner, mock_model, mock_metrics):
    """Test that CLI app runs without errors."""
    with (
        patch("drone_detector_mlops.workflows.test.torch.cuda.is_available", return_value=False),
        patch("drone_detector_mlops.workflows.test.torch.backends.mps.is_available", return_value=False),
        patch("drone_detector_mlops.workflows.test.get_model", return_value=mock_model),
        patch("drone_detector_mlops.workflows.test.torch.load"),
        patch("drone_detector_mlops.workflows.test.get_dataloaders") as mock_dataloaders,
        patch("drone_detector_mlops.workflows.test.evaluate_model", return_value=mock_metrics),
    ):
        mock_dataloaders.return_value = (None, None, MagicMock())
        result = runner.invoke(app, [])
        assert result.exit_code == 0


def test_evaluate_model_returns_correct_keys(simple_model, simple_dataloader):
    """Test that evaluate_model returns dict with correct keys."""
    device = torch.device("cpu")
    criterion = nn.CrossEntropyLoss()
    metrics = evaluate_model(simple_model, simple_dataloader, criterion, device)
    assert "loss" in metrics
    assert "accuracy" in metrics
    assert "drone_accuracy" in metrics
    assert "bird_accuracy" in metrics


def test_evaluate_model_accuracy_values_in_range(simple_model, simple_dataloader):
    """Test that accuracy values are between 0 and 1."""
    device = torch.device("cpu")
    criterion = nn.CrossEntropyLoss()
    metrics = evaluate_model(simple_model, simple_dataloader, criterion, device)
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["drone_accuracy"] <= 1
    assert 0 <= metrics["bird_accuracy"] <= 1


def test_evaluate_model_sets_eval_mode(simple_model, simple_dataloader):
    """Test that model is set to eval mode."""
    device = torch.device("cpu")
    criterion = nn.CrossEntropyLoss()
    simple_model.train()
    assert simple_model.training
    evaluate_model(simple_model, simple_dataloader, criterion, device)
    assert not simple_model.training


def test_evaluate_model_loss_is_positive(simple_model, simple_dataloader):
    """Test that loss is a positive number."""
    device = torch.device("cpu")
    criterion = nn.CrossEntropyLoss()
    metrics = evaluate_model(simple_model, simple_dataloader, criterion, device)
    assert metrics["loss"] > 0
