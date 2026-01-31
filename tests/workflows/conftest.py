import pytest
from unittest.mock import MagicMock, patch
from torch import nn
from omegaconf import DictConfig


@pytest.fixture
def make_config():
    """Factory fixture to create DictConfig for tests."""

    def _make_config(epochs=1, batch_size=32, lr=0.001, multirun=False, profile=False):
        return DictConfig(
            {
                "training": {
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "lr": lr,
                },
                "multirun": multirun,
                "profile": profile,
            }
        )

    return _make_config


@pytest.fixture
def mock_training_setup():
    """Mock all training dependencies."""
    with (
        patch("drone_detector_mlops.workflows.train.get_storage") as mock_storage,
        patch("drone_detector_mlops.workflows.train.setup_training") as mock_setup,
        patch("drone_detector_mlops.workflows.train.get_dataloaders") as mock_dataloaders,
        patch("drone_detector_mlops.workflows.train.wandb") as mock_wandb,
        patch("drone_detector_mlops.workflows.train.train_epoch") as mock_train_epoch,
        patch("drone_detector_mlops.workflows.train.validate_epoch") as mock_val_epoch,
        patch("drone_detector_mlops.workflows.train.torch.cuda.is_available", return_value=False),
        patch("drone_detector_mlops.workflows.train.torch.backends.mps.is_available", return_value=False),
    ):
        storage = MagicMock()
        storage.mode = "local"
        storage.data_dir = "data"
        storage.splits_dir = "data/splits"
        storage.models_dir = "models"
        storage.save_model.return_value = "models/model-test.pth"
        mock_storage.return_value = storage

        model = MagicMock(spec=nn.Module)
        optimizer = MagicMock()
        criterion = MagicMock()
        mock_setup.return_value = (model, optimizer, criterion)

        train_loader = MagicMock()
        val_loader = MagicMock()
        test_loader = MagicMock()
        mock_dataloaders.return_value = (train_loader, val_loader, test_loader)

        mock_wandb.run = MagicMock()
        mock_wandb.run.project = "test-project"
        mock_wandb.run.id = "test-run-id"

        mock_train_epoch.return_value = {"loss": 0.5, "accuracy": 0.85}
        mock_val_epoch.return_value = {"loss": 0.4, "accuracy": 0.90}

        yield {
            "storage": storage,
            "mock_storage": mock_storage,
            "model": model,
            "optimizer": optimizer,
            "criterion": criterion,
            "mock_setup": mock_setup,
            "train_loader": train_loader,
            "val_loader": val_loader,
            "mock_dataloaders": mock_dataloaders,
            "mock_wandb": mock_wandb,
            "mock_train_epoch": mock_train_epoch,
            "mock_val_epoch": mock_val_epoch,
        }
