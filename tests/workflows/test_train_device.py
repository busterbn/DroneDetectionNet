from unittest.mock import patch

from drone_detector_mlops.workflows.train import main


def test_main_function_runs_successfully(mock_training_setup, make_config):
    """Should run training workflow without errors."""
    cfg = make_config(epochs=2, batch_size=32, lr=0.001)
    main(cfg)
    assert mock_training_setup["mock_train_epoch"].called
    assert mock_training_setup["mock_val_epoch"].called


def test_uses_correct_device_cpu(mock_training_setup, make_config):
    """Should use CPU when CUDA and MPS are unavailable."""
    cfg = make_config(epochs=1, batch_size=32, lr=0.001)
    main(cfg)
    call_args = mock_training_setup["mock_setup"].call_args
    device = call_args[0][0]
    assert device.type == "cpu"


@patch("drone_detector_mlops.workflows.train.torch.cuda.is_available", return_value=True)
def test_uses_cuda_when_available(mock_cuda, mock_training_setup, make_config):
    """Should use CUDA when available."""
    cfg = make_config(epochs=1, batch_size=32, lr=0.001)
    main(cfg)
    call_args = mock_training_setup["mock_setup"].call_args
    device = call_args[0][0]
    assert device.type == "cuda"


@patch("drone_detector_mlops.workflows.train.torch.cuda.is_available", return_value=False)
@patch("drone_detector_mlops.workflows.train.torch.backends.mps.is_available", return_value=True)
def test_uses_mps_when_available(mock_mps, mock_cuda, mock_training_setup, make_config):
    """Should use MPS when CUDA unavailable but MPS available."""
    cfg = make_config(epochs=1, batch_size=32, lr=0.001)
    main(cfg)
    call_args = mock_training_setup["mock_setup"].call_args
    device = call_args[0][0]
    assert device.type == "mps"


def test_gets_dataloaders_with_correct_params(mock_training_setup, make_config):
    """Should get dataloaders with correct parameters."""
    cfg = make_config(epochs=1, batch_size=16, lr=0.001)
    main(cfg)
    mock_dataloaders = mock_training_setup["mock_dataloaders"]
    mock_dataloaders.assert_called_once()
    call_kwargs = mock_dataloaders.call_args[1]
    assert call_kwargs["batch_size"] == 16
    assert "transforms_dict" in call_kwargs


def test_runs_correct_number_of_epochs(mock_training_setup, make_config):
    """Should run training for specified number of epochs."""
    cfg = make_config(epochs=3, batch_size=32, lr=0.001)
    main(cfg)
    assert mock_training_setup["mock_train_epoch"].call_count == 3
    assert mock_training_setup["mock_val_epoch"].call_count == 3


def test_passes_learning_rate_to_setup(mock_training_setup, make_config):
    """Should pass learning rate to setup_training."""
    cfg = make_config(epochs=1, batch_size=32, lr=0.005)
    main(cfg)
    mock_setup = mock_training_setup["mock_setup"]
    call_args = mock_setup.call_args
    lr = call_args[0][1]
    assert lr == 0.005


def test_calls_storage_in_correct_mode(mock_training_setup, make_config):
    """Should initialize storage context correctly."""
    cfg = make_config(epochs=1, batch_size=32, lr=0.001)
    main(cfg)
    mock_storage = mock_training_setup["mock_storage"]
    mock_storage.assert_called_once()
