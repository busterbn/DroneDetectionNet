from drone_detector_mlops.workflows.train import main


def test_initializes_wandb(mock_training_setup, make_config):
    """Should initialize W&B with correct config."""
    cfg = make_config(epochs=5, batch_size=64, lr=0.002)
    main(cfg)
    mock_wandb = mock_training_setup["mock_wandb"]
    mock_wandb.login.assert_called_once()
    mock_wandb.init.assert_called_once()
    init_call = mock_wandb.init.call_args
    config = init_call[1]["config"]
    assert config["epochs"] == 5
    assert config["batch_size"] == 64
    assert config["learning_rate"] == 0.002


def test_logs_metrics_to_wandb(mock_training_setup, make_config):
    """Should log training metrics to W&B."""
    cfg = make_config(epochs=2, batch_size=32, lr=0.001)
    main(cfg)
    mock_wandb = mock_training_setup["mock_wandb"]
    assert mock_wandb.log.call_count >= 2
    first_log = mock_wandb.log.call_args_list[0][0][0]
    assert "train_loss" in first_log
    assert "train_accuracy" in first_log
    assert "val_loss" in first_log
    assert "val_accuracy" in first_log


def test_logs_artifacts_to_wandb_local_mode(mock_training_setup, make_config):
    """Should log model artifacts to W&B in local mode."""
    mock_training_setup["storage"].mode = "local"
    cfg = make_config(epochs=1, batch_size=32, lr=0.001)
    main(cfg)
    mock_wandb = mock_training_setup["mock_wandb"]
    mock_wandb.log_artifact.assert_called()
    artifact_call = mock_wandb.log_artifact.call_args[0][0]
    assert artifact_call.add_file.called


def test_logs_artifacts_to_wandb_cloud_mode(mock_training_setup, make_config):
    """Should log model artifacts to W&B in cloud mode."""
    mock_training_setup["storage"].mode = "cloud"
    cfg = make_config(epochs=1, batch_size=32, lr=0.001)
    main(cfg)
    mock_wandb = mock_training_setup["mock_wandb"]
    mock_wandb.log_artifact.assert_called()
    artifact_call = mock_wandb.log_artifact.call_args[0][0]
    assert artifact_call.add_reference.called


def test_finishes_wandb_run(mock_training_setup, make_config):
    """Should finish W&B run at the end."""
    cfg = make_config(epochs=1, batch_size=32, lr=0.001)
    main(cfg)
    mock_wandb = mock_training_setup["mock_wandb"]
    mock_wandb.finish.assert_called_once()
