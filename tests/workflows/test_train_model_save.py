from drone_detector_mlops.workflows.train import main


def test_saves_best_model_when_val_loss_improves(mock_training_setup, make_config):
    """Should save model when validation loss improves."""
    mock_val_epoch = mock_training_setup["mock_val_epoch"]
    mock_val_epoch.side_effect = [
        {"loss": 0.5, "accuracy": 0.85},
        {"loss": 0.3, "accuracy": 0.90},
    ]
    cfg = make_config(epochs=2, batch_size=32, lr=0.001)
    main(cfg)
    storage = mock_training_setup["storage"]
    assert storage.save_model.call_count == 2


def test_does_not_save_model_when_val_loss_worsens(mock_training_setup, make_config):
    """Should not save model when validation loss worsens."""
    mock_val_epoch = mock_training_setup["mock_val_epoch"]
    mock_val_epoch.side_effect = [
        {"loss": 0.3, "accuracy": 0.90},
        {"loss": 0.5, "accuracy": 0.85},
    ]
    cfg = make_config(epochs=2, batch_size=32, lr=0.001)
    main(cfg)
    storage = mock_training_setup["storage"]
    assert storage.save_model.call_count == 1
