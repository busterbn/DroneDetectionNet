from drone_detector_mlops.utils.settings import Settings, settings


def test_settings_instance_exists():
    """Test that settings singleton is created."""
    assert settings is not None
    assert isinstance(settings, Settings)


def test_settings_default_values():
    """Test that default values are correct."""
    assert settings.RANDOM_SEED == 42
    assert settings.IMAGENET_MEAN == [0.485, 0.456, 0.406]
    assert settings.IMAGENET_STD == [0.229, 0.224, 0.225]


def test_settings_can_be_instantiated():
    """Test that Settings class can create new instances."""
    new_settings = Settings()
    assert new_settings.RANDOM_SEED == 42
