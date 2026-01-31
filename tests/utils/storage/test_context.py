import pytest
from pathlib import Path
from unittest.mock import patch

from drone_detector_mlops.utils.storage import StorageContext, get_storage


class TestStorageContext:
    """Tests for StorageContext class."""

    def test_initialization_local_mode(self):
        """Should initialize with local mode."""
        storage = StorageContext(mode="local")
        assert storage.mode == "local"

    def test_initialization_cloud_mode(self):
        """Should initialize with cloud mode."""
        storage = StorageContext(mode="cloud")
        assert storage.mode == "cloud"

    def test_data_dir_local_mode(self):
        """Should return local path for data_dir in local mode."""
        storage = StorageContext(mode="local")
        assert storage.data_dir == Path("data")

    @patch("drone_detector_mlops.utils.storage.settings")
    def test_data_dir_cloud_mode(self, mock_settings):
        """Should return GCS URI for data_dir in cloud mode."""
        mock_settings.GCS_DATA_PATH = "gs://bucket/data"
        storage = StorageContext(mode="cloud")
        assert storage.data_dir == "gs://bucket/data"

    def test_splits_dir_local_mode(self):
        """Should return local path for splits_dir in local mode."""
        storage = StorageContext(mode="local")
        assert storage.splits_dir == Path("data/splits")

    @patch("drone_detector_mlops.utils.storage.settings")
    def test_splits_dir_cloud_mode(self, mock_settings):
        """Should return GCS URI for splits_dir in cloud mode."""
        mock_settings.GCS_DATA_PATH = "gs://bucket/data"
        storage = StorageContext(mode="cloud")
        assert storage.splits_dir == "gs://bucket/data/splits"

    def test_models_dir_local_mode(self):
        """Should return local path for models_dir in local mode."""
        storage = StorageContext(mode="local")
        assert storage.models_dir == Path("models")

    @patch("drone_detector_mlops.utils.storage.settings")
    def test_models_dir_cloud_mode(self, mock_settings):
        """Should return GCS URI for models_dir in cloud mode."""
        mock_settings.GCS_MODELS_BUCKET = "gs://bucket/models"
        storage = StorageContext(mode="cloud")
        assert storage.models_dir == "gs://bucket/models/checkpoints"

    def test_gcs_client_lazy_loading_local_mode(self):
        """Should not initialize GCS client in local mode."""
        storage = StorageContext(mode="local")
        assert storage.gcs_client is None

    @patch("drone_detector_mlops.utils.storage.gcs.Client")
    @patch("drone_detector_mlops.utils.storage.settings")
    def test_gcs_client_lazy_loading_cloud_mode(self, mock_settings, mock_gcs_client):
        """Should lazily initialize GCS client in cloud mode."""
        mock_settings.GCP_PROJECT = "test-project"
        storage = StorageContext(mode="cloud")

        # Client should not be initialized yet
        assert storage._gcs_client is None

        # Access gcs_client property
        client = storage.gcs_client

        # Client should now be initialized
        mock_gcs_client.assert_called_once_with(project="test-project")
        assert client is not None


class TestGetStorage:
    """Tests for get_storage factory function."""

    @patch("drone_detector_mlops.utils.storage.settings")
    def test_get_storage_local_mode(self, mock_settings):
        """Should return StorageContext with local mode."""
        mock_settings.MODE = "local"

        storage = get_storage()

        assert isinstance(storage, StorageContext)
        assert storage.mode == "local"

    @patch("drone_detector_mlops.utils.storage.settings")
    def test_get_storage_cloud_mode(self, mock_settings):
        """Should return StorageContext with cloud mode."""
        mock_settings.MODE = "cloud"

        storage = get_storage()

        assert isinstance(storage, StorageContext)
        assert storage.mode == "cloud"

    @patch("drone_detector_mlops.utils.storage.settings")
    def test_get_storage_invalid_mode(self, mock_settings):
        """Should raise ValueError for invalid mode."""
        mock_settings.MODE = "invalid"

        with pytest.raises(ValueError, match="Invalid MODE"):
            get_storage()
