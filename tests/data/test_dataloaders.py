import pytest
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler

from drone_detector_mlops.data.data import get_dataloaders


class TestGetDataloaders:
    """Tests for get_dataloaders function."""

    @pytest.fixture
    def setup_splits(self, tmp_data_dir, tmp_splits_dir, sample_images):
        """Create split files for testing."""
        for split in ["train", "val", "test"]:
            split_file = tmp_splits_dir / f"{split}_files.txt"
            lines = ["drone/drone_0.jpg", "bird/bird_0.jpg"]
            split_file.write_text("\n".join(lines))
        return tmp_splits_dir

    def test_returns_three_dataloaders(self, tmp_data_dir, setup_splits):
        """Should return train, val, and test dataloaders."""
        train, val, test = get_dataloaders(
            data_dir=tmp_data_dir,
            splits_dir=setup_splits,
            batch_size=1,
            num_workers=0,
        )

        assert isinstance(train, DataLoader)
        assert isinstance(val, DataLoader)
        assert isinstance(test, DataLoader)

    def test_train_loader_shuffles(self, tmp_data_dir, setup_splits):
        """Train loader should shuffle data."""
        train, _, _ = get_dataloaders(
            data_dir=tmp_data_dir,
            splits_dir=setup_splits,
            batch_size=1,
            num_workers=0,
        )

        assert isinstance(train.sampler, RandomSampler)

    def test_val_loader_does_not_shuffle(self, tmp_data_dir, setup_splits):
        """Val loader should not shuffle data."""
        _, val, _ = get_dataloaders(
            data_dir=tmp_data_dir,
            splits_dir=setup_splits,
            batch_size=1,
            num_workers=0,
        )

        assert isinstance(val.sampler, SequentialSampler)

    def test_test_loader_does_not_shuffle(self, tmp_data_dir, setup_splits):
        """Test loader should not shuffle data."""
        _, _, test = get_dataloaders(
            data_dir=tmp_data_dir,
            splits_dir=setup_splits,
            batch_size=1,
            num_workers=0,
        )

        assert isinstance(test.sampler, SequentialSampler)

    def test_batch_size_is_respected(self, tmp_data_dir, setup_splits):
        """Batch size should be applied correctly."""
        batch_size = 2
        train, val, test = get_dataloaders(
            data_dir=tmp_data_dir,
            splits_dir=setup_splits,
            batch_size=batch_size,
            num_workers=0,
        )

        assert train.batch_size == batch_size
        assert val.batch_size == batch_size
        assert test.batch_size == batch_size
