import pytest
from PIL import Image

from drone_detector_mlops.data.stats import show_sample_images, show_before_and_new_images


class TestShowSampleImages:
    """Tests for show_sample_images function."""

    @pytest.fixture
    def mock_dataset(self, tmp_data_dir, sample_images):
        """Create a mock dataset with known images."""
        from drone_detector_mlops.data.data import DroneVsBirdDataset

        split_file = tmp_data_dir.parent / "split.txt"
        lines = [f"drone/drone_{i}.jpg" for i in range(10)]
        split_file.write_text("\n".join(lines))
        return DroneVsBirdDataset(tmp_data_dir, split_file, transform=None)

    def test_returns_matplotlib_figure(self, mock_dataset):
        """Should return a matplotlib Figure object."""
        from matplotlib.figure import Figure

        fig = show_sample_images(mock_dataset, n_samples=4)
        assert isinstance(fig, Figure)

    def test_creates_correct_number_of_subplots(self, mock_dataset):
        """Should create grid with correct layout."""
        fig = show_sample_images(mock_dataset, n_samples=8)
        axes = fig.get_axes()
        assert len(axes) == 8

    def test_handles_fewer_samples_than_requested(self, tmp_data_dir, tmp_path):
        """Should handle dataset smaller than requested samples."""
        from drone_detector_mlops.data.data import DroneVsBirdDataset

        (tmp_data_dir / "drone").mkdir(parents=True, exist_ok=True)
        img = Image.new("RGB", (100, 100), color=(255, 0, 0))
        img.save(tmp_data_dir / "drone" / "only_one.jpg")

        split_file = tmp_path / "split.txt"
        split_file.write_text("drone/only_one.jpg")
        dataset = DroneVsBirdDataset(tmp_data_dir, split_file, transform=None)

        fig = show_sample_images(dataset, n_samples=4)
        assert fig is not None

    def test_hides_empty_subplots(self, mock_dataset):
        """Should hide empty subplots when n_samples doesn't fill grid."""
        fig = show_sample_images(mock_dataset, n_samples=5)
        axes = fig.get_axes()
        assert len(axes) == 8
        assert not axes[5].axison
        assert not axes[6].axison
        assert not axes[7].axison


class TestShowBeforeAndNewImages:
    """Tests for show_before_and_new_images function."""

    @pytest.fixture
    def setup_datasets(self, tmp_path):
        """Setup before and after datasets."""
        from drone_detector_mlops.data.data import DroneVsBirdDataset

        before_data = tmp_path / "before_data"
        after_data = tmp_path / "after_data"
        (before_data / "drone").mkdir(parents=True)
        (before_data / "bird").mkdir(parents=True)
        (after_data / "drone").mkdir(parents=True)
        (after_data / "bird").mkdir(parents=True)

        for i in range(5):
            img = Image.new("RGB", (100, 100), color=(255, 0, 0))
            img.save(before_data / "drone" / f"drone_{i}.jpg")
            img.save(after_data / "drone" / f"drone_{i}.jpg")

        for i in range(3):
            img = Image.new("RGB", (100, 100), color=(0, 0, 255))
            img.save(before_data / "bird" / f"bird_{i}.jpg")
            img.save(after_data / "bird" / f"bird_{i}.jpg")

        img = Image.new("RGB", (100, 100), color=(0, 255, 0))
        img.save(after_data / "drone" / "new_drone.jpg")
        img.save(after_data / "bird" / "new_bird.jpg")

        before_split = tmp_path / "before_split.txt"
        after_split = tmp_path / "after_split.txt"
        before_split.write_text("\n".join([f"drone/drone_{i}.jpg" for i in range(5)]))
        after_split.write_text("\n".join([f"drone/drone_{i}.jpg" for i in range(5)] + ["drone/new_drone.jpg"]))

        before_dataset = DroneVsBirdDataset(before_data, before_split, transform=None)
        after_dataset = DroneVsBirdDataset(after_data, after_split, transform=None)

        return before_dataset, after_dataset, after_data

    def test_returns_matplotlib_figure(self, setup_datasets):
        """Should return a matplotlib Figure object."""
        from matplotlib.figure import Figure

        before_ds, after_ds, after_data = setup_datasets
        new_files = ["drone/new_drone.jpg"]
        fig = show_before_and_new_images(before_ds, after_ds, new_files, after_data, n_before=4, n_new=4)
        assert isinstance(fig, Figure)

    def test_handles_no_new_images(self, setup_datasets):
        """Should handle case with no new images."""
        from matplotlib.figure import Figure

        before_ds, after_ds, after_data = setup_datasets
        fig = show_before_and_new_images(before_ds, after_ds, [], after_data, n_before=4, n_new=4)
        assert isinstance(fig, Figure)

    def test_handles_missing_new_image_file(self, setup_datasets):
        """Should handle missing image files gracefully."""
        before_ds, after_ds, after_data = setup_datasets
        new_files = ["drone/nonexistent.jpg"]
        fig = show_before_and_new_images(before_ds, after_ds, new_files, after_data, n_before=2, n_new=2)
        assert fig is not None

    def test_handles_single_row_layout(self, setup_datasets):
        """Should handle single row when very few samples requested."""
        before_ds, after_ds, after_data = setup_datasets
        fig = show_before_and_new_images(before_ds, after_ds, [], after_data, n_before=2, n_new=0)
        assert fig is not None
        assert len(fig.get_axes()) == 8

    def test_handles_edge_case_single_row(self, setup_datasets):
        """Should handle edge case with exactly one row total."""
        before_ds, after_ds, after_data = setup_datasets
        # n_before=0 gives before_rows=0, n_new=0 gives new_rows=1, total=1
        fig = show_before_and_new_images(before_ds, after_ds, [], after_data, n_before=0, n_new=0)
        assert fig is not None
        # Should create 1 row x 4 cols = 4 axes
        assert len(fig.get_axes()) == 4
