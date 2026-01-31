import pytest
from unittest.mock import patch
from PIL import Image

from drone_detector_mlops.data.stats import get_image_stats, plot_label_distribution


class TestGetImageStats:
    """Tests for get_image_stats function."""

    @pytest.fixture
    def mock_dataset(self, tmp_data_dir, sample_images):
        """Create a mock dataset with known image sizes."""
        from drone_detector_mlops.data.data import DroneVsBirdDataset

        split_file = tmp_data_dir.parent / "split.txt"
        lines = [f"drone/drone_{i}.jpg" for i in range(10)]
        split_file.write_text("\n".join(lines))
        return DroneVsBirdDataset(tmp_data_dir, split_file, transform=None)

    def test_returns_dict_with_correct_keys(self, mock_dataset):
        """Should return dict with min, max, avg for width and height."""
        stats = get_image_stats(mock_dataset, n_samples=5)
        assert "min_width" in stats
        assert "max_width" in stats
        assert "avg_width" in stats
        assert "min_height" in stats
        assert "max_height" in stats
        assert "avg_height" in stats

    def test_all_same_size_images(self, mock_dataset):
        """Should handle all images having the same size."""
        stats = get_image_stats(mock_dataset, n_samples=5)
        assert stats["min_width"] == 100
        assert stats["max_width"] == 100
        assert stats["avg_width"] == 100
        assert stats["min_height"] == 100
        assert stats["max_height"] == 100
        assert stats["avg_height"] == 100

    def test_respects_n_samples_parameter(self, mock_dataset):
        """Should sample at most n_samples images."""
        with patch.object(mock_dataset, "_build_image_path") as mock_build:
            with patch.object(mock_dataset, "_open_image") as mock_open:
                mock_open.return_value = Image.new("RGB", (100, 100))
                get_image_stats(mock_dataset, n_samples=3)
                assert mock_build.call_count <= 3


class TestPlotLabelDistribution:
    """Tests for plot_label_distribution function."""

    def test_returns_matplotlib_figure(self):
        """Should return a matplotlib Figure object."""
        labels = [0, 0, 1, 1, 1]
        fig = plot_label_distribution(labels, "Test Distribution")
        from matplotlib.figure import Figure

        assert isinstance(fig, Figure)

    def test_handles_all_same_label(self):
        """Should handle all labels being the same class."""
        labels = [0, 0, 0, 0]
        fig = plot_label_distribution(labels, "All Drones")
        assert fig is not None

    def test_handles_single_label(self):
        """Should handle single label."""
        labels = [0]
        fig = plot_label_distribution(labels, "Single Label")
        assert fig is not None
