import pytest
from PIL import Image

from drone_detector_mlops.data.stats import plot_comparison_distribution


class TestPlotComparisonDistribution:
    """Tests for plot_comparison_distribution function."""

    @pytest.fixture
    def setup_comparison_datasets(self, tmp_path):
        """Setup before and after datasets for comparison."""
        from drone_detector_mlops.data.data import DroneVsBirdDataset

        before_data = tmp_path / "before"
        after_data = tmp_path / "after"

        for data_dir in [before_data, after_data]:
            (data_dir / "drone").mkdir(parents=True)
            (data_dir / "bird").mkdir(parents=True)

        for i in range(10):
            img = Image.new("RGB", (100, 100))
            img.save(before_data / "drone" / f"drone_{i}.jpg")
            img.save(after_data / "drone" / f"drone_{i}.jpg")
        for i in range(5):
            img = Image.new("RGB", (100, 100))
            img.save(before_data / "bird" / f"bird_{i}.jpg")
            img.save(after_data / "bird" / f"bird_{i}.jpg")

        for i in range(3):
            img = Image.new("RGB", (100, 100))
            img.save(after_data / "drone" / f"new_drone_{i}.jpg")

        def create_datasets(data_dir, prefix):
            splits = {}
            for split_name, drone_range, bird_range in [
                ("Train", range(7), range(3)),
                ("Validation", range(7, 9), range(3, 4)),
                ("Test", range(9, 10), range(4, 5)),
            ]:
                split_file = tmp_path / f"{prefix}_{split_name.lower()}.txt"
                lines = [f"drone/drone_{i}.jpg" for i in drone_range]
                lines += [f"bird/bird_{i}.jpg" for i in bird_range]
                split_file.write_text("\n".join(lines))
                splits[split_name] = DroneVsBirdDataset(data_dir, split_file, transform=None)
            return splits

        before_datasets = create_datasets(before_data, "before")

        for split_name, drone_range, bird_range in [
            ("Train", list(range(7)) + list(range(10, 13)), range(3)),
            ("Validation", range(7, 9), range(3, 4)),
            ("Test", range(9, 10), range(4, 5)),
        ]:
            split_file = tmp_path / f"after_{split_name.lower()}.txt"
            lines = [f"drone/drone_{i}.jpg" for i in drone_range if i < 10]
            lines += [f"drone/new_drone_{i - 10}.jpg" for i in drone_range if i >= 10]
            lines += [f"bird/bird_{i}.jpg" for i in bird_range]
            split_file.write_text("\n".join(lines))

        after_datasets = {}
        for split_name in ["Train", "Validation", "Test"]:
            split_file = tmp_path / f"after_{split_name.lower()}.txt"
            after_datasets[split_name] = DroneVsBirdDataset(after_data, split_file, transform=None)

        return before_datasets, after_datasets

    def test_returns_matplotlib_figure(self, setup_comparison_datasets):
        """Should return a matplotlib Figure object."""
        from matplotlib.figure import Figure

        before_ds, after_ds = setup_comparison_datasets
        fig = plot_comparison_distribution(before_ds, after_ds)
        assert isinstance(fig, Figure)

    def test_creates_three_subplots(self, setup_comparison_datasets):
        """Should create 3 subplots for train/val/test."""
        before_ds, after_ds = setup_comparison_datasets
        fig = plot_comparison_distribution(before_ds, after_ds)
        axes = fig.get_axes()
        assert len(axes) == 3
