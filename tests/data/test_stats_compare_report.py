import pytest
from PIL import Image

from drone_detector_mlops.data.stats import print_comparison_markdown_report


class TestPrintComparisonMarkdownReport:
    """Tests for print_comparison_markdown_report function."""

    @pytest.fixture
    def setup_report_data(self, tmp_path):
        """Setup data for comparison report."""
        from drone_detector_mlops.data.data import DroneVsBirdDataset

        data_dir = tmp_path / "data"
        (data_dir / "drone").mkdir(parents=True)
        (data_dir / "bird").mkdir(parents=True)

        for i in range(6):
            img = Image.new("RGB", (100, 100))
            img.save(data_dir / "drone" / f"drone_{i}.jpg")
        for i in range(4):
            img = Image.new("RGB", (100, 100))
            img.save(data_dir / "bird" / f"bird_{i}.jpg")

        def create_datasets(prefix):
            datasets = {}
            for split_name, indices in [("Train", [0, 1, 2, 3]), ("Validation", [4]), ("Test", [5])]:
                split_file = tmp_path / f"{prefix}_{split_name.lower()}.txt"
                lines = [f"drone/drone_{i}.jpg" for i in indices if i < 6]
                lines += [f"bird/bird_{i}.jpg" for i in indices if i < 4]
                split_file.write_text("\n".join(lines))
                datasets[split_name] = DroneVsBirdDataset(data_dir, split_file, transform=None)
            return datasets

        before_datasets = create_datasets("before")
        after_datasets = create_datasets("after")

        img_stats = {
            "min_width": 100,
            "max_width": 100,
            "avg_width": 100,
            "min_height": 100,
            "max_height": 100,
            "avg_height": 100,
        }

        return {
            "before_datasets": before_datasets,
            "after_datasets": after_datasets,
            "before_img_stats": img_stats,
            "after_img_stats": img_stats,
            "new_files": ["drone/new.jpg"],
            "before_dir_counts": {"drone": 5, "bird": 3},
            "after_dir_counts": {"drone": 6, "bird": 4},
            "new_images_in_dirs": {"drone": ["new_drone.jpg"], "bird": ["new_bird.jpg"]},
        }

    def test_prints_markdown_headers(self, setup_report_data, capsys):
        """Should print markdown formatted headers."""
        print_comparison_markdown_report(**setup_report_data)
        captured = capsys.readouterr()
        assert "# 📊 Drone vs Bird Dataset Changes" in captured.out
        assert "## Raw Data Changes" in captured.out
        assert "## Split Sizes" in captured.out

    def test_prints_new_images_section(self, setup_report_data, capsys):
        """Should list new images when present."""
        print_comparison_markdown_report(**setup_report_data)
        captured = capsys.readouterr()
        assert "New Images Added" in captured.out

    def test_handles_no_new_files(self, setup_report_data, capsys):
        """Should handle case with no new files."""
        setup_report_data["new_files"] = []
        setup_report_data["new_images_in_dirs"] = {"drone": [], "bird": []}
        print_comparison_markdown_report(**setup_report_data)
        captured = capsys.readouterr()
        assert "Drone vs Bird Dataset Changes" in captured.out

    def test_handles_many_new_files(self, setup_report_data, capsys):
        """Should truncate display when many new files."""
        setup_report_data["new_files"] = [f"drone/img_{i}.jpg" for i in range(25)]
        print_comparison_markdown_report(**setup_report_data)
        captured = capsys.readouterr()
        assert "... and" in captured.out
        assert "more" in captured.out

    def test_truncates_new_images_per_class(self, setup_report_data, capsys):
        """Should truncate when more than 5 new images per class."""
        setup_report_data["new_images_in_dirs"] = {
            "drone": [f"new_drone_{i}.jpg" for i in range(8)],
            "bird": [f"new_bird_{i}.jpg" for i in range(3)],
        }
        print_comparison_markdown_report(**setup_report_data)
        captured = capsys.readouterr()
        assert "**drone:** 8 new images" in captured.out
        assert "... and 3 more" in captured.out
