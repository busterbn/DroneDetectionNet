import pytest

from drone_detector_mlops.data.stats import print_markdown_report, print_console_report


class TestPrintMarkdownReport:
    """Tests for print_markdown_report function."""

    @pytest.fixture
    def mock_datasets(self, tmp_data_dir, sample_images):
        """Create mock datasets for testing."""
        from drone_detector_mlops.data.data import DroneVsBirdDataset

        split_file = tmp_data_dir.parent / "split.txt"
        lines = ["drone/drone_0.jpg", "bird/bird_0.jpg"]
        split_file.write_text("\n".join(lines))
        dataset = DroneVsBirdDataset(tmp_data_dir, split_file, transform=None)
        return dataset, dataset, dataset

    @pytest.fixture
    def img_stats(self):
        """Standard image stats for testing."""
        return {
            "min_width": 100,
            "max_width": 200,
            "avg_width": 150,
            "min_height": 100,
            "max_height": 200,
            "avg_height": 150,
        }

    def test_prints_markdown_headers(self, mock_datasets, img_stats, capsys):
        """Should print markdown formatted headers."""
        train, val, test = mock_datasets
        print_markdown_report(train, val, test, img_stats)
        captured = capsys.readouterr()
        assert "# 📊" in captured.out
        assert "## Split Sizes" in captured.out
        assert "## Class Distribution" in captured.out
        assert "## Image Statistics" in captured.out

    def test_prints_split_sizes_table(self, mock_datasets, img_stats, capsys):
        """Should print split sizes in table format."""
        train, val, test = mock_datasets
        print_markdown_report(train, val, test, img_stats)
        captured = capsys.readouterr()
        assert "| Train |" in captured.out
        assert "| Validation |" in captured.out
        assert "| Test |" in captured.out
        assert "| **Total** |" in captured.out


class TestPrintConsoleReport:
    """Tests for print_console_report function."""

    @pytest.fixture
    def mock_datasets(self, tmp_data_dir, sample_images):
        """Create mock datasets for testing."""
        from drone_detector_mlops.data.data import DroneVsBirdDataset

        split_file = tmp_data_dir.parent / "split.txt"
        lines = ["drone/drone_0.jpg", "bird/bird_0.jpg"]
        split_file.write_text("\n".join(lines))
        dataset = DroneVsBirdDataset(tmp_data_dir, split_file, transform=None)
        return dataset, dataset, dataset

    @pytest.fixture
    def img_stats(self):
        """Standard image stats for testing."""
        return {
            "min_width": 100,
            "max_width": 200,
            "avg_width": 150,
            "min_height": 100,
            "max_height": 200,
            "avg_height": 150,
        }

    def test_prints_console_formatted_output(self, mock_datasets, img_stats, capsys, tmp_data_dir):
        """Should print console formatted output."""
        train, val, test = mock_datasets
        print_console_report(train, val, test, img_stats, tmp_data_dir, tmp_data_dir.parent)
        captured = capsys.readouterr()
        assert "DRONE VS BIRD DATASET STATISTICS" in captured.out
        assert "SPLIT SIZES" in captured.out
        assert "CLASS DISTRIBUTION" in captured.out
        assert "IMAGE STATISTICS" in captured.out

    def test_prints_data_directories(self, mock_datasets, img_stats, capsys, tmp_data_dir):
        """Should print data and splits directory paths."""
        train, val, test = mock_datasets
        splits_dir = tmp_data_dir.parent / "splits"
        print_console_report(train, val, test, img_stats, tmp_data_dir, splits_dir)
        captured = capsys.readouterr()
        assert str(tmp_data_dir) in captured.out
        assert str(splits_dir) in captured.out
