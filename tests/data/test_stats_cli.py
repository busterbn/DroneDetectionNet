import pytest
from PIL import Image
from typer.testing import CliRunner

from drone_detector_mlops.data.stats import app


@pytest.fixture
def cli_runner():
    """Typer CLI test runner."""
    return CliRunner()


@pytest.fixture
def setup_test_data(tmp_path):
    """Create test data structure with images and splits."""
    data_dir = tmp_path / "data"
    splits_dir = tmp_path / "splits"
    output_dir = tmp_path / "output"

    (data_dir / "drone").mkdir(parents=True)
    (data_dir / "bird").mkdir(parents=True)
    splits_dir.mkdir()

    for i in range(10):
        img = Image.new("RGB", (100, 100), color=(255, 0, 0))
        img.save(data_dir / "drone" / f"drone_{i}.jpg")
    for i in range(10):
        img = Image.new("RGB", (100, 100), color=(0, 0, 255))
        img.save(data_dir / "bird" / f"bird_{i}.jpg")

    train = [f"drone/drone_{i}.jpg" for i in range(7)] + [f"bird/bird_{i}.jpg" for i in range(7)]
    val = [f"drone/drone_{i}.jpg" for i in range(7, 9)] + [f"bird/bird_{i}.jpg" for i in range(7, 9)]
    test = ["drone/drone_9.jpg", "bird/bird_9.jpg"]

    (splits_dir / "train_files.txt").write_text("\n".join(train))
    (splits_dir / "val_files.txt").write_text("\n".join(val))
    (splits_dir / "test_files.txt").write_text("\n".join(test))

    return data_dir, splits_dir, output_dir


class TestDatasetStatisticsCLI:
    """Tests for dataset_statistics CLI command."""

    def test_cli_runs_successfully(self, cli_runner, setup_test_data):
        """Should run without errors."""
        data_dir, splits_dir, output_dir = setup_test_data
        result = cli_runner.invoke(
            app,
            [
                "--data-dir",
                str(data_dir),
                "--splits-dir",
                str(splits_dir),
                "--output-dir",
                str(output_dir),
            ],
        )
        assert result.exit_code == 0

    def test_cli_creates_output_figures(self, cli_runner, setup_test_data):
        """Should create output figures."""
        data_dir, splits_dir, output_dir = setup_test_data
        cli_runner.invoke(
            app,
            [
                "--data-dir",
                str(data_dir),
                "--splits-dir",
                str(splits_dir),
                "--output-dir",
                str(output_dir),
            ],
        )
        assert (output_dir / "sample_images.png").exists()
        assert (output_dir / "train_label_distribution.png").exists()
        assert (output_dir / "all_splits_distribution.png").exists()

    def test_cli_markdown_mode(self, cli_runner, setup_test_data):
        """Should output markdown format when --markdown flag used."""
        data_dir, splits_dir, output_dir = setup_test_data
        result = cli_runner.invoke(
            app,
            [
                "--data-dir",
                str(data_dir),
                "--splits-dir",
                str(splits_dir),
                "--output-dir",
                str(output_dir),
                "--markdown",
            ],
        )
        assert result.exit_code == 0
        assert "# 📊" in result.stdout
        assert "## Split Sizes" in result.stdout

    def test_cli_console_output(self, cli_runner, setup_test_data):
        """Should print console formatted output by default."""
        data_dir, splits_dir, output_dir = setup_test_data
        result = cli_runner.invoke(
            app,
            [
                "--data-dir",
                str(data_dir),
                "--splits-dir",
                str(splits_dir),
                "--output-dir",
                str(output_dir),
            ],
        )
        assert "DRONE VS BIRD DATASET STATISTICS" in result.stdout
        assert "SPLIT SIZES" in result.stdout


class TestDatasetStatisticsComparisonCLI:
    """Tests for dataset_statistics CLI in comparison mode."""

    @pytest.fixture
    def setup_comparison_data(self, tmp_path):
        """Create before/after test data for comparison mode."""
        before_data = tmp_path / "before_data"
        after_data = tmp_path / "after_data"
        before_splits = tmp_path / "before_splits"
        after_splits = tmp_path / "after_splits"
        output_dir = tmp_path / "output"

        for data_dir in [before_data, after_data]:
            (data_dir / "drone").mkdir(parents=True)
            (data_dir / "bird").mkdir(parents=True)

        for splits_dir in [before_splits, after_splits]:
            splits_dir.mkdir()

        for i in range(8):
            img = Image.new("RGB", (100, 100))
            img.save(before_data / "drone" / f"drone_{i}.jpg")
            img.save(after_data / "drone" / f"drone_{i}.jpg")
        for i in range(6):
            img = Image.new("RGB", (100, 100))
            img.save(before_data / "bird" / f"bird_{i}.jpg")
            img.save(after_data / "bird" / f"bird_{i}.jpg")

        for i in range(2):
            img = Image.new("RGB", (100, 100))
            img.save(after_data / "drone" / f"new_drone_{i}.jpg")

        before_train = [f"drone/drone_{i}.jpg" for i in range(6)] + [f"bird/bird_{i}.jpg" for i in range(4)]
        before_val = ["drone/drone_6.jpg", "bird/bird_4.jpg"]
        before_test = ["drone/drone_7.jpg", "bird/bird_5.jpg"]

        after_train = before_train + ["drone/new_drone_0.jpg", "drone/new_drone_1.jpg"]
        after_val = before_val
        after_test = before_test

        (before_splits / "train_files.txt").write_text("\n".join(before_train))
        (before_splits / "val_files.txt").write_text("\n".join(before_val))
        (before_splits / "test_files.txt").write_text("\n".join(before_test))

        (after_splits / "train_files.txt").write_text("\n".join(after_train))
        (after_splits / "val_files.txt").write_text("\n".join(after_val))
        (after_splits / "test_files.txt").write_text("\n".join(after_test))

        return before_data, after_data, before_splits, after_splits, output_dir

    def test_comparison_mode_runs(self, cli_runner, setup_comparison_data):
        """Should run comparison mode without errors."""
        before_data, after_data, before_splits, after_splits, output_dir = setup_comparison_data
        result = cli_runner.invoke(
            app,
            [
                "--data-dir",
                str(after_data),
                "--splits-dir",
                str(after_splits),
                "--output-dir",
                str(output_dir),
                "--before-data-dir",
                str(before_data),
                "--before-splits-dir",
                str(before_splits),
            ],
        )
        assert result.exit_code == 0

    def test_comparison_mode_markdown(self, cli_runner, setup_comparison_data):
        """Should output comparison markdown report."""
        before_data, after_data, before_splits, after_splits, output_dir = setup_comparison_data
        result = cli_runner.invoke(
            app,
            [
                "--data-dir",
                str(after_data),
                "--splits-dir",
                str(after_splits),
                "--output-dir",
                str(output_dir),
                "--before-data-dir",
                str(before_data),
                "--before-splits-dir",
                str(before_splits),
                "--markdown",
            ],
        )
        assert result.exit_code == 0
        assert "Dataset Changes" in result.stdout
        assert "Raw Data Changes" in result.stdout
