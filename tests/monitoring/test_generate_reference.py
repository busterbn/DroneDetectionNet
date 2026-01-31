"""Tests for generate_reference_dataset script."""

import json

import pandas as pd
import pytest
from PIL import Image

from drone_detector_mlops.monitoring.generate_reference_dataset import generate_reference_dataset


@pytest.fixture
def setup_split_data(tmp_path):
    """Create test data structure with split file."""
    data_dir = tmp_path / "data"
    drone_dir = data_dir / "drone"
    bird_dir = data_dir / "bird"
    drone_dir.mkdir(parents=True)
    bird_dir.mkdir(parents=True)

    # Create some test images
    for i in range(5):
        img = Image.new("RGB", (100, 100), color=(255, 0, 0))
        img.save(drone_dir / f"drone_{i}.jpg")

    for i in range(3):
        img = Image.new("RGB", (100, 100), color=(0, 0, 255))
        img.save(bird_dir / f"bird_{i}.jpg")

    # Create split file
    split_file = data_dir / "train_split.txt"
    image_paths = [f"drone/drone_{i}.jpg" for i in range(5)] + [f"bird/bird_{i}.jpg" for i in range(3)]
    split_file.write_text("\n".join(image_paths))

    return data_dir


class TestGenerateReferenceDataset:
    """Tests for generate_reference_dataset function."""

    def test_creates_reference_csv(self, setup_split_data, tmp_path):
        """Test that reference dataset CSV is created."""
        output_path = tmp_path / "reference.csv"

        generate_reference_dataset(
            data_dir=str(setup_split_data),
            output_path=str(output_path),
            split="train",
        )

        assert output_path.exists()
        df = pd.read_csv(output_path)
        assert len(df) == 8  # 5 drone + 3 bird

    def test_csv_has_required_columns(self, setup_split_data, tmp_path):
        """Test that CSV has required columns."""
        output_path = tmp_path / "reference.csv"

        generate_reference_dataset(
            data_dir=str(setup_split_data),
            output_path=str(output_path),
            split="train",
        )

        df = pd.read_csv(output_path)
        required_columns = ["timestamp", "class_name", "class_int", "image_path"]
        for col in required_columns:
            assert col in df.columns

    def test_extracts_image_features(self, setup_split_data, tmp_path):
        """Test that image features are extracted."""
        output_path = tmp_path / "reference.csv"

        generate_reference_dataset(
            data_dir=str(setup_split_data),
            output_path=str(output_path),
            split="train",
        )

        df = pd.read_csv(output_path)
        # Check for some expected feature columns
        assert "brightness_mean" in df.columns
        assert "contrast" in df.columns
        assert "aspect_ratio" in df.columns

    def test_class_labels_correct(self, setup_split_data, tmp_path):
        """Test that class labels are assigned correctly."""
        output_path = tmp_path / "reference.csv"

        generate_reference_dataset(
            data_dir=str(setup_split_data),
            output_path=str(output_path),
            split="train",
        )

        df = pd.read_csv(output_path)
        drone_rows = df[df["class_name"] == "drone"]
        bird_rows = df[df["class_name"] == "bird"]

        assert len(drone_rows) == 5
        assert len(bird_rows) == 3
        assert all(drone_rows["class_int"] == 0)
        assert all(bird_rows["class_int"] == 1)

    def test_respects_max_samples(self, setup_split_data, tmp_path):
        """Test that max_samples parameter limits output."""
        output_path = tmp_path / "reference.csv"

        generate_reference_dataset(
            data_dir=str(setup_split_data),
            output_path=str(output_path),
            split="train",
            max_samples=3,
        )

        df = pd.read_csv(output_path)
        assert len(df) == 3

    def test_creates_metadata_file(self, setup_split_data, tmp_path):
        """Test that metadata JSON file is created."""
        output_path = tmp_path / "reference.csv"
        metadata_path = tmp_path / "reference.json"

        generate_reference_dataset(
            data_dir=str(setup_split_data),
            output_path=str(output_path),
            split="train",
        )

        assert metadata_path.exists()
        with open(metadata_path) as f:
            metadata = json.load(f)

        assert "generated_at" in metadata
        assert "split" in metadata
        assert "total_samples" in metadata
        assert metadata["split"] == "train"
        assert metadata["total_samples"] == 8

    def test_metadata_has_class_distribution(self, setup_split_data, tmp_path):
        """Test that metadata includes class distribution."""
        output_path = tmp_path / "reference.csv"
        metadata_path = tmp_path / "reference.json"

        generate_reference_dataset(
            data_dir=str(setup_split_data),
            output_path=str(output_path),
            split="train",
        )

        with open(metadata_path) as f:
            metadata = json.load(f)

        assert "class_distribution" in metadata
        assert metadata["class_distribution"]["drone"] == 5
        assert metadata["class_distribution"]["bird"] == 3

    def test_handles_missing_split_file(self, tmp_path):
        """Test that missing split file raises error."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        with pytest.raises(FileNotFoundError):
            generate_reference_dataset(
                data_dir=str(data_dir),
                output_path=str(tmp_path / "reference.csv"),
                split="train",
            )

    def test_skips_missing_images(self, setup_split_data, tmp_path):
        """Test that missing images are skipped gracefully."""
        # Add a non-existent image to split file
        split_file = setup_split_data / "train_split.txt"
        current_content = split_file.read_text()
        split_file.write_text(current_content + "\ndrone/nonexistent.jpg")

        output_path = tmp_path / "reference.csv"

        generate_reference_dataset(
            data_dir=str(setup_split_data),
            output_path=str(output_path),
            split="train",
        )

        df = pd.read_csv(output_path)
        # Should only have the 8 valid images, not the missing one
        assert len(df) == 8

    def test_creates_output_directory(self, setup_split_data, tmp_path):
        """Test that output directory is created if it doesn't exist."""
        output_path = tmp_path / "nested" / "dir" / "reference.csv"

        generate_reference_dataset(
            data_dir=str(setup_split_data),
            output_path=str(output_path),
            split="train",
        )

        assert output_path.exists()
        assert output_path.parent.exists()
