import pytest
from PIL import Image
import torch
from torchvision import transforms

from drone_detector_mlops.data.data import DroneVsBirdDataset


class TestDroneVsBirdDataset:
    """Tests for DroneVsBirdDataset class."""

    def test_is_valid_pytorch_dataset(self, tmp_data_dir, sample_split_file):
        """Dataset should be a valid PyTorch Dataset."""
        from torch.utils.data import Dataset

        dataset = DroneVsBirdDataset(tmp_data_dir, sample_split_file)
        assert isinstance(dataset, Dataset)

    def test_length_matches_split_file(self, tmp_data_dir, sample_split_file):
        """Dataset length should match number of entries in split file."""
        dataset = DroneVsBirdDataset(tmp_data_dir, sample_split_file)
        assert len(dataset) == 4

    def test_empty_split_file_returns_empty_dataset(self, tmp_data_dir, tmp_path):
        """Empty split file should create empty dataset."""
        empty_split = tmp_path / "empty.txt"
        empty_split.write_text("")

        dataset = DroneVsBirdDataset(tmp_data_dir, empty_split)
        assert len(dataset) == 0

    def test_whitespace_lines_are_skipped(self, tmp_data_dir, tmp_path, sample_images):
        """Whitespace-only lines should be skipped."""
        split_file = tmp_path / "split.txt"
        split_file.write_text("drone/drone_0.jpg\n\n  \n\ndrone/drone_1.jpg\n")

        dataset = DroneVsBirdDataset(tmp_data_dir, split_file)
        assert len(dataset) == 2

    def test_drone_label_is_zero(self, tmp_data_dir, tmp_path, sample_images):
        """Drone images should have label 0."""
        split_file = tmp_path / "split.txt"
        split_file.write_text("drone/drone_0.jpg\n")

        dataset = DroneVsBirdDataset(tmp_data_dir, split_file)
        _, label = dataset[0]
        assert label == 0

    def test_bird_label_is_one(self, tmp_data_dir, tmp_path, sample_images):
        """Bird images should have label 1."""
        split_file = tmp_path / "split.txt"
        split_file.write_text("bird/bird_0.jpg\n")

        dataset = DroneVsBirdDataset(tmp_data_dir, split_file)
        _, label = dataset[0]
        assert label == 1

    def test_getitem_returns_pil_image_without_transform(self, tmp_data_dir, tmp_path, sample_images):
        """Without transform, getitem should return PIL Image."""
        split_file = tmp_path / "split.txt"
        split_file.write_text("drone/drone_0.jpg\n")

        dataset = DroneVsBirdDataset(tmp_data_dir, split_file, transform=None)
        image, _ = dataset[0]
        assert isinstance(image, Image.Image)

    def test_getitem_applies_transform(self, tmp_data_dir, tmp_path, sample_images):
        """Transform should be applied to images."""
        split_file = tmp_path / "split.txt"
        split_file.write_text("drone/drone_0.jpg\n")

        transform = transforms.ToTensor()
        dataset = DroneVsBirdDataset(tmp_data_dir, split_file, transform=transform)
        image, _ = dataset[0]

        assert isinstance(image, torch.Tensor)

    def test_getitem_converts_to_rgb(self, tmp_data_dir, tmp_path):
        """Grayscale images should be converted to RGB."""
        gray_img = Image.new("L", (100, 100), color=128)
        gray_path = tmp_data_dir / "drone" / "gray.jpg"
        gray_img.save(gray_path)

        split_file = tmp_path / "split.txt"
        split_file.write_text("drone/gray.jpg\n")

        dataset = DroneVsBirdDataset(tmp_data_dir, split_file, transform=None)
        image, _ = dataset[0]

        assert image.mode == "RGB"

    def test_index_out_of_bounds_raises_error(self, tmp_data_dir, sample_split_file):
        """Accessing invalid index should raise IndexError."""
        dataset = DroneVsBirdDataset(tmp_data_dir, sample_split_file)

        with pytest.raises(IndexError):
            _ = dataset[100]

    def test_negative_indexing(self, tmp_data_dir, sample_split_file):
        """Negative indexing should work."""
        dataset = DroneVsBirdDataset(tmp_data_dir, sample_split_file)
        _, label = dataset[-1]
        assert label in [0, 1]

    def test_missing_image_raises_error(self, tmp_data_dir, tmp_path):
        """Missing image file should raise FileNotFoundError."""
        split_file = tmp_path / "split.txt"
        split_file.write_text("drone/nonexistent.jpg\n")

        dataset = DroneVsBirdDataset(tmp_data_dir, split_file)

        with pytest.raises(FileNotFoundError):
            _ = dataset[0]
