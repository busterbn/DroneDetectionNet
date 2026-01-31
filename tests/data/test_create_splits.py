from PIL import Image

from drone_detector_mlops.data.create_splits import (
    create_splits,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
)


class TestCreateSplits:
    """Tests for create_splits function."""

    def test_creates_output_directory(self, tmp_data_dir, tmp_path, sample_images):
        """Output directory should be created if it doesn't exist."""
        output_dir = tmp_path / "new_splits"
        create_splits(tmp_data_dir, output_dir)
        assert output_dir.exists()

    def test_creates_three_split_files(self, tmp_data_dir, tmp_splits_dir, sample_images):
        """Should create train, val, and test split files."""
        create_splits(tmp_data_dir, tmp_splits_dir)

        assert (tmp_splits_dir / "train_files.txt").exists()
        assert (tmp_splits_dir / "val_files.txt").exists()
        assert (tmp_splits_dir / "test_files.txt").exists()

    def test_split_ratios_are_correct(self, tmp_data_dir, tmp_splits_dir, sample_images):
        """Split ratios should match TRAIN_RATIO, VAL_RATIO, TEST_RATIO."""
        create_splits(tmp_data_dir, tmp_splits_dir)

        train = (tmp_splits_dir / "train_files.txt").read_text().strip().split("\n")
        val = (tmp_splits_dir / "val_files.txt").read_text().strip().split("\n")
        test = (tmp_splits_dir / "test_files.txt").read_text().strip().split("\n")

        total = len(train) + len(val) + len(test)

        assert abs(len(train) / total - TRAIN_RATIO) < 0.1
        assert abs(len(val) / total - VAL_RATIO) < 0.1
        assert abs(len(test) / total - TEST_RATIO) < 0.1

    def test_no_data_loss(self, tmp_data_dir, tmp_splits_dir, sample_images):
        """All images should appear in exactly one split."""
        create_splits(tmp_data_dir, tmp_splits_dir)

        all_files = set()
        for split in ["train", "val", "test"]:
            content = (tmp_splits_dir / f"{split}_files.txt").read_text().strip()
            if content:
                files = set(content.split("\n"))
                # No overlap with previous splits
                assert len(all_files & files) == 0
                all_files.update(files)

        assert len(all_files) == 20  # 10 drone + 10 bird

    def test_reproducibility_with_same_seed(self, tmp_data_dir, tmp_path, sample_images):
        """Same seed should produce identical splits."""
        splits1 = tmp_path / "splits1"
        splits2 = tmp_path / "splits2"

        create_splits(tmp_data_dir, splits1)
        create_splits(tmp_data_dir, splits2)

        for split in ["train", "val", "test"]:
            content1 = (splits1 / f"{split}_files.txt").read_text()
            content2 = (splits2 / f"{split}_files.txt").read_text()
            assert content1 == content2

    def test_only_jpg_files_included(self, tmp_data_dir, tmp_splits_dir):
        """Only .jpg/.jpeg files should be included."""
        # Create various file types
        (tmp_data_dir / "drone" / "img.jpg").touch()
        (tmp_data_dir / "drone" / "img.JPG").touch()
        (tmp_data_dir / "drone" / "img.jpeg").touch()
        (tmp_data_dir / "drone" / "img.png").touch()
        (tmp_data_dir / "drone" / "img.txt").touch()

        # Create enough files for split
        for i in range(10):
            Image.new("RGB", (10, 10)).save(tmp_data_dir / "drone" / f"d{i}.jpg")
            Image.new("RGB", (10, 10)).save(tmp_data_dir / "bird" / f"b{i}.jpg")

        create_splits(tmp_data_dir, tmp_splits_dir)

        all_content = ""
        for split in ["train", "val", "test"]:
            all_content += (tmp_splits_dir / f"{split}_files.txt").read_text()

        assert "png" not in all_content
        assert "txt" not in all_content

    def test_paths_are_relative(self, tmp_data_dir, tmp_splits_dir, sample_images):
        """Paths in split files should be relative."""
        create_splits(tmp_data_dir, tmp_splits_dir)

        content = (tmp_splits_dir / "train_files.txt").read_text()
        lines = [line for line in content.strip().split("\n") if line]

        for line in lines:
            assert not line.startswith("/")
            assert line.startswith("drone/") or line.startswith("bird/")

    def test_ignores_files_in_data_dir_root(self, tmp_data_dir, tmp_splits_dir, sample_images):
        """Should skip files (non-directories) in data_dir root."""
        (tmp_data_dir / "readme.txt").touch()
        (tmp_data_dir / "metadata.json").touch()

        create_splits(tmp_data_dir, tmp_splits_dir)

        all_content = ""
        for split in ["train", "val", "test"]:
            all_content += (tmp_splits_dir / f"{split}_files.txt").read_text()

        assert "readme.txt" not in all_content
        assert "metadata.json" not in all_content
        assert "drone/" in all_content
        assert "bird/" in all_content
