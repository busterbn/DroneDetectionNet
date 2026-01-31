from drone_detector_mlops.data.stats import (
    get_all_files_from_splits,
    find_new_files,
    LABEL_NAMES,
    IMAGE_EXTENSIONS,
)


class TestGetAllFilesFromSplits:
    """Tests for get_all_files_from_splits function."""

    def test_reads_all_split_files(self, tmp_path):
        """Should read all three split files."""
        splits_dir = tmp_path / "splits"
        splits_dir.mkdir()
        (splits_dir / "train_files.txt").write_text("drone/img1.jpg\ndrone/img2.jpg")
        (splits_dir / "val_files.txt").write_text("bird/img3.jpg")
        (splits_dir / "test_files.txt").write_text("drone/img4.jpg")
        all_files = get_all_files_from_splits(splits_dir)
        assert len(all_files) == 4
        assert "drone/img1.jpg" in all_files
        assert "bird/img3.jpg" in all_files

    def test_handles_missing_split_files(self, tmp_path):
        """Should handle missing split files gracefully."""
        splits_dir = tmp_path / "splits"
        splits_dir.mkdir()
        all_files = get_all_files_from_splits(splits_dir)
        assert len(all_files) == 0

    def test_returns_set(self, tmp_path):
        """Should return a set."""
        splits_dir = tmp_path / "splits"
        splits_dir.mkdir()
        (splits_dir / "train_files.txt").write_text("drone/img1.jpg")
        all_files = get_all_files_from_splits(splits_dir)
        assert isinstance(all_files, set)

    def test_handles_duplicate_entries(self, tmp_path):
        """Should handle duplicate entries across files."""
        splits_dir = tmp_path / "splits"
        splits_dir.mkdir()
        (splits_dir / "train_files.txt").write_text("drone/img1.jpg")
        (splits_dir / "val_files.txt").write_text("drone/img1.jpg")
        (splits_dir / "test_files.txt").write_text("bird/img2.jpg")
        all_files = get_all_files_from_splits(splits_dir)
        assert len(all_files) == 2


class TestFindNewFiles:
    """Tests for find_new_files function."""

    def test_finds_files_in_after_but_not_before(self, tmp_path):
        """Should find files that exist in after but not before."""
        before_splits = tmp_path / "before_splits"
        after_splits = tmp_path / "after_splits"
        before_splits.mkdir()
        after_splits.mkdir()
        (before_splits / "train_files.txt").write_text("drone/img1.jpg")
        (before_splits / "val_files.txt").write_text("")
        (before_splits / "test_files.txt").write_text("")
        (after_splits / "train_files.txt").write_text("drone/img1.jpg\ndrone/img2.jpg")
        (after_splits / "val_files.txt").write_text("bird/img3.jpg")
        (after_splits / "test_files.txt").write_text("")
        new_files = find_new_files(before_splits, after_splits)
        assert len(new_files) == 2
        assert "drone/img2.jpg" in new_files
        assert "bird/img3.jpg" in new_files

    def test_returns_sorted_list(self, tmp_path):
        """Should return a sorted list."""
        before_splits = tmp_path / "before_splits"
        after_splits = tmp_path / "after_splits"
        before_splits.mkdir()
        after_splits.mkdir()
        (before_splits / "train_files.txt").write_text("")
        (before_splits / "val_files.txt").write_text("")
        (before_splits / "test_files.txt").write_text("")
        (after_splits / "train_files.txt").write_text("drone/zzz.jpg\ndrone/aaa.jpg")
        (after_splits / "val_files.txt").write_text("")
        (after_splits / "test_files.txt").write_text("")
        new_files = find_new_files(before_splits, after_splits)
        assert new_files == sorted(new_files)

    def test_returns_empty_list_when_no_new_files(self, tmp_path):
        """Should return empty list when no new files."""
        before_splits = tmp_path / "before_splits"
        after_splits = tmp_path / "after_splits"
        before_splits.mkdir()
        after_splits.mkdir()
        (before_splits / "train_files.txt").write_text("drone/img1.jpg")
        (before_splits / "val_files.txt").write_text("")
        (before_splits / "test_files.txt").write_text("")
        (after_splits / "train_files.txt").write_text("drone/img1.jpg")
        (after_splits / "val_files.txt").write_text("")
        (after_splits / "test_files.txt").write_text("")
        new_files = find_new_files(before_splits, after_splits)
        assert len(new_files) == 0


class TestConstants:
    """Tests for module constants."""

    def test_label_names_defined(self):
        """LABEL_NAMES should be properly defined."""
        assert LABEL_NAMES[0] == "drone"
        assert LABEL_NAMES[1] == "bird"

    def test_image_extensions_defined(self):
        """IMAGE_EXTENSIONS should contain common image formats."""
        assert ".jpg" in IMAGE_EXTENSIONS
        assert ".jpeg" in IMAGE_EXTENSIONS
        assert ".png" in IMAGE_EXTENSIONS
        assert isinstance(IMAGE_EXTENSIONS, set)
