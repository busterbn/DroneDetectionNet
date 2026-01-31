from drone_detector_mlops.data.stats import (
    count_images_in_directory,
    find_new_images_in_directory,
)


class TestCountImagesInDirectory:
    """Tests for count_images_in_directory function."""

    def test_counts_jpg_images(self, tmp_path):
        """Should count .jpg images in drone and bird directories."""
        data_dir = tmp_path / "data"
        (data_dir / "drone").mkdir(parents=True)
        (data_dir / "bird").mkdir(parents=True)
        for i in range(3):
            (data_dir / "drone" / f"img{i}.jpg").touch()
        for i in range(5):
            (data_dir / "bird" / f"img{i}.jpg").touch()
        counts = count_images_in_directory(data_dir)
        assert counts["drone"] == 3
        assert counts["bird"] == 5

    def test_counts_multiple_image_extensions(self, tmp_path):
        """Should count various image extensions."""
        data_dir = tmp_path / "data"
        (data_dir / "drone").mkdir(parents=True)
        (data_dir / "bird").mkdir(parents=True)
        (data_dir / "drone" / "img1.jpg").touch()
        (data_dir / "drone" / "img2.jpeg").touch()
        (data_dir / "drone" / "img3.png").touch()
        (data_dir / "bird" / "img1.JPG").touch()
        (data_dir / "bird" / "img2.gif").touch()
        counts = count_images_in_directory(data_dir)
        assert counts["drone"] == 3
        assert counts["bird"] == 2

    def test_ignores_non_image_files(self, tmp_path):
        """Should ignore non-image files."""
        data_dir = tmp_path / "data"
        (data_dir / "drone").mkdir(parents=True)
        (data_dir / "drone" / "img.jpg").touch()
        (data_dir / "drone" / "file.txt").touch()
        (data_dir / "drone" / "data.csv").touch()
        counts = count_images_in_directory(data_dir)
        assert counts["drone"] == 1

    def test_handles_missing_directories(self, tmp_path):
        """Should return zero for missing directories."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        counts = count_images_in_directory(data_dir)
        assert counts["drone"] == 0
        assert counts["bird"] == 0

    def test_returns_dict_with_both_classes(self, tmp_path):
        """Should always return dict with both drone and bird keys."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        counts = count_images_in_directory(data_dir)
        assert "drone" in counts
        assert "bird" in counts


class TestFindNewImagesInDirectory:
    """Tests for find_new_images_in_directory function."""

    def test_finds_new_images_in_after_directory(self, tmp_path):
        """Should find images that exist in after but not before."""
        before_dir = tmp_path / "before"
        after_dir = tmp_path / "after"
        (before_dir / "drone").mkdir(parents=True)
        (after_dir / "drone").mkdir(parents=True)
        (before_dir / "drone" / "img1.jpg").touch()
        (after_dir / "drone" / "img1.jpg").touch()
        (after_dir / "drone" / "img2.jpg").touch()
        new_images = find_new_images_in_directory(before_dir, after_dir)
        assert "img2.jpg" in new_images["drone"]
        assert "img1.jpg" not in new_images["drone"]

    def test_handles_before_directory_not_exists(self, tmp_path):
        """Should treat missing before directory as empty."""
        before_dir = tmp_path / "before"
        after_dir = tmp_path / "after"
        (after_dir / "bird").mkdir(parents=True)
        (after_dir / "bird" / "img.jpg").touch()
        new_images = find_new_images_in_directory(before_dir, after_dir)
        assert "img.jpg" in new_images["bird"]

    def test_returns_empty_when_no_new_images(self, tmp_path):
        """Should return empty lists when no new images."""
        before_dir = tmp_path / "before"
        after_dir = tmp_path / "after"
        (before_dir / "drone").mkdir(parents=True)
        (after_dir / "drone").mkdir(parents=True)
        (before_dir / "drone" / "img.jpg").touch()
        (after_dir / "drone" / "img.jpg").touch()
        new_images = find_new_images_in_directory(before_dir, after_dir)
        assert len(new_images["drone"]) == 0
        assert len(new_images["bird"]) == 0

    def test_finds_new_images_in_both_classes(self, tmp_path):
        """Should find new images in both drone and bird directories."""
        before_dir = tmp_path / "before"
        after_dir = tmp_path / "after"
        (before_dir / "drone").mkdir(parents=True)
        (before_dir / "bird").mkdir(parents=True)
        (after_dir / "drone").mkdir(parents=True)
        (after_dir / "bird").mkdir(parents=True)
        (after_dir / "drone" / "new_drone.jpg").touch()
        (after_dir / "bird" / "new_bird.jpg").touch()
        new_images = find_new_images_in_directory(before_dir, after_dir)
        assert "new_drone.jpg" in new_images["drone"]
        assert "new_bird.jpg" in new_images["bird"]
