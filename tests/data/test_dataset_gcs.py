from unittest.mock import patch, MagicMock
from io import BytesIO
from PIL import Image

from drone_detector_mlops.data.data import DroneVsBirdDataset, get_dataloaders


class TestDroneVsBirdDatasetGCS:
    """Tests for GCS path handling in DroneVsBirdDataset."""

    def test_is_gcs_path_detection(self, tmp_path):
        """Should identify GCS and local paths correctly."""
        split_file = tmp_path / "split.txt"
        split_file.write_text("drone/img.jpg")
        dataset = DroneVsBirdDataset(tmp_path, split_file)
        assert dataset._is_gcs_path("gs://bucket/path")
        assert not dataset._is_gcs_path("/local/path")
        assert not dataset._is_gcs_path("relative/path")

    def test_gcsfs_lazy_loading(self, tmp_path):
        """Should lazy-load gcsfs filesystem."""
        split_file = tmp_path / "split.txt"
        split_file.write_text("drone/img.jpg")
        dataset = DroneVsBirdDataset(tmp_path, split_file)

        assert dataset._gcsfs is None

        with patch("drone_detector_mlops.data.data.gcsfs.GCSFileSystem") as mock_gcsfs:
            mock_fs = MagicMock()
            mock_gcsfs.return_value = mock_fs
            fs = dataset.gcsfs
            assert fs == mock_fs
            mock_gcsfs.assert_called_once()

    @patch("drone_detector_mlops.data.data.gcsfs.GCSFileSystem")
    def test_read_split_file_from_gcs(self, mock_gcsfs_class, tmp_path):
        """Should read split file from GCS."""
        mock_fs = MagicMock()
        mock_gcsfs_class.return_value = mock_fs
        mock_fs.open.return_value.__enter__ = MagicMock(return_value=iter(["drone/img1.jpg\n", "bird/img2.jpg\n"]))
        mock_fs.open.return_value.__exit__ = MagicMock(return_value=False)

        local_split = tmp_path / "local_split.txt"
        local_split.write_text("drone/dummy.jpg")
        dataset = DroneVsBirdDataset(tmp_path, local_split)

        dataset._read_split_file("gs://bucket/splits/train.txt")

        mock_fs.open.assert_called_with("gs://bucket/splits/train.txt", "r")

    def test_build_image_path_gcs(self, tmp_path):
        """Should build GCS image path correctly."""
        split_file = tmp_path / "split.txt"
        split_file.write_text("drone/img.jpg")
        dataset = DroneVsBirdDataset("gs://bucket/data", split_file)

        path = dataset._build_image_path(0)
        assert path == "gs://bucket/data/drone/img.jpg"

    def test_build_image_path_local(self, tmp_path):
        """Should build local image path correctly."""
        split_file = tmp_path / "split.txt"
        split_file.write_text("drone/img.jpg")
        dataset = DroneVsBirdDataset(tmp_path, split_file)

        path = dataset._build_image_path(0)
        assert path == tmp_path / "drone" / "img.jpg"

    @patch("drone_detector_mlops.data.data.gcsfs.GCSFileSystem")
    def test_open_image_from_gcs(self, mock_gcsfs_class, tmp_path):
        """Should open image from GCS."""
        mock_fs = MagicMock()
        mock_gcsfs_class.return_value = mock_fs

        img = Image.new("RGB", (100, 100), color=(255, 0, 0))
        img_bytes = BytesIO()
        img.save(img_bytes, format="JPEG")
        img_bytes.seek(0)

        mock_fs.open.return_value.__enter__ = MagicMock(return_value=img_bytes)
        mock_fs.open.return_value.__exit__ = MagicMock(return_value=False)

        split_file = tmp_path / "split.txt"
        split_file.write_text("drone/img.jpg")
        dataset = DroneVsBirdDataset(tmp_path, split_file)

        result = dataset._open_image("gs://bucket/data/drone/img.jpg")

        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"
        mock_fs.open.assert_called_with("gs://bucket/data/drone/img.jpg", "rb")


class TestGetDataloadersGCS:
    """Tests for GCS path handling in get_dataloaders."""

    @patch("drone_detector_mlops.data.data.gcsfs.GCSFileSystem")
    def test_handles_gcs_string_splits_dir(self, mock_gcsfs_class, tmp_path):
        """Should handle GCS string path for splits_dir."""
        mock_fs = MagicMock()
        mock_gcsfs_class.return_value = mock_fs

        (tmp_path / "drone").mkdir()
        (tmp_path / "bird").mkdir()
        img = Image.new("RGB", (100, 100))
        img.save(tmp_path / "drone" / "img.jpg")
        img.save(tmp_path / "bird" / "img.jpg")

        for split in ["train", "val", "test"]:
            split_file = tmp_path / f"{split}_files.txt"
            split_file.write_text("drone/img.jpg\nbird/img.jpg")

        train, val, test = get_dataloaders(
            data_dir=tmp_path,
            splits_dir=tmp_path,
            batch_size=1,
            num_workers=0,
        )

        assert train is not None
        assert val is not None
        assert test is not None

    def test_handles_string_splits_dir_path(self, tmp_path):
        """Should handle string path (non-GCS) for splits_dir."""
        (tmp_path / "drone").mkdir()
        (tmp_path / "bird").mkdir()
        img = Image.new("RGB", (100, 100))
        img.save(tmp_path / "drone" / "img.jpg")
        img.save(tmp_path / "bird" / "img.jpg")

        for split in ["train", "val", "test"]:
            split_file = tmp_path / f"{split}_files.txt"
            split_file.write_text("drone/img.jpg\nbird/img.jpg")

        train, val, test = get_dataloaders(
            data_dir=tmp_path,
            splits_dir=str(tmp_path),
            batch_size=1,
            num_workers=0,
        )

        assert train is not None
