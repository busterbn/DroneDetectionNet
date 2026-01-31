import numpy as np
import pytest
from PIL import Image

from drone_detector_mlops.monitoring.feature_extraction import ImageFeatureExtractor


class TestImageFeatureExtractor:
    """Tests for ImageFeatureExtractor class."""

    @pytest.fixture
    def sample_image(self):
        """Create a sample RGB image."""
        return Image.new("RGB", (100, 100), color=(128, 128, 128))

    @pytest.fixture
    def red_image(self):
        """Create a red image."""
        return Image.new("RGB", (100, 100), color=(255, 0, 0))

    @pytest.fixture
    def bright_image(self):
        """Create a bright image."""
        return Image.new("RGB", (100, 100), color=(240, 240, 240))

    @pytest.fixture
    def dark_image(self):
        """Create a dark image."""
        return Image.new("RGB", (100, 100), color=(20, 20, 20))

    def test_extract_features_returns_dict(self, sample_image):
        """Test that extract_features returns a dictionary."""
        features = ImageFeatureExtractor.extract_features(sample_image)
        assert isinstance(features, dict)

    def test_extract_features_has_all_required_keys(self, sample_image):
        """Test that all required feature keys are present."""
        features = ImageFeatureExtractor.extract_features(sample_image)
        expected_keys = ImageFeatureExtractor.get_feature_names()

        assert set(features.keys()) == set(expected_keys)

    def test_extract_features_values_are_floats(self, sample_image):
        """Test that all feature values are floats."""
        features = ImageFeatureExtractor.extract_features(sample_image)

        for key, value in features.items():
            assert isinstance(value, float), f"{key} is not a float"

    def test_brightness_mean_in_valid_range(self, sample_image):
        """Test that brightness mean is in valid range 0-255."""
        features = ImageFeatureExtractor.extract_features(sample_image)
        assert 0 <= features["brightness_mean"] <= 255

    def test_brightness_mean_bright_vs_dark(self, bright_image, dark_image):
        """Test that bright images have higher brightness than dark images."""
        bright_features = ImageFeatureExtractor.extract_features(bright_image)
        dark_features = ImageFeatureExtractor.extract_features(dark_image)

        assert bright_features["brightness_mean"] > dark_features["brightness_mean"]

    def test_rgb_means_for_red_image(self, red_image):
        """Test RGB means for a pure red image."""
        features = ImageFeatureExtractor.extract_features(red_image)

        assert features["r_mean"] > 200
        assert features["g_mean"] < 50
        assert features["b_mean"] < 50

    def test_rgb_means_in_valid_range(self, sample_image):
        """Test that RGB means are in valid range 0-255."""
        features = ImageFeatureExtractor.extract_features(sample_image)

        assert 0 <= features["r_mean"] <= 255
        assert 0 <= features["g_mean"] <= 255
        assert 0 <= features["b_mean"] <= 255

    def test_rgb_stds_non_negative(self, sample_image):
        """Test that RGB standard deviations are non-negative."""
        features = ImageFeatureExtractor.extract_features(sample_image)

        assert features["r_std"] >= 0
        assert features["g_std"] >= 0
        assert features["b_std"] >= 0

    def test_edge_density_in_valid_range(self, sample_image):
        """Test that edge density is between 0 and 1."""
        features = ImageFeatureExtractor.extract_features(sample_image)
        assert 0 <= features["edge_density"] <= 1

    def test_aspect_ratio_square_image(self):
        """Test aspect ratio for square image."""
        square_image = Image.new("RGB", (100, 100), color=(128, 128, 128))
        features = ImageFeatureExtractor.extract_features(square_image)
        assert abs(features["aspect_ratio"] - 1.0) < 0.01

    def test_aspect_ratio_rectangular_image(self):
        """Test aspect ratio for rectangular image."""
        rect_image = Image.new("RGB", (200, 100), color=(128, 128, 128))
        features = ImageFeatureExtractor.extract_features(rect_image)
        assert abs(features["aspect_ratio"] - 2.0) < 0.01

    def test_size_pixels_correct(self):
        """Test that size_pixels matches actual image size."""
        image = Image.new("RGB", (100, 50), color=(128, 128, 128))
        features = ImageFeatureExtractor.extract_features(image)
        assert features["size_pixels"] == 5000

    def test_sharpness_non_negative(self, sample_image):
        """Test that sharpness is non-negative."""
        features = ImageFeatureExtractor.extract_features(sample_image)
        assert features["sharpness"] >= 0

    def test_contrast_non_negative(self, sample_image):
        """Test that contrast is non-negative."""
        features = ImageFeatureExtractor.extract_features(sample_image)
        assert features["contrast"] >= 0

    def test_saturation_mean_in_valid_range(self, sample_image):
        """Test that saturation mean is in valid range."""
        features = ImageFeatureExtractor.extract_features(sample_image)
        assert 0 <= features["saturation_mean"] <= 255

    def test_value_mean_in_valid_range(self, sample_image):
        """Test that value mean is in valid range."""
        features = ImageFeatureExtractor.extract_features(sample_image)
        assert 0 <= features["value_mean"] <= 255


class TestExtractFeaturesFromArray:
    """Tests for extract_features_from_array method."""

    def test_extract_from_numpy_array(self):
        """Test extracting features from numpy array."""
        array = np.ones((100, 100, 3), dtype=np.uint8) * 128
        features = ImageFeatureExtractor.extract_features_from_array(array)

        assert isinstance(features, dict)
        assert len(features) > 0

    def test_array_and_pil_produce_same_features(self):
        """Test that numpy array and PIL image produce same features."""
        array = np.ones((100, 100, 3), dtype=np.uint8) * 128
        pil_image = Image.fromarray(array)

        features_from_array = ImageFeatureExtractor.extract_features_from_array(array)
        features_from_pil = ImageFeatureExtractor.extract_features(pil_image)

        for key in features_from_array:
            assert abs(features_from_array[key] - features_from_pil[key]) < 0.01


class TestExtractFeaturesFromPath:
    """Tests for extract_features_from_path method."""

    def test_extract_from_file_path(self, tmp_path):
        """Test extracting features from file path."""
        image = Image.new("RGB", (100, 100), color=(128, 128, 128))
        image_path = tmp_path / "test_image.jpg"
        image.save(image_path)

        features = ImageFeatureExtractor.extract_features_from_path(str(image_path))

        assert isinstance(features, dict)
        assert len(features) > 0

    def test_extract_from_path_has_all_features(self, tmp_path):
        """Test that extracting from path includes all features."""
        image = Image.new("RGB", (100, 100), color=(128, 128, 128))
        image_path = tmp_path / "test_image.jpg"
        image.save(image_path)

        features = ImageFeatureExtractor.extract_features_from_path(str(image_path))
        expected_keys = ImageFeatureExtractor.get_feature_names()

        assert set(features.keys()) == set(expected_keys)


class TestGetFeatureNames:
    """Tests for get_feature_names method."""

    def test_returns_list(self):
        """Test that get_feature_names returns a list."""
        feature_names = ImageFeatureExtractor.get_feature_names()
        assert isinstance(feature_names, list)

    def test_returns_non_empty_list(self):
        """Test that feature names list is not empty."""
        feature_names = ImageFeatureExtractor.get_feature_names()
        assert len(feature_names) > 0

    def test_all_names_are_strings(self):
        """Test that all feature names are strings."""
        feature_names = ImageFeatureExtractor.get_feature_names()
        assert all(isinstance(name, str) for name in feature_names)

    def test_expected_features_present(self):
        """Test that expected feature names are present."""
        feature_names = ImageFeatureExtractor.get_feature_names()

        expected_features = [
            "brightness_mean",
            "brightness_std",
            "contrast",
            "sharpness",
            "r_mean",
            "g_mean",
            "b_mean",
            "edge_density",
            "aspect_ratio",
        ]

        for expected in expected_features:
            assert expected in feature_names
