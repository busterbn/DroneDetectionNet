"""Image feature extraction for data drift monitoring.

This module extracts numerical features from images that can be used
for statistical drift detection using Evidently AI.
"""

import numpy as np
from PIL import Image
import cv2


class ImageFeatureExtractor:
    """Extract numerical features from images for drift monitoring."""

    @staticmethod
    def extract_features(image: Image.Image) -> dict[str, float]:
        """Extract all numerical features from an image.

        Args:
            image: PIL Image object

        Returns:
            Dictionary containing extracted features:
            - brightness_mean: Average brightness (0-255)
            - brightness_std: Brightness standard deviation
            - contrast: RMS contrast
            - sharpness: Laplacian variance (higher = sharper)
            - r_mean, g_mean, b_mean: Average RGB channel values
            - r_std, g_std, b_std: RGB channel standard deviations
            - saturation_mean: Average HSV saturation
            - value_mean: Average HSV value
            - edge_density: Proportion of edge pixels
            - aspect_ratio: Width/height ratio
            - size_pixels: Total number of pixels
        """
        # Convert to numpy array
        img_array = np.array(image)

        # Convert to different color spaces
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        img_hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)

        # Extract features
        features = {}

        # Brightness statistics (from grayscale)
        features["brightness_mean"] = float(np.mean(img_gray))
        features["brightness_std"] = float(np.std(img_gray))

        # Contrast (RMS contrast)
        features["contrast"] = float(np.std(img_gray))

        # Sharpness (Laplacian variance - higher values = sharper)
        laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
        features["sharpness"] = float(np.var(laplacian))

        # RGB channel statistics
        features["r_mean"] = float(np.mean(img_array[:, :, 0]))
        features["g_mean"] = float(np.mean(img_array[:, :, 1]))
        features["b_mean"] = float(np.mean(img_array[:, :, 2]))

        features["r_std"] = float(np.std(img_array[:, :, 0]))
        features["g_std"] = float(np.std(img_array[:, :, 1]))
        features["b_std"] = float(np.std(img_array[:, :, 2]))

        # HSV statistics
        features["saturation_mean"] = float(np.mean(img_hsv[:, :, 1]))
        features["value_mean"] = float(np.mean(img_hsv[:, :, 2]))

        # Edge detection (Canny)
        edges = cv2.Canny(img_gray, threshold1=100, threshold2=200)
        features["edge_density"] = float(np.sum(edges > 0) / edges.size)

        # Image dimensions
        height, width = img_gray.shape
        features["aspect_ratio"] = float(width / height)
        features["size_pixels"] = float(width * height)

        return features

    @staticmethod
    def extract_features_from_array(img_array: np.ndarray) -> dict[str, float]:
        """Extract features from numpy array.

        Args:
            img_array: Image as numpy array (H, W, C)

        Returns:
            Dictionary of extracted features
        """
        image = Image.fromarray(img_array.astype("uint8"))
        return ImageFeatureExtractor.extract_features(image)

    @staticmethod
    def extract_features_from_path(image_path: str) -> dict[str, float]:
        """Extract features from image file path.

        Args:
            image_path: Path to image file

        Returns:
            Dictionary of extracted features
        """
        image = Image.open(image_path).convert("RGB")
        return ImageFeatureExtractor.extract_features(image)

    @staticmethod
    def get_feature_names() -> list[str]:
        """Get list of all feature names that will be extracted.

        Returns:
            List of feature names
        """
        return [
            "brightness_mean",
            "brightness_std",
            "contrast",
            "sharpness",
            "r_mean",
            "g_mean",
            "b_mean",
            "r_std",
            "g_std",
            "b_std",
            "saturation_mean",
            "value_mean",
            "edge_density",
            "aspect_ratio",
            "size_pixels",
        ]
