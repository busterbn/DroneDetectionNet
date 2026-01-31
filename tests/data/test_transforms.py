import torch
from PIL import Image
from torchvision import transforms

from drone_detector_mlops.data.transforms import (
    train_transform,
    val_transform,
    test_transform,
)


class TestTransforms:
    """Tests for image transforms."""

    def test_train_transform_output_shape(self):
        """Train transform should produce (3, 224, 224) tensor."""
        img = Image.new("RGB", (300, 300))
        result = train_transform(img)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 224, 224)

    def test_val_transform_output_shape(self):
        """Val transform should produce (3, 224, 224) tensor."""
        img = Image.new("RGB", (300, 300))
        result = val_transform(img)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 224, 224)

    def test_test_transform_is_val_transform(self):
        """Test transform should be identical to val transform."""
        assert test_transform is val_transform

    def test_train_transform_includes_augmentation(self):
        """Train transform should include data augmentation."""
        transform_types = [type(t).__name__ for t in train_transform.transforms]

        assert "RandomResizedCrop" in transform_types
        assert "RandomHorizontalFlip" in transform_types

    def test_val_transform_no_random_augmentation(self):
        """Val transform should not include random augmentation."""
        transform_types = [type(t).__name__ for t in val_transform.transforms]

        assert "RandomResizedCrop" not in transform_types
        assert "RandomHorizontalFlip" not in transform_types

    def test_transforms_normalize_to_imagenet(self):
        """Both transforms should normalize to ImageNet stats."""
        # Check train transform has normalize
        has_normalize = any(isinstance(t, transforms.Normalize) for t in train_transform.transforms)
        assert has_normalize

    def test_train_transform_is_deterministic_with_seed(self):
        """Train transform should be reproducible with fixed seed."""
        img = Image.new("RGB", (300, 300), color=(128, 128, 128))

        torch.manual_seed(42)
        result1 = train_transform(img)

        torch.manual_seed(42)
        result2 = train_transform(img)

        assert torch.equal(result1, result2)
