import pytest
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Create a temporary data directory with drone and bird subdirectories."""
    data_dir = tmp_path / "data"
    (data_dir / "drone").mkdir(parents=True)
    (data_dir / "bird").mkdir(parents=True)
    return data_dir


@pytest.fixture
def tmp_splits_dir(tmp_path):
    """Create a temporary splits directory."""
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir(parents=True)
    return splits_dir


@pytest.fixture
def sample_images(tmp_data_dir):
    """Create sample image files in data directory."""
    images = []
    for i in range(10):
        img_path = tmp_data_dir / "drone" / f"drone_{i}.jpg"
        img = Image.new("RGB", (100, 100), color=(255, 0, 0))
        img.save(img_path)
        images.append(img_path)

    for i in range(10):
        img_path = tmp_data_dir / "bird" / f"bird_{i}.jpg"
        img = Image.new("RGB", (100, 100), color=(0, 0, 255))
        img.save(img_path)
        images.append(img_path)

    return images


@pytest.fixture
def sample_split_file(tmp_path, sample_images):
    """Create a sample split file."""
    split_file = tmp_path / "split.txt"
    lines = [
        "drone/drone_0.jpg",
        "drone/drone_1.jpg",
        "bird/bird_0.jpg",
        "bird/bird_1.jpg",
    ]
    split_file.write_text("\n".join(lines))
    return split_file


@pytest.fixture
def device():
    """Get appropriate device for testing."""
    return torch.device("cpu")


@pytest.fixture
def simple_model(device):
    """Create a simple model for testing."""
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 224 * 224, 2)).to(device)
    return model


@pytest.fixture
def simple_dataloader():
    """Create a simple dataloader for testing."""
    images = torch.randn(8, 3, 224, 224)
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=4, shuffle=False)
