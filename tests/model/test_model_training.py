import torch
import torch.nn as nn

from drone_detector_mlops.model import DroneDetectorModel


def test_drone_detector_model_gradient_flow():
    """Test that gradients flow through the model (can be trained)."""
    model = DroneDetectorModel(num_classes=2, pretrained=False)
    model.train()
    x = torch.randn(2, 3, 224, 224)
    target = torch.tensor([0, 1])
    output = model(x)
    loss = nn.CrossEntropyLoss()(output, target)
    loss.backward()
    has_gradients = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert has_gradients


def test_drone_detector_model_train_eval_modes():
    """Test that model can switch between train and eval modes."""
    model = DroneDetectorModel(num_classes=2, pretrained=False)
    model.train()
    assert model.training
    model.eval()
    assert not model.training


def test_drone_detector_model_has_parameters():
    """Test that model has trainable parameters."""
    model = DroneDetectorModel(num_classes=2, pretrained=False)
    params = list(model.parameters())
    assert len(params) > 0
    trainable_params = [p for p in params if p.requires_grad]
    assert len(trainable_params) > 0


def test_drone_detector_model_parameter_count():
    """Test that model has expected number of parameters (ResNet18 size)."""
    model = DroneDetectorModel(num_classes=2, pretrained=False)
    total_params = sum(p.numel() for p in model.parameters())
    assert 10_000_000 < total_params < 15_000_000


def test_drone_detector_model_device_transfer():
    """Test that model can be moved to different devices."""
    model = DroneDetectorModel(num_classes=2, pretrained=False)
    model_cpu = model.to("cpu")
    assert next(model_cpu.parameters()).device.type == "cpu"
    if torch.cuda.is_available():
        model_cuda = model.to("cuda")
        assert next(model_cuda.parameters()).device.type == "cuda"
