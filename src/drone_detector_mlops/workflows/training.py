import torch
from torch import nn
from torch.utils.data import DataLoader

from drone_detector_mlops.utils.logger import get_logger
from drone_detector_mlops.model import get_model

logger = get_logger(__name__)


def setup_training(
    device: torch.device,
    learning_rate: float = 1e-3,
):
    """Setup model, optimizer, loss for training."""
    model = get_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    return model, optimizer, criterion


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    profiler=None,
    epoch: int = 0,
) -> dict:
    """Train for one epoch."""
    model.train()
    total_loss = torch.tensor(0.0, device=device)
    correct = torch.tensor(0, device=device)
    total = 0

    num_batches = len(dataloader)
    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Step profiler (this marks end of training step)
        if profiler:
            profiler.step()

        # Compute metrics (accumulate tensors, minimize .item() calls)
        total_loss += loss.detach()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().detach()
        total += labels.size(0)

        # Log progress every 10 batches
        if batch_idx % 10 == 0:
            logger.info(
                f"Epoch {epoch} [Train] batch {batch_idx}/{num_batches}",
                loss=f"{total_loss.item() / (batch_idx + 1):.4f}",
                acc=f"{correct.item() / total:.4f}",
            )

    return {
        "loss": total_loss.item() / len(dataloader),
        "accuracy": correct.item() / total,
    }


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int = 0,
) -> dict:
    """Validate for one epoch."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    num_batches = len(dataloader)
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            # Compute metrics (call .item() only once per tensor)
            loss_val = loss.item()
            total_loss += loss_val
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            # Log progress every 10 batches
            if batch_idx % 10 == 0:
                logger.info(
                    f"Epoch {epoch} [Val] batch {batch_idx}/{num_batches}",
                    loss=f"{loss_val:.4f}",
                    acc=f"{correct / total:.4f}",
                )

    return {
        "loss": total_loss / len(dataloader),
        "accuracy": correct / total,
    }
