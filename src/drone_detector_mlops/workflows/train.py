from contextlib import nullcontext
from pathlib import Path
import time

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf

from drone_detector_mlops.data.data import get_dataloaders
from drone_detector_mlops.data.transforms import train_transform, val_transform
from drone_detector_mlops.utils.logger import get_logger
from drone_detector_mlops.utils.profiling import get_profiler
from drone_detector_mlops.utils.settings import settings
from drone_detector_mlops.utils.storage import get_storage
from drone_detector_mlops.workflows.training import (
    setup_training,
    train_epoch,
    validate_epoch,
)

logger = get_logger(__name__)

# Determine config path - use /app/configs in Docker, relative path locally
CONFIG_PATH = (
    "/app/configs" if Path("/app/configs").exists() else str(Path(__file__).parent.parent.parent.parent / "configs")
)
timestamp = time.strftime("%Y%m%d-%H%M%S")


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg: DictConfig) -> float:
    """Run training with Hydra config."""
    logger.info("Training config:\n" + OmegaConf.to_yaml(cfg))

    # Extract config values
    epochs = cfg.training.epochs
    batch_size = cfg.training.batch_size
    lr = cfg.training.lr
    profile = cfg.get("profile", False)
    is_multirun = cfg.get("multirun", False)

    logger.info(
        "Starting training",
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        profile=profile,
    )

    # Setup
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    logger.info("Using device", device=str(device))

    model, optimizer, criterion = setup_training(device, lr)

    # W&B
    wandb.login(key=settings.WANDB_API_KEY)
    wandb.init(
        project=settings.WANDB_PROJECT_NAME,
        name=f"resnet18-lr{lr}-bs{batch_size}-e{epochs}-{timestamp}",
        tags=["sweep", "optuna"] if is_multirun else ["single-run"],
        config={
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "architecture": "resnet18",
            "dataset": "drone-vs-bird",
            "device": str(device),
            "mode": settings.MODE,
            "profile": profile,
        },
    )
    logger.success("W&B initialized", project=wandb.run.project, run_id=wandb.run.id)

    # Initialize storage
    storage = get_storage()

    train_loader, val_loader, _ = get_dataloaders(
        data_dir=storage.data_dir,
        splits_dir=storage.splits_dir,
        batch_size=batch_size,
        num_workers=0,
        transforms_dict={
            "train": train_transform,
            "val": val_transform,
            "test": val_transform,
        },
    )

    # Setup profiler if enabled (nullcontext if disabled)
    profiler_ctx = get_profiler(f"profiler-{timestamp}") if profile else nullcontext()

    if profile:
        logger.info("PyTorch profiler enabled", output_dir=f"profiler-{timestamp}")

    # Training loop
    best_val_loss = float("inf")
    with profiler_ctx as profiler:
        for epoch in range(epochs):
            train_metrics = train_epoch(
                model, train_loader, optimizer, criterion, device, profiler if profile else None, epoch + 1
            )
            val_metrics = validate_epoch(model, val_loader, criterion, device, epoch + 1)

            # Log to console
            logger.info(
                f"Epoch {epoch + 1}/{epochs}",
                train_loss=train_metrics["loss"],
                train_acc=train_metrics["accuracy"],
                val_loss=val_metrics["loss"],
                val_acc=val_metrics["accuracy"],
            )

            # Log to W&B
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_metrics["loss"],
                    "train_accuracy": train_metrics["accuracy"],
                    "val_loss": val_metrics["loss"],
                    "val_accuracy": val_metrics["accuracy"],
                }
            )

            # Save best model (only if validation loss improved)
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                model_filename = f"model-lr{lr}-bs{batch_size}-e{epochs}-{timestamp}"

                model_path = storage.save_model(model, model_filename)
                logger.success("Best model saved", path=str(model_path), val_loss=best_val_loss)

                # Log model artifacts to W&B
                artifact = wandb.Artifact(
                    name=f"model-lr{lr}-bs{batch_size}-e{epochs}-{timestamp}",
                    type="model",
                    description=f"ResNet18 drone detector (val_loss={best_val_loss:.4f})",
                    metadata={
                        "learning_rate": lr,
                        "batch_size": batch_size,
                        "epochs": epochs,
                        "val_loss": best_val_loss,
                        "val_acc": val_metrics["accuracy"],
                        "architecture": "resnet18",
                    },
                )

                # Add models based on storage mode
                if storage.mode == "local":
                    artifact.add_file(str(model_path))
                    onnx_path = str(model_path).replace(".pth", ".onnx")
                    artifact.add_file(onnx_path)
                    if not is_multirun:
                        artifact.add_file("models/model-latest.pth")
                        artifact.add_file("models/model-latest.onnx")
                else:
                    artifact.add_reference(str(model_path))
                    artifact.add_reference(str(model_path).replace(".pth", ".onnx"))

                wandb.log_artifact(artifact)

    if profile:
        logger.success("Profiling complete", output_dir=f"profiler-{timestamp}")
        logger.info("View results: tensorboard --logdir=profiler-" + timestamp)

    wandb.finish()
    logger.success(
        "Training completed",
        best_val_loss=best_val_loss,
        hyperparameters={"lr": lr, "batch_size": batch_size, "epochs": epochs},
    )

    return best_val_loss  # Return for Optuna optimization


if __name__ == "__main__":
    main()
