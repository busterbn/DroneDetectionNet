"""Generate reference dataset from training data for drift monitoring.

This script extracts features from training images and saves them as
a reference dataset for comparison with production data.
"""

import json
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import typer
from tqdm import tqdm

from drone_detector_mlops.monitoring.feature_extraction import ImageFeatureExtractor
from drone_detector_mlops.utils.logger import get_logger

logger = get_logger(__name__)
app = typer.Typer()


@app.command()
def generate_reference_dataset(
    data_dir: str = "data",
    output_path: str = "data/reference_dataset.csv",
    split: str = "train",
    max_samples: int | None = None,
):
    """Generate reference dataset from training data.

    Args:
        data_dir: Directory containing structured data (drone/, bird/ folders)
        output_path: Path to save the reference dataset CSV
        split: Which split to use (train/val/test)
        max_samples: Maximum number of samples to process (None = all)
    """
    logger.info(f"Generating reference dataset from {split} split")
    data_dir = Path(data_dir)

    # Read split file
    split_file = data_dir / f"{split}_split.txt"
    if not split_file.exists():
        logger.error(f"Split file not found: {split_file}")
        raise FileNotFoundError(f"Split file not found: {split_file}")

    with open(split_file) as f:
        image_paths = [line.strip() for line in f if line.strip()]

    if max_samples:
        image_paths = image_paths[:max_samples]
        logger.info(f"Limiting to {max_samples} samples")

    logger.info(f"Processing {len(image_paths)} images")

    # Extract features from all images
    records = []
    extractor = ImageFeatureExtractor()

    for rel_path in tqdm(image_paths, desc="Extracting features"):
        full_path = data_dir / rel_path

        if not full_path.exists():
            logger.warning(f"Image not found: {full_path}")
            continue

        try:
            # Extract features
            features = extractor.extract_features_from_path(str(full_path))

            # Determine class from path
            class_name = "drone" if "drone" in rel_path.lower() else "bird"
            class_int = 0 if class_name == "drone" else 1

            # Add metadata
            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "class_name": class_name,
                "class_int": class_int,
                "image_path": rel_path,
                **features,
            }

            records.append(record)

        except Exception as e:
            logger.error(f"Failed to process {full_path}: {e}")
            continue

    # Create DataFrame and save
    df = pd.DataFrame(records)
    logger.info(f"Extracted features from {len(df)} images")

    # Save to CSV
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.success(f"Reference dataset saved to {output_path}")

    # Print statistics
    logger.info("\nDataset Statistics:")
    logger.info(f"Total samples: {len(df)}")
    logger.info(f"Class distribution:\n{df['class_name'].value_counts()}")
    logger.info(f"\nFeature columns: {list(df.columns)}")

    # Save metadata
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "split": split,
        "total_samples": len(df),
        "class_distribution": df["class_name"].value_counts().to_dict(),
        "feature_names": ImageFeatureExtractor.get_feature_names(),
        "data_dir": str(data_dir),
    }

    metadata_path = output_path.with_suffix(".json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.success(f"Metadata saved to {metadata_path}")


if __name__ == "__main__":
    app()
