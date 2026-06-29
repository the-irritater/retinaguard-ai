"""
RetinaGuard AI — PyTorch Dataset
==================================
Purpose: PyTorch Dataset class for loading IDRiD retinal fundus images
with proper preprocessing and augmentation handling.

Supports:
- Binary classification (DR grade ≥ 2 detection).
- Multiclass classification (DR grades 0-4, secondary analysis).
- Train/val/test modes with appropriate transforms.
- Cross-validation fold selection.

Usage:
    from src.data.dataset import IDRiDDataset
    dataset = IDRiDDataset(metadata_csv, mode='train', fold=0, ...)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.preprocessing import (
    get_transforms,
    preprocess_image,
    read_image_safely,
)

logger = logging.getLogger("retinaguard.dataset")


class IDRiDDataset(Dataset):
    """PyTorch Dataset for IDRiD retinal fundus images.

    Handles loading images, applying preprocessing, and returning
    labels for both binary and multiclass tasks.

    Attributes:
        data: DataFrame with image metadata and labels.
        mode: One of 'train', 'val', 'test'.
        task: One of 'binary', 'multiclass'.
        transform: Albumentations transform pipeline.
        image_size: Target image size.
        crop_borders: Whether to crop black borders.
    """

    def __init__(
        self,
        metadata_csv: str | Path,
        mode: str = "train",
        task: str = "binary",
        fold: Optional[int] = None,
        config: Optional[dict[str, Any]] = None,
        transform: Optional[Any] = None,
        image_size: int = 384,
        crop_borders: bool = True,
        border_threshold: int = 10,
    ) -> None:
        """Initialise the IDRiD dataset.

        Args:
            metadata_csv: Path to the split metadata CSV (idrid_splits.csv).
            mode: 'train', 'val', or 'test'.
            task: 'binary' (DR grade ≥ 2) or 'multiclass' (grades 0-4).
            fold: CV fold index for validation. When mode='train', all folds
                  except this one are used for training. When mode='val',
                  only this fold is used. Ignored when mode='test'.
            config: Configuration dictionary for transforms.
            transform: Override transform pipeline (if None, built from config).
            image_size: Target image size for preprocessing.
            crop_borders: Whether to crop black borders.
            border_threshold: Threshold for border detection.

        Raises:
            FileNotFoundError: If metadata CSV does not exist.
            ValueError: If mode or task is invalid.
        """
        super().__init__()

        if mode not in ("train", "val", "test", "inference"):
            raise ValueError(f"Invalid mode: {mode}. Use 'train', 'val', 'test', or 'inference'.")
        if task not in ("binary", "multiclass"):
            raise ValueError(f"Invalid task: {task}. Use 'binary' or 'multiclass'.")

        self.mode = mode
        self.task = task
        self.image_size = image_size
        self.crop_borders = crop_borders
        self.border_threshold = border_threshold

        # Load metadata
        metadata_path = Path(metadata_csv)
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata CSV not found: {metadata_path}")

        df = pd.read_csv(metadata_path)
        logger.info(f"Loaded metadata: {len(df)} total records")

        # Filter by mode and fold
        self.data = self._filter_data(df, mode, fold)
        logger.info(
            f"Dataset [{mode}] (task={task}): {len(self.data)} images"
        )

        # Set up transforms
        if transform is not None:
            self.transform = transform
        elif config is not None:
            self.transform = get_transforms(mode, config)
        else:
            # Minimal default config
            default_config = {
                "preprocessing": {
                    "image_size": image_size,
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                },
                "augmentation": {
                    "horizontal_flip": False,
                    "rotation_limit": 15,
                    "translate_percent": 0.05,
                    "scale_range": [0.9, 1.1],
                    "brightness_limit": 0.1,
                    "contrast_limit": 0.1,
                },
            }
            self.transform = get_transforms(mode, default_config)

    def _filter_data(
        self,
        df: pd.DataFrame,
        mode: str,
        fold: Optional[int],
    ) -> pd.DataFrame:
        """Filter DataFrame by mode and fold assignment.

        For cross-validation:
        - train: all folds except the specified fold.
        - val: only the specified fold.
        - test: official test set (fold == -1).

        Args:
            df: Full metadata DataFrame.
            mode: Dataset mode.
            fold: CV fold index.

        Returns:
            Filtered DataFrame.
        """
        # Filter to matched records with valid labels
        if "match_status" in df.columns:
            df = df[df["match_status"] == "matched"].copy()

        label_col = "binary_label" if self.task == "binary" else "dr_grade"
        if label_col in df.columns:
            df = df[df[label_col].notna()].copy()

        if mode == "test":
            # Official test set
            filtered = df[df["partition"] == "test"].copy()
        elif mode == "train":
            if fold is not None:
                # All training folds except the validation fold
                filtered = df[
                    (df["partition"] == "train") & (df["fold"] != fold) & (df["fold"] >= 0)
                ].copy()
            else:
                # Full training set (for final refit)
                filtered = df[df["partition"] == "train"].copy()
        elif mode == "val":
            if fold is not None:
                # Only the validation fold
                filtered = df[
                    (df["partition"] == "train") & (df["fold"] == fold)
                ].copy()
            else:
                raise ValueError("fold must be specified for validation mode")
        elif mode == "inference":
            filtered = df.copy()
        else:
            raise ValueError(f"Unknown mode: {mode}")

        filtered = filtered.reset_index(drop=True)
        return filtered

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a single sample.

        Returns:
            Dictionary with keys:
            - 'image': Preprocessed image tensor.
            - 'label': Integer label (binary or multiclass).
            - 'image_id': Image identifier string.
            - 'image_path': Path to original image.
            - 'dr_grade': Original DR grade (0-4).
            - 'dme_grade': DME grade.
            - 'binary_label': Binary label.
        """
        row = self.data.iloc[idx]

        # Read image
        image_path = row.get("full_path", "")
        image = read_image_safely(image_path)

        if image is None:
            # Return a black image as fallback (logged in read_image_safely)
            image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

        # Preprocess (border cropping)
        image = preprocess_image(
            image,
            image_size=self.image_size,
            border_threshold=self.border_threshold,
            crop_borders=self.crop_borders,
        )

        # Apply transforms (augmentation + normalisation + tensor conversion)
        if self.transform is not None:
            transformed = self.transform(image=image)
            image_tensor = transformed["image"]
        else:
            # Fallback: simple tensor conversion
            image_tensor = torch.from_numpy(
                image.transpose(2, 0, 1).astype(np.float32) / 255.0
            )

        # Get label
        if self.task == "binary":
            label = int(row.get("binary_label", 0))
        else:
            label = int(row.get("dr_grade", 0))

        return {
            "image": image_tensor,
            "label": label,
            "image_id": str(row.get("stem", row.get("image_id", f"unknown_{idx}"))),
            "image_path": str(image_path),
            "dr_grade": int(row.get("dr_grade", -1)) if pd.notna(row.get("dr_grade")) else -1,
            "dme_grade": int(row.get("dme_grade", -1)) if pd.notna(row.get("dme_grade")) else -1,
            "binary_label": int(row.get("binary_label", -1)) if pd.notna(row.get("binary_label")) else -1,
        }

    def get_labels(self) -> np.ndarray:
        """Return all labels as a numpy array (for computing class weights).

        Returns:
            Array of integer labels.
        """
        label_col = "binary_label" if self.task == "binary" else "dr_grade"
        return self.data[label_col].astype(int).values

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for loss function.

        Returns:
            Tensor of class weights.
        """
        labels = self.get_labels()
        n_classes = len(np.unique(labels))
        class_counts = np.bincount(labels, minlength=n_classes)
        # Inverse frequency, normalised
        weights = 1.0 / (class_counts + 1e-6)
        weights = weights / weights.sum() * n_classes
        return torch.FloatTensor(weights)

    def get_class_distribution(self) -> dict[int, int]:
        """Return class distribution as a dictionary.

        Returns:
            Dictionary mapping class labels to counts.
        """
        labels = self.get_labels()
        unique, counts = np.unique(labels, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))
