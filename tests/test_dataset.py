"""
RetinaGuard AI — PyTorch Dataset Unit Tests
===========================================
Purpose: Verify that IDRiDDataset loads data, handles CV folds, and returns
properly shaped images and labels.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from src.data.dataset import IDRiDDataset


@pytest.fixture
def mock_dataset_environment() -> Tuple[Path, Path]:
    """Create a temporary directory structure with dummy images and metadata CSV.

    Returns:
        Tuple of (temp_dir_path, metadata_csv_path).
    """
    temp_dir = tempfile.TemporaryDirectory()
    temp_path = Path(temp_dir.name)

    # Subdirs
    images_dir = temp_path / "images"
    images_dir.mkdir()

    # Create dummy images
    img_ids = [f"IDRiD_{i:03d}" for i in range(10)]
    img_paths = []
    for img_id in img_ids:
        p = images_dir / f"{img_id}.jpg"
        # 100x100 white image
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        cv2.imwrite(str(p), img)
        img_paths.append(p)

    # Create metadata CSV
    rows = []
    for i, img_id in enumerate(img_ids):
        # 5 train, 5 test
        partition = "train" if i < 8 else "test"
        # Fold assignment for training images
        fold = i % 4 if partition == "train" else -1
        rows.append({
            "image_id": img_id,
            "stem": img_id,
            "filename": f"{img_id}.jpg",
            "full_path": str(img_paths[i]),
            "relative_path": f"images/{img_id}.jpg",
            "partition": partition,
            "dr_grade": i % 5,  # Grades 0 to 4
            "binary_label": 1 if (i % 5) >= 2 else 0,
            "match_status": "matched",
            "readable": True,
            "fold": fold,
        })
    df = pd.DataFrame(rows)
    csv_path = temp_path / "idrid_splits.csv"
    df.to_csv(csv_path, index=False)

    # Keep reference to temp_dir so it doesn't clean up immediately
    pytest.temp_dir_ref = temp_dir
    return temp_path, csv_path


def test_dataset_loading_binary(mock_dataset_environment) -> None:
    """Verify loading binary dataset in train mode."""
    _, csv_path = mock_dataset_environment
    
    # Train dataset (should include folds 1, 2, 3; exclude fold 0)
    dataset = IDRiDDataset(
        metadata_csv=csv_path,
        mode="train",
        task="binary",
        fold=0,
        image_size=64,
        crop_borders=False,
    )
    
    # Total train matches = 8. Fold 0 has index 0 and 4.
    # Excluded count = 2. Expected size = 6.
    assert len(dataset) == 6
    
    # Test getting item
    item = dataset[0]
    assert isinstance(item["image"], torch.Tensor)
    assert item["image"].shape == (3, 64, 64)
    assert item["label"] in (0, 1)
    assert "image_id" in item
    assert "image_path" in item


def test_dataset_loading_val(mock_dataset_environment) -> None:
    """Verify loading validation dataset (single fold)."""
    _, csv_path = mock_dataset_environment
    
    # Val dataset for fold 0 (should include only fold 0 training images)
    dataset = IDRiDDataset(
        metadata_csv=csv_path,
        mode="val",
        task="binary",
        fold=0,
        image_size=64,
        crop_borders=False,
    )
    
    # Fold 0 has indices 0, 4 (2 images)
    assert len(dataset) == 2
    
    # Ensure correct target values
    labels = dataset.get_labels()
    assert len(labels) == 2


def test_dataset_loading_test(mock_dataset_environment) -> None:
    """Verify loading official test set."""
    _, csv_path = mock_dataset_environment
    
    dataset = IDRiDDataset(
        metadata_csv=csv_path,
        mode="test",
        task="binary",
        image_size=64,
        crop_borders=False,
    )
    
    # Test partition has 2 images (idx 8, 9)
    assert len(dataset) == 2


def test_dataset_multiclass(mock_dataset_environment) -> None:
    """Verify multiclass target loading."""
    _, csv_path = mock_dataset_environment
    
    dataset = IDRiDDataset(
        metadata_csv=csv_path,
        mode="inference",
        task="multiclass",
        image_size=64,
        crop_borders=False,
    )
    
    # Inference mode loader holds all 10 images
    assert len(dataset) == 10
    for i in range(10):
        item = dataset[i]
        assert item["label"] == i % 5  # Multiclass is original grade (0-4)
