"""
RetinaGuard AI - Data Splitting Unit Tests
-
Purpose: Verify that the data splitting logic is correct:
- Official test partition is untouched.
- Training set is split into stratified k-folds.
- Class proportion is balanced across folds.
- No duplicate images exist across partitions.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.splitting import create_cv_folds, verify_split_integrity


@pytest.fixture
def dummy_master_df() -> pd.DataFrame:
    """Create a dummy master metadata dataframe representing the audit output.

    Contains 100 train images and 25 test images, with binary labels.
    """
    rows = []
    # Train set (100 images)
    for i in range(100):
        # 30% positive rate
        label = 1 if i % 10 < 3 else 0
        rows.append({
            "image_id": f"IDRiD_{i:03d}",
            "stem": f"IDRiD_{i:03d}",
            "partition": "train",
            "binary_label": label,
            "match_status": "matched",
            "readable": True,
        })
    # Test set (25 images)
    for i in range(100, 125):
        label = 1 if i % 10 < 3 else 0
        rows.append({
            "image_id": f"IDRiD_{i:03d}",
            "stem": f"IDRiD_{i:03d}",
            "partition": "test",
            "binary_label": label,
            "match_status": "matched",
            "readable": True,
        })
    return pd.DataFrame(rows)


def test_cv_folds_creation(dummy_master_df: pd.DataFrame) -> None:
    """Verify that create_cv_folds properly assigns train partition to folds and keeps test untouched."""
    n_folds = 5
    seed = 42
    
    split_df = create_cv_folds(
        master=dummy_master_df,
        n_folds=n_folds,
        seed=seed,
        stratify_col="binary_label",
    )

    # Check that test set remains fold = -1
    test_rows = split_df[split_df["partition"] == "test"]
    assert (test_rows["fold"] == -1).all(), "Test set images must have fold = -1"

    # Check training rows are divided into 0 to 4 folds
    train_rows = split_df[split_df["partition"] == "train"]
    assert (train_rows["fold"] >= 0).all()
    assert (train_rows["fold"] < n_folds).all()

    # Check fold distribution counts (100 images / 5 folds = 20 images per fold)
    fold_counts = train_rows["fold"].value_counts()
    for fold in range(n_folds):
        assert fold_counts[fold] == 20, f"Expected 20 images in fold {fold}, got {fold_counts[fold]}"


def test_cv_folds_stratification(dummy_master_df: pd.DataFrame) -> None:
    """Verify that class stratification is maintained across folds."""
    n_folds = 5
    seed = 42
    
    split_df = create_cv_folds(
        master=dummy_master_df,
        n_folds=n_folds,
        seed=seed,
        stratify_col="binary_label",
    )

    # Overall positive class rate is 30% (30 positive / 100 total)
    train_rows = split_df[split_df["partition"] == "train"]
    for fold in range(n_folds):
        fold_data = train_rows[train_rows["fold"] == fold]
        n_pos = fold_data["binary_label"].sum()
        # Stratification should assign exactly 6 positive cases to each fold (30% of 20)
        assert n_pos == 6, f"Expected exactly 6 positive cases in fold {fold}, got {n_pos}"


def test_verify_split_integrity(dummy_master_df: pd.DataFrame) -> None:
    """Test the integrity verification helper function."""
    n_folds = 5
    seed = 42
    
    split_df = create_cv_folds(
        master=dummy_master_df,
        n_folds=n_folds,
        seed=seed,
        stratify_col="binary_label",
    )

    # Integrity verification should pass
    issues = verify_split_integrity(split_df, n_folds)
    assert len(issues) == 0, f"Expected no integrity issues, got: {issues}"

    # Artificially introduce a leakage error (test image placed in fold 1)
    split_df.loc[split_df["partition"] == "test", "fold"] = 1
    issues = verify_split_integrity(split_df, n_folds)
    assert any("test images assigned to CV folds" in issue for issue in issues), "Failed to detect test leakage"
