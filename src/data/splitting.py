"""
RetinaGuard AI — Data Splitting
================================
Purpose: Create stratified cross-validation folds from the official IDRiD
training partition while preserving the official test set.

Key design decisions:
- Official test set (103 images) is NEVER modified or used for development.
- 5-fold stratified CV is created from the official training set (413 images).
- Patient-level splitting is NOT performed because IDRiD does not provide
  a clearly documented patient-grouping variable.
- Binary label (DR grade ≥ 2) is used for stratification.

Usage:
    python -m src.data.splitting --config configs/base_config.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger("retinaguard.splitting")


def load_config(config_path: Path) -> dict[str, Any]:
    """Load YAML configuration."""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_master_csv(metadata_dir: Path) -> pd.DataFrame:
    """Load the master metadata CSV produced by the audit.

    Args:
        metadata_dir: Directory containing idrid_master.csv.

    Returns:
        Master DataFrame.

    Raises:
        FileNotFoundError: If master CSV is missing.
    """
    master_path = metadata_dir / "idrid_master.csv"
    if not master_path.exists():
        raise FileNotFoundError(
            f"Master CSV not found: {master_path}\n"
            f"Run the data audit first: python -m src.data.audit_dataset"
        )
    df = pd.read_csv(master_path)
    logger.info(f"Loaded master CSV with {len(df)} records")
    return df


def create_cv_folds(
    master: pd.DataFrame,
    n_folds: int,
    seed: int,
    stratify_col: str = "binary_label",
) -> pd.DataFrame:
    """Create stratified k-fold CV splits from the official training set.

    The official test set is preserved without modification.

    Args:
        master: Master DataFrame with partition and label columns.
        n_folds: Number of CV folds.
        seed: Random seed for reproducibility.
        stratify_col: Column to stratify by.

    Returns:
        DataFrame with added 'fold' column (-1 for test set).
    """
    # Separate train and test
    matched = master[master["match_status"] == "matched"].copy()

    # Initialise fold column: -1 = test set (untouched)
    matched["fold"] = -1

    train_mask = matched["partition"] == "train"
    test_mask = matched["partition"] == "test"

    train_data = matched[train_mask].copy()
    test_data = matched[test_mask].copy()

    logger.info(f"Official training set: {len(train_data)} images")
    logger.info(f"Official test set: {len(test_data)} images (preserved)")

    # Validate stratification column exists
    if stratify_col not in train_data.columns:
        raise ValueError(f"Stratification column '{stratify_col}' not found")

    # Drop rows with missing labels
    valid_train = train_data[train_data[stratify_col].notna()].copy()
    excluded = len(train_data) - len(valid_train)
    if excluded > 0:
        logger.warning(f"Excluded {excluded} training images with missing {stratify_col}")

    # Create stratified folds
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    y = valid_train[stratify_col].astype(int).values

    for fold_idx, (_, val_indices) in enumerate(skf.split(valid_train, y)):
        valid_train.iloc[val_indices, valid_train.columns.get_loc("fold")] = fold_idx

    # Recombine
    matched.loc[valid_train.index, "fold"] = valid_train["fold"]

    # Log fold distribution
    for fold in range(n_folds):
        fold_data = valid_train[valid_train["fold"] == fold]
        fold_dist = fold_data[stratify_col].value_counts().sort_index()
        logger.info(f"  Fold {fold}: {len(fold_data)} images, "
                     f"class distribution: {fold_dist.to_dict()}")

    return matched


def verify_split_integrity(
    split_df: pd.DataFrame,
    n_folds: int,
) -> list[str]:
    """Verify the integrity of the data split.

    Checks:
    1. No image appears in multiple folds.
    2. Official test images are untouched (fold = -1).
    3. Class distributions are approximately balanced across folds.
    4. All training images are assigned to a fold.

    Args:
        split_df: DataFrame with fold assignments.
        n_folds: Expected number of folds.

    Returns:
        List of warning/error messages (empty if all checks pass).
    """
    issues: list[str] = []

    # Check 1: No duplicates
    train_data = split_df[split_df["fold"] >= 0]
    if "stem" in train_data.columns:
        dup_check = train_data["stem"].duplicated().sum()
        if dup_check > 0:
            issues.append(f"ERROR: {dup_check} duplicate images across folds!")
    elif "image_id" in train_data.columns:
        dup_check = train_data["image_id"].duplicated().sum()
        if dup_check > 0:
            issues.append(f"ERROR: {dup_check} duplicate images across folds!")

    # Check 2: Test set preserved
    test_data = split_df[split_df["partition"] == "test"]
    test_in_folds = test_data[test_data["fold"] >= 0]
    if len(test_in_folds) > 0:
        issues.append(f"ERROR: {len(test_in_folds)} test images assigned to CV folds!")

    # Check 3: Class balance across folds
    if "binary_label" in train_data.columns:
        fold_props = []
        for fold in range(n_folds):
            fold_data = train_data[train_data["fold"] == fold]
            if len(fold_data) > 0:
                prop = fold_data["binary_label"].mean()
                fold_props.append(prop)
        if fold_props:
            prop_range = max(fold_props) - min(fold_props)
            if prop_range > 0.15:
                issues.append(
                    f"WARNING: Class proportion varies by {prop_range:.2%} across folds"
                )

    # Check 4: All training images assigned
    unassigned = split_df[
        (split_df["partition"] == "train")
        & (split_df["fold"].isna() | (split_df["fold"] < 0))
        & (split_df["binary_label"].notna())
    ]
    if len(unassigned) > 0:
        issues.append(f"WARNING: {len(unassigned)} training images not assigned to any fold")

    for issue in issues:
        if issue.startswith("ERROR"):
            logger.error(issue)
        else:
            logger.warning(issue)

    if not issues:
        logger.info("All integrity checks passed ✓")

    return issues


def generate_split_summary(
    split_df: pd.DataFrame,
    n_folds: int,
    output_dir: Path,
) -> None:
    """Generate split summary table and save to files.

    Args:
        split_df: DataFrame with fold assignments.
        n_folds: Number of CV folds.
        output_dir: Directory for output files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Summary table
    rows = []
    for fold in range(n_folds):
        fold_data = split_df[split_df["fold"] == fold]
        n_total = len(fold_data)
        if "binary_label" in fold_data.columns:
            n_pos = int(fold_data["binary_label"].sum())
            n_neg = n_total - n_pos
        else:
            n_pos = n_neg = "N/A"
        rows.append({
            "partition": f"CV fold {fold}",
            "n_images": n_total,
            "n_negative": n_neg,
            "n_positive": n_pos,
        })

    # Test set
    test_data = split_df[split_df["partition"] == "test"]
    n_test = len(test_data)
    if "binary_label" in test_data.columns:
        n_test_pos = int(test_data["binary_label"].sum())
        n_test_neg = n_test - n_test_pos
    else:
        n_test_pos = n_test_neg = "N/A"
    rows.append({
        "partition": "Official test",
        "n_images": n_test,
        "n_negative": n_test_neg,
        "n_positive": n_test_pos,
    })

    summary_df = pd.DataFrame(rows)
    summary_path = output_dir / "split_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Split summary saved to {summary_path}")

    # Save image IDs per fold
    for fold in range(n_folds):
        fold_data = split_df[split_df["fold"] == fold]
        id_col = "stem" if "stem" in fold_data.columns else "image_id"
        if id_col in fold_data.columns:
            ids = fold_data[id_col].dropna().tolist()
            fold_path = output_dir / f"fold_{fold}_ids.csv"
            pd.DataFrame({"image_id": ids}).to_csv(fold_path, index=False)

    # Save test IDs
    id_col = "stem" if "stem" in test_data.columns else "image_id"
    if id_col in test_data.columns:
        test_ids = test_data[id_col].dropna().tolist()
        test_path = output_dir / "test_ids.csv"
        pd.DataFrame({"image_id": test_ids}).to_csv(test_path, index=False)

    logger.info(f"Image ID lists saved to {output_dir}")


def run_splitting(config_path: Path) -> pd.DataFrame:
    """Execute the complete data splitting pipeline.

    Args:
        config_path: Path to YAML configuration.

    Returns:
        DataFrame with fold assignments.
    """
    config = load_config(config_path)
    project_root = config_path.parent.parent
    metadata_dir = project_root / config["paths"]["metadata_dir"]
    split_config = config["splitting"]
    seed = config["seed"]
    n_folds = split_config["n_folds"]

    logger.info("=" * 60)
    logger.info("RetinaGuard AI — Data Splitting")
    logger.info(f"  Folds: {n_folds}")
    logger.info(f"  Seed: {seed}")
    logger.info(f"  Stratify by: {split_config['stratify_by']}")
    logger.info(f"  Patient-level: {split_config['patient_level']}")
    if not split_config["patient_level"]:
        logger.info(f"  Note: {split_config.get('patient_level_note', 'N/A')}")
    logger.info("=" * 60)

    # Load master CSV
    master = load_master_csv(metadata_dir)

    # Create CV folds
    split_df = create_cv_folds(
        master=master,
        n_folds=n_folds,
        seed=seed,
        stratify_col=split_config["stratify_by"],
    )

    # Verify integrity
    issues = verify_split_integrity(split_df, n_folds)
    if any(i.startswith("ERROR") for i in issues):
        raise RuntimeError(
            "Split integrity check failed. See errors above."
        )

    # Save split metadata
    split_output_dir = metadata_dir / "splits"
    generate_split_summary(split_df, n_folds, split_output_dir)

    # Save complete split DataFrame
    split_path = metadata_dir / "idrid_splits.csv"
    split_df.to_csv(split_path, index=False)
    logger.info(f"Complete split data saved to {split_path}")

    return split_df


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="RetinaGuard AI — Data Splitting"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/base_config.yaml"),
        help="Path to YAML configuration file.",
    )
    args = parser.parse_args()

    try:
        run_splitting(args.config)
    except Exception as e:
        logger.error(f"Splitting failed: {e}", exc_info=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
