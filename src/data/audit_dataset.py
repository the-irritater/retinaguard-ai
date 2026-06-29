"""
RetinaGuard AI - IDRiD Dataset Audit
-
Purpose: Discover, validate, and characterise all images and labels in the
IDRiD dataset. Produces a master metadata CSV and an audit summary report.

This script:
- Finds all image files in the IDRiD directory structure.
- Reads DR and DME label CSV files.
- Matches image IDs to labels.
- Identifies missing images, missing labels, unreadable images, duplicates.
- Calculates file hashes (MD5) for exact duplicate detection.
- Records image dimensions, aspect ratio, brightness, contrast, sharpness,
  and black-border proportion.
- Produces data/metadata/idrid_master.csv.
- Produces reports/data_audit_summary.md.
- Fails with clear error messages if required files are absent.

Usage:
    python -m src.data.audit_dataset --config configs/base_config.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import pandas as pd
import yaml

# -
# Logging
# -
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("retinaguard.audit")


# -
# Configuration loader
# -
def load_config(config_path: Path) -> dict[str, Any]:
    """Load YAML configuration file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Dictionary with configuration values.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# -
# Image discovery
# -
def discover_images(
    root_dir: Path, extensions: list[str]
) -> list[dict[str, Any]]:
    """Recursively find all image files under root_dir.

    Args:
        root_dir: Root directory to search.
        extensions: List of valid image file extensions (e.g. ['.jpg', '.png']).

    Returns:
        List of dicts with keys: filename, stem, suffix, full_path, relative_path.
    """
    if not root_dir.exists():
        raise FileNotFoundError(
            f"Data root directory not found: {root_dir}\n"
            f"Please download IDRiD and place it in {root_dir}"
        )

    images: list[dict[str, Any]] = []
    ext_set = {e.lower() for e in extensions}

    for fpath in sorted(root_dir.rglob("*")):
        if fpath.is_file() and fpath.suffix.lower() in ext_set:
            images.append(
                {
                    "filename": fpath.name,
                    "stem": fpath.stem,
                    "suffix": fpath.suffix.lower(),
                    "full_path": str(fpath),
                    "relative_path": str(fpath.relative_to(root_dir)),
                }
            )

    logger.info(f"Discovered {len(images)} image files under {root_dir}")
    return images


# -
# Label loading
# -
def load_labels(
    labels_dir: Path,
    train_csv_name: str,
    test_csv_name: str,
    image_id_col: str,
    dr_col: str,
    dme_col: str,
) -> pd.DataFrame:
    """Load and combine training and testing label CSV files.

    Args:
        labels_dir: Directory containing label CSV files.
        train_csv_name: Filename of training labels CSV.
        test_csv_name: Filename of testing labels CSV.
        image_id_col: Column name for image identifiers.
        dr_col: Column name for DR grade.
        dme_col: Column name for DME grade.

    Returns:
        Combined DataFrame with partition column.

    Raises:
        FileNotFoundError: If either CSV file is missing.
    """
    train_path = labels_dir / train_csv_name
    test_path = labels_dir / test_csv_name

    if not train_path.exists():
        raise FileNotFoundError(
            f"Training labels not found: {train_path}\n"
            f"Expected IDRiD label file at this location."
        )
    if not test_path.exists():
        raise FileNotFoundError(
            f"Testing labels not found: {test_path}\n"
            f"Expected IDRiD label file at this location."
        )

    train_df = pd.read_csv(train_path)
    train_df["partition"] = "train"
    logger.info(f"Loaded {len(train_df)} training labels from {train_path}")

    test_df = pd.read_csv(test_path)
    test_df["partition"] = "test"
    logger.info(f"Loaded {len(test_df)} testing labels from {test_path}")

    # Standardise column names (handle trailing spaces in IDRiD CSVs)
    for df in [train_df, test_df]:
        df.columns = df.columns.str.strip()

    combined = pd.concat([train_df, test_df], ignore_index=True)

    # Rename columns to standard names
    col_map = {}
    for col in combined.columns:
        stripped = col.strip()
        if stripped == image_id_col.strip():
            col_map[col] = "image_id"
        elif stripped == dr_col.strip():
            col_map[col] = "dr_grade"
        elif stripped == dme_col.strip():
            col_map[col] = "dme_grade"
    combined.rename(columns=col_map, inplace=True)

    # Clean image IDs
    if "image_id" in combined.columns:
        combined["image_id"] = combined["image_id"].astype(str).str.strip()

    logger.info(f"Combined labels: {len(combined)} total records")
    return combined


# -
# File hash calculation
# -
def calculate_md5(filepath: str) -> str:
    """Calculate MD5 hash of a file for duplicate detection.

    Args:
        filepath: Path to the file.

    Returns:
        Hexadecimal MD5 hash string.
    """
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


# -
# Image quality metrics
# -
def compute_image_metrics(filepath: str) -> dict[str, Any]:
    """Compute quality metrics for a single image.

    Args:
        filepath: Path to the image file.

    Returns:
        Dictionary with width, height, aspect_ratio, mean_brightness,
        contrast, sharpness, black_border_proportion, and readable flag.
    """
    result: dict[str, Any] = {
        "width": None,
        "height": None,
        "aspect_ratio": None,
        "mean_brightness": None,
        "contrast": None,
        "sharpness": None,
        "black_border_proportion": None,
        "readable": False,
        "read_error": None,
    }

    try:
        img = cv2.imread(filepath, cv2.IMREAD_COLOR)
        if img is None:
            result["read_error"] = "cv2.imread returned None"
            return result

        result["readable"] = True
        h, w = img.shape[:2]
        result["width"] = w
        result["height"] = h
        result["aspect_ratio"] = round(w / h, 4) if h > 0 else None

        # Convert to grayscale for quality metrics
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Mean brightness
        result["mean_brightness"] = round(float(gray.mean()), 2)

        # Contrast (standard deviation of pixel intensities)
        result["contrast"] = round(float(gray.std()), 2)

        # Sharpness (variance of Laplacian)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        result["sharpness"] = round(float(laplacian.var()), 2)

        # Black border proportion
        # Count pixels below a brightness threshold
        border_threshold = 10
        total_pixels = h * w
        black_pixels = int(np.sum(gray < border_threshold))
        result["black_border_proportion"] = round(
            black_pixels / total_pixels, 4
        )

    except Exception as e:
        result["read_error"] = str(e)

    return result


# -
# Matching images to labels
# -
def match_images_to_labels(
    images: list[dict[str, Any]], labels: pd.DataFrame
) -> pd.DataFrame:
    """Match discovered image files to label records.

    Args:
        images: List of image dicts from discover_images().
        labels: DataFrame from load_labels().

    Returns:
        Master DataFrame with matched and unmatched records.
    """
    img_df = pd.DataFrame(images)

    # IDRiD image names are like IDRiD_001, matching the label CSV
    label_ids = set(labels["image_id"].values) if "image_id" in labels.columns else set()

    # Try matching by stem
    img_df["matched_label"] = img_df["stem"].isin(label_ids)

    # Merge with labels
    master = img_df.merge(
        labels, left_on="stem", right_on="image_id", how="outer",
        indicator=True, suffixes=("_img", "_label"),
    )

    master["match_status"] = master["_merge"].map(
        {"both": "matched", "left_only": "image_no_label", "right_only": "label_no_image"}
    )

    logger.info(
        f"Match results: "
        f"{(master['match_status'] == 'matched').sum()} matched, "
        f"{(master['match_status'] == 'image_no_label').sum()} images without labels, "
        f"{(master['match_status'] == 'label_no_image').sum()} labels without images"
    )

    return master


# -
# Duplicate detection
# -
def detect_duplicates(master: pd.DataFrame) -> dict[str, Any]:
    """Detect filename and content duplicates.

    Args:
        master: Master DataFrame with file paths.

    Returns:
        Dictionary with duplicate information.
    """
    results: dict[str, Any] = {
        "filename_duplicates": [],
        "hash_duplicates": [],
    }

    # Filename duplicates
    if "filename" in master.columns:
        name_counts = master["filename"].dropna().value_counts()
        dupes = name_counts[name_counts > 1]
        if len(dupes) > 0:
            results["filename_duplicates"] = dupes.to_dict()
            logger.warning(f"Found {len(dupes)} duplicate filenames")

    # Hash duplicates (only for matched files)
    hash_col = "md5_hash"
    if hash_col in master.columns:
        hash_counts = master[hash_col].dropna().value_counts()
        hash_dupes = hash_counts[hash_counts > 1]
        if len(hash_dupes) > 0:
            results["hash_duplicates"] = hash_dupes.to_dict()
            logger.warning(f"Found {len(hash_dupes)} content-duplicate hashes")

    return results


# -
# Binary target creation
# -
def create_binary_target(master: pd.DataFrame) -> pd.DataFrame:
    """Add binary classification target column.

    Binary target definition (study-specific):
        DR grades 0-1 → 0 (DR grade < 2)
        DR grades 2-4 → 1 (DR grade ≥ 2)

    This is an operational research definition, NOT a universal clinical
    referral standard.

    Args:
        master: Master DataFrame with dr_grade column.

    Returns:
        DataFrame with added binary_label column.
    """
    if "dr_grade" in master.columns:
        master["binary_label"] = (
            master["dr_grade"]
            .apply(lambda x: 1 if pd.notna(x) and int(x) >= 2 else (0 if pd.notna(x) else None))
        )
        master["binary_label_name"] = master["binary_label"].map(
            {0: "DR grade < 2", 1: "DR grade >= 2"}
        )
        counts = master["binary_label"].value_counts()
        logger.info(f"Binary target distribution: {counts.to_dict()}")
    return master


# -
# Generate audit summary report
# -
def generate_audit_report(
    master: pd.DataFrame,
    duplicates: dict[str, Any],
    output_path: Path,
    config: dict[str, Any],
) -> None:
    """Write the data audit summary as a Markdown report.

    Args:
        master: Complete master DataFrame.
        duplicates: Duplicate detection results.
        output_path: Path for the output report.
        config: Project configuration.
    """
    matched = master[master["match_status"] == "matched"]
    readable = matched[matched["readable"] == True]

    lines: list[str] = []
    lines.append("# IDRiD Data Audit Summary\n")
    lines.append(f"**Generated:** {datetime.now().isoformat()}\n")
    lines.append(f"**Dataset:** {config.get('dataset', {}).get('name', 'IDRiD')}\n")
    lines.append("---\n")

    # Overview
    lines.append("## 1. Overview\n")
    lines.append(f"| Metric | Count |")
    lines.append(f"|-|-|")
    lines.append(f"| Total image files discovered | {master['filename'].notna().sum()} |")
    lines.append(f"| Total label records | {master['image_id'].notna().sum()} |")
    lines.append(f"| Matched (image + label) | {(master['match_status'] == 'matched').sum()} |")
    lines.append(f"| Images without labels | {(master['match_status'] == 'image_no_label').sum()} |")
    lines.append(f"| Labels without images | {(master['match_status'] == 'label_no_image').sum()} |")
    lines.append(f"| Readable images | {readable.shape[0]} |")
    lines.append(f"| Unreadable images | {matched.shape[0] - readable.shape[0]} |")
    lines.append("")

    # Partition breakdown
    lines.append("## 2. Partition Breakdown\n")
    if "partition" in matched.columns:
        part_counts = matched["partition"].value_counts()
        lines.append("| Partition | Images |")
        lines.append("|-|-|")
        for part, count in sorted(part_counts.items()):
            lines.append(f"| {part} | {count} |")
        lines.append("")

    # DR grade distribution
    lines.append("## 3. DR Grade Distribution\n")
    if "dr_grade" in matched.columns:
        dr_counts = matched["dr_grade"].value_counts().sort_index()
        lines.append("| DR Grade | Count | Percentage |")
        lines.append("|-|-|-|")
        total = dr_counts.sum()
        for grade, count in dr_counts.items():
            pct = round(100 * count / total, 1)
            lines.append(f"| {int(grade)} | {count} | {pct}% |")
        lines.append("")

    # Binary target distribution
    lines.append("## 4. Binary Target Distribution\n")
    lines.append("> **Study-specific definition:** DR grades 0-1 → Class 0; "
                 "DR grades 2-4 → Class 1. This is NOT a universal referral standard.\n")
    if "binary_label" in matched.columns:
        bin_counts = matched["binary_label"].value_counts().sort_index()
        lines.append("| Binary Label | Count | Percentage |")
        lines.append("|-|-|-|")
        total = bin_counts.sum()
        for label, count in bin_counts.items():
            name = "DR grade < 2" if label == 0 else "DR grade >= 2"
            pct = round(100 * count / total, 1)
            lines.append(f"| {name} | {count} | {pct}% |")
        lines.append("")

    # DME distribution
    lines.append("## 5. DME Grade Distribution\n")
    if "dme_grade" in matched.columns:
        dme_counts = matched["dme_grade"].value_counts().sort_index()
        lines.append("| DME Grade | Count | Percentage |")
        lines.append("|-|-|-|")
        total = dme_counts.sum()
        for grade, count in dme_counts.items():
            pct = round(100 * count / total, 1)
            lines.append(f"| {int(grade)} | {count} | {pct}% |")
        lines.append("")

    # Image dimensions
    lines.append("## 6. Image Dimensions\n")
    if "width" in readable.columns:
        lines.append(f"| Statistic | Width | Height |")
        lines.append(f"|-|-|-|")
        for stat in ["min", "max", "mean", "median", "std"]:
            w = readable["width"].agg(stat)
            h = readable["height"].agg(stat)
            lines.append(f"| {stat.capitalize()} | {w:.0f} | {h:.0f} |")
        lines.append("")

    # Quality metrics
    lines.append("## 7. Image Quality Metrics\n")
    for metric, label in [
        ("mean_brightness", "Mean Brightness"),
        ("contrast", "Contrast (Std Dev)"),
        ("sharpness", "Sharpness (Laplacian Var)"),
        ("black_border_proportion", "Black Border Proportion"),
    ]:
        if metric in readable.columns:
            vals = readable[metric].dropna()
            lines.append(f"### {label}\n")
            lines.append(f"| Statistic | Value |")
            lines.append(f"|-|-|")
            lines.append(f"| Min | {vals.min():.2f} |")
            lines.append(f"| Max | {vals.max():.2f} |")
            lines.append(f"| Mean | {vals.mean():.2f} |")
            lines.append(f"| Median | {vals.median():.2f} |")
            lines.append(f"| Std | {vals.std():.2f} |")
            lines.append("")

    # Duplicates
    lines.append("## 8. Duplicate Detection\n")
    if duplicates["filename_duplicates"]:
        lines.append(f"**WARNING:** Found {len(duplicates['filename_duplicates'])} "
                     f"duplicate filenames.\n")
        for name, count in duplicates["filename_duplicates"].items():
            lines.append(f"- `{name}`: {count} occurrences")
    else:
        lines.append("No duplicate filenames detected.\n")

    if duplicates["hash_duplicates"]:
        lines.append(f"\n**WARNING:** Found {len(duplicates['hash_duplicates'])} "
                     f"content-identical file pairs.\n")
    else:
        lines.append("No content-identical duplicates detected.\n")

    # Unreadable images
    lines.append("## 9. Unreadable Images\n")
    unreadable = matched[matched["readable"] != True]
    if len(unreadable) > 0:
        lines.append(f"**WARNING:** {len(unreadable)} images could not be read.\n")
        for _, row in unreadable.iterrows():
            lines.append(f"- `{row.get('filename', 'unknown')}`: "
                         f"{row.get('read_error', 'unknown error')}")
    else:
        lines.append("All matched images are readable.\n")

    # Integrity checks
    lines.append("## 10. Integrity Checks\n")
    checks = []
    n_matched = (master["match_status"] == "matched").sum()
    n_no_label = (master["match_status"] == "image_no_label").sum()
    n_no_image = (master["match_status"] == "label_no_image").sum()
    n_unreadable = matched.shape[0] - readable.shape[0]

    checks.append(("All labels have images", "PASS" if n_no_image == 0 else "FAIL"))
    checks.append(("All images have labels", "PASS" if n_no_label == 0 else "WARN"))
    checks.append(("All images readable", "PASS" if n_unreadable == 0 else "FAIL"))
    checks.append(("No filename duplicates",
                    "PASS" if not duplicates["filename_duplicates"] else "WARN"))
    checks.append(("No content duplicates",
                    "PASS" if not duplicates["hash_duplicates"] else "WARN"))

    lines.append("| Check | Status |")
    lines.append("|-|-|")
    for check, status in checks:
        emoji = "" if status == "PASS" else ("" if status == "WARN" else "")
        lines.append(f"| {check} | {emoji} {status} |")
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Audit report written to {output_path}")


# -
# Main audit pipeline
# -
def run_audit(config_path: Path) -> pd.DataFrame:
    """Execute the complete dataset audit pipeline.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Master DataFrame with all metadata.

    Raises:
        FileNotFoundError: If required files or directories are missing.
    """
    config = load_config(config_path)
    project_root = config_path.parent.parent  # configs/ -> project root

    # Resolve paths
    data_root = project_root / config["paths"]["data_root"]
    metadata_dir = project_root / config["paths"]["metadata_dir"]
    reports_dir = project_root / config["paths"]["reports_dir"]
    ds_config = config["dataset"]

    logger.info("=" * 60)
    logger.info("RetinaGuard AI - IDRiD Dataset Audit")
    logger.info("=" * 60)

    # Step 1: Discover images
    logger.info("Step 1: Discovering image files...")
    extensions = ds_config["image_extensions"]
    images = discover_images(data_root, extensions)

    if len(images) == 0:
        raise FileNotFoundError(
            f"No image files found in {data_root}.\n"
            f"Please download IDRiD and extract it into {data_root}.\n"
            f"Expected structure:\n"
            f"  {data_root}/{ds_config['train_images_dir']}/\n"
            f"  {data_root}/{ds_config['test_images_dir']}/\n"
            f"  {data_root}/{ds_config['labels_dir']}/"
        )

    # Step 2: Load labels
    logger.info("Step 2: Loading label files...")
    labels_dir = data_root / ds_config["labels_dir"]
    labels = load_labels(
        labels_dir=labels_dir,
        train_csv_name=ds_config["train_labels_csv"],
        test_csv_name=ds_config["test_labels_csv"],
        image_id_col=ds_config["image_id_col"],
        dr_col=ds_config["dr_grade_col"],
        dme_col=ds_config["dme_grade_col"],
    )

    # Step 3: Match images to labels
    logger.info("Step 3: Matching images to labels...")
    master = match_images_to_labels(images, labels)

    # Step 4: Compute file hashes (matched images only)
    logger.info("Step 4: Computing file hashes...")
    matched_mask = master["match_status"] == "matched"
    hashes = []
    for idx, row in master.iterrows():
        if matched_mask[idx] and pd.notna(row.get("full_path")):
            hashes.append(calculate_md5(row["full_path"]))
        else:
            hashes.append(None)
    master["md5_hash"] = hashes

    # Step 5: Compute image quality metrics
    logger.info("Step 5: Computing image quality metrics...")
    metrics_list = []
    for idx, row in master.iterrows():
        if matched_mask[idx] and pd.notna(row.get("full_path")):
            metrics = compute_image_metrics(row["full_path"])
            metrics_list.append(metrics)
        else:
            metrics_list.append({
                "width": None, "height": None, "aspect_ratio": None,
                "mean_brightness": None, "contrast": None, "sharpness": None,
                "black_border_proportion": None, "readable": None,
                "read_error": None,
            })

    metrics_df = pd.DataFrame(metrics_list, index=master.index)
    for col in metrics_df.columns:
        master[col] = metrics_df[col]

    # Step 6: Detect duplicates
    logger.info("Step 6: Detecting duplicates...")
    duplicates = detect_duplicates(master)

    # Step 7: Create binary target
    logger.info("Step 7: Creating binary classification target...")
    master = create_binary_target(master)

    # Step 8: Determine partition from image path
    logger.info("Step 8: Verifying partitions...")
    train_dir_pattern = ds_config["train_images_dir"].lower()
    test_dir_pattern = ds_config["test_images_dir"].lower()

    for idx, row in master.iterrows():
        if pd.notna(row.get("relative_path")):
            rel = str(row["relative_path"]).lower()
            if "training" in rel or "train" in rel:
                if pd.isna(row.get("partition")):
                    master.at[idx, "partition"] = "train"
            elif "testing" in rel or "test" in rel:
                if pd.isna(row.get("partition")):
                    master.at[idx, "partition"] = "test"

    # Step 9: Save master CSV
    logger.info("Step 9: Saving master metadata CSV...")
    metadata_dir.mkdir(parents=True, exist_ok=True)
    master_path = metadata_dir / "idrid_master.csv"
    master.to_csv(master_path, index=False)
    logger.info(f"Master CSV saved to {master_path} ({len(master)} records)")

    # Step 10: Generate audit report
    logger.info("Step 10: Generating audit summary report...")
    report_path = reports_dir / "data_audit_summary.md"
    generate_audit_report(master, duplicates, report_path, config)

    # Summary
    logger.info("=" * 60)
    logger.info("Audit complete!")
    logger.info(f"  Master CSV: {master_path}")
    logger.info(f"  Report:     {report_path}")
    logger.info("=" * 60)

    return master


# -
# CLI entry point
# -
def main() -> None:
    """CLI entry point for the data audit."""
    parser = argparse.ArgumentParser(
        description="RetinaGuard AI - IDRiD Dataset Audit"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/base_config.yaml"),
        help="Path to YAML configuration file.",
    )
    args = parser.parse_args()

    try:
        run_audit(args.config)
    except FileNotFoundError as e:
        logger.error(f"Missing required files:\n{e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Audit failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
