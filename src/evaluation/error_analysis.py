"""
RetinaGuard AI — Systematic Error Analysis
============================================
Purpose: Analyse classification errors, categorise failure modes, and evaluate
performance across image quality quartiles and disease grades.

This module:
- Generates a comprehensive error table (CSV).
- Produces detailed reports for:
  - False negatives (FN)
  - False positives (FP)
  - Severe undergrading (for multiclass)
  - High-confidence mistakes (errors with probability >= 0.80 or <= 0.20)
  - Low-confidence correct predictions (correct with probability between 0.40 and 0.60)
  - Error rates broken down by DR grade, DME grade, brightness quartile, and sharpness quartile.
- Produces reports/error_analysis_report.md.

Usage:
    python -m src.evaluation.error_analysis --config configs/base_config.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger("retinaguard.error_analysis")


def categorize_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> List[str]:
    """Categorise prediction outcomes into TP, TN, FP, FN.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        y_prob: Positive class probabilities.
        threshold: Classification threshold.

    Returns:
        List of category strings.
    """
    categories = []
    for t, p in zip(y_true, y_pred):
        if t == 1 and p == 1:
            categories.append("TP")
        elif t == 0 and p == 0:
            categories.append("TN")
        elif t == 0 and p == 1:
            categories.append("FP")
        elif t == 1 and p == 0:
            categories.append("FN")
        else:
            categories.append("Unknown")
    return categories


def analyze_errors(
    predictions_df: pd.DataFrame,
    threshold: float = 0.5,
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Perform detailed error analysis on model predictions.

    Args:
        predictions_df: DataFrame containing at least:
            - image_id
            - true_label (binary)
            - predicted_probability
            - dr_grade (original 0-4)
            - dme_grade (original 0-2)
            - mean_brightness
            - sharpness
            - full_path
        threshold: Classification threshold.
        output_dir: Directory to save generated reports.

    Returns:
        DataFrame with error categories and metrics added.
    """
    df = predictions_df.copy()

    # Derived prediction and confidence
    df["predicted_label"] = (df["predicted_probability"] >= threshold).astype(int)
    df["error_category"] = categorize_errors(
        df["true_label"].values,
        df["predicted_label"].values,
        df["predicted_probability"].values,
        threshold,
    )
    df["is_error"] = (df["predicted_label"] != df["true_label"]).astype(int)
    
    # Model confidence: |predicted_probability - 0.5| * 2 (ranges from 0 to 1)
    df["model_confidence"] = np.abs(df["predicted_probability"] - 0.5) * 2

    # High-confidence mistakes: wrong predictions with confidence >= 0.6 (probability >= 0.8 or <= 0.2)
    df["high_confidence_mistake"] = (
        (df["is_error"] == 1) & (df["model_confidence"] >= 0.6)
    ).astype(int)

    # Low-confidence correct: correct predictions with confidence <= 0.2 (probability between 0.4 and 0.6)
    df["low_confidence_correct"] = (
        (df["is_error"] == 0) & (df["model_confidence"] <= 0.2)
    ).astype(int)

    # Image quality quartiles (using training distributions or current sample if not available)
    for col in ["mean_brightness", "sharpness"]:
        if col in df.columns and df[col].notna().sum() > 0:
            if df[col].nunique() >= 4:
                df[f"{col}_quartile"] = pd.qcut(
                    df[col], q=4, labels=["Q1 (Low)", "Q2", "Q3", "Q4 (High)"], duplicates="drop"
                )
            else:
                # Fallback for datasets with constant or few unique values (e.g. synthetic data)
                df[f"{col}_quartile"] = "Q1 (Low)"

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save master error CSV
        error_table_path = output_dir / "error_analysis_table.csv"
        df.to_csv(error_table_path, index=False)
        logger.info(f"Error analysis table saved to {error_table_path}")

        # Generate markdown report
        generate_markdown_report(df, output_dir / "error_analysis_report.md", threshold)

    return df


def generate_markdown_report(df: pd.DataFrame, report_path: Path, threshold: float) -> None:
    """Generate and save error analysis markdown report.

    Args:
        df: Processed predictions DataFrame.
        report_path: Path to write the report.
        threshold: Classification threshold.
    """
    total = len(df)
    errors = df[df["is_error"] == 1]
    n_errors = len(errors)
    error_rate = n_errors / total if total > 0 else 0

    lines: list[str] = []
    lines.append("# RetinaGuard AI — Systematic Error Analysis Report\n")
    lines.append(f"**Classification Threshold:** {threshold:.4f}\n")
    lines.append(f"**Total Samples Evaluated:** {total}\n")
    lines.append(f"**Total Errors:** {n_errors} (Error Rate: {error_rate:.2%})\n")
    lines.append("---\n")

    # 1. Error Categories breakdown
    lines.append("## 1. Error Categories Breakdown\n")
    cat_counts = df["error_category"].value_counts()
    lines.append("| Category | Count | Percentage of Total |")
    lines.append("|---|---|---|")
    for cat in ["TP", "TN", "FP", "FN"]:
        count = cat_counts.get(cat, 0)
        pct = count / total if total > 0 else 0
        lines.append(f"| {cat} | {count} | {pct:.2%} |")
    lines.append("\n")

    # 2. High-confidence mistakes
    lines.append("## 2. High-Confidence Mistakes\n")
    lines.append("> Defined as incorrect predictions where predicted probability is ≥ 0.80 or ≤ 0.20.\n")
    hcm = df[df["high_confidence_mistake"] == 1]
    lines.append(f"**Total Count:** {len(hcm)}\n")
    if len(hcm) > 0:
        lines.append("| Image ID | True Label | True DR | Prob | Type | Brightness | Sharpness |")
        lines.append("|---|---|---|---|---|---|---|")
        for _, row in hcm.head(10).iterrows():
            lines.append(
                f"| `{row.get('image_id', 'N/A')}` | {row.get('true_label')} | "
                f"Grade {row.get('dr_grade')} | {row.get('predicted_probability'):.4f} | "
                f"{row.get('error_category')} | {row.get('mean_brightness', 0):.1f} | "
                f"{row.get('sharpness', 0):.1f} |"
            )
    else:
        lines.append("No high-confidence mistakes found ✓\n")
    lines.append("\n")

    # 3. Low-confidence correct predictions
    lines.append("## 3. Low-Confidence Correct Predictions\n")
    lines.append("> Defined as correct predictions where predicted probability is between 0.40 and 0.60.\n")
    lcc = df[df["low_confidence_correct"] == 1]
    lines.append(f"**Total Count:** {len(lcc)}\n")
    if len(lcc) > 0:
        lines.append("| Image ID | True Label | True DR | Prob | Type | Brightness | Sharpness |")
        lines.append("|---|---|---|---|---|---|---|")
        for _, row in lcc.head(10).iterrows():
            lines.append(
                f"| `{row.get('image_id', 'N/A')}` | {row.get('true_label')} | "
                f"Grade {row.get('dr_grade')} | {row.get('predicted_probability'):.4f} | "
                f"{row.get('error_category')} | {row.get('mean_brightness', 0):.1f} | "
                f"{row.get('sharpness', 0):.1f} |"
            )
    else:
        lines.append("No low-confidence correct predictions found.\n")
    lines.append("\n")

    # 4. Error rates by disease grade
    lines.append("## 4. Error Analysis by Disease Grade\n")
    if "dr_grade" in df.columns:
        grade_stats = []
        for g in sorted(df["dr_grade"].dropna().unique()):
            sub = df[df["dr_grade"] == g]
            errs = sub["is_error"].sum()
            n_sub = len(sub)
            rate = errs / n_sub if n_sub > 0 else 0
            grade_stats.append(f"| Grade {int(g)} | {n_sub} | {errs} | {rate:.2%} |")
        
        lines.append("| DR Grade | Count | Errors | Error Rate |")
        lines.append("|---|---|---|---|")
        lines.extend(grade_stats)
    lines.append("\n")

    # 5. Errors by DME grade
    lines.append("## 5. Error Analysis by DME Grade\n")
    if "dme_grade" in df.columns:
        dme_stats = []
        for g in sorted(df["dme_grade"].dropna().unique()):
            sub = df[df["dme_grade"] == g]
            errs = sub["is_error"].sum()
            n_sub = len(sub)
            rate = errs / n_sub if n_sub > 0 else 0
            dme_stats.append(f"| DME Risk {int(g)} | {n_sub} | {errs} | {rate:.2%} |")
        
        lines.append("| DME Grade | Count | Errors | Error Rate |")
        lines.append("|---|---|---|---|")
        lines.extend(dme_stats)
    lines.append("\n")

    # 6. Errors by image quality quartiles
    lines.append("## 6. Error Analysis by Image Quality\n")
    
    # Brightness
    if "mean_brightness_quartile" in df.columns:
        lines.append("### Errors by Brightness Quartile\n")
        lines.append("| Brightness Quartile | Count | Errors | Error Rate |")
        lines.append("|---|---|---|---|")
        for q in sorted(df["mean_brightness_quartile"].dropna().unique()):
            sub = df[df["mean_brightness_quartile"] == q]
            errs = sub["is_error"].sum()
            n_sub = len(sub)
            rate = errs / n_sub if n_sub > 0 else 0
            lines.append(f"| {q} | {n_sub} | {errs} | {rate:.2%} |")
        lines.append("\n")

    # Sharpness
    if "sharpness_quartile" in df.columns:
        lines.append("### Errors by Sharpness Quartile\n")
        lines.append("| Sharpness Quartile | Count | Errors | Error Rate |")
        lines.append("|---|---|---|---|")
        for q in sorted(df["sharpness_quartile"].dropna().unique()):
            sub = df[df["sharpness_quartile"] == q]
            errs = sub["is_error"].sum()
            n_sub = len(sub)
            rate = errs / n_sub if n_sub > 0 else 0
            lines.append(f"| {q} | {n_sub} | {errs} | {rate:.2%} |")
        lines.append("\n")

    # 7. Actionable findings
    lines.append("## 7. Actionable Findings & Recommendations\n")
    lines.append("- **False Negatives (FN):** These are cases of DR grade ≥ 2 that the model missed. Check if they have small localized lesions (like isolated microaneurysms) that might be lost during image resizing.")
    lines.append("- **False Positives (FP):** Check if they have artifacts, vessel crossings, or bright spots that mimic exudates.")
    lines.append("- **Image Quality:** If error rates are significantly higher in the Q1 brightness or sharpness quartiles, it suggests a need for stricter quality filtering or custom normalization.")

    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Markdown error analysis report saved to {report_path}")
