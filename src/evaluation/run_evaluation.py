"""
RetinaGuard AI — Model Evaluation Orchestrator
===============================================
Purpose: Load the trained model and perform the final test evaluation,
calibration, bootstrapping, error analysis, and Grad-CAM generation.

Key design decisions:
- The test set is evaluated EXACTLY ONCE.
- Recalibration temperature scaling is fitted on the validation set only,
  and applied to test probabilities.
- Runs 2,000 resamples for bootstrap confidence intervals.
- Generates all curves: ROC, PR, Calibration, Threshold-vs-Sensitivity/Specificity.
- Generates the systematic error reports.
- Generates Grad-CAM overlays with the required medical disclaimer.

Usage:
    python -m src.evaluation.run_evaluation --config configs/base_config.yaml --checkpoint models/checkpoints/final/best_model_fold0_binary.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

from src.data.dataset import IDRiDDataset
from src.models.architectures import create_model
from src.evaluation.metrics import (
    compute_binary_metrics,
    compute_roc_curve,
    compute_pr_curve,
    select_threshold,
)
from src.evaluation.calibration import (
    TemperatureScaler,
    compute_calibration_stats,
)
from src.evaluation.bootstrap import bootstrap_metrics, format_bootstrap_results
from src.evaluation.error_analysis import analyze_errors
from src.explainability.gradcam import GradCAMExplainer, plot_gradcam_panel

logger = logging.getLogger("retinaguard.evaluation")


# ---------------------------------------------------------------------------
# Plotting Helpers
# ---------------------------------------------------------------------------
def plot_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    calibrated_prob: np.ndarray,
    threshold: float,
    figures_dir: Path,
) -> None:
    """Generate and save all diagnostic evaluation plots.

    Plots:
    - ROC Curve
    - Precision-Recall Curve
    - Calibration Curve (uncalibrated vs calibrated)
    - Threshold vs Sensitivity / Specificity

    Args:
        y_true: True binary labels.
        y_prob: Uncalibrated probabilities.
        calibrated_prob: Calibrated probabilities.
        threshold: Classification threshold.
        figures_dir: Output directory for plots.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. ROC Curve
    plt.figure(figsize=(6, 5))
    fpr, tpr, _ = roc_curve(y_true, calibrated_prob)
    from sklearn.metrics import auc
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(figures_dir / "roc_curve.png", dpi=150)
    plt.close()

    # 2. PR Curve
    plt.figure(figsize=(6, 5))
    from sklearn.metrics import precision_recall_curve, average_precision_score
    precision, recall, _ = precision_recall_curve(y_true, calibrated_prob)
    pr_auc = average_precision_score(y_true, calibrated_prob)
    plt.plot(recall, precision, color="forestgreen", lw=2, label=f"PR curve (AUC = {pr_auc:.4f})")
    plt.xlabel("Recall (Sensitivity)")
    plt.ylabel("Precision")
    plt.title("Precision-Recall (PR) Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(figures_dir / "precision_recall_curve.png", dpi=150)
    plt.close()

    # 3. Calibration Curve
    plt.figure(figsize=(6, 5))
    from sklearn.calibration import calibration_curve
    prob_true_uncal, prob_pred_uncal = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
    prob_true_cal, prob_pred_cal = calibration_curve(y_true, calibrated_prob, n_bins=10, strategy="uniform")
    
    plt.plot(prob_pred_uncal, prob_true_uncal, "s-", color="red", label="Uncalibrated")
    plt.plot(prob_pred_cal, prob_true_cal, "o-", color="blue", label="Calibrated (Temp Scaled)")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect Calibration")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Curve")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(figures_dir / "calibration_curve.png", dpi=150)
    plt.close()

    # 4. Threshold vs Sensitivity/Specificity
    thresholds = np.linspace(0.01, 0.99, 100)
    sensitivities = []
    specificities = []
    
    for t in thresholds:
        preds = (calibrated_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivities.append(sens)
        specificities.append(spec)

    plt.figure(figsize=(7, 5))
    plt.plot(thresholds, sensitivities, label="Sensitivity", color="crimson", lw=2)
    plt.plot(thresholds, specificities, label="Specificity", color="dodgerblue", lw=2)
    plt.axvline(threshold, color="green", linestyle="--", label=f"Selected Threshold ({threshold:.3f})")
    plt.xlabel("Probability Threshold")
    plt.ylabel("Metric Value")
    plt.title("Threshold vs. Sensitivity & Specificity")
    plt.legend(loc="lower center")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "threshold_tradeoff.png", dpi=150)
    plt.close()

    logger.info(f"Evaluation plots saved to {figures_dir}")


# ---------------------------------------------------------------------------
# Evaluation orchestrator
# ---------------------------------------------------------------------------
def run_evaluation(
    config_path: Path,
    checkpoint_path: Path,
    task: str = "binary",
    use_synthetic: bool = False,
) -> None:
    """Run full evaluation on validation and test set.

    Args:
        config_path: Path to YAML config.
        checkpoint_path: Path to PyTorch model checkpoint.
        task: Classification task ('binary' or 'multiclass').
        use_synthetic: Force generation of synthetic metadata if raw files missing.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    project_root = config_path.parent.parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Output directories
    figures_dir = project_root / config["paths"]["figures_dir"]
    tables_dir = project_root / config["paths"]["tables_dir"]
    metadata_dir = project_root / config["paths"]["metadata_dir"]
    splits_csv = metadata_dir / "idrid_splits.csv"

    # Validation of data splits existence
    if not splits_csv.exists():
        if use_synthetic:
            logger.info("Generating synthetic dataset for demonstration purposes...")
            generate_synthetic_splits(metadata_dir)
        else:
            raise FileNotFoundError(
                f"Splits file not found: {splits_csv}.\n"
                f"Please run data audit and splitting first, or use --synthetic flag."
            )

    logger.info("=" * 60)
    logger.info("RetinaGuard AI — Final Test Evaluation")
    logger.info(f"  Config:     {config_path}")
    logger.info(f"  Checkpoint: {checkpoint_path}")
    logger.info(f"  Task:       {task}")
    logger.info("=" * 60)

    # 1. Load data loaders
    image_size = config["preprocessing"]["image_size"]
    
    val_dataset = IDRiDDataset(
        metadata_csv=splits_csv,
        mode="val",
        task=task,
        fold=0,  # Calibration fold
        config=config,
    )
    test_dataset = IDRiDDataset(
        metadata_csv=splits_csv,
        mode="test",
        task=task,
        config=config,
    )

    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)

    # 2. Load model
    num_classes = 2 if task == "binary" else 5
    architecture = config["model"]["architecture"] if "model" in config else "efficientnet_b0"
    model = create_model(architecture=architecture, num_classes=num_classes, pretrained=False)
    
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Successfully loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        logger.warning(
            f"Checkpoint not found at {checkpoint_path}.\n"
            f"Using random model weights for code verification."
        )

    model = model.to(device)
    model.eval()

    # 3. Extract validation predictions (for threshold selection & calibration fitting)
    logger.info("Extracting validation predictions for calibration...")
    val_logits = []
    val_probs = []
    val_labels = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            labels = batch["label"]
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            val_logits.append(logits.cpu().numpy())
            val_probs.append(probs.cpu().numpy())
            val_labels.extend(labels.tolist())

    val_logits = np.concatenate(val_logits, axis=0)
    val_probs = np.concatenate(val_probs, axis=0)
    val_labels = np.array(val_labels)

    # Fit temperature scaling calibration on validation set
    temp_scaler = TemperatureScaler()
    temp_scaler.fit(val_logits, val_labels)

    # Apply calibration
    calibrated_val_probs = temp_scaler.scale_probabilities(val_probs)

    # Select threshold using validation set calibrated probabilities
    threshold_method = config["threshold"].get("methods", ["youden"])[0]
    val_pos_probs = calibrated_val_probs[:, 1] if task == "binary" else calibrated_val_probs[:, 1]
    
    locked_threshold = select_threshold(
        y_true=val_labels,
        y_prob=val_pos_probs,
        method=threshold_method,
        target_sensitivity=config["threshold"].get("sensitivity_target", 0.90),
        fixed_value=config["threshold"].get("fixed_value", 0.50),
    )
    logger.info(f"Locked classification threshold: {locked_threshold:.4f}")

    # 4. Extract test predictions (ONE-TIME evaluation)
    logger.info("Running inference on official test set...")
    test_logits = []
    test_probs = []
    test_labels = []
    test_meta = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            labels = batch["label"]
            logits = model(images)
            probs = torch.softmax(logits, dim=1)

            test_logits.append(logits.cpu().numpy())
            test_probs.append(probs.cpu().numpy())
            test_labels.extend(labels.tolist())

            # Save batch metadata for error analysis
            for i in range(len(labels)):
                test_meta.append({
                    "image_id": batch["image_id"][i],
                    "dr_grade": int(batch["dr_grade"][i]),
                    "dme_grade": int(batch["dme_grade"][i]),
                    "mean_brightness": float(batch["dr_grade"][i]), # Fallback dummy if not present
                    "sharpness": 100.0,
                    "full_path": batch["image_path"][i],
                })

    test_logits = np.concatenate(test_logits, axis=0)
    test_probs = np.concatenate(test_probs, axis=0)
    test_labels = np.array(test_labels)

    # Recalibrate test predictions using the validation scaler
    calibrated_test_probs = temp_scaler.scale_probabilities(test_probs)

    # 5. Compute test metrics & Bootstrap CIs
    logger.info("Computing metrics with bootstrap confidence intervals...")
    y_test_prob = calibrated_test_probs[:, 1] if task == "binary" else calibrated_test_probs[:, 1]
    
    bootstrap_results = bootstrap_metrics(
        y_true=test_labels,
        y_prob=y_test_prob,
        threshold=locked_threshold,
        n_resamples=config["evaluation"]["bootstrap"]["n_resamples"],
        confidence_level=config["evaluation"]["bootstrap"]["confidence_level"],
        random_state=config["evaluation"]["bootstrap"]["random_state"],
    )

    # Format and save CIs table
    ci_table = format_bootstrap_results(bootstrap_results)
    tables_dir.mkdir(parents=True, exist_ok=True)
    with open(tables_dir / "test_bootstrap_ci.md", "w") as f:
        f.write(f"# Final Test Bootstrap Metrics (95% CIs)\n\n")
        f.write(f"**Locked Threshold:** {locked_threshold:.4f}\n\n")
        f.write(ci_table)
    logger.info(f"Bootstrap CIs table written to {tables_dir / 'test_bootstrap_ci.md'}")

    # 6. Plot curves
    plot_curves(
        y_true=test_labels,
        y_prob=test_probs[:, 1] if task == "binary" else test_probs[:, 1],
        calibrated_prob=y_test_prob,
        threshold=locked_threshold,
        figures_dir=figures_dir,
    )

    # 7. Calibration Stats
    cal_stats = compute_calibration_stats(test_labels, y_test_prob, n_bins=10)
    logger.info(
        f"Test Calibration Stats: ECE={cal_stats['ece']:.4f}, "
        f"Slope={cal_stats['slope']:.4f}, Intercept={cal_stats['intercept']:.4f}"
    )
    with open(tables_dir / "calibration_summary.json", "w") as f:
        json.dump({
            "ece": cal_stats["ece"],
            "slope": cal_stats["slope"],
            "intercept": cal_stats["intercept"],
            "temperature": float(temp_scaler.temperature.item()),
        }, f, indent=2)

    # 8. Systematic Error Analysis
    logger.info("Running systematic error analysis...")
    test_df = pd.DataFrame(test_meta)
    test_df["true_label"] = test_labels
    test_df["predicted_probability"] = y_test_prob
    test_df["uncalibrated_probability"] = test_probs[:, 1] if task == "binary" else test_probs[:, 1]

    # Populate actual brightness/sharpness from master CSV if available
    master_csv = metadata_dir / "idrid_master.csv"
    if master_csv.exists():
        master_df = pd.read_csv(master_csv)
        for col in ["mean_brightness", "sharpness"]:
            if col in master_df.columns:
                val_map = dict(zip(master_df["stem"].astype(str), master_df[col]))
                test_df[col] = test_df["image_id"].astype(str).map(val_map)

    analyze_errors(test_df, threshold=locked_threshold, output_dir=project_root / "reports")

    # 9. Explainability (Grad-CAM generation)
    logger.info("Generating Grad-CAM visualizations...")
    explainer = GradCAMExplainer(model, use_cuda=torch.cuda.is_available())
    
    # Select representative cases
    # We want one TP, TN, FP, FN, and high-confidence mistake if available
    err_df = analyze_errors(test_df, threshold=locked_threshold)
    
    for category in ["TP", "TN", "FP", "FN"]:
        sub = err_df[err_df["error_category"] == category]
        if len(sub) > 0:
            row = sub.iloc[0]
            img_id = row["image_id"]
            img_path = row["full_path"]
            
            # Find item in test dataset
            idx = int(np.where(err_df["image_id"] == img_id)[0][0])
            sample = test_dataset[idx]
            
            overlay, heatmap = explainer.generate(
                sample["image"],
                target_category=1 if task == "binary" else sample["label"]
            )
            
            # Construct original image (unnormalised)
            img_np = sample["image"].numpy().transpose(1, 2, 0)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_np = std * img_np + mean
            img_np = np.clip(img_np, 0.0, 1.0)
            
            panel_path = figures_dir / f"gradcam_{category.lower()}_{img_id}.png"
            plot_gradcam_panel(
                original_img=img_np,
                overlay_img=overlay,
                grayscale_cam=heatmap,
                true_label="DR grade >= 2" if sample["binary_label"] == 1 else "DR grade < 2",
                pred_label="DR grade >= 2" if row["predicted_label"] == 1 else "DR grade < 2",
                probability=row["predicted_probability"],
                image_id=img_id,
                output_path=panel_path,
            )

    logger.info("=" * 60)
    logger.info("Final Evaluation Complete!")
    logger.info(f"  Curves:      {figures_dir}")
    logger.info(f"  CIs table:   {tables_dir / 'test_bootstrap_ci.md'}")
    logger.info(f"  Error rep:   {project_root / 'reports/error_analysis_report.md'}")
    logger.info("=" * 60)


def generate_synthetic_splits(output_dir: Path) -> None:
    """Helper to generate dummy splits for pipeline testing when raw data is missing."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    
    # Create 40 training and 10 testing images
    for i in range(50):
        partition = "train" if i < 40 else "test"
        fold = i % 5 if partition == "train" else -1
        img_id = f"IDRiD_{i+1:03d}"
        
        # Create temp blank image
        temp_img_dir = output_dir.parent / "raw/images"
        temp_img_dir.mkdir(parents=True, exist_ok=True)
        img_path = temp_img_dir / f"{img_id}.jpg"
        if not img_path.exists():
            import cv2
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.circle(img, (50, 50), 40, (0, 255, 0), -1)
            cv2.imwrite(str(img_path), img)

        rows.append({
            "image_id": img_id,
            "stem": img_id,
            "filename": f"{img_id}.jpg",
            "full_path": str(img_path),
            "relative_path": f"raw/images/{img_id}.jpg",
            "partition": partition,
            "dr_grade": i % 5,
            "binary_label": 1 if (i % 5) >= 2 else 0,
            "match_status": "matched",
            "readable": True,
            "fold": fold,
            "dme_grade": i % 3,
            "mean_brightness": 120.0,
            "sharpness": 50.0,
        })
        
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "idrid_splits.csv", index=False)
    df.to_csv(output_dir / "idrid_master.csv", index=False)
    logger.info(f"Created synthetic splits CSV at {output_dir / 'idrid_splits.csv'}")


if __name__ == "__main__":
    from sklearn.metrics import roc_curve, confusion_matrix
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="RetinaGuard AI Orchestrator")
    parser.add_argument("--config", type=Path, default=Path("configs/base_config.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/checkpoints/final/best_model_fold0_binary.pt"))
    parser.add_argument("--task", default="binary")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic dataset if missing raw files")
    args = parser.parse_args()

    run_evaluation(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        task=args.task,
        use_synthetic=args.synthetic,
    )
