"""
RetinaGuard AI - Evaluation Metrics
-
Purpose: Comprehensive metrics for binary and multiclass DR classification.

Binary metrics (DR grade ≥ 2 detection):
- Sensitivity, specificity, precision, NPV, F1, balanced accuracy, MCC
- ROC-AUC, PR-AUC, Brier score

Multiclass metrics (DR grades 0-4, secondary analysis):
- Accuracy, balanced accuracy, macro/weighted F1, per-class P/R/F1
- Confusion matrix, QWK, MAE, within-one-grade, severe undergrading

Threshold selection uses ONLY validation data.

Usage:
    from src.evaluation.metrics import compute_binary_metrics, select_threshold
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    average_precision_score,
)

logger = logging.getLogger("retinaguard.metrics")


# -
# Threshold selection (validation set only)
# -
def select_threshold_youden(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> float:
    """Select threshold using Youden's J index (max sensitivity + specificity - 1).

    Must be applied to VALIDATION data only. Never to test data.

    Args:
        y_true: True binary labels.
        y_prob: Predicted probabilities for the positive class.

    Returns:
        Optimal threshold value.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = float(thresholds[best_idx])
    logger.info(
        f"Youden threshold: {best_threshold:.4f} "
        f"(sensitivity={tpr[best_idx]:.4f}, specificity={1-fpr[best_idx]:.4f})"
    )
    return best_threshold


def select_threshold_sensitivity_target(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_sensitivity: float = 0.90,
) -> float:
    """Select threshold to achieve a target sensitivity.

    Must be applied to VALIDATION data only.

    Args:
        y_true: True binary labels.
        y_prob: Predicted probabilities for the positive class.
        target_sensitivity: Desired minimum sensitivity.

    Returns:
        Threshold that achieves at least the target sensitivity.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)

    # Find thresholds where sensitivity >= target
    valid_idx = np.where(tpr >= target_sensitivity)[0]
    if len(valid_idx) == 0:
        logger.warning(
            f"Cannot achieve target sensitivity {target_sensitivity:.2f}. "
            f"Max achieved: {tpr.max():.4f}"
        )
        return float(thresholds[np.argmax(tpr)])

    # Among valid thresholds, pick the one with highest specificity
    best_idx = valid_idx[np.argmax(1 - fpr[valid_idx])]
    best_threshold = float(thresholds[best_idx])
    logger.info(
        f"Sensitivity-target threshold: {best_threshold:.4f} "
        f"(sensitivity={tpr[best_idx]:.4f}, specificity={1-fpr[best_idx]:.4f})"
    )
    return best_threshold


def select_uncertainty_bounds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    target_sensitivity: float = 0.95,
    target_specificity: float = 0.95,
) -> tuple[float, float]:
    """Derive uncertainty boundaries (L, U) around the threshold from validation data.

    Lower bound L: maximum threshold where validation sensitivity >= target_sensitivity.
    Upper bound U: minimum threshold where validation specificity >= target_specificity.
    
    If validation scores fall in [L, U], the prediction is classified as Indeterminate.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    
    # Lower bound: we want sensitivity >= target_sensitivity
    idx_sens = np.where(tpr >= target_sensitivity)[0]
    if len(idx_sens) > 0:
        candidate_thresholds = thresholds[idx_sens]
        valid_candidates = candidate_thresholds[candidate_thresholds <= threshold]
        if len(valid_candidates) > 0:
            lower_bound = float(np.max(valid_candidates))
        else:
            lower_bound = float(np.min(candidate_thresholds))
    else:
        lower_bound = threshold - 0.05
        
    # Upper bound: we want specificity >= target_specificity
    idx_spec = np.where((1 - fpr) >= target_specificity)[0]
    if len(idx_spec) > 0:
        candidate_thresholds = thresholds[idx_spec]
        valid_candidates = candidate_thresholds[candidate_thresholds >= threshold]
        if len(valid_candidates) > 0:
            upper_bound = float(np.min(valid_candidates))
        else:
            upper_bound = float(np.max(candidate_thresholds))
    else:
        upper_bound = threshold + 0.05
        
    # Safety constraints to prevent bounds collapse
    lower_bound = max(0.01, min(lower_bound, threshold - 0.01))
    upper_bound = min(0.99, max(upper_bound, threshold + 0.01))
    
    # Make sure bounds cover the threshold properly
    if not (lower_bound < threshold < upper_bound):
        lower_bound = max(0.01, threshold - 0.05)
        upper_bound = min(0.99, threshold + 0.05)
        
    logger.info(f"Derived validation uncertainty zone: [{lower_bound:.4f}, {upper_bound:.4f}] around threshold {threshold:.4f}")
    return lower_bound, upper_bound


def select_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    method: str = "youden",
    **kwargs: Any,
) -> float:
    """Select probability threshold using the specified method.

    ALL threshold selection must use VALIDATION data only.
    The test set is NEVER used for threshold selection.

    Args:
        y_true: True binary labels (validation set).
        y_prob: Predicted probabilities (validation set).
        method: 'youden', 'sensitivity_target', or 'fixed'.

    Returns:
        Selected threshold value.
    """
    if method == "youden":
        return select_threshold_youden(y_true, y_prob)
    elif method == "sensitivity_target":
        target = kwargs.get("target_sensitivity", 0.90)
        return select_threshold_sensitivity_target(y_true, y_prob, target)
    elif method == "fixed":
        threshold = kwargs.get("fixed_value", 0.50)
        logger.info(f"Using fixed threshold: {threshold}")
        return float(threshold)
    else:
        raise ValueError(f"Unknown threshold method: {method}")


# -
# Binary metrics
# -
def compute_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute comprehensive binary classification metrics.

    Args:
        y_true: True binary labels.
        y_prob: Predicted probabilities for the positive class.
        threshold: Classification threshold.

    Returns:
        Dictionary of metric names and values.
    """
    y_pred = (y_prob >= threshold).astype(int)
    y_true = np.asarray(y_true, dtype=int)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # Core metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    metrics = {
        "threshold": threshold,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "npv": npv,
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }

    # AUC metrics (threshold-independent)
    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics["roc_auc"] = None

    try:
        metrics["pr_auc"] = average_precision_score(y_true, y_prob)
    except ValueError:
        metrics["pr_auc"] = None

    # Calibration
    metrics["brier_score"] = brier_score_loss(y_true, y_prob)

    return metrics


# -
# Multiclass metrics
# -
def compute_multiclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    num_classes: int = 5,
) -> dict[str, Any]:
    """Compute comprehensive multiclass classification metrics.

    This is a SECONDARY exploratory analysis for DR grades 0-4.

    Args:
        y_true: True labels (0-4).
        y_pred: Predicted labels (0-4).
        y_prob: Predicted probabilities (optional, shape: [N, num_classes]).
        num_classes: Number of classes.

    Returns:
        Dictionary of metric names and values.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    metrics: dict[str, Any] = {}

    # Overall metrics
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
    metrics["macro_precision"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["macro_recall"] = recall_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["macro_f1"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["weighted_f1"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # Per-class metrics
    report = classification_report(
        y_true, y_pred,
        labels=list(range(num_classes)),
        output_dict=True,
        zero_division=0,
    )
    metrics["per_class"] = {
        str(i): report.get(str(i), {}) for i in range(num_classes)
    }

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    metrics["confusion_matrix"] = cm.tolist()

    # Quadratic weighted kappa (treats grades as ordered)
    metrics["quadratic_weighted_kappa"] = cohen_kappa_score(
        y_true, y_pred, weights="quadratic"
    )

    # Mean absolute grade error
    metrics["mean_absolute_grade_error"] = float(np.mean(np.abs(y_true - y_pred)))

    # Percentage within one grade
    within_one = np.mean(np.abs(y_true - y_pred) <= 1)
    metrics["pct_within_one_grade"] = float(within_one)

    # Severe undergrading: predicted grade ≤ 1 when true grade ≥ 3
    severe_undergrade_mask = (y_pred <= 1) & (y_true >= 3)
    metrics["severe_undergrading_count"] = int(severe_undergrade_mask.sum())
    metrics["severe_undergrading_pct"] = float(
        severe_undergrade_mask.mean() if len(y_true) > 0 else 0
    )

    return metrics


# -
# ROC and PR curve data
# -
def compute_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute ROC curve data points.

    Args:
        y_true: True binary labels.
        y_prob: Predicted probabilities.

    Returns:
        Dictionary with fpr, tpr, and thresholds arrays.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return {"fpr": fpr, "tpr": tpr, "thresholds": thresholds}


def compute_pr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute precision-recall curve data points.

    Args:
        y_true: True binary labels.
        y_prob: Predicted probabilities.

    Returns:
        Dictionary with precision, recall, and thresholds arrays.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    return {"precision": precision, "recall": recall, "thresholds": thresholds}
