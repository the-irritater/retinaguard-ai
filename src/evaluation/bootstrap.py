"""
RetinaGuard AI - Bootstrap Evaluation
-
Purpose: Calculate 95% confidence intervals for binary classification metrics
using stratified bootstrap resampling (2,000 resamples).

Key design decisions:
- Computes 95% CIs for sensitivity, specificity, ROC-AUC, PR-AUC, F1, balanced accuracy, and Brier score.
- Supports patient-level bootstrapping if patient IDs are available, otherwise defaults to image-level.
- In IDRiD, patient IDs are not clearly documented, so we perform image-level bootstrapping while stating this limitation.

Usage:
    from src.evaluation.bootstrap import bootstrap_metrics
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.utils import resample

from src.evaluation.metrics import compute_binary_metrics

logger = logging.getLogger("retinaguard.bootstrap")


def bootstrap_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    n_resamples: int = 2000,
    confidence_level: float = 0.95,
    patient_ids: Optional[np.ndarray] = None,
    random_state: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Calculate point estimates and bootstrap confidence intervals for binary metrics.

    Args:
        y_true: Ground truth binary labels.
        y_prob: Predicted probabilities for the positive class.
        threshold: Classification threshold.
        n_resamples: Number of bootstrap iterations.
        confidence_level: CI confidence level (e.g. 0.95).
        patient_ids: Optional array of patient identifiers for group-based bootstrapping.
        random_state: Random seed for reproducibility.

    Returns:
        Dictionary mapping metric names to dicts containing 'estimate', 'ci_lower', 'ci_upper'.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n_samples = len(y_true)

    # Compute point estimates on full dataset
    point_estimates = compute_binary_metrics(y_true, y_prob, threshold)
    
    # Store bootstrap metric values
    bootstrap_results: Dict[str, List[float]] = {
        metric: [] for metric in point_estimates.keys() if isinstance(point_estimates[metric], (int, float))
    }

    # Set up generator
    rng = np.random.default_rng(random_state)

    logger.info(f"Running {n_resamples} bootstrap iterations (stratified)...")
    
    # Stratified bootstrap: separate indices by class
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    
    for i in range(n_resamples):
        if patient_ids is not None:
            # Group-based patient bootstrap
            unique_patients = np.unique(patient_ids)
            booted_patients = rng.choice(unique_patients, size=len(unique_patients), replace=True)
            
            # Map back to image indices
            boot_idx = []
            for pat in booted_patients:
                boot_idx.extend(np.where(patient_ids == pat)[0])
            boot_idx = np.array(boot_idx)
        else:
            # Stratified image-level bootstrap
            boot_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
            boot_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
            boot_idx = np.concatenate([boot_pos, boot_neg])

        y_true_b = y_true[boot_idx]
        y_prob_b = y_prob[boot_idx]

        try:
            metrics_b = compute_binary_metrics(y_true_b, y_prob_b, threshold)
            for k in bootstrap_results.keys():
                val = metrics_b.get(k)
                if val is not None:
                    bootstrap_results[k].append(val)
        except Exception as e:
            # Skip iterations that error (e.g. if bootstrap sample lacks class representation, rare with stratified)
            continue

    # Calculate percentiles for CIs
    alpha = 1.0 - confidence_level
    lower_pct = (alpha / 2.0) * 100
    upper_pct = (1.0 - alpha / 2.0) * 100

    ci_results: Dict[str, Dict[str, float]] = {}
    for metric, estimate in point_estimates.items():
        if not isinstance(estimate, (int, float)):
            continue
            
        values = bootstrap_results.get(metric, [])
        if len(values) > 0:
            ci_lower = float(np.percentile(values, lower_pct))
            ci_upper = float(np.percentile(values, upper_pct))
        else:
            ci_lower = ci_upper = estimate

        ci_results[metric] = {
            "estimate": float(estimate),
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        }

    return ci_results


def format_bootstrap_results(ci_results: Dict[str, Dict[str, float]]) -> str:
    """Format confidence intervals as a clean Markdown table.

    Args:
        ci_results: Dictionary from bootstrap_metrics().

    Returns:
        Markdown table string.
    """
    lines = [
        "| Metric | Point Estimate | 95% Confidence Interval |",
        "| :--- | :---: | :---: |"
    ]
    
    for metric, vals in ci_results.items():
        if metric in ["tp", "tn", "fp", "fn", "threshold"]:
            continue
        name = metric.replace("_", " ").title()
        est = vals["estimate"]
        lower = vals["ci_lower"]
        upper = vals["ci_upper"]
        
        # Display as percentages for appropriate metrics
        pct_metrics = ["sensitivity", "specificity", "precision", "npv", "f1_score", "balanced_accuracy", "accuracy", "roc_auc", "pr_auc"]
        if metric in pct_metrics:
            lines.append(f"| {name} | {est:.2%} | [{lower:.2%}, {upper:.2%}] |")
        else:
            lines.append(f"| {name} | {est:.4f} | [{lower:.4f}, {upper:.4f}] |")

    return "\n".join(lines)
