"""
RetinaGuard AI - Evaluation Metrics Unit Tests
-
Purpose: Verify that metric computations, threshold selection, calibration
error calculations, and bootstrapping function correctly under dummy scenarios.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.metrics import (
    compute_binary_metrics,
    select_threshold_youden,
    select_threshold_sensitivity_target,
)
from src.evaluation.calibration import compute_ece, compute_calibration_stats
from src.evaluation.bootstrap import bootstrap_metrics


def test_binary_metrics_perfect_predictions() -> None:
    """Verify metrics for perfect classification."""
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    y_prob = np.array([0.1, 0.2, 0.9, 0.8, 0.05, 0.95, 0.15, 0.85])
    
    metrics = compute_binary_metrics(y_true, y_prob, threshold=0.5)
    
    assert metrics["accuracy"] == 1.0
    assert metrics["sensitivity"] == 1.0
    assert metrics["specificity"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["npv"] == 1.0
    assert metrics["f1_score"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["tp"] == 4
    assert metrics["tn"] == 4
    assert metrics["fp"] == 0
    assert metrics["fn"] == 0
    assert metrics["brier_score"] < 0.05  # Perfect probabilities should have low Brier score


def test_binary_metrics_imperfect_predictions() -> None:
    """Verify metrics for realistic (imperfect) classification."""
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    # 1 FP at 0.5 (idx 1), 1 FN at 0.5 (idx 3)
    y_prob = np.array([0.1, 0.7, 0.9, 0.3, 0.05, 0.95, 0.15, 0.85])
    
    metrics = compute_binary_metrics(y_true, y_prob, threshold=0.5)
    
    # TP=3, TN=3, FP=1, FN=1
    assert metrics["tp"] == 3
    assert metrics["tn"] == 3
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["sensitivity"] == 0.75
    assert metrics["specificity"] == 0.75
    assert metrics["precision"] == 0.75
    assert metrics["npv"] == 0.75
    assert metrics["accuracy"] == 0.75


def test_threshold_selection_youden() -> None:
    """Verify Youden threshold selection finds the point maximizing J = TPR - FPR."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.35, 0.4, 0.8, 0.9, 0.95])
    
    # True positives: probabilities [0.4, 0.8, 0.9, 0.95]
    # False positives: probabilities [0.1, 0.2, 0.3, 0.35]
    # A threshold of 0.4 gives:
    # TPR: 1.0 (all positive cases >= 0.4 detected)
    # FPR: 0.0 (no negative cases >= 0.4 detected)
    # Youden J = 1.0 - 0.0 = 1.0
    threshold = select_threshold_youden(y_true, y_prob)
    assert 0.3 < threshold <= 0.4


def test_threshold_selection_sensitivity_target() -> None:
    """Verify sensitivity target threshold selection achieves required sensitivity."""
    y_true = np.array([0, 0, 0, 1, 1, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.5, 0.4, 0.6, 0.7, 0.8, 0.9])
    
    # Total positive cases = 5. Target sensitivity = 80% (needs 4 positives detected)
    # Positives are [0.4, 0.6, 0.7, 0.8, 0.9]
    # To detect 4 positives, we need threshold <= 0.6
    threshold = select_threshold_sensitivity_target(y_true, y_prob, target_sensitivity=0.8)
    assert threshold <= 0.6001


def test_ece_computation() -> None:
    """Verify ECE computation returns expected values."""
    # Perfectly calibrated case: probability matches actual rate
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.1, 0.9, 0.9])
    # Accuracy in [0, 0.2] is 0/2 = 0.0, Confidence is 0.1. Absolute difference = 0.1
    # Accuracy in [0.8, 1.0] is 2/2 = 1.0, Confidence is 0.9. Absolute difference = 0.1
    # ECE should be 0.5 * 0.1 + 0.5 * 0.1 = 0.1
    ece = compute_ece(y_true, y_prob, n_bins=5)
    assert np.isclose(ece, 0.1)


def test_calibration_stats() -> None:
    """Verify calibration slope/intercept fit works."""
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.15, 0.85, 0.25, 0.75])
    
    stats = compute_calibration_stats(y_true, y_prob, n_bins=3)
    assert "ece" in stats
    assert "slope" in stats
    assert "intercept" in stats
    assert not np.isnan(stats["slope"])
    assert not np.isnan(stats["intercept"])


def test_bootstrap_metrics() -> None:
    """Verify bootstrap confidence intervals generation."""
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    y_prob = np.array([0.1, 0.2, 0.9, 0.8, 0.05, 0.95, 0.15, 0.85])
    
    ci_results = bootstrap_metrics(
        y_true, y_prob, threshold=0.5, n_resamples=50, confidence_level=0.95, random_state=42
    )
    
    assert "sensitivity" in ci_results
    assert "specificity" in ci_results
    assert "roc_auc" in ci_results
    
    for metric in ["sensitivity", "specificity", "roc_auc"]:
        est = ci_results[metric]["estimate"]
        lower = ci_results[metric]["ci_lower"]
        upper = ci_results[metric]["ci_upper"]
        assert lower <= est <= upper
