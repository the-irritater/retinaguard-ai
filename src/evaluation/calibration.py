"""
RetinaGuard AI — Model Calibration Analysis
============================================
Purpose: Assess and adjust model calibration. Computes ECE, Brier score,
calibration slope, and calibration intercept. Implements post-hoc
temperature scaling.

Key design decisions:
- Fits temperature scaling on validation set logits/probabilities only.
- Never fits temperature scaling or calibration models on test data.
- Calibration slope and intercept are estimated via unregularised logistic
  regression of labels on log-odds.
- Expected Calibration Error (ECE) is calculated using adaptive or equal-width binning.

Usage:
    from src.evaluation.calibration import TemperatureScaler, analyze_calibration
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger("retinaguard.calibration")


# ---------------------------------------------------------------------------
# ECE Calculation
# ---------------------------------------------------------------------------
def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE).

    Args:
        y_true: True binary labels (0 or 1).
        y_prob: Predicted probabilities for the positive class.
        n_bins: Number of bins.

    Returns:
        Expected Calibration Error as a float.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        # Find samples in current bin
        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            confidence_in_bin = np.mean(y_prob[in_bin])
            ece += prop_in_bin * np.abs(accuracy_in_bin - confidence_in_bin)

    return float(ece)


# ---------------------------------------------------------------------------
# Calibration Curve, Slope, and Intercept
# ---------------------------------------------------------------------------
def compute_calibration_stats(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Calculate calibration curve data, ECE, slope, and intercept.

    Calibration slope (b) and intercept (a) are computed by fitting:
    logit(p) = a + b * logit(y_prob)
    Perfect calibration is a=0, b=1.

    Args:
        y_true: True binary labels.
        y_prob: Predicted probabilities.
        n_bins: Number of bins.

    Returns:
        Dictionary with ECE, slope, intercept, and curve coordinate arrays.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    # Get curve coordinates
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")

    # Avoid boundary problems for logit calculation
    eps = 1e-7
    y_prob_clipped = np.clip(y_prob, eps, 1.0 - eps)
    logits = np.log(y_prob_clipped / (1.0 - y_prob_clipped))

    # Fit unregularised logistic regression to get slope and intercept
    # logit(P(y=1)) = intercept + slope * logits
    try:
        lr = LogisticRegression(penalty=None, solver="lbfgs")
        lr.fit(logits.reshape(-1, 1), y_true)
        # Note: scikit-learn fits logit(p) = coef_ * X + intercept_
        slope = float(lr.coef_[0, 0])
        intercept = float(lr.intercept_[0])
    except Exception as e:
        logger.warning(f"Failed to fit calibration regression: {e}")
        slope, intercept = np.nan, np.nan

    ece = compute_ece(y_true, y_prob, n_bins=n_bins)

    return {
        "ece": ece,
        "slope": slope,
        "intercept": intercept,
        "prob_true": prob_true,
        "prob_pred": prob_pred,
    }


# ---------------------------------------------------------------------------
# Temperature Scaling
# ---------------------------------------------------------------------------
class TemperatureScaler(nn.Module):
    """Temperature scaling post-processing module.

    Learns a single scalar temperature parameter (T) on validation logits
    to recalibrate probabilities:
    P(y=1) = sigmoid(logits / T)

    Attributes:
        temperature: nn.Parameter holding the temperature value.
    """

    def __init__(self) -> None:
        """Initialise temperature scaler."""
        super().__init__()
        # Initialise temperature to 1.0 (no scaling)
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling to logits.

        Args:
            logits: Logits tensor.

        Returns:
            Scaled logits.
        """
        # Ensure temperature is positive
        temp = torch.clamp(self.temperature, min=0.01)
        return logits / temp

    def fit(
        self,
        val_logits: np.ndarray,
        val_labels: np.ndarray,
        lr: float = 0.01,
        max_iter: int = 50,
    ) -> float:
        """Fit temperature parameter on validation set only.

        Args:
            val_logits: Validation set logits (or pre-sigmoid log-odds).
            val_labels: Validation set ground-truth labels.
            lr: Learning rate.
            max_iter: Maximum optimization iterations.

        Returns:
            Optimised temperature value.
        """
        logits_t = torch.from_numpy(val_logits).float()
        labels_t = torch.from_numpy(val_labels).float()

        # Handle multiclass vs binary logits format
        is_binary = len(logits_t.shape) == 1 or logits_t.shape[1] == 1
        
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        
        if is_binary:
            if len(logits_t.shape) == 2:
                logits_t = logits_t.squeeze(1)
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()
            labels_t = labels_t.long()

        def eval_loss():
            optimizer.zero_grad()
            temp = torch.clamp(self.temperature, min=0.01)
            scaled = logits_t / temp
            loss = criterion(scaled, labels_t)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        
        final_temp = float(torch.clamp(self.temperature, min=0.01).item())
        logger.info(f"Temperature scaling fitted on validation set: T = {final_temp:.4f}")
        return final_temp

    def scale_probabilities(self, probs: np.ndarray) -> np.ndarray:
        """Scale probabilities using the fitted temperature.

        Args:
            probs: Uncalibrated probabilities (shape: [N] or [N, C]).

        Returns:
            Calibrated probabilities.
        """
        eps = 1e-7
        probs_clipped = np.clip(probs, eps, 1.0 - eps)
        
        if len(probs.shape) == 1 or probs.shape[1] == 2:
            # Binary probabilities
            p = probs_clipped if len(probs.shape) == 1 else probs_clipped[:, 1]
            logits = np.log(p / (1.0 - p))
            temp = max(0.01, self.temperature.item())
            scaled_logits = logits / temp
            calibrated_p = 1.0 / (1.0 + np.exp(-scaled_logits))
            
            if len(probs.shape) == 1:
                return calibrated_p
            else:
                out = np.zeros_like(probs)
                out[:, 0] = 1.0 - calibrated_p
                out[:, 1] = calibrated_p
                return out
        else:
            # Multiclass probabilities
            logits = np.log(probs_clipped)
            temp = max(0.01, self.temperature.item())
            scaled_logits = logits / temp
            exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
            return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
