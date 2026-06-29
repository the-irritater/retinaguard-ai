"""
RetinaGuard AI - Loss Functions
-
Purpose: Loss functions for DR classification including weighted
cross-entropy and focal loss.

Focal loss is provided as a sensitivity analysis option, not the default.
The primary loss is weighted cross-entropy.

Usage:
    from src.models.losses import create_loss_function
    criterion = create_loss_function('weighted_ce', class_weights=weights)
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("retinaguard.losses")


class FocalLoss(nn.Module):
    """Focal loss for addressing class imbalance.

    Focal loss down-weights easy examples and focuses training on hard cases.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    This is provided as a sensitivity analysis option. The primary loss
    function is weighted cross-entropy.

    Reference:
        Lin et al. "Focal Loss for Dense Object Detection." ICCV 2017.

    Attributes:
        gamma: Focusing parameter (default 2.0).
        alpha: Class balancing weights (optional).
        reduction: Loss reduction method.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> None:
        """Initialise focal loss.

        Args:
            gamma: Focusing parameter. Higher values focus more on hard examples.
            alpha: Per-class weights tensor. If None, all classes weighted equally.
            reduction: 'mean', 'sum', or 'none'.
        """
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute focal loss.

        Args:
            inputs: Model logits of shape (B, C).
            targets: Ground truth labels of shape (B,).

        Returns:
            Scalar loss value.
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        p_t = torch.exp(-ce_loss)  # probability of correct class

        focal_weight = (1 - p_t) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha.to(inputs.device)[targets]
            focal_weight = alpha_t * focal_weight

        loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


def create_loss_function(
    loss_type: str = "weighted_cross_entropy",
    class_weights: Optional[torch.Tensor] = None,
    focal_gamma: float = 2.0,
    focal_alpha: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
) -> nn.Module:
    """Factory function to create loss functions.

    Args:
        loss_type: Type of loss function.
            'weighted_cross_entropy': Cross-entropy with class weights (primary).
            'cross_entropy': Standard cross-entropy.
            'focal': Focal loss (sensitivity analysis).
        class_weights: Per-class weight tensor for weighted CE.
        focal_gamma: Gamma parameter for focal loss.
        focal_alpha: Alpha weights for focal loss.
        label_smoothing: Label smoothing factor (0.0 = no smoothing).

    Returns:
        Loss function module.
    """
    if loss_type in ("weighted_cross_entropy", "weighted_ce"):
        if class_weights is not None:
            logger.info(f"Using weighted cross-entropy with weights: {class_weights.tolist()}")
        else:
            logger.info("Using cross-entropy (no class weights provided)")
        return nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing,
        )

    elif loss_type == "cross_entropy":
        logger.info("Using standard cross-entropy loss")
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    elif loss_type == "focal":
        logger.info(
            f"Using focal loss (gamma={focal_gamma}) - "
            f"sensitivity analysis option"
        )
        return FocalLoss(
            gamma=focal_gamma,
            alpha=focal_alpha,
        )

    else:
        raise ValueError(
            f"Unknown loss type: {loss_type}. "
            f"Options: 'weighted_cross_entropy', 'cross_entropy', 'focal'"
        )
