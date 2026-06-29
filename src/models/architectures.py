"""
RetinaGuard AI — Model Architectures
======================================
Purpose: Configurable CNN architectures for DR classification with
transfer learning and two-stage training support.

Primary model: EfficientNet-B0
Secondary: MobileNetV3 (deployment comparison)
Optional: ResNet50 (only if needed for research questions)

All models use ImageNet-pretrained backbones with a custom classification
head for binary (DR grade ≥ 2) or multiclass (DR grades 0-4) tasks.

Usage:
    from src.models.architectures import create_model
    model = create_model('efficientnet_b0', num_classes=2)
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
import torch.nn as nn
import timm

logger = logging.getLogger("retinaguard.architectures")


class DRClassifier(nn.Module):
    """DR Classification model with transfer-learning backbone.

    Supports:
    - EfficientNet-B0 (primary)
    - MobileNetV3-Large (deployment comparison)
    - ResNet50 (optional)
    - ResNet18 (baseline)
    - Two-stage training (freeze/unfreeze backbone)

    Attributes:
        backbone: Feature extraction network.
        classifier: Classification head.
        backbone_name: Name of the backbone architecture.
        num_classes: Number of output classes.
    """

    def __init__(
        self,
        backbone_name: str = "efficientnet_b0",
        num_classes: int = 2,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
    ) -> None:
        """Initialise the DR classifier.

        Args:
            backbone_name: Name of the backbone architecture.
                Supported: 'efficientnet_b0', 'mobilenetv3_large_100',
                           'resnet50', 'resnet18'.
            num_classes: Number of output classes (2 for binary, 5 for multiclass).
            pretrained: Whether to use ImageNet-pretrained weights.
            dropout_rate: Dropout probability for the classification head.
        """
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes

        # Create backbone using timm
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,  # Remove original classifier
        )

        # Get feature dimension from backbone
        with torch.no_grad():
            dummy = torch.randn(1, 3, 384, 384)
            features = self.backbone(dummy)
            n_features = features.shape[-1]

        logger.info(
            f"Backbone: {backbone_name}, features: {n_features}, "
            f"pretrained: {pretrained}"
        )

        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(n_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(256, num_classes),
        )

        # Initialise classifier weights
        self._init_classifier()

    def _init_classifier(self) -> None:
        """Initialise classification head with Kaiming initialisation."""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, 3, H, W).

        Returns:
            Logits of shape (B, num_classes).
        """
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters for Stage 1 training."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info(f"Backbone frozen: {self.backbone_name}")

    def unfreeze_backbone(self, unfreeze_from: Optional[str] = None) -> None:
        """Unfreeze backbone parameters for Stage 2 fine-tuning.

        Args:
            unfreeze_from: If specified, only unfreeze layers from this name
                onwards. If None, unfreeze the entire backbone.
        """
        if unfreeze_from is None:
            for param in self.backbone.parameters():
                param.requires_grad = True
            logger.info(f"Entire backbone unfrozen: {self.backbone_name}")
        else:
            found = False
            for name, param in self.backbone.named_parameters():
                if unfreeze_from in name:
                    found = True
                param.requires_grad = found
            n_unfrozen = sum(p.requires_grad for p in self.backbone.parameters())
            n_total = sum(1 for _ in self.backbone.parameters())
            logger.info(
                f"Backbone partially unfrozen from '{unfreeze_from}': "
                f"{n_unfrozen}/{n_total} parameters trainable"
            )

    def get_grad_cam_target_layer(self) -> nn.Module:
        """Return the target layer for Grad-CAM.

        Returns the last convolutional block for each architecture.

        Returns:
            Target layer module.
        """
        if "efficientnet" in self.backbone_name:
            # Last block of EfficientNet
            return self.backbone.blocks[-1]
        elif "mobilenetv3" in self.backbone_name:
            return self.backbone.blocks[-1]
        elif "resnet" in self.backbone_name:
            return self.backbone.layer4
        else:
            # Fallback: try to find last convolutional layer
            layers = [
                m for m in self.backbone.modules()
                if isinstance(m, (nn.Conv2d,))
            ]
            if layers:
                return layers[-1]
            raise ValueError(
                f"Cannot determine Grad-CAM target layer for {self.backbone_name}"
            )

    def count_parameters(self) -> dict[str, int]:
        """Count model parameters.

        Returns:
            Dictionary with total, trainable, and frozen parameter counts.
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        return {
            "total": total,
            "trainable": trainable,
            "frozen": frozen,
        }


def create_model(
    architecture: str = "efficientnet_b0",
    num_classes: int = 2,
    pretrained: bool = True,
    dropout_rate: float = 0.3,
) -> DRClassifier:
    """Factory function to create a DR classifier.

    Args:
        architecture: Model architecture name.
        num_classes: Number of output classes.
        pretrained: Whether to use pretrained weights.
        dropout_rate: Dropout rate for classification head.

    Returns:
        Configured DRClassifier model.
    """
    # Map friendly names to timm model names
    arch_map = {
        "efficientnet_b0": "efficientnet_b0",
        "efficientnet-b0": "efficientnet_b0",
        "mobilenetv3": "mobilenetv3_large_100",
        "mobilenetv3_large": "mobilenetv3_large_100",
        "resnet50": "resnet50",
        "resnet18": "resnet18",
    }

    backbone_name = arch_map.get(architecture.lower(), architecture)

    model = DRClassifier(
        backbone_name=backbone_name,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate,
    )

    params = model.count_parameters()
    logger.info(
        f"Created model: {backbone_name}\n"
        f"  Classes: {num_classes}\n"
        f"  Parameters: {params['total']:,} total, "
        f"{params['trainable']:,} trainable"
    )

    return model
