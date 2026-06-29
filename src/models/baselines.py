"""
RetinaGuard AI - Baseline Classifiers
-
Purpose: Simple and classical baseline models for DR classification.
- Majority class classifier: predicts the most frequent class.
- Frozen pretrained ResNet18 feature extractor with simple linear classifier.
- Embeddings-based Logistic Regression: extracts features using ResNet18
  and fits a scikit-learn Logistic Regression on the training fold.

Reported metrics are evaluated on the validation fold only. Test set is NOT used.

Usage:
    from src.models.baselines import MajorityClassifier, LogisticRegressionBaseline
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import torchvision.models as models

logger = logging.getLogger("retinaguard.baselines")


# -
# 1. Majority Class Baseline
# -
class MajorityClassifier:
    """Predicts the most frequent class in the training labels.

    Attributes:
        majority_class: The class index that is most common.
        num_classes: Total number of classes.
    """

    def __init__(self, num_classes: int = 2) -> None:
        """Initialise majority classifier."""
        self.majority_class: Optional[int] = None
        self.num_classes = num_classes

    def fit(self, y_train: np.ndarray) -> None:
        """Determine the majority class.

        Args:
            y_train: Training labels.
        """
        counts = np.bincount(y_train, minlength=self.num_classes)
        self.majority_class = int(np.argmax(counts))
        logger.info(f"Majority classifier fitted: class = {self.majority_class}")

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Args:
            x: Input feature/image array of shape (N, ...).

        Returns:
            Predicted class labels.
        """
        if self.majority_class is None:
            raise ValueError("Classifier must be fitted first.")
        n_samples = len(x)
        return np.full(n_samples, self.majority_class)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Outputs 1.0 for majority class and 0.0 for others.

        Args:
            x: Input array.

        Returns:
            Array of shape (N, num_classes).
        """
        if self.majority_class is None:
            raise ValueError("Classifier must be fitted first.")
        n_samples = len(x)
        probs = np.zeros((n_samples, self.num_classes))
        probs[:, self.majority_class] = 1.0
        return probs


# -
# 2 & 3. Classical Logistic Regression on Pretrained Image Embeddings
# -
class LogisticRegressionBaseline:
    """Extracts features using a frozen ResNet18 backbone, and fits a Logistic Regression model.

    Attributes:
        device: Torch compute device.
        backbone: Pretrained ResNet18 model with classifier removed.
        clf: Scikit-learn Logistic Regression model.
    """

    def __init__(self, device: Optional[torch.device] = None) -> None:
        """Initialise embeddings baseline."""
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load frozen ResNet18 backbone
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        resnet = models.resnet18(weights=weights)
        
        # Remove classification layer to get feature extractor
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.backbone.to(self.device)
        self.backbone.eval()
        
        # Scikit-learn classifier
        self.clf = LogisticRegression(max_iter=1000, C=1.0)

    @torch.no_grad()
    def _extract_features(self, dataloader: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Extract embeddings from backbone.

        Args:
            dataloader: PyTorch DataLoader yielding batches with 'image' and 'label'.

        Returns:
            Tuple of:
            - features: Numpy array of shape (N, 512).
            - labels: Numpy array of shape (N,).
        """
        features_list = []
        labels_list = []

        for batch in dataloader:
            images = batch["image"].to(self.device)
            labels = batch["label"]
            
            # Forward pass through feature extractor
            feats = self.backbone(images)
            # Flatten to vector (B, 512)
            feats = torch.flatten(feats, 1)
            
            features_list.append(feats.cpu().numpy())
            labels_list.append(labels.numpy())

        features = np.concatenate(features_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)
        return features, labels

    def fit(self, train_loader: Any) -> None:
        """Extract features and fit the logistic regression model.

        Args:
            train_loader: Training PyTorch DataLoader.
        """
        logger.info("Extracting embeddings for training...")
        x_train, y_train = self._extract_features(train_loader)
        
        logger.info(f"Fitting Logistic Regression on {x_train.shape[0]} samples...")
        self.clf.fit(x_train, y_train)
        train_acc = self.clf.score(x_train, y_train)
        logger.info(f"Baseline fit complete. Training Accuracy: {train_acc:.4f}")

    def evaluate(self, val_loader: Any) -> Dict[str, float]:
        """Extract features and evaluate on validation set.

        Args:
            val_loader: Validation PyTorch DataLoader.

        Returns:
            Dictionary of validation metrics.
        """
        logger.info("Extracting embeddings for validation...")
        x_val, y_val = self._extract_features(val_loader)
        
        preds = self.clf.predict(x_val)
        probs = self.clf.predict_proba(x_val)
        
        acc = accuracy_score(y_val, preds)
        
        # Binary or Multiclass AUC
        n_classes = len(np.unique(y_val))
        if n_classes == 2:
            auc = roc_auc_score(y_val, probs[:, 1])
        else:
            auc = roc_auc_score(y_val, probs, multi_class="ovr")

        metrics = {
            "accuracy": float(acc),
            "auc": float(auc),
        }
        
        logger.info(f"Baseline Validation Metrics: Acc={acc:.4f}, AUC={auc:.4f}")
        return metrics
