"""
RetinaGuard AI — Baseline Models Unit Tests
===========================================
Purpose: Verify that MajorityClassifier and LogisticRegressionBaseline fit and
predict correctly.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from src.models.baselines import MajorityClassifier, LogisticRegressionBaseline


class MockDataLoaderDataset(Dataset):
    """Simple mock dataset yielding random tensors and labels."""
    def __init__(self, size: int = 10) -> None:
        self.size = size
        # Random ImageNet-like tensors (3, 224, 224)
        self.images = torch.randn(size, 3, 224, 224)
        self.labels = torch.tensor([1 if i % 3 == 0 else 0 for i in range(size)], dtype=torch.long)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict:
        return {
            "image": self.images[idx],
            "label": self.labels[idx],
        }


def test_majority_classifier() -> None:
    """Verify that MajorityClassifier predicts the most frequent training class."""
    # Training set has more 0s than 1s
    y_train = np.array([0, 0, 1, 0, 1, 0])
    
    clf = MajorityClassifier(num_classes=2)
    clf.fit(y_train)
    
    assert clf.majority_class == 0
    
    # Test predictions
    x_test = np.zeros((3, 10))
    preds = clf.predict(x_test)
    assert np.array_equal(preds, [0, 0, 0])
    
    # Probabilities should be 1.0 for index 0 and 0.0 for index 1
    probs = clf.predict_proba(x_test)
    assert np.array_equal(probs, [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])


def test_logistic_regression_baseline() -> None:
    """Verify that LogisticRegressionBaseline can extract features and fit."""
    train_dataset = MockDataLoaderDataset(size=6)
    val_dataset = MockDataLoaderDataset(size=3)
    
    train_loader = DataLoader(train_dataset, batch_size=2)
    val_loader = DataLoader(val_dataset, batch_size=2)
    
    # Run on CPU for testing safety
    device = torch.device("cpu")
    
    baseline = LogisticRegressionBaseline(device=device)
    baseline.fit(train_loader)
    
    metrics = baseline.evaluate(val_loader)
    assert "accuracy" in metrics
    assert "auc" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0
