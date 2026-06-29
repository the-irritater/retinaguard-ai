"""
RetinaGuard AI — Preprocessing Unit Tests
==========================================
Purpose: Verify that the preprocessing pipeline behaves correctly:
- Correct output size (384x384 by default).
- No augmentation is applied during validation or testing.
- Images are read and processed with correct RGB channel order.
- Reproducible when random seeds are fixed.
"""

from __future__ import annotations

import albumentations as A
import numpy as np
import pytest
import torch

from src.data.preprocessing import (
    get_transforms,
    preprocess_image,
    read_image_safely,
)


@pytest.fixture
def dummy_image() -> np.ndarray:
    """Create a dummy RGB image with a simulated circular retina and black borders.

    Returns:
        Numpy array representing the image (H, W, 3).
    """
    # Create 500x500 black image
    img = np.zeros((500, 500, 3), dtype=np.uint8)
    # Draw a green circle (simulated retina)
    img = np.ascontiguousarray(img)
    import cv2
    cv2.circle(img, (250, 250), 200, (0, 255, 0), -1)
    return img


def test_preprocess_image_output_size(dummy_image: np.ndarray) -> None:
    """Verify that preprocess_image correctly resizes the image."""
    sizes = [224, 384, 512]
    for size in sizes:
        processed = preprocess_image(dummy_image, image_size=size, crop_borders=True)
        assert processed.shape == (size, size, 3), f"Expected shape ({size}, {size}, 3), got {processed.shape}"


def test_no_augmentation_in_eval(dummy_image: np.ndarray) -> None:
    """Verify that validation and test transforms apply NO random augmentations."""
    config = {
        "preprocessing": {
            "image_size": 224,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "augmentation": {
            "horizontal_flip": False,
            "rotation_limit": 15,
            "translate_percent": 0.05,
            "scale_range": [0.9, 1.1],
            "brightness_limit": 0.1,
            "contrast_limit": 0.1,
        },
    }

    # Eval transforms
    val_transform = get_transforms("val", config)
    test_transform = get_transforms("test", config)

    # Apply transform twice and check if outputs are identical
    res1_val = val_transform(image=dummy_image)["image"]
    res2_val = val_transform(image=dummy_image)["image"]
    assert torch.allclose(res1_val, res2_val), "Validation transform is non-deterministic (has augmentation)"

    res1_test = test_transform(image=dummy_image)["image"]
    res2_test = test_transform(image=dummy_image)["image"]
    assert torch.allclose(res1_test, res2_test), "Test transform is non-deterministic (has augmentation)"


def test_augmentation_determinism_with_seed(dummy_image: np.ndarray) -> None:
    """Verify that training augmentations are reproducible if seeds are set/controlled."""
    config = {
        "preprocessing": {
            "image_size": 224,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "augmentation": {
            "horizontal_flip": False,
            "rotation_limit": 15,
            "translate_percent": 0.05,
            "scale_range": [0.9, 1.1],
            "brightness_limit": 0.1,
            "contrast_limit": 0.1,
        },
    }

    # Set seed and create transform 1
    import random
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    train_transform1 = get_transforms("train", config)
    train_transform1.set_random_seed(42)
    res1 = train_transform1(image=dummy_image.copy())["image"]

    # Set seed and create transform 2
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    train_transform2 = get_transforms("train", config)
    train_transform2.set_random_seed(42)
    res2 = train_transform2(image=dummy_image.copy())["image"]

    assert torch.allclose(res1, res2, atol=1e-4, rtol=1e-4), "Training transforms differ despite identical seed state"


def test_rgb_channel_order(dummy_image: np.ndarray) -> None:
    """Verify that the preprocessing returns RGB channel order.

    In the dummy image, we drew a green circle (BGR: 0, 255, 0, RGB: 0, 255, 0).
    Let's draw a red circle: BGR (0, 0, 255) vs RGB (255, 0, 0).
    """
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    # Draw pure red in BGR: (0, 0, 255)
    img[:, :] = [0, 0, 255]

    # Convert to RGB (normal read path)
    import cv2
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    assert img_rgb[0, 0, 0] == 255  # Red channel is first
    assert img_rgb[0, 0, 2] == 0    # Blue channel is last

    processed = preprocess_image(img_rgb, image_size=64, crop_borders=False)
    # Still RGB
    assert processed[0, 0, 0] == 255
    assert processed[0, 0, 2] == 0
