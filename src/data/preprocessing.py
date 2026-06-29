"""
RetinaGuard AI — Image Preprocessing Pipeline
===============================================
Purpose: Safely read, crop, resize, normalise, and augment retinal fundus
images for model training and inference.

Key design decisions:
- Black border detection preserves the full circular retinal field.
- Training-only augmentations are medically conservative.
- Horizontal flip is EXCLUDED from baseline (changes retinal laterality).
- Validation and test images receive NO augmentation.
- All parameters are configurable via YAML.
- Reproducible with fixed random seeds.

Usage:
    from src.data.preprocessing import get_transforms, preprocess_image
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2

logger = logging.getLogger("retinaguard.preprocessing")


# ---------------------------------------------------------------------------
# Black border detection and cropping
# ---------------------------------------------------------------------------
def detect_and_crop_borders(
    image: np.ndarray,
    threshold: int = 10,
    min_crop_ratio: float = 0.5,
) -> np.ndarray:
    """Detect and crop excessive black borders while preserving the retinal field.

    Uses a grayscale threshold to find the bounding box of non-black content,
    then crops with a safety margin to ensure the full circular retinal field
    is preserved.

    Args:
        image: Input BGR image as numpy array.
        threshold: Pixel intensity below which pixels are considered black border.
        min_crop_ratio: Minimum ratio of cropped size to original size (safety).

    Returns:
        Cropped image preserving the retinal field.
    """
    if image is None or image.size == 0:
        return image

    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find non-black region
    mask = gray > threshold
    coords = np.argwhere(mask)

    if len(coords) == 0:
        logger.warning("Image appears entirely black — returning original")
        return image

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Add small margin (2% of dimension)
    margin_y = int(h * 0.02)
    margin_x = int(w * 0.02)
    y_min = max(0, y_min - margin_y)
    x_min = max(0, x_min - margin_x)
    y_max = min(h - 1, y_max + margin_y)
    x_max = min(w - 1, x_max + margin_x)

    crop_h = y_max - y_min
    crop_w = x_max - x_min

    # Safety check: don't crop too aggressively
    if crop_h / h < min_crop_ratio or crop_w / w < min_crop_ratio:
        logger.debug(
            f"Crop would be too aggressive ({crop_h/h:.2f}, {crop_w/w:.2f}), "
            f"keeping original"
        )
        return image

    cropped = image[y_min:y_max + 1, x_min:x_max + 1]
    return cropped


# ---------------------------------------------------------------------------
# Safe image reading
# ---------------------------------------------------------------------------
def read_image_safely(
    filepath: str | Path,
) -> Optional[np.ndarray]:
    """Safely read an image file and convert to RGB.

    Args:
        filepath: Path to the image file.

    Returns:
        RGB numpy array, or None if reading fails.
    """
    filepath = str(filepath)
    try:
        img = cv2.imread(filepath, cv2.IMREAD_COLOR)
        if img is None:
            logger.error(f"Failed to read image: {filepath}")
            return None
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    except Exception as e:
        logger.error(f"Error reading image {filepath}: {e}")
        return None


# ---------------------------------------------------------------------------
# Preprocessing pipeline
# ---------------------------------------------------------------------------
def preprocess_image(
    image: np.ndarray,
    image_size: int = 384,
    border_threshold: int = 10,
    min_crop_ratio: float = 0.5,
    crop_borders: bool = True,
) -> np.ndarray:
    """Apply preprocessing pipeline to a single image.

    Steps:
    1. Convert to RGB (if BGR).
    2. Detect and crop black borders.
    3. Resize to target size.

    Note: Normalisation and augmentation are handled by the transform pipeline.

    Args:
        image: Input image as numpy array (RGB).
        image_size: Target square size for resizing.
        border_threshold: Threshold for border detection.
        min_crop_ratio: Safety ratio for border cropping.
        crop_borders: Whether to crop black borders.

    Returns:
        Preprocessed image as numpy array.
    """
    if crop_borders:
        # Need BGR for border detection (uses cv2.cvtColor internally)
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        img_bgr = detect_and_crop_borders(
            img_bgr,
            threshold=border_threshold,
            min_crop_ratio=min_crop_ratio,
        )
        image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Resize
    image = cv2.resize(
        image,
        (image_size, image_size),
        interpolation=cv2.INTER_LANCZOS4,
    )

    return image


# ---------------------------------------------------------------------------
# Albumentations transform pipelines
# ---------------------------------------------------------------------------
def get_training_transforms(
    image_size: int = 384,
    mean: list[float] | None = None,
    std: list[float] | None = None,
    rotation_limit: int = 15,
    translate_percent: float = 0.05,
    scale_range: tuple[float, float] = (0.9, 1.1),
    brightness_limit: float = 0.1,
    contrast_limit: float = 0.1,
    horizontal_flip: bool = False,
) -> A.Compose:
    """Create training augmentation pipeline.

    Augmentations are medically conservative:
    - Rotation ±15° (default).
    - Small translation and scaling.
    - Mild brightness and contrast adjustment.
    - Horizontal flip EXCLUDED by default (changes retinal laterality).

    Args:
        image_size: Target image size.
        mean: Normalisation mean (ImageNet default if None).
        std: Normalisation std (ImageNet default if None).
        rotation_limit: Maximum rotation in degrees.
        translate_percent: Maximum translation as fraction of image size.
        scale_range: (min_scale, max_scale) tuple.
        brightness_limit: Maximum brightness change.
        contrast_limit: Maximum contrast change.
        horizontal_flip: Whether to include horizontal flip (default False).

    Returns:
        Albumentations Compose pipeline.
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]

    transforms_list = [
        A.Resize(image_size, image_size, interpolation=cv2.INTER_LANCZOS4),
    ]

    # Geometric augmentations
    if horizontal_flip:
        transforms_list.append(A.HorizontalFlip(p=0.5))
        logger.warning(
            "Horizontal flip enabled — this changes retinal laterality. "
            "Use only as ablation experiment."
        )

    transforms_list.extend([
        A.ShiftScaleRotate(
            shift_limit=translate_percent,
            scale_limit=(scale_range[0] - 1.0, scale_range[1] - 1.0),
            rotate_limit=rotation_limit,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            p=0.7,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=brightness_limit,
            contrast_limit=contrast_limit,
            p=0.5,
        ),
    ])

    # Normalisation and tensor conversion
    transforms_list.extend([
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

    return A.Compose(transforms_list)


def get_eval_transforms(
    image_size: int = 384,
    mean: list[float] | None = None,
    std: list[float] | None = None,
) -> A.Compose:
    """Create evaluation (validation/test) transform pipeline.

    NO augmentation is applied. Only resize, normalise, and tensor conversion.

    Args:
        image_size: Target image size.
        mean: Normalisation mean (ImageNet default if None).
        std: Normalisation std (ImageNet default if None).

    Returns:
        Albumentations Compose pipeline.
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]

    return A.Compose([
        A.Resize(image_size, image_size, interpolation=cv2.INTER_LANCZOS4),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def get_transforms(
    mode: str,
    config: dict[str, Any],
) -> A.Compose:
    """Get the appropriate transform pipeline based on mode.

    Args:
        mode: One of 'train', 'val', 'test'.
        config: Configuration dictionary.

    Returns:
        Albumentations Compose pipeline.

    Raises:
        ValueError: If mode is not recognized.
    """
    preproc = config.get("preprocessing", {})
    aug_config = config.get("augmentation", {})

    image_size = preproc.get("image_size", 384)
    mean = preproc.get("mean", [0.485, 0.456, 0.406])
    std = preproc.get("std", [0.229, 0.224, 0.225])

    if mode == "train":
        return get_training_transforms(
            image_size=image_size,
            mean=mean,
            std=std,
            rotation_limit=aug_config.get("rotation_limit", 15),
            translate_percent=aug_config.get("translate_percent", 0.05),
            scale_range=tuple(aug_config.get("scale_range", [0.9, 1.1])),
            brightness_limit=aug_config.get("brightness_limit", 0.1),
            contrast_limit=aug_config.get("contrast_limit", 0.1),
            horizontal_flip=aug_config.get("horizontal_flip", False),
        )
    elif mode in ("val", "test", "inference"):
        return get_eval_transforms(
            image_size=image_size,
            mean=mean,
            std=std,
        )
    else:
        raise ValueError(
            f"Unknown mode '{mode}'. Expected 'train', 'val', 'test', or 'inference'."
        )
