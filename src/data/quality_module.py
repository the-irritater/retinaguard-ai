"""
RetinaGuard AI - Synthetic Image-Degradation Module
-
Purpose: Assess and simulate quality degradations in fundus images.

WARNING: This module is purely experimental. IDRiD contains only expert-selected
images of adequate clinical quality. It does NOT contain a representative set
of ungradable images. This module is NOT clinically validated.

Simulated degradations supported:
- Blur (Gaussian blur)
- Underexposure (brightness reduction)
- Overexposure (brightness enhancement)
- Cropping (excessive center crop)
- Glare (simulated light reflection)

Usage:
    from src.data.quality_module import ImageDegradationSimulator, detect_synthetic_degradations
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import cv2
import numpy as np

logger = logging.getLogger("retinaguard.quality")


class ImageDegradationSimulator:
    """Simulates clinical image quality degradations on retinal fundus photographs.

    Used for checking model robustness against artificial quality defects.
    """

    @staticmethod
    def apply_blur(image: np.ndarray, kernel_size: int = 15) -> np.ndarray:
        """Apply Gaussian blur to simulate focus issues."""
        if kernel_size % 2 == 0:
            kernel_size += 1
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    @staticmethod
    def apply_underexposure(image: np.ndarray, factor: float = 0.4) -> np.ndarray:
        """Simulate underexposure by reducing brightness."""
        # Convert to HSV, scale V channel, convert back
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hsv = hsv.astype(np.float32)
        hsv[:, :, 2] = hsv[:, :, 2] * factor
        hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    @staticmethod
    def apply_overexposure(image: np.ndarray, factor: float = 1.6) -> np.ndarray:
        """Simulate overexposure by increasing brightness."""
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hsv = hsv.astype(np.float32)
        hsv[:, :, 2] = hsv[:, :, 2] * factor
        hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    @staticmethod
    def apply_glare(
        image: np.ndarray,
        center: Optional[Tuple[int, int]] = None,
        radius: int = 80,
        intensity: int = 200,
    ) -> np.ndarray:
        """Simulate camera lens glare or reflection."""
        h, w = image.shape[:2]
        if center is None:
            # Default to top-right quadrant
            center = (int(w * 0.7), int(h * 0.3))

        # Create white glare mask with radial falloff
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.circle(mask, center, radius, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (radius | 1, radius | 1), 0)

        # Broadcast mask
        mask = np.expand_dims(mask, axis=2)
        
        # Add intensity
        glared = image.astype(np.float32) + (mask * intensity)
        return np.clip(glared, 0, 255).astype(np.uint8)

    @staticmethod
    def apply_cropping(image: np.ndarray, crop_percent: float = 0.3) -> np.ndarray:
        """Simulate field-of-view cropping issues (cut off retina)."""
        h, w = image.shape[:2]
        crop_h = int(h * crop_percent)
        crop_w = int(w * crop_percent)
        
        # Crop from edges
        cropped = image[crop_h // 2: h - crop_h // 2, crop_w // 2: w - crop_w // 2]
        # Pad back to original dimensions with black
        padded = cv2.copyMakeBorder(
            cropped,
            crop_h // 2, crop_h - crop_h // 2,
            crop_w // 2, crop_w - crop_w // 2,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0)
        )
        return padded


# -
# Simple Heuristic Degradation Detector (Experimental)
# -
def detect_synthetic_degradations(
    image: np.ndarray,
    min_sharpness: float = 15.0,
    min_brightness: float = 40.0,
    max_brightness: float = 210.0,
) -> Dict[str, Any]:
    """Analyze image and flag potential degradations (experimental).

    Args:
        image: RGB image numpy array.
        min_sharpness: Sharpness threshold (variance of Laplacian).
        min_brightness: Lower bound for average brightness.
        max_brightness: Upper bound for average brightness.

    Returns:
        Dictionary indicating quality metrics and warning flags.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    # Metrics
    brightness = float(gray.mean())
    contrast = float(gray.std())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Warnings
    is_blurry = sharpness < min_sharpness
    is_underexposed = brightness < min_brightness
    is_overexposed = brightness > max_brightness

    # Glare flag (bright spots that are localized)
    _, max_val, _, max_loc = cv2.minMaxLoc(gray)
    is_glare = max_val >= 250 and contrast > 60

    passed = not (is_blurry or is_underexposed or is_overexposed)

    return {
        "passed": passed,
        "brightness": brightness,
        "contrast": contrast,
        "sharpness": sharpness,
        "warnings": {
            "blurry": is_blurry,
            "underexposed": is_underexposed,
            "overexposed": is_overexposed,
            "glare": is_glare,
        },
        "disclaimer": (
            "EXPERIMENTAL RESULTS. Quality assessment has not been clinically "
            "validated and is for research purposes only."
        ),
    }
