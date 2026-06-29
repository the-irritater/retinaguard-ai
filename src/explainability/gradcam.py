"""
RetinaGuard AI — Grad-CAM visual explanations
================================================
Purpose: Generate visual explanations for model predictions using Grad-CAM.

Key design decisions:
- Computes Grad-CAM overlays on target convolutional layers.
- Outputs must display the original image, heatmap, and overlay side-by-side.
- Every Grad-CAM output contains the mandatory disclaimer:
  "The highlighted regions influenced the model's prediction. They are not confirmed lesion boundaries."
- Grad-CAM is never described as a lesion detector or lesion boundaries locator.

Usage:
    from src.explainability.gradcam import GradCAMExplainer
    explainer = GradCAMExplainer(model, target_layer)
    overlay, heatmap = explainer.generate(image_tensor)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# Import pytorch-grad-cam with safety check
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
    HAS_PYTORCH_GRAD_CAM = True
except ImportError:
    HAS_PYTORCH_GRAD_CAM = False
    logger = logging.getLogger("retinaguard.gradcam")
    logger.warning("pytorch-grad-cam library not found. Will use custom Grad-CAM fallback.")

logger = logging.getLogger("retinaguard.gradcam")

# Mandatory warning text required for all explainability visualisations
DISCLAIMER_TEXT = (
    "The highlighted regions influenced the model's prediction. "
    "They are not confirmed lesion boundaries."
)


class GradCAMExplainer:
    """Wrapper class for generating Grad-CAM visualizations.

    Supports both pytorch-grad-cam library and a PyTorch fallback.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[nn.Module] = None,
        use_cuda: bool = False,
    ) -> None:
        """Initialise the Grad-CAM explainer.

        Args:
            model: Trained DR Classifier model.
            target_layer: The target convolutional layer (e.g. model.backbone.layer4).
                If None, auto-detected from model architecture.
            use_cuda: Whether to use GPU.
        """
        self.model = model
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        
        # If target layer is not provided, try to get it from the model
        if target_layer is None and hasattr(model, "get_grad_cam_target_layer"):
            self.target_layer = model.get_grad_cam_target_layer()
        else:
            self.target_layer = target_layer

        logger.info(f"Grad-CAM explainer initialized. Target layer: {self.target_layer}")

    def generate(
        self,
        image_tensor: torch.Tensor,
        target_category: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate Grad-CAM heatmap and overlay on the original image.

        Args:
            image_tensor: Input image tensor of shape (3, H, W).
            target_category: Class index to compute CAM for (e.g. 1 for positive class).

        Returns:
            Tuple of:
            - overlay: RGB image with CAM overlay (shape: [H, W, 3]).
            - grayscale_cam: Grayscale heatmap (shape: [H, W]).
        """
        self.model.eval()
        
        # Ensure tensor has batch dimension: (1, 3, H, W)
        if len(image_tensor.shape) == 3:
            input_tensor = image_tensor.unsqueeze(0)
        else:
            input_tensor = image_tensor
            
        input_tensor = input_tensor.to(self.device)

        # Reconstruct normalized image for visualization overlay (0.0 to 1.0)
        # Assumes standard ImageNet normalization
        img_np = image_tensor.cpu().numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0.0, 1.0)

        if HAS_PYTORCH_GRAD_CAM and self.target_layer is not None:
            # Use the pytorch-grad-cam library
            targets = [ClassifierOutputTarget(target_category)]
            
            with GradCAM(
                model=self.model,
                target_layers=[self.target_layer],
            ) as cam:
                grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
                overlay = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
                
            return overlay, grayscale_cam
        else:
            # Fallback: simple native implementation of Grad-CAM
            return self._fallback_gradcam(input_tensor, img_np, target_category)

    def _fallback_gradcam(
        self,
        input_tensor: torch.Tensor,
        img_np: np.ndarray,
        target_category: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Custom PyTorch native fallback for Grad-CAM when library is missing.

        Args:
            input_tensor: Image tensor of shape (1, 3, H, W).
            img_np: Unnormalized numpy image (shape: [H, W, 3]).
            target_category: Target class index.

        Returns:
            Tuple of (overlay, grayscale_cam).
        """
        gradients: list[torch.Tensor] = []
        activations: list[torch.Tensor] = []

        def save_gradient(grad):
            gradients.append(grad)

        # Hook to capture forward activations and backward gradients
        def hook_fn(module, input, output):
            activations.append(output)
            output.register_hook(save_gradient)

        # Register hook on target layer
        if self.target_layer is not None:
            handle = self.target_layer.register_forward_hook(hook_fn)
        else:
            # If no target layer, register on the last layer of the backbone
            if hasattr(self.model, "backbone") and hasattr(self.model.backbone, "conv_head"):
                handle = self.model.backbone.conv_head.register_forward_hook(hook_fn)
            else:
                return (img_np * 255).astype(np.uint8), np.zeros(img_np.shape[:2])

        # Forward pass
        logits = self.model(input_tensor)
        score = logits[0, target_category]

        # Backward pass
        self.model.zero_grad()
        score.backward()

        # Remove hook
        handle.remove()

        if len(gradients) == 0 or len(activations) == 0:
            logger.warning("Fallback Grad-CAM hook failed. Returning original image.")
            return (img_np * 255).astype(np.uint8), np.zeros(img_np.shape[:2])

        # Compute weights: spatial average of gradients
        grads = gradients[0].cpu().data.numpy()[0]
        acts = activations[0].cpu().data.numpy()[0]
        
        weights = np.mean(grads, axis=(1, 2))
        
        # Weighted combination of activations
        cam = np.zeros(acts.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * acts[i, :, :]

        # ReLU on CAM to keep only positive influence
        cam = np.maximum(cam, 0)
        
        # Resize to original image size
        cam = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))
        
        # Normalize
        cam_max = cam.max()
        if cam_max > 0:
            cam = cam / cam_max

        # Create overlay
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
        
        # Superimpose
        overlay = 0.5 * heatmap + 0.5 * img_np
        overlay = np.clip(overlay, 0.0, 1.0)
        overlay = (overlay * 255).astype(np.uint8)

        return overlay, cam


# ---------------------------------------------------------------------------
# Explainability Plotting Utility
# ---------------------------------------------------------------------------
def plot_gradcam_panel(
    original_img: np.ndarray,
    overlay_img: np.ndarray,
    grayscale_cam: np.ndarray,
    true_label: int | str,
    pred_label: int | str,
    probability: float,
    image_id: str,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """Create a panel plot displaying the original image, heatmap, and overlay.

    Always embeds the mandatory disclaimer text in the figure.

    Args:
        original_img: RGB original image as numpy array.
        overlay_img: RGB Grad-CAM overlay image.
        grayscale_cam: Grayscale heatmap array.
        true_label: True label value or name.
        pred_label: Predicted label value or name.
        probability: Predicted class probability.
        image_id: Image identifier.
        output_path: Optional path to save the generated panel plot.

    Returns:
        Matplotlib Figure object.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    # Plot original image
    axes[0].imshow(original_img)
    axes[0].set_title(f"Original Image: {image_id}")
    axes[0].axis("off")

    # Plot heatmap
    axes[1].imshow(grayscale_cam, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")

    # Plot overlay
    axes[2].imshow(overlay_img)
    axes[2].set_title("Grad-CAM Overlay")
    axes[2].axis("off")

    # Add information titles/captions
    info_text = (
        f"True: {true_label} | Predicted: {pred_label} (p = {probability:.4f})"
    )
    plt.suptitle(info_text, fontsize=14, y=0.92, fontweight="bold")

    # Add mandatory disclaimer at the bottom of the figure
    fig.text(
        0.5, 0.02,
        f"Disclaimer: {DISCLAIMER_TEXT}",
        ha="center", fontsize=11, color="red",
        style="italic", bbox=dict(facecolor="white", alpha=0.8, edgecolor="red")
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.9])

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved Grad-CAM panel plot to {output_path}")
        plt.close(fig)

    return fig
