"""
RetinaGuard AI — Streamlit Demonstration Application
=====================================================
Purpose: Interactive web interface for demonstrating the DR classification
system. Allows uploading retinal images, viewing predictions, confidence,
Grad-CAM overlays, and patient education information.

⚠️ THIS IS A RESEARCH DEMONSTRATION — NOT A DIAGNOSTIC TOOL ⚠️

Usage:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import streamlit as st
import torch
import yaml

# Configure page
st.set_page_config(
    page_title="RetinaGuard AI — DR Screening Research Demo",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

logger = logging.getLogger("retinaguard.app")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DISCLAIMER_BANNER = """
⚠️ **RESEARCH PROTOTYPE — NOT A DIAGNOSTIC TOOL**

This system is an academic research demonstration. It is **NOT** a medical device,
**NOT** clinically validated, and **NOT** approved by any regulatory authority
(FDA, CE, TGA, or equivalent). It must **NOT** be used to make diagnostic or
treatment decisions. All outputs require independent verification by a
qualified ophthalmologist.
"""

GRADCAM_DISCLAIMER = (
    "The highlighted regions influenced the model's prediction. "
    "They are **not** confirmed lesion boundaries."
)

PATIENT_EDUCATION = """
### 📚 About Diabetic Retinopathy

Diabetic retinopathy (DR) is a complication of diabetes that affects the blood
vessels in the retina — the light-sensitive tissue at the back of the eye.

**Key Facts:**
- DR is the leading cause of preventable blindness in working-age adults.
- Early stages often have **no symptoms** — regular screening is essential.
- Effective treatments exist, especially when DR is detected early.
- Good blood sugar, blood pressure, and cholesterol control can slow progression.

**DR Severity Grades:**
| Grade | Description | Typical Features |
|---|---|---|
| 0 | No DR | Normal retina |
| 1 | Mild NPDR | A few microaneurysms |
| 2 | Moderate NPDR | More microaneurysms, dot/blot haemorrhages, hard exudates |
| 3 | Severe NPDR | Extensive haemorrhages, venous beading, IRMA |
| 4 | Proliferative DR | Neovascularisation, vitreous haemorrhage |

**What should you do?**
- 🩺 See an ophthalmologist for regular dilated eye exams.
- 📊 Maintain good glycaemic control (HbA1c < 7% as advised by your doctor).
- 🚫 Do not rely on any AI system for medical decisions.

> **Important:** This educational information is general in nature. Consult your
> healthcare provider for advice specific to your situation.
"""


# ---------------------------------------------------------------------------
# Model Loading (cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model(config_path: str, checkpoint_path: str):
    """Load model and config from disk (cached across sessions)."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    
    from src.models.architectures import create_model
    
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    architecture = config.get("model", {}).get("architecture", "efficientnet_b0")
    num_classes = 2
    model = create_model(architecture=architecture, num_classes=num_classes, pretrained=False)
    
    checkpoint_p = Path(checkpoint_path)
    if checkpoint_p.exists():
        checkpoint = torch.load(checkpoint_p, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        st.sidebar.success(f"✅ Model loaded from checkpoint")
    else:
        st.sidebar.warning("⚠️ No checkpoint found — using random weights for demo")
    
    model = model.to(device)
    model.eval()
    return model, config, device


def preprocess_for_inference(image_rgb: np.ndarray, config: dict) -> torch.Tensor:
    """Preprocess an uploaded image for model inference."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    
    from src.data.preprocessing import preprocess_image, get_transforms
    
    image_size = config["preprocessing"]["image_size"]
    processed = preprocess_image(image_rgb, image_size=image_size, crop_borders=True)
    
    transform = get_transforms("test", config)
    result = transform(image=processed)
    return result["image"]


def generate_gradcam(model, image_tensor, device):
    """Generate Grad-CAM overlay for the given image."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    
    from src.explainability.gradcam import GradCAMExplainer
    
    explainer = GradCAMExplainer(model, use_cuda=torch.cuda.is_available())
    overlay, heatmap = explainer.generate(image_tensor.to(device), target_category=1)
    return overlay, heatmap


# ---------------------------------------------------------------------------
# UI Layout
# ---------------------------------------------------------------------------
def main():
    """Main Streamlit application."""
    
    # --- Sidebar ---
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/retina-scan.png", width=64)
        st.title("RetinaGuard AI")
        st.caption("Research Demonstration v1.0")
        
        st.divider()
        
        # Config paths
        project_root = Path(__file__).resolve().parent.parent
        config_path = st.text_input(
            "Config Path",
            value=str(project_root / "configs" / "base_config.yaml"),
        )
        checkpoint_path = st.text_input(
            "Checkpoint Path",
            value=str(project_root / "models" / "checkpoints" / "final" / "best_model_fold0_binary.pt"),
        )
        
        st.divider()
        
        # Threshold
        threshold = st.slider(
            "Classification Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.50,
            step=0.01,
            help="Probability threshold for DR grade ≥ 2 classification. "
                 "Selected on validation data using Youden's J statistic.",
        )
        
        show_gradcam = st.checkbox("Show Grad-CAM Overlay", value=True)
        show_education = st.checkbox("Show Patient Education", value=False)
        
        st.divider()
        st.markdown("📋 [View Model Card](reports/model_card.md)")
        st.markdown("📄 [View Study Protocol](reports/study_protocol.md)")
    
    # --- Main Content ---
    st.markdown(DISCLAIMER_BANNER)
    st.title("👁️ RetinaGuard AI")
    st.subheader("Explainable Deep Learning for Diabetic Retinopathy Research")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload a retinal fundus photograph",
        type=["jpg", "jpeg", "png", "tif", "tiff"],
        help="Upload a colour retinal fundus image for analysis.",
    )
    
    if uploaded_file is not None:
        # Read and decode the image
        file_bytes = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image_bgr is None:
            st.error("❌ Could not read the uploaded image. Please try a different file.")
            return
            
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # Display original image
        col1, col2 = st.columns(2)
        with col1:
            st.image(image_rgb, caption="Uploaded Retinal Image", use_container_width=True)
        
        # Load model
        with st.spinner("Loading model..."):
            try:
                model, config, device = load_model(config_path, checkpoint_path)
            except Exception as e:
                st.error(f"❌ Failed to load model: {e}")
                return
        
        # Preprocess
        with st.spinner("Preprocessing image..."):
            try:
                image_tensor = preprocess_for_inference(image_rgb, config)
            except Exception as e:
                st.error(f"❌ Preprocessing failed: {e}")
                return
        
        # Inference
        with st.spinner("Running inference..."):
            model.eval()
            with torch.no_grad():
                input_tensor = image_tensor.unsqueeze(0).to(device)
                logits = model(input_tensor)
                probs = torch.softmax(logits, dim=1)
                prob_positive = probs[0, 1].item()
                prob_negative = probs[0, 0].item()
        
        # Classification result
        prediction = "DR Grade ≥ 2 (Potentially Referable)" if prob_positive >= threshold else "DR Grade < 2 (Non-Referable)"
        confidence = max(prob_positive, prob_negative)
        
        with col2:
            # Result card
            if prob_positive >= threshold:
                st.error(f"🔴 **{prediction}**")
            else:
                st.success(f"🟢 **{prediction}**")
            
            st.metric("Probability (DR ≥ 2)", f"{prob_positive:.4f}")
            st.metric("Model Confidence", f"{confidence:.2%}")
            st.metric("Threshold Used", f"{threshold:.4f}")
            
            # Uncertainty indicator
            if confidence < 0.6:
                st.warning("⚠️ **Low confidence prediction** — model is uncertain about this image.")
            elif confidence < 0.8:
                st.info("ℹ️ Moderate confidence — consider additional review.")
        
        # Professional recommendation
        st.divider()
        st.markdown("### 🩺 Professional Assessment Recommendation")
        st.info(
            "**This result requires verification by a qualified ophthalmologist.** "
            "This system is a research prototype and cannot replace professional "
            "clinical assessment. Please consult an eye care professional for "
            "diagnosis and treatment planning."
        )
        
        # Grad-CAM
        if show_gradcam:
            st.divider()
            st.markdown("### 🔍 Grad-CAM Visual Explanation")
            st.warning(f"⚠️ {GRADCAM_DISCLAIMER}")
            
            with st.spinner("Generating Grad-CAM overlay..."):
                try:
                    overlay, heatmap = generate_gradcam(model, image_tensor, device)
                    
                    gcol1, gcol2, gcol3 = st.columns(3)
                    
                    # Original (resized)
                    img_np = image_tensor.numpy().transpose(1, 2, 0)
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    img_display = std * img_np + mean
                    img_display = np.clip(img_display, 0.0, 1.0)
                    
                    with gcol1:
                        st.image(img_display, caption="Preprocessed Image", use_container_width=True)
                    with gcol2:
                        st.image(heatmap, caption="Grad-CAM Heatmap", use_container_width=True, clamp=True)
                    with gcol3:
                        st.image(overlay, caption="Grad-CAM Overlay", use_container_width=True)
                        
                except Exception as e:
                    st.error(f"Grad-CAM generation failed: {e}")
        
        # Patient Education
        if show_education:
            st.divider()
            st.markdown(PATIENT_EDUCATION)
    
    else:
        # Welcome state
        st.markdown("---")
        st.markdown(
            "### How to Use\n"
            "1. **Upload** a retinal fundus photograph using the file uploader above.\n"
            "2. **View** the model's prediction, confidence, and classification.\n"
            "3. **Explore** the Grad-CAM overlay to see which regions influenced the prediction.\n"
            "4. **Read** the patient education section for general information about DR.\n"
            "5. **Always** consult a qualified ophthalmologist for clinical decisions.\n"
        )
        
        st.markdown("---")
        st.markdown(
            "### ⚡ System Capabilities\n"
            "| Feature | Description |\n"
            "|---|---|\n"
            "| **Binary Classification** | Detects DR grade ≥ 2 (moderate NPDR or worse) |\n"
            "| **Confidence Reporting** | Reports prediction probability and model confidence |\n"
            "| **Grad-CAM Explanations** | Visual heatmaps showing influential image regions |\n"
            "| **Calibrated Probabilities** | Temperature-scaled predictions for better calibration |\n"
            "| **Patient Education** | Fixed educational content about diabetic retinopathy |\n"
            "| **Professional Referral** | Always recommends ophthalmologist assessment |\n"
        )
    
    # Footer
    st.markdown("---")
    st.caption(
        "RetinaGuard AI v1.0 — Research Prototype | "
        "NOT a medical device | NOT clinically validated | "
        "Consult a qualified ophthalmologist for all clinical decisions"
    )


if __name__ == "__main__":
    main()
