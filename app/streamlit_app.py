"""
RetinaGuard AI - Streamlit Demonstration Application
-
Purpose: Interactive web interface for demonstrating the DR classification
system. Allows uploading retinal images, viewing predictions, confidence,
Grad-CAM overlays, and patient education information.

 THIS IS A RESEARCH DEMONSTRATION - NOT A DIAGNOSTIC TOOL 

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
import json
import yaml

# Configure page
st.set_page_config(
    page_title="RetinaGuard AI - DR Screening Research Demo",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

logger = logging.getLogger("retinaguard.app")

# -
# Constants
# -
DISCLAIMER_BANNER = """
> **RESEARCH PROTOTYPE – NOT FOR CLINICAL USE**
>
> This software has not been clinically validated, reviewed, cleared or approved by any regulatory authority. It must not be used for diagnosis, treatment decisions, patient management or emergency assessment. All outputs require interpretation by a qualified eye-care professional.
"""

GRADCAM_DISCLAIMER = (
    "The highlighted regions influenced the model's prediction. "
    "They are not confirmed lesion boundaries."
)

PATIENT_EDUCATION = """
### About Diabetic Retinopathy

Diabetic retinopathy (DR) is a complication of diabetes that affects the blood
vessels in the retina - the light-sensitive tissue at the back of the eye.

**Key Facts:**
- Diabetic retinopathy is a major cause of vision impairment and blindness among working-age adults. Early detection, follow-up and timely treatment can reduce the risk of vision loss.
- Early stages often have no symptoms. The National Eye Institute states that early diabetic retinopathy may have no symptoms, diagnosis is based on professional eye examination and treatment can include injections, laser procedures or surgery.
- Effective treatments exist, especially when DR is detected early.
- Good blood sugar, blood pressure, and cholesterol control can slow progression.

**What should you do?**
- See an ophthalmologist for regular dilated eye exams.
- Work with your diabetes-care team to establish appropriate individual goals for blood glucose, blood pressure and cholesterol management.
- Do not rely on any AI system for medical decisions.

The following severity grades are provided for general education. This application does not determine or confirm an individual clinical DR grade. The present model is binary and does not assess all causes of referral, including diabetic macular oedema and other retinal conditions.

**DR Severity Grades:**
| Grade | Description | Typical Features |
|---|---|---|
| 0 | No DR | Normal retina |
| 1 | Mild NPDR | A few microaneurysms |
| 2 | Moderate NPDR | More microaneurysms, dot/blot haemorrhages, hard exudates |
| 3 | Severe NPDR | Extensive haemorrhages, venous beading, IRMA |
| 4 | Proliferative DR | Neovascularisation, vitreous haemorrhage |

> **Important:** This educational information is general in nature. Consult your
> healthcare provider for advice specific to your situation.
"""


# -
# Model Loading (cached)
# -
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
        st.sidebar.success(f" Model loaded from checkpoint")
    else:
        st.sidebar.warning(" No checkpoint found - using random weights for demo")
    
    model = model.to(device)
    model.eval()
    return model, config, device


def load_calibration_params(project_root: Path) -> tuple[float, float, float]:
    """Load locked threshold, lower_bound, and upper_bound from calibration_summary.json.

    Fallback to threshold=0.50, lower=0.45, upper=0.55 if not found.
    """
    cal_path = project_root / "ppc" / "reports" / "tables" / "calibration_summary.json"
    if not cal_path.exists():
        cal_path = project_root / "reports" / "tables" / "calibration_summary.json"
        
    if cal_path.exists():
        try:
            with open(cal_path) as f:
                data = json.load(f)
            threshold = data.get("threshold", 0.50)
            lower_bound = data.get("lower_bound", threshold - 0.05)
            upper_bound = data.get("upper_bound", threshold + 0.05)
            return threshold, lower_bound, upper_bound
        except Exception as e:
            logger.warning(f"Failed to load calibration summary: {e}")
    return 0.50, 0.45, 0.55
def assess_image_suitability(image_rgb: np.ndarray) -> tuple[str, list[str]]:
    """Assess image quality and compatibility with IDRiD dataset.

    Checks:
    - Resolution and shape
    - Colour channel ratios (detects non-retinal invalid inputs)
    - Centring and field completeness (via mask center of mass)
    - Average brightness
    - Contrast (std of grayscale)
    - Blur/sharpness (Laplacian variance)
    """
    reasons = []
    
    # 1. Check shape and resolution
    h, w, c = image_rgb.shape
    if h < 200 or w < 200:
        reasons.append("Resolution is too low for reliable classification.")
        return "Rejected", reasons
        
    # 2. Colour channel ratio checking (retinal photographs are predominantly red/green)
    r_mean = float(np.mean(image_rgb[:, :, 0]))
    g_mean = float(np.mean(image_rgb[:, :, 1]))
    b_mean = float(np.mean(image_rgb[:, :, 2]))
    
    r_b_ratio = r_mean / b_mean if b_mean > 0 else 0
    g_b_ratio = g_mean / b_mean if b_mean > 0 else 0
    
    if r_b_ratio < 1.2 or g_b_ratio < 0.8:
        reasons.append("Image does not match the expected colour signature of a retinal fundus photograph.")
        return "Rejected", reasons

    # Convert to grayscale for metric calculations
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    
    # 3. Centring and field completeness check
    _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    M = cv2.moments(thresh)
    if M["m00"] > 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        
        # Relative distance of center of mass from geometric center of image
        dist_x = abs(cX - w / 2) / w
        dist_y = abs(cY - h / 2) / h
        total_dist = np.sqrt(dist_x**2 + dist_y**2)
        
        if total_dist > 0.020:
            reasons.append("Retinal field appears off-centre or incomplete.")
    else:
        reasons.append("Could not detect any circular retinal structure.")
        return "Rejected", reasons

    # 4. Brightness check
    mean_brightness = float(np.mean(gray))
    if mean_brightness < 40:
        reasons.append("Image is too dark (average brightness below threshold).")
    elif mean_brightness > 215:
        reasons.append("Image is overexposed (average brightness above threshold).")
        
    # 5. Contrast check
    contrast = float(np.std(gray))
    if contrast < 20:
        reasons.append("Image has extremely low contrast (flat histogram).")
        
    # 6. Sharpness check using Laplacian variance
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if sharpness < 10.0:
        reasons.append("Image is excessively blurry or out of focus.")
        
    if len(reasons) > 0:
        return "Uncertain", reasons
    else:
        return "Acceptable", []


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
    overlay, heatmap, is_meaningful = explainer.generate(image_tensor.to(device), target_category=1)
    return overlay, heatmap, is_meaningful


# -
# UI Layout
# -
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
        
        # Load validation parameters
        locked_threshold, lower_bound, upper_bound = load_calibration_params(project_root)
        
        st.divider()
        
        # Threshold
        threshold = st.slider(
            "Classification Threshold",
            min_value=0.0,
            max_value=1.0,
            value=locked_threshold,
            step=0.01,
            help="Probability threshold for DR grade >= 2 classification. "
                 "Selected on validation data using Youden's J statistic.",
        )
        
        st.caption(f"Locked threshold: **{locked_threshold:.4f}**")
        st.caption(f"Locked uncertainty zone: **[{lower_bound:.4f}, {upper_bound:.4f}]**")
        
        show_gradcam = st.checkbox("Show Grad-CAM Overlay", value=True)
        show_education = st.checkbox("Show Patient Education", value=False)
        
        st.divider()
        
        # Local file path text input for testing/automation support
        local_image_path = st.text_input(
            "Local Image Path (for automation/testing)",
            value="",
            help="Direct path to an image file on the local filesystem. Used in automated tests.",
        )
        
        st.divider()
        st.markdown(" [View Model Card](ppc/reports/model_card.md)")
        st.markdown(" [View Study Protocol](ppc/reports/study_protocol.md)")
    
    # --- Main Content ---
    st.markdown(DISCLAIMER_BANNER)
    st.title(" RetinaGuard AI")
    st.subheader("Explainable Deep Learning for Diabetic Retinopathy Research")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload a retinal fundus photograph",
        type=["jpg", "jpeg", "png", "tif", "tiff"],
        help="Upload a colour retinal fundus image for analysis.",
    )
    
    image_bgr = None
    if local_image_path:
        local_p = Path(local_image_path)
        if local_p.exists() and local_p.is_file():
            image_bgr = cv2.imread(str(local_p))
            if image_bgr is None:
                st.sidebar.error("Failed to read image from local path.")
        else:
            st.sidebar.warning("Local image path does not exist or is not a file.")
    elif uploaded_file is not None:
        file_bytes = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image_bgr is None:
            st.error(" Could not read the uploaded image. Please try a different file.")
            return

    if image_bgr is not None:
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
                st.error(f" Failed to load model: {e}")
                return
        
        # Preprocess
        with st.spinner("Preprocessing image..."):
            try:
                image_tensor = preprocess_for_inference(image_rgb, config)
            except Exception as e:
                st.error(f" Preprocessing failed: {e}")
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
        
        # Assess image suitability
        suitability, suitability_reasons = assess_image_suitability(image_rgb)
        
        # Calculate uncertainty boundaries around selected threshold
        # We shift them proportionally to the user's selected threshold if they adjusted it,
        # but preserve the size of the uncertainty zone.
        zone_half_width = (upper_bound - lower_bound) / 2.0
        current_lower = max(0.01, threshold - zone_half_width)
        current_upper = min(0.99, threshold + zone_half_width)
        
        # Classification using uncertainty zone
        if current_lower <= prob_positive <= current_upper:
            classification = "Indeterminate"
            uncertainty_level = "High"
            interpretation = (
                "The model score is extremely close to the decision threshold. The system "
                "cannot provide a reliable positive or negative classification for this "
                "image."
            )
            if prob_positive >= threshold:
                model_output_desc = "Score slightly above the study threshold for DR grade >= 2"
            else:
                model_output_desc = "Score slightly below the study threshold for DR grade >= 2"
        elif prob_positive < current_lower:
            classification = "Below study threshold"
            uncertainty_level = "Low"
            interpretation = (
                "Model score is below the locked study threshold. This suggests a lower "
                "likelihood of potentially referable diabetic retinopathy (DR grade >= 2)."
            )
            model_output_desc = "Score below the study threshold for DR grade >= 2"
        else:
            classification = "Above study threshold"
            uncertainty_level = "Low"
            interpretation = (
                "Model score is above the locked study threshold. This suggests a higher "
                "likelihood of potentially referable diabetic retinopathy (DR grade >= 2)."
            )
            model_output_desc = "Score above the study threshold for DR grade >= 2"

        decision_margin = prob_positive - threshold
        margin_sign = "+" if decision_margin >= 0 else ""
        
        with col2:
            st.markdown("### Analysis Result")
            
            # Suitability Display
            st.markdown("**Image suitability:**")
            if suitability == "Acceptable":
                st.success("Accepted for experimental analysis")
            elif suitability == "Uncertain":
                st.warning("Uncertain suitability")
                for reason in suitability_reasons:
                    st.caption(f"- {reason}")
            else:
                st.error("Rejected suitability")
                for reason in suitability_reasons:
                    st.caption(f"- {reason}")
            st.caption("*Quality and dataset compatibility remain uncertain.*")
            st.caption(
                "Image quality and compatibility checking are experimental. A "
                "successfully uploaded image is not necessarily suitable for reliable analysis."
            )
            
            st.divider()
            
            # Screening classification
            st.markdown("**Screening classification:**")
            if classification == "Indeterminate":
                st.info(f"**{classification}**")
            elif classification == "Above study threshold":
                st.error(f"**{classification}**")
            else:
                st.success(f"**{classification}**")
                
            # Decision Metrics
            st.metric("Calibrated model score", f"{prob_positive:.4f}")
            st.caption(
                "The model score estimates the study-defined outcome under the development "
                "dataset. It is not an individual clinical probability or diagnosis."
            )
            
            st.metric("Locked study threshold", f"{threshold:.4f}")
            st.metric("Decision margin", f"{margin_sign}{decision_margin:.4f}")
            st.metric("Uncertainty", uncertainty_level)
            
            st.divider()
            
            # Interpretation & Actions
            st.markdown("**Model output:**")
            st.caption(model_output_desc)
            
            st.markdown("**Interpretation:**")
            st.write(interpretation)
            
            st.markdown("**Recommended action:**")
            st.info(
                "Do not interpret this output as a diagnosis. A qualified ophthalmologist "
                "should assess the retinal image and determine whether diabetic "
                "retinopathy or another eye condition is present."
            )
            
            st.markdown("**Model limitation:**")
            st.caption(
                "This image may differ from the photographs used to develop the model. "
                "Differences in camera type, illumination, colour, resolution and image "
                "processing can affect model output."
            )
        
        # Grad-CAM
        if show_gradcam:
            st.divider()
            st.markdown("### Grad-CAM Visual Explanation")
            st.warning(f"{GRADCAM_DISCLAIMER}")
            
            with st.spinner("Generating Grad-CAM overlay..."):
                try:
                    overlay, heatmap, is_meaningful = generate_gradcam(model, image_tensor, device)
                    
                    if not is_meaningful:
                        st.info(
                            "The Grad-CAM heatmap has no meaningful spatial variation. "
                            "This typically means the model has not been trained. "
                            "Load a trained checkpoint to see informative heatmaps."
                        )
                    
                    gcol1, gcol2, gcol3 = st.columns(3)
                    
                    # Original (resized)
                    img_np = image_tensor.numpy().transpose(1, 2, 0)
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    img_display = std * img_np + mean
                    img_display = np.clip(img_display, 0.0, 1.0)
                    
                    # Apply JET colormap to grayscale heatmap for proper
                    # visualisation.  The raw heatmap is a float32 array in
                    # [0, 1]; without a colormap Streamlit renders it as a
                    # near-black greyscale image.
                    import matplotlib.cm as cm
                    heatmap_colored = cm.jet(heatmap)[:, :, :3]  # drop alpha
                    
                    with gcol1:
                        st.image(img_display, caption="Preprocessed Image", use_container_width=True)
                    with gcol2:
                        st.image(heatmap_colored, caption="Grad-CAM Heatmap", use_container_width=True, clamp=True)
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
            "###  System Capabilities\n"
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
        "RetinaGuard AI v1.0 - Research Prototype | "
        "NOT a medical device | NOT clinically validated | "
        "Consult a qualified ophthalmologist for all clinical decisions"
    )


if __name__ == "__main__":
    main()
