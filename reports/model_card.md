# RetinaGuard AI — Model Card

> **Version:** 1.0.0  
> **Last Updated:** 2026-06-29  
> **Status:** Research Prototype — NOT a diagnostic tool

---

## ⚠️ Intended Use Disclaimer

**This system is an academic research prototype. It is NOT a medical device, NOT clinically validated, and NOT approved by any regulatory body (FDA, CE, TGA, or equivalent). It must NOT be used to make diagnostic or treatment decisions. All outputs require independent verification by a qualified ophthalmologist.**

---

## 1. Model Overview

| Property | Value |
|---|---|
| **Model Name** | RetinaGuard AI Binary DR Classifier |
| **Architecture** | EfficientNet-B0 (transfer learning from ImageNet) |
| **Task** | Binary classification: DR grade ≥ 2 detection |
| **Input** | Retinal fundus photograph (RGB, 384×384 pixels) |
| **Output** | Probability of DR grade ≥ 2, Grad-CAM heatmap |
| **Training Dataset** | IDRiD (Indian Diabetic Retinopathy Image Dataset) |
| **Training Set Size** | 413 images (official training partition) |
| **Test Set Size** | 103 images (official test partition, evaluated once) |
| **Framework** | PyTorch 2.x, timm |

## 2. Intended Use

### Primary Use Case
- Academic research into explainable deep learning for DR screening.
- Portfolio demonstration of responsible AI practices.
- Educational tool for understanding model behaviour and limitations.

### Out-of-Scope Uses
- **Clinical diagnosis or screening** — the model has NOT been validated for clinical use.
- **Treatment planning or patient management** — the model is NOT a decision support tool.
- **Deployment on populations not represented by IDRiD** — the model was trained on a single Indian dataset and may not generalise.
- **Use without ophthalmologist oversight** — all model outputs must be reviewed by a qualified professional.

## 3. Training Methodology

### Data Pipeline
1. **Official split preservation:** IDRiD's 413-image training and 103-image test partitions are strictly maintained.
2. **Cross-validation:** Stratified 5-fold CV within the 413 training images for model selection.
3. **Binary target:** DR grade ≥ 2 (moderate NPDR or worse) mapped to positive class.
4. **Preprocessing:** Black border cropping → resize to 384×384 → ImageNet normalisation.
5. **Augmentation (training only):** Rotation (±15°), shift/scale, brightness/contrast adjustment. **No horizontal flip** (preserves retinal laterality).

### Training Strategy
1. **Stage 1 — Head training:** Backbone frozen, only classifier head trained (5 epochs).
2. **Stage 2 — Fine-tuning:** Last backbone layers unfrozen, end-to-end training with reduced LR (15 epochs).
3. **Early stopping:** Patience of 5 epochs on validation balanced accuracy.
4. **Loss function:** Weighted cross-entropy with class-frequency-based weights.

### Model Selection Protocol
- Architecture, loss, preprocessing, and threshold are selected using **validation data only**.
- The test set is **never** used for model selection.
- The complete pipeline is **locked** before test evaluation.

## 4. Evaluation Metrics

### Binary Classification (DR grade ≥ 2 detection)

Metrics are reported with 95% bootstrap confidence intervals (2,000 stratified resamples).

| Metric | Description |
|---|---|
| Sensitivity (Recall) | Proportion of true positives correctly identified |
| Specificity | Proportion of true negatives correctly identified |
| Precision (PPV) | Proportion of positive predictions that are correct |
| Negative Predictive Value (NPV) | Proportion of negative predictions that are correct |
| F1 Score | Harmonic mean of precision and recall |
| Balanced Accuracy | Average of sensitivity and specificity |
| ROC-AUC | Area under receiver operating characteristic curve |
| PR-AUC | Area under precision-recall curve |
| Brier Score | Calibration metric (lower is better) |
| Matthews Correlation Coefficient | Balanced measure for imbalanced datasets |
| Expected Calibration Error (ECE) | Deviation of predicted probabilities from observed frequencies |

### Threshold Selection
- Method: Youden's J statistic (maximises sensitivity + specificity - 1).
- Alternative: Sensitivity-target method (≥ 90% sensitivity).
- **Selected on validation set only.**

### Calibration
- Post-hoc temperature scaling fitted on validation logits.
- Both uncalibrated and calibrated probabilities are preserved in outputs.

## 5. Limitations and Risks

### Dataset Limitations
- **Small dataset:** Only 516 images total (413 train + 103 test).
- **Single source:** All images from Aravind Eye Hospital, India; single camera/protocol.
- **No patient-level metadata:** Cannot verify patient independence between train/test.
- **Expert-selected quality:** IDRiD contains only adequately gradable images; ungradable image handling is experimental.
- **Class imbalance:** Uneven distribution across DR grades 0–4.

### Model Limitations
- **Not validated for clinical use** in any jurisdiction.
- **Domain shift risk:** Performance may degrade on images from different cameras, populations, or imaging protocols.
- **No laterality awareness:** The model does not distinguish left vs. right eye.
- **Grad-CAM explanations** highlight regions that influenced the prediction; they are **NOT confirmed lesion boundaries**.
- **Binary simplification:** The model reduces a 5-grade ordinal scale to a binary decision, losing clinical nuance.

### Ethical Considerations
- **Automation bias:** Users may over-rely on model predictions.
- **Health equity:** Training data from a single geographic/ethnic population limits generalisability.
- **Transparency:** All model limitations, training data characteristics, and evaluation methodology are documented.

## 6. Responsible AI Practices

| Practice | Implementation |
|---|---|
| **Disclaimers** | Present on every output, Grad-CAM panel, and UI screen |
| **Confidence reporting** | Probability and uncertainty reported alongside every prediction |
| **Calibration** | Temperature scaling ensures probabilities approximate true frequencies |
| **Error analysis** | Systematic breakdown by DR grade, image quality, and error type |
| **Reproducibility** | Fixed seeds, locked pipeline, version-controlled configuration |
| **Explainability** | Grad-CAM visual explanations with mandatory disclaimer |
| **Professional referral** | System recommends ophthalmologist assessment for all findings |

## 7. Maintenance and Updates

- **Model versioning:** Checkpoints are saved with epoch, fold, and timestamp.
- **Configuration tracking:** All hyperparameters stored in `configs/base_config.yaml`.
- **Retraining:** Full retraining requires re-running the training pipeline with the locked configuration.

## 8. Contact and Citation

- **Project:** RetinaGuard AI Research Prototype
- **License:** MIT License with Medical Disclaimer (see LICENSE file)
- **Dataset Citation:** Porwal, P., et al. "IDRiD: Diabetic Retinopathy – Segmentation and Grading Challenge." Medical Image Analysis, 2020.

---

> **Reminder:** This model card documents a research prototype. The system described here has NOT been clinically validated and must NOT be used for medical diagnosis.
