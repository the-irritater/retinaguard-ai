# RetinaGuard AI: Study Protocol

**Version:** 1.0.0  
**Date:** 2026-06-29  
**Status:** Pre-registration draft  

> **Disclaimer:** This document describes an academic research prototype.
> The system described herein is NOT a medical device, has NOT received
> regulatory approval from any authority, and MUST NOT be used for clinical
> diagnosis, treatment decisions, or patient management.

---

## 1. Background

Diabetic retinopathy (DR) is a microvascular complication of diabetes mellitus
and a leading cause of preventable blindness worldwide. The International
Diabetes Federation estimates that over 100 million people globally have some
form of DR. Early detection through systematic screening can reduce the risk
of severe vision loss, but screening programmes require trained graders and
equipment that may not be available in all settings.

Deep learning has shown promise in automated analysis of retinal fundus
photographs. However, most published systems lack transparency regarding
their decision-making process, rigorous calibration analysis, and honest
reporting of limitations. A responsible research prototype should demonstrate
not only classification performance but also explainability, uncertainty
quantification, calibration, and clear communication of what the system
can and cannot do.

This project uses the Indian Diabetic Retinopathy Image Dataset (IDRiD),
a publicly available, expert-annotated dataset of retinal fundus photographs
from an Indian diabetic population.

## 2. Research Problem

Can a transfer-learning-based convolutional neural network, trained on a
small expert-annotated dataset of retinal fundus photographs, reliably
distinguish images showing signs consistent with DR grade ≥ 2 from those
graded 0–1, while providing calibrated probabilities and visual explanations?

## 3. Aim

To develop and evaluate an explainable deep-learning research prototype for
binary detection of DR grade ≥ 2 in retinal fundus photographs, with
transparent reporting of performance, calibration, limitations, and failure
modes.

## 4. Objectives

### 4.1 Primary Objectives

1. Train a binary classifier to detect DR grade ≥ 2 using transfer learning
   on IDRiD retinal fundus photographs.
2. Evaluate classification performance on the official IDRiD test set with
   bootstrap confidence intervals.
3. Assess model calibration and apply post-hoc recalibration if needed.
4. Generate Grad-CAM visual explanations for model predictions.

### 4.2 Secondary Objectives

5. Develop a five-class DR severity grading model as an exploratory
   secondary analysis.
6. Perform systematic error analysis to identify failure modes.
7. Compare model performance across image quality characteristics.
8. Create a Streamlit demonstration interface with appropriate disclaimers.

## 5. Research Questions

1. What sensitivity and specificity does the binary classifier achieve for
   detecting DR grade ≥ 2 on the IDRiD test set?
2. Is the model's predicted probability well-calibrated?
3. Which image regions drive the model's predictions?
4. How does the model perform across different DR severity grades?
5. What image characteristics are associated with classification errors?
6. Does image resolution affect screening performance?

## 6. Intended Users

This research prototype is intended for:

- Machine-learning researchers studying medical image classification.
- Ophthalmology researchers evaluating AI-assisted screening concepts.
- Students and educators in biomedical AI.

This prototype is **NOT** intended for:

- Clinical practitioners making diagnostic or treatment decisions.
- Patients seeking medical advice.
- Screening programme operators.
- Any deployment without appropriate regulatory clearance.

## 7. Intended Use

- Academic research and education only.
- Analysis of retinal fundus photographs from datasets with expert labels.
- Demonstration of responsible AI development practices.

## 8. Prohibited Use

- Clinical diagnosis of any eye disease.
- Treatment or medication recommendations.
- Patient triage without qualified clinical oversight.
- Deployment in clinical settings without regulatory approval.
- Use as a substitute for professional ophthalmic examination.
- Permanent storage or sharing of patient images.
- Any use that implies the system is a validated diagnostic tool.

## 9. Input Definition

- **Format:** Colour retinal fundus photographs (JPG, JPEG, PNG).
- **Source:** IDRiD dataset images captured with a Kowa VX-10α fundus camera
  at a macula-centred 50° field of view.
- **Resolution:** Variable; original images approximately 4288 × 2848 pixels.
- **Quality:** Images in IDRiD were selected by experts as adequate quality.
  The dataset does not contain representative ungradable images.

## 10. Output Definition

For each input image, the system produces:

1. **Binary prediction:** DR grade ≥ 2 detected or not detected.
2. **Predicted probability:** Calibrated probability of DR grade ≥ 2.
3. **Confidence level:** Qualitative confidence indicator.
4. **Grad-CAM overlay:** Visual explanation of regions influencing the
   prediction (not lesion boundaries).
5. **Five-class grade (secondary):** Exploratory DR severity estimate (0–4).
6. **Disclaimer:** Statement that results require professional confirmation.

## 11. Primary Endpoint

**Sensitivity for detecting DR grade ≥ 2** on the official IDRiD test set,
evaluated at the threshold selected using the development set only.

## 12. Secondary Endpoints

- Specificity at the selected threshold.
- Area under the receiver operating characteristic curve (ROC-AUC).
- Area under the precision–recall curve (PR-AUC).
- F1-score.
- Balanced accuracy.
- Brier score.
- Expected calibration error (ECE).
- Quadratic weighted kappa (for five-class grading).
- Mean absolute grade error (for five-class grading).

## 13. Dataset Description

### 13.1 Source

**Indian Diabetic Retinopathy Image Dataset (IDRiD)**  
Published by: Prasanna Porwal, Samiksha Pachade, et al.  
Available from: IEEE DataPort  
DOI: 10.21227/H25W98

### 13.2 Composition

| Partition | Images | Purpose |
|-----------|--------|---------|
| Training  | 413    | Model development |
| Testing   | 103    | Final evaluation (one-time) |
| **Total** | **516** | |

### 13.3 Annotations

- **DR grading:** Expert-assigned severity grades 0–4 per the International
  Clinical Diabetic Retinopathy Scale (No DR, Mild NPDR, Moderate NPDR,
  Severe NPDR, PDR).
- **DME grading:** Expert-assigned risk grades 0–2 for diabetic macular
  oedema.
- **Lesion masks:** Pixel-level annotations for 81 images covering
  microaneurysms, haemorrhages, hard exudates, and soft exudates.

### 13.4 Acquisition

- **Camera:** Kowa VX-10α
- **Field of view:** 50°, macula-centred
- **Population:** Indian diabetic patients
- **Setting:** Eye clinic

### 13.5 Limitations

- Single camera and single clinical site.
- Expert-selected adequate-quality images only.
- Small sample size limits statistical power and generalisability.
- Indian diabetic population may not represent other demographics.
- No patient-level identifiers for verifying data independence.

## 14. Inclusion Criteria

- Images present in the official IDRiD dataset.
- Images with valid DR grading labels (0–4).
- Images readable by standard image-processing libraries.

## 15. Exclusion Criteria

- Images without corresponding DR grade labels.
- Corrupt or unreadable image files.
- Images with ambiguous or conflicting labels (none expected in IDRiD).

## 16. Label Definitions

### 16.1 Original DR Grades

| Grade | Description | ICDR Equivalent |
|-------|-------------|-----------------|
| 0 | No apparent retinopathy | No DR |
| 1 | Mild non-proliferative DR | Mild NPDR |
| 2 | Moderate non-proliferative DR | Moderate NPDR |
| 3 | Severe non-proliferative DR | Severe NPDR |
| 4 | Proliferative DR | PDR |

### 16.2 Binary Target (Study-Specific)

| Class | DR Grades | Label |
|-------|-----------|-------|
| 0 | 0, 1 | DR grade < 2 |
| 1 | 2, 3, 4 | DR grade ≥ 2 |

> **Important:** This binary target is an operational research definition
> created for this study. It is NOT a universal clinical referral standard.
> Referral decisions in clinical practice depend on multiple factors including
> DME status, image quality, patient symptoms, and other clinical findings.
> IDRiD provides separate DME labels; DR grade alone does not represent
> every possible referral pathway.

### 16.3 DME Grades (Descriptive Analysis Only)

| Grade | Description |
|-------|-------------|
| 0 | No apparent DME |
| 1 | DME present but not clinically significant |
| 2 | Clinically significant DME |

## 17. Data-Splitting Strategy

### 17.1 Official Split

The official IDRiD training set (413 images) and test set (103 images) are
preserved without modification. The test set is used exactly once for final
evaluation after all development decisions are locked.

### 17.2 Development Strategy

**Stratified 5-fold cross-validation** within the official training set for:

- Architecture comparison.
- Loss-function comparison.
- Resolution comparison (224, 384, 512).
- Augmentation ablation.
- Hyperparameter selection.
- Threshold selection.

After selection, the chosen model is **retrained on the full official
training set** and evaluated once on the test set.

### 17.3 Patient-Level Considerations

IDRiD documentation describes deidentified, renamed images and does not
provide a clearly documented patient-grouping variable in the standard
grading CSV files. Therefore:

- Patient-level data splitting cannot be performed.
- Patient-level leakage cannot be independently verified.
- This limitation is acknowledged and reported.
- Patient identity is not inferred from filenames, retinal appearance,
  or left/right eye characteristics.

## 18. Preprocessing Plan

### 18.1 Pipeline

1. Read image safely; verify readability.
2. Convert to RGB colour space.
3. Detect and crop excessive black borders while preserving the full
   circular retinal field.
4. Resize to the selected image resolution (determined by CV experiment).
5. Normalise pixel values using ImageNet statistics.

### 18.2 Augmentation (Training Only)

Medically conservative augmentations applied during training only:

- Rotation: ±15°.
- Translation: ±5%.
- Scaling: 0.9–1.1×.
- Brightness adjustment: ±0.1.
- Contrast adjustment: ±0.1.

**Excluded from baseline:** Horizontal flipping. Horizontal flipping changes
retinal laterality and swaps optic-disc and macular orientation. It may be
tested as an ablation experiment only.

**Never applied to validation or test images.**

## 19. Model-Development Plan

### 19.1 Baselines

1. **Majority-class classifier:** Predicts the most frequent class for all
   images.
2. **Frozen ResNet18:** Pretrained ImageNet features with a trained
   classification head.

### 19.2 Primary Model

**EfficientNet-B0** with transfer learning from ImageNet weights.

Two-stage training:
- Stage 1: Freeze backbone, train classification head.
- Stage 2: Unfreeze upper layers, fine-tune with reduced learning rate.

### 19.3 Model Comparison (Version 2)

MobileNetV3 for deployment-efficiency comparison, evaluated on cross-
validation folds. ResNet50 only if earlier models leave a clear research
question unanswered.

### 19.4 Selection Procedure

```
Train with 5-fold CV
        ↓
Compare validation metrics across folds
        ↓
Select architecture, loss, resolution, threshold
        ↓
Lock pipeline completely
        ↓
Refit on full official training set
        ↓
Evaluate ONCE on official test set
```

**The test set is never used for model selection.**

## 20. Evaluation Plan

### 20.1 Development Phase

All evaluation during development uses cross-validation folds or the
held-out validation fold within the training set.

Metrics computed during development:
- Sensitivity, specificity, precision, NPV.
- F1-score, balanced accuracy.
- ROC-AUC, PR-AUC.
- Brier score.
- Calibration plot and ECE.

### 20.2 Final Test Evaluation

After locking the pipeline, evaluate once on the official test set (103
images).

Report with **2,000 stratified bootstrap resamples** for 95% confidence
intervals on:
- Sensitivity, specificity.
- ROC-AUC, PR-AUC.
- F1-score, balanced accuracy.
- Brier score.

## 21. Statistical-Analysis Plan

### 21.1 Primary Analysis

- Point estimates with 95% bootstrap confidence intervals.
- Effect sizes and confidence intervals as main evidence.

### 21.2 Comparisons (When Justified)

- McNemar's test: only when comparing two locked models on the same cases.
- DeLong's test: only when a formally justified AUC comparison is required.

### 21.3 Avoided

- Multiple pairwise tests across architectures, losses, and thresholds.
- Significance testing without correction for multiple comparisons.
- Post-hoc comparisons not specified in this protocol.

## 22. Explainability Plan

### 22.1 Method

Grad-CAM applied to the final convolutional layer of the selected model.

### 22.2 Outputs

For selected test images (covering TP, TN, FP, FN, and high-confidence
errors):

- Original image.
- Grad-CAM heatmap.
- Overlay visualisation.
- True class, predicted class, and probability.

### 22.3 Interpretation Statement

Every Grad-CAM output includes:

> "The highlighted regions influenced the model's prediction. They are not
> confirmed lesion boundaries."

Grad-CAM is **not** called lesion segmentation or lesion detection.

## 23. Ethics

### 23.1 Dataset Ethics

IDRiD is a publicly available research dataset published through IEEE
DataPort. The original study received appropriate ethical clearance. Images
are deidentified.

### 23.2 Responsible AI

- No diagnostic claims.
- No treatment recommendations.
- Clear disclaimer on all outputs.
- Transparent limitation reporting.
- Human oversight required for any clinical application.

### 23.3 ICMR Considerations

Consistent with ICMR guidance on medical AI, this project emphasises:

- Ethical review and accountability.
- Privacy and data protection.
- Transparency in methods and limitations.
- Human oversight requirements.

## 24. Privacy

- No personal health information is collected or stored.
- IDRiD images are deidentified.
- The Streamlit application does not permanently store uploaded images.
- Uploaded images are processed in memory and deleted after analysis.

## 25. Limitations

1. **Small dataset:** 516 images total; limited statistical power.
2. **Single site and camera:** Kowa VX-10α from one Indian clinic.
3. **Population bias:** Indian diabetic patients only; may not generalise.
4. **No patient IDs:** Cannot verify patient-level data independence.
5. **Adequate quality only:** IDRiD lacks genuinely ungradable images.
6. **No prospective validation:** Retrospective analysis only.
7. **No clinical validation:** Not tested in a screening programme.
8. **Binary simplification:** DR grade ≥ 2 is a study-specific target,
   not a clinical referral standard.
9. **Five-class instability:** Small class sizes for severe DR grades.
10. **Calibration generalisability:** Temperature scaling fitted on
    development data may not transfer to new populations.

## 26. External-Validation Plan

### 26.1 Strategy

1. Create a dataset adapter for a second retinal dataset (e.g., APTOS,
   Messidor-2, EyePACS subset).
2. Evaluate the frozen final model on the external dataset without
   retraining.
3. Report performance delta relative to IDRiD test results.
4. Document dataset and camera differences.

### 26.2 Expected Limitations

External performance is expected to degrade due to:

- Different cameras and acquisition protocols.
- Different patient populations.
- Different grading standards or annotators.
- Different image quality distributions.

---

## References

1. Porwal P, Pachade S, et al. "IDRiD: Diabetic Retinopathy – Segmentation
   and Grading Challenge." Medical Image Analysis. 2020.
2. Collins GS, et al. "TRIPOD+AI statement." BMJ. 2024.
3. Sounderajah V, et al. "STARD-AI: Standards for Reporting of Diagnostic
   Accuracy Studies – Artificial Intelligence." BMJ. 2024.
4. ICMR. "Ethical Guidelines for Application of Artificial Intelligence in
   Biomedical Research and Healthcare." 2023.
