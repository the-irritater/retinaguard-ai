# RetinaGuard AI

## An Explainable Deep-Learning Research Prototype for Diabetic Retinopathy Screening and Patient Education

> **RESEARCH PROTOTYPE – NOT FOR CLINICAL USE**
>
> RetinaGuard AI has not been clinically validated, reviewed, cleared or approved by any regulatory authority. It must not be used for diagnosis, treatment decisions, patient management or emergency assessment. All outputs require interpretation by a qualified eye-care professional.

## Project Overview

RetinaGuard AI is an academic medical-imaging research prototype that analyses retinal fundus photographs to:

* Detect moderate-or-worse diabetic retinopathy, operationally defined in this study as DR grade ≥2.
* Classify images below this threshold as DR grade <2.
* Produce a calibrated model score for the study-defined outcome.
* Generate Grad-CAM visualisations showing image regions that influenced the model prediction.
* Provide fixed educational information about diabetic retinopathy.
* Direct users toward appropriate professional eye care.
* Document model limitations, uncertainty and known failure modes.

Within this project, DR grade ≥2 is described as "potentially referable diabetic retinopathy." This is a study-specific research definition. It is not a complete clinical referral rule because referral decisions may also depend on diabetic macular oedema, image quality, symptoms, medical history and professional examination.

Grad-CAM heatmaps are visual explanations of model influence. They are not confirmed lesions, segmentation masks or clinical evidence.

## Intended Use

RetinaGuard AI is intended for:

* Academic study of retinal-image classification.
* Machine-learning and biostatistical education.
* Reproducible evaluation of an experimental screening model.
* Demonstration of calibration, uncertainty analysis and explainability.

## Prohibited Use

RetinaGuard AI must not be used for:

* Clinical diagnosis.
* Treatment or medication decisions.
* Replacing an ophthalmologist or retinal specialist.
* Emergency triage.
* Screening real patients outside an approved research protocol.
* Making decisions about insurance, employment or access to healthcare.

## Repository Structure

```
retinaguard-ai/
├── app/                          # Streamlit demonstration application
│   └── streamlit_app.py
├── configs/                      # YAML configuration files
│   └── base_config.yaml
├── src/                          # Source code
│   ├── data/                     # Data processing pipeline
│   │   ├── audit_dataset.py      # Dataset audit and validation
│   │   ├── splitting.py          # Stratified CV fold creation
│   │   ├── preprocessing.py      # Image preprocessing and augmentation
│   │   ├── dataset.py            # PyTorch Dataset class
│   │   ├── quality_module.py     # Synthetic degradation detector (experimental)
│   │   └── external_adapter.py   # External dataset adapter
│   ├── models/                   # Model definitions and training
│   │   ├── architectures.py      # EfficientNet-B0 transfer learning
│   │   ├── baselines.py          # Majority class and logistic regression baselines
│   │   ├── losses.py             # Weighted cross-entropy and focal loss
│   │   └── train.py              # Training engine with cross-validation
│   ├── evaluation/               # Evaluation and analysis
│   │   ├── metrics.py            # Binary and multiclass metrics
│   │   ├── bootstrap.py          # Bootstrap confidence intervals
│   │   ├── calibration.py        # ECE, temperature scaling
│   │   ├── error_analysis.py     # Systematic error categorisation
│   │   └── run_evaluation.py     # Evaluation orchestrator
│   └── explainability/           # Visual explanations
│       └── gradcam.py            # Grad-CAM generation
├── ppc/                          # Supporting project files
│   ├── data/                     # Dataset directory (raw images not tracked)
│   ├── models/                   # Saved model checkpoints
│   ├── reports/                  # Generated reports and figures
│   └── tests/                    # Unit test suite
├── requirements.txt
├── runtime.txt                   # Python 3.11 for Streamlit Cloud
├── LICENSE                       # MIT with medical disclaimer
└── README.md
```

## Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/the-irritater/retinaguard-ai.git
cd retinaguard-ai
pip install -r requirements.txt
```

### 2. Download IDRiD Dataset

Download the [IDRiD dataset](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid) and place it inside `ppc/data/raw/`:

```
ppc/data/raw/
├── 1. Original Images/
│   ├── a. Training Set/
│   │   ├── IDRiD_001.jpg
│   │   ├── IDRiD_002.jpg
│   │   └── ...
│   └── b. Testing Set/
│       ├── IDRiD_001.jpg
│       └── ...
└── 2. Groundtruths/
    ├── a. IDRiD_Disease Grading_Training Labels.csv
    └── b. IDRiD_Disease Grading_Testing Labels.csv
```

### 3. Run Data Audit

```bash
PYTHONPATH=. python -m src.data.audit_dataset --config configs/base_config.yaml
```

### 4. Create CV Splits

```bash
PYTHONPATH=. python -m src.data.splitting --config configs/base_config.yaml
```

### 5. Train Model

```bash
PYTHONPATH=. python -m src.models.train --config configs/base_config.yaml
```

### 6. Evaluate

```bash
PYTHONPATH=. python -m src.evaluation.run_evaluation \
    --config configs/base_config.yaml \
    --checkpoint ppc/models/checkpoints/final/best_model_fold0_binary.pt
```

### 7. Launch Streamlit Demo

```bash
streamlit run app/streamlit_app.py
```

## Running Tests

```bash
PYTHONPATH=. pytest ppc/tests/ -v
```

All 20 tests should pass:

```
ppc/tests/test_baselines.py            2 passed
ppc/tests/test_dataset.py              4 passed
ppc/tests/test_metrics.py              7 passed
ppc/tests/test_preprocessing.py        4 passed
ppc/tests/test_splitting.py            3 passed
                                      20 passed
```

## Methodology

### Model Selection Protocol

```
Train EfficientNet-B0 (5-fold stratified CV)
                    |
Compare using VALIDATION data only
                    |
Select architecture, loss, preprocessing, threshold
                    |
Lock the complete pipeline
                    |
Evaluate ONCE on official test set
```

### Key Design Decisions

| Decision | Rationale |
|---|---|
| DR grade ≥2 binary target | Study-specific threshold for moderate NPDR |
| No horizontal flip | Preserves retinal laterality (OD vs OS) |
| Validation-only threshold | Prevents test set contamination |
| Stratified bootstrap CIs | Accounts for small test set size |
| Temperature scaling | Post-hoc calibration fitted on validation set |
| Grad-CAM disclaimer | Heatmaps are not confirmed lesion boundaries |

## Documentation

| Document | Description |
|---|---|
| [Study Protocol](ppc/reports/study_protocol.md) | Research scope, ethics and methodology |
| [Model Card](ppc/reports/model_card.md) | Model documentation and limitations |
| [Error Analysis](ppc/reports/error_analysis_report.md) | Systematic error categorisation |
| [Bootstrap CIs](ppc/reports/tables/test_bootstrap_ci.md) | Test metrics with confidence intervals |

## License

MIT License with Medical Disclaimer. See [LICENSE](LICENSE).

## Citation

If you use this project in your research, please cite:

```bibtex
@misc{retinaguard2026,
  title={RetinaGuard AI: An Explainable Deep-Learning System for
         Diabetic Retinopathy Screening},
  year={2026},
  note={Research prototype. Not clinically validated.}
}
```

Dataset:

```bibtex
@article{porwal2020idrid,
  title={IDRiD: Diabetic Retinopathy -- Segmentation and Grading Challenge},
  author={Porwal, Prasanna and others},
  journal={Medical Image Analysis},
  year={2020}
}
```

> RetinaGuard AI is a research prototype. It has not been clinically validated and must not be used for medical diagnosis or treatment decisions.
