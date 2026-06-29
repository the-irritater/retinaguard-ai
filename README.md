# 👁️ RetinaGuard AI

**An Explainable Deep-Learning System for Diabetic Retinopathy Screening and Patient Education**

> ⚠️ **RESEARCH PROTOTYPE — NOT A DIAGNOSTIC TOOL.** This system is NOT a medical device, NOT clinically validated, and NOT approved by any regulatory authority. It must NOT be used for diagnosis or treatment decisions.

---

## 📋 Project Overview

RetinaGuard AI is an academic research prototype that analyses retinal fundus photographs to:

1. **Classify** diabetic retinopathy as non-referable (DR grade < 2) or potentially referable (DR grade ≥ 2).
2. **Report** model confidence and calibrated probability estimates.
3. **Generate** Grad-CAM visual explanations showing which image regions influenced the prediction.
4. **Provide** fixed patient-education information about diabetic retinopathy.
5. **Recommend** professional ophthalmologist assessment for all findings.
6. **State clearly** that the system is not a diagnostic tool.

## 🏗️ Architecture

```
retinaguard-ai/
├── app/                          # Streamlit demonstration application
│   └── streamlit_app.py
├── configs/                      # YAML configuration files
│   └── base_config.yaml
├── data/                         # Data directory (not tracked in git)
│   ├── raw/                      # Raw IDRiD images (user-provided)
│   └── metadata/                 # Generated audit and split CSVs
├── models/                       # Saved model checkpoints
│   └── checkpoints/
├── notebooks/                    # Jupyter notebooks for analysis
├── reports/                      # Generated reports and figures
│   ├── figures/                  # ROC, PR, calibration, Grad-CAM plots
│   ├── tables/                   # Bootstrap CIs, comparison tables
│   ├── model_card.md             # Model documentation
│   ├── study_protocol.md         # Research protocol
│   └── error_analysis_report.md  # Systematic error analysis
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
│   │   ├── baselines.py          # Majority class + LogReg baselines
│   │   ├── losses.py             # Weighted CE + Focal loss
│   │   └── train.py              # Training engine with CV
│   ├── evaluation/               # Evaluation and analysis
│   │   ├── metrics.py            # Binary and multiclass metrics
│   │   ├── bootstrap.py          # Bootstrap confidence intervals
│   │   ├── calibration.py        # ECE, temperature scaling
│   │   ├── error_analysis.py     # Systematic error categorisation
│   │   └── run_evaluation.py     # Evaluation orchestrator
│   └── explainability/           # Visual explanations
│       └── gradcam.py            # Grad-CAM generation
├── tests/                        # Unit tests
│   ├── test_preprocessing.py
│   ├── test_splitting.py
│   ├── test_metrics.py
│   ├── test_baselines.py
│   └── test_dataset.py
├── requirements.txt
├── LICENSE                       # MIT + medical disclaimer
└── README.md
```

## 🚀 Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/your-username/retinaguard-ai.git
cd retinaguard-ai
pip install -r requirements.txt
```

### 2. Download IDRiD Dataset

Download the [IDRiD dataset](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid) and place it inside `data/raw/`:

```
data/raw/
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
    --checkpoint models/checkpoints/final/best_model_fold0_binary.pt
```

### 7. Launch Streamlit Demo

```bash
streamlit run app/streamlit_app.py
```

## 🧪 Running Tests

```bash
PYTHONPATH=. pytest tests/ -v
```

All 20 tests should pass:

```
tests/test_baselines.py           ✓ 2 passed
tests/test_dataset.py             ✓ 4 passed
tests/test_metrics.py             ✓ 7 passed
tests/test_preprocessing.py       ✓ 4 passed
tests/test_splitting.py           ✓ 3 passed
======================== 20 passed ========================
```

## 📊 Methodology

### Model Selection Protocol

```
Train EfficientNet-B0 (5-fold stratified CV)
                    ↓
Compare using VALIDATION data only
                    ↓
Select architecture, loss, preprocessing, threshold
                    ↓
Lock the complete pipeline
                    ↓
Evaluate ONCE on official test set
```

### Key Design Decisions

| Decision | Rationale |
|---|---|
| **DR grade ≥ 2 binary target** | Clinically meaningful threshold for moderate NPDR |
| **No horizontal flip** | Preserves retinal laterality (OD vs OS) |
| **Validation-only threshold** | Prevents test set contamination |
| **Stratified bootstrap CIs** | Accounts for small test set size |
| **Temperature scaling** | Post-hoc calibration fitted on validation set |
| **Grad-CAM disclaimer** | Heatmaps are NOT confirmed lesion boundaries |

## 📄 Documentation

| Document | Description |
|---|---|
| [Study Protocol](reports/study_protocol.md) | Research scope, ethics, and methodology |
| [Model Card](reports/model_card.md) | Model documentation and limitations |
| [Error Analysis](reports/error_analysis_report.md) | Systematic error categorisation |
| [Bootstrap CIs](reports/tables/test_bootstrap_ci.md) | Test metrics with confidence intervals |

## ⚖️ License

MIT License with Medical Disclaimer — see [LICENSE](LICENSE).

## 📚 Citation

If you use this project in your research, please cite:

```bibtex
@misc{retinaguard2026,
  title={RetinaGuard AI: An Explainable Deep-Learning System for Diabetic Retinopathy Screening},
  year={2026},
  note={Research prototype — not clinically validated}
}
```

**Dataset:**
```bibtex
@article{porwal2020idrid,
  title={IDRiD: Diabetic Retinopathy -- Segmentation and Grading Challenge},
  author={Porwal, Prasanna and others},
  journal={Medical Image Analysis},
  year={2020}
}
```

---

> **This is a research prototype. It is NOT a medical device and must NOT be used for clinical diagnosis or treatment decisions.**
