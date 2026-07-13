
# Self-Healing ML Pipeline: Observability Phase 1

This repository contains the starting Phase 1 training pipeline for the Observability Platform. The script `train_pipeline.py` will autonomously download data from a Google Drive URL (or load a local CSV), process it through scikit-learn preprocessing steps, train an ensemble model (`RandomForestClassifier` or `XGBoost`), and save the inference-ready pipeline.

## Prerequisites
- Python 3.9+
- A Virtual Environment (Recommended)

## Setup Guidelines

**1. Create and Activate a Virtual Environment**
Open your terminal (PowerShell or Command Prompt) and run:
```bash
python -m venv venv
.\venv\Scripts\activate
```

**2. Install Dependencies**
Install the necessary ML packages included in `requirements.txt`:
```bash
pip install -r requirements.txt
```

**3. Run the Training Pipeline**
The script uses the `gdown` package under the hood, which correctly handles Google Drive URLs for large files (>200MB) without failing on the standard 'Google virus scan limit' warning.

Run the pipeline (no augmentation — default):
```bash
python -m src.train_pipeline --data-source "YOUR_GOOGLE_DRIVE_LINK_HERE" --target-column "TARGET" --model-type "xgb"
```

Run with Gaussian-noise augmentation on the training split:
```bash
python -m src.train_pipeline \
  --data-source "data/historical_data.csv" \
  --target-column "TARGET" \
  --model-type "rf" \
  --augment \
  --aug-sigma 0.05 \
  --random-state 42
```

### CLI Arguments — Training Pipeline:
| Argument | Default | Description |
|---|---|---|
| `--data-source` | *(required)* | Local CSV path or Google Drive URL |
| `--target-column` | `TARGET` | Name of the label column |
| `--model-type` | `rf` | `rf` (Random Forest) or `xgb` (XGBoost) |
| `--model-path` | `pipeline_model.joblib` | Output path for the saved pipeline |
| `--augment` | `False` | Enable Gaussian-noise augmentation (training split only) |
| `--aug-sigma` | `0.05` | Noise magnitude as a fraction of each column's std-dev |
| `--random-state` | `42` | Seed for train/val split (keep fixed across experiments) |

**4. Verification**
Once the script completes successfully, a `.joblib` file is saved in your directory. This file bundles imputation methods, one-hot encoders, standardisations, and the decision model tree together into a neat sklearn pipeline object.

---

## Dataset Augmentation

> **Augmentation is applied strictly to the training split. Validation and test data are never modified.**

### Overview

The augmentation system lives in [`src/augmentation.py`](src/augmentation.py). It is designed to be:

- **Training-only**: `apply_augmentation()` accepts only the train split by API design. Validation indices are passed as `val_index` — any index overlap triggers an immediate `AugmentationLeakageError`.
- **Configurable**: All behaviour is controlled via the `AugmentationConfig` dataclass — no hidden global state.
- **Safe by default**: Augmentation is **disabled** unless explicitly enabled.

### Available Augmentation Methods

| Method | Key | Description | Best for |
|---|---|---|---|
| **Gaussian Noise** | `"gaussian_noise"` | Appends noisy copies of training rows. Noise is zero-mean, with σ = `noise_sigma × column_std`. | Dense numeric features; adding mild invariance |
| **SMOTE** | `"smote"` | Synthetic Minority Over-sampling Technique. Synthesises new minority-class samples via k-NN interpolation. Requires `pip install imbalanced-learn`. | Imbalanced binary classification |
| **Combined** | `"combined"` | Applies SMOTE first, then adds Gaussian noise on top. | Imbalanced + noisy feature spaces |

### Configuration Options — `AugmentationConfig`

```python
from src.augmentation import AugmentationConfig

config = AugmentationConfig(
    enabled       = True,             # Master switch (default: True when created; False disables all augmentation)
    method        = "gaussian_noise", # "gaussian_noise" | "smote" | "combined"
    seed          = 42,               # RNG seed — keep fixed across experiment arms for fair comparison
    noise_sigma   = 0.05,             # Noise std-dev as fraction of each column's std-dev (default: 5%)
    noise_fraction= 1.0,              # Fraction of training rows to duplicate with noise (default: 100%)
    smote_k_neighbors = 5,            # k-NN neighbours for SMOTE (default: 5)
)
```

### Usage Examples

#### 1. Programmatic — in Python code

```python
from src.augmentation import AugmentationConfig, apply_augmentation
from sklearn.model_selection import train_test_split

# Split your data first
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Configure augmentation
config = AugmentationConfig(enabled=True, method="gaussian_noise", seed=42, noise_sigma=0.05)

# Apply — val_index is the leakage guard; AugmentationLeakageError raised if indices overlap
X_train_aug, y_train_aug = apply_augmentation(X_train, y_train, config, val_index=X_val.index)

# Train only on augmented training data — val set untouched
model.fit(X_train_aug, y_train_aug)
score = model.score(X_val, y_val)
```

#### 2. Via the training pipeline

```python
from src.augmentation import AugmentationConfig
from src.train_pipeline import train_and_save_model

aug_config = AugmentationConfig(enabled=True, method="gaussian_noise", seed=42, noise_sigma=0.05)

result = train_and_save_model(
    data_source   = "data/historical_data.csv",
    target_column = "TARGET",
    model_path    = "models/augmented_model.joblib",
    model_type    = "rf",
    aug_config    = aug_config,
    random_state  = 42,
)
print(f"ROC-AUC: {result.roc_auc:.4f}  F1: {result.f1:.4f}")
```

#### 3. Controlled A/B experiment (baseline vs augmented)

```bash
# Uses --use-synthetic for offline testing; omit to use your own CSV
python scripts/run_augmentation_experiment.py \
    --data-source data/historical_data.csv \
    --model-type rf \
    --noise-sigma 0.05 \
    --random-state 42
```

This prints a side-by-side metrics table, writes `experiments/augmentation_experiment_<timestamp>.json`, and auto-generates [`docs/model_card.md`](docs/model_card.md).

#### 4. Demo script (no real dataset required)

```bash
python notebooks/augmentation_demo.py
```

---

## Leakage Prevention Guarantees

| Guard | Where | What it does |
|---|---|---|
| **Train-only API** | `src/augmentation.py::apply_augmentation` | Function signature accepts only the training split |
| **Index overlap check** | `src/augmentation.py::check_no_leakage` | Raises `AugmentationLeakageError` if val indices appear in train |
| **Post-aug diagnostics** | `src/augmentation.py::_run_diagnostics` | Asserts `len(X) == len(y)` and no NaN labels after augmentation |
| **Experiment design** | `scripts/run_augmentation_experiment.py` | Val split computed before augmentation; passed only to `.predict` |
| **Integration test** | `tests/test_augmentation.py::test_val_data_never_augmented` | Byte-compares val set before and after a full training call |

---

## Running Tests

```bash
# All tests (23 total)
pytest tests/ -v

# Augmentation-specific tests only (8 tests)
pytest tests/test_augmentation.py -v

# With coverage
pytest tests/ --cov=src --cov-report=term-missing
```

---

## Project Structure

```
self-healing-ml-pipeline/
├── src/
│   ├── augmentation.py          # Augmentation module (training-only, leakage-safe)
│   ├── train_pipeline.py        # Training orchestrator (returns TrainingResult)
│   ├── drift_detector.py        # KS-test based drift detection
│   └── app.py                   # FastAPI inference + drift API
├── scripts/
│   ├── prepare_data.py          # Downloads & prepares historical/current CSV splits
│   ├── run_augmentation_experiment.py  # Controlled A/B experiment runner
│   └── simulate_inference.py    # Simulates production inference batches
├── tests/
│   ├── test_augmentation.py     # 8 unit + integration augmentation tests
│   ├── test_pipeline.py         # Pipeline build & load tests
│   ├── test_drift_detector.py   # Drift detection tests
│   └── test_app.py              # FastAPI endpoint tests
├── notebooks/
│   ├── 01_eda.ipynb             # Exploratory data analysis
│   └── augmentation_demo.py     # Runnable augmentation demo (no dataset needed)
├── docs/
│   └── model_card.md            # Auto-generated model card (baseline vs augmented)
├── experiments/                 # JSON experiment results (git-ignored)
├── models/                      # Saved .joblib pipelines
└── requirements.txt
```
