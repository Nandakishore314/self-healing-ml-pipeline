"""
Controlled Augmentation Experiment Runner
==========================================
Runs two training arms on the **same** train/val split with the **same** random
seed so results are directly comparable:

    Arm A (baseline)  – raw training data, no augmentation
    Arm B (augmented) – Gaussian-noise augmentation on training data only

Outputs
-------
* Console: side-by-side metrics table
* experiments/augmentation_experiment_<timestamp>.json  – machine-readable
* docs/model_card.md – auto-generated / overwritten model card

Usage
-----
    python scripts/run_augmentation_experiment.py \\
        --data-source data/historical_data.csv \\
        --model-type rf \\
        --noise-sigma 0.05 \\
        --random-state 42

    # Use a tiny synthetic dataset (no local CSV needed) for quick CI checks:
    python scripts/run_augmentation_experiment.py --use-synthetic
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Make sure src/ is importable when script is run from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.augmentation import AugmentationConfig
from src.train_pipeline import train_and_save_model

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
DOCS_DIR = PROJECT_ROOT / "docs"
MODELS_DIR = PROJECT_ROOT / "models"

EXPERIMENTS_DIR.mkdir(exist_ok=True)
DOCS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Synthetic dataset (fallback for CI / offline use)
# ---------------------------------------------------------------------------

def _make_synthetic_dataset(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Generate a small synthetic credit-risk-like dataset for testing."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "AMT_INCOME_TOTAL": rng.normal(180_000, 60_000, n).clip(10_000),
            "AMT_CREDIT": rng.normal(500_000, 200_000, n).clip(50_000),
            "AMT_ANNUITY": rng.normal(25_000, 8_000, n).clip(2_000),
            "DAYS_BIRTH": rng.integers(-25000, -6000, n),
            "DAYS_EMPLOYED": rng.integers(-10000, 0, n),
            "EXT_SOURCE_2": rng.uniform(0, 1, n),
            "EXT_SOURCE_3": rng.uniform(0, 1, n),
            "CNT_CHILDREN": rng.integers(0, 5, n),
            "CODE_GENDER": rng.choice(["M", "F"], n),
            "FLAG_OWN_CAR": rng.choice(["Y", "N"], n),
            "FLAG_OWN_REALTY": rng.choice(["Y", "N"], n),
            "NAME_CONTRACT_TYPE": rng.choice(["Cash loans", "Revolving loans"], n),
        }
    )
    # Simple synthetic target: higher EXT_SOURCE → less likely to default
    prob_default = 1 - (df["EXT_SOURCE_2"] * 0.5 + df["EXT_SOURCE_3"] * 0.5)
    df["TARGET"] = (rng.uniform(0, 1, n) < prob_default * 0.25).astype(int)
    return df


# ---------------------------------------------------------------------------
# Metrics table printer
# ---------------------------------------------------------------------------

def _print_comparison(baseline, augmented) -> None:
    """Print a formatted side-by-side comparison to the console."""
    metrics = ["accuracy", "roc_auc", "f1", "precision", "recall",
               "train_rows", "val_rows"]
    col_w = 14

    header = f"{'Metric':<18} {'Baseline':>{col_w}} {'Augmented':>{col_w}} {'Diff(aug-base)':>{col_w}}"
    sep = "-" * len(header)
    print("\n" + sep)
    print("  AUGMENTATION EXPERIMENT - RESULTS COMPARISON")
    print(sep)
    print(header)
    print(sep)

    b = baseline.as_dict()
    a = augmented.as_dict()

    for m in metrics:
        bv = b.get(m, 0)
        av = a.get(m, 0)
        if isinstance(bv, float):
            delta = av - bv
            sign = "+" if delta >= 0 else ""
            print(
                f"  {m:<16} {bv:>{col_w}.4f} {av:>{col_w}.4f} "
                f"{sign + f'{delta:.4f}':>{col_w}}"
            )
        else:
            delta = av - bv
            sign = "+" if delta >= 0 else ""
            print(
                f"  {m:<16} {bv:>{col_w}d} {av:>{col_w}d} "
                f"{sign + str(delta):>{col_w}}"
            )
    print(sep + "\n")


# ---------------------------------------------------------------------------
# Model card writer
# ---------------------------------------------------------------------------

def _write_model_card(
    baseline,
    augmented,
    aug_config: AugmentationConfig,
    model_type: str,
    data_source: str,
    random_state: int,
    output_path: Path,
) -> None:
    """Render and write a model card markdown file."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    b = baseline.as_dict()
    a = augmented.as_dict()

    def delta_str(metric: str, fmt: str = ".4f") -> str:
        d = a[metric] - b[metric]
        sign = "+" if d >= 0 else ""
        return f"{sign}{d:{fmt}}"

    def winner(metric: str) -> str:
        if a[metric] > b[metric]:
            return "🟢 Augmented"
        elif a[metric] < b[metric]:
            return "🔴 Baseline"
        return "🟡 Tie"

    aug_details = (
        f"- **Method**: `{aug_config.method}`\n"
        f"- **Noise σ** (fraction of column std-dev): `{aug_config.noise_sigma}`\n"
        f"- **Noise fraction** (rows augmented): `{aug_config.noise_fraction}`\n"
        f"- **Seed**: `{aug_config.seed}`\n"
    )

    conclusions = []
    if a["roc_auc"] > b["roc_auc"] + 0.002:
        conclusions.append(
            f"Augmentation improved ROC-AUC by **{delta_str('roc_auc')}**, "
            "suggesting better generalisation on the held-out set."
        )
    elif a["roc_auc"] < b["roc_auc"] - 0.002:
        conclusions.append(
            f"Augmentation slightly reduced ROC-AUC by **{delta_str('roc_auc')}**. "
            "Consider tuning `noise_sigma` or switching to SMOTE for class imbalance."
        )
    else:
        conclusions.append(
            "ROC-AUC difference is negligible (< 0.002). "
            "Augmentation had no significant impact on discrimination power."
        )

    if a["f1"] > b["f1"] + 0.005:
        conclusions.append(
            f"F1 score improved by **{delta_str('f1')}**, indicating better minority-class recall."
        )

    conclusion_text = "\n\n".join(conclusions) if conclusions else (
        "No significant performance difference was observed between arms."
    )

    card = f"""# Model Card — Credit Default Risk Classifier

> Auto-generated by `scripts/run_augmentation_experiment.py` on **{now}**

---

## 1. Model Details

| Field | Value |
|---|---|
| **Model Type** | `{model_type.upper()}` ({'Random Forest' if model_type == 'rf' else 'XGBoost'}) |
| **Task** | Binary classification (0 = repay, 1 = default) |
| **Dataset** | Home Credit Default Risk (subset of 13 features) |
| **Data Source** | `{data_source}` |
| **Train / Val Split** | 80 % / 20 %, stratified by `TARGET` |
| **Random State** | `{random_state}` (fixed across both experiment arms) |
| **Generated** | {now} |

---

## 2. Intended Use

- **Primary use**: Score credit applications and estimate probability of default.
- **Intended users**: Data scientists and ML engineers evaluating augmentation strategies in the self-healing pipeline.
- **Out-of-scope**: Production credit decisions without additional fairness and bias evaluation.

---

## 3. Training Data & Augmentation Strategy

### 3.1 Dataset Summary

The dataset is a subset of the [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) competition data,
pre-processed in `scripts/prepare_data.py`. Only 12 predictive features are retained.

### 3.2 Augmentation Method

{aug_details}

> **Leakage Prevention**: Augmentation is applied **exclusively** to the training split.
> The validation split is determined before any augmentation occurs and its indices
> are passed to `apply_augmentation()` as `val_index`. An `AugmentationLeakageError`
> is raised immediately if any overlap is detected — ensuring evaluation integrity.

---

## 4. Controlled Experiment Design

Both arms use the **identical** train/val split (`random_state={random_state}`, `test_size=0.20`).
No hyper-parameters are changed between arms — only whether augmentation is applied.

| Arm | Training Rows | Val Rows | Augmentation |
|---|---|---|---|
| **Baseline** | {b['train_rows']:,} | {b['val_rows']:,} | None |
| **Augmented** | {a['train_rows']:,} | {a['val_rows']:,} | {aug_config.method} |

---

## 5. Evaluation Metrics — Baseline vs Augmented

> All metrics computed on the **held-out validation set** (never seen during training or augmentation).

| Metric | Baseline | Augmented | Δ (aug − base) | Better |
|---|---|---|---|---|
| **Accuracy** | {b['accuracy']:.4f} | {a['accuracy']:.4f} | {delta_str('accuracy')} | {winner('accuracy')} |
| **ROC-AUC** | {b['roc_auc']:.4f} | {a['roc_auc']:.4f} | {delta_str('roc_auc')} | {winner('roc_auc')} |
| **F1** | {b['f1']:.4f} | {a['f1']:.4f} | {delta_str('f1')} | {winner('f1')} |
| **Precision** | {b['precision']:.4f} | {a['precision']:.4f} | {delta_str('precision')} | {winner('precision')} |
| **Recall** | {b['recall']:.4f} | {a['recall']:.4f} | {delta_str('recall')} | {winner('recall')} |

---

## 6. Findings & Conclusions

{conclusion_text}

### Recommendations

- If ROC-AUC improvement > 0.005, promote augmentation to the default training config.
- If recall on the minority class (default = 1) improves, augmentation helps the pipeline
  catch more at-risk applicants — valuable for the self-healing retraining loop.
- Run the experiment on the full dataset (not just the 13-feature subset) for a more
  definitive assessment.

---

## 7. Limitations & Caveats

- Results are specific to this dataset subset and feature set.
- Gaussian noise is a mild augmentation; stronger methods (SMOTE, MixUp) may yield different outcomes.
- Class imbalance (~8–9 % positive rate) means accuracy is a poor primary metric; prioritise ROC-AUC and Recall.
- This model card is auto-generated and should be reviewed by a domain expert before any production decision.

---

## 8. Leakage Prevention Guarantees

| Guard | Location | Mechanism |
|---|---|---|
| Train-only API | `src/augmentation.py::apply_augmentation` | Function signature only accepts training split |
| Index overlap check | `src/augmentation.py::check_no_leakage` | Raises `AugmentationLeakageError` on any val index overlap |
| Post-aug shape check | `src/augmentation.py::_run_diagnostics` | Asserts `len(X_aug) == len(y_aug)`, no NaN labels |
| Experiment design | `scripts/run_augmentation_experiment.py` | Val set split before augmentation, passed only to `.predict` |
| Integration test | `tests/test_augmentation.py::test_val_data_never_augmented` | Verifies val DataFrame is byte-identical before/after training call |

---

*For questions or updates, re-run `scripts/run_augmentation_experiment.py`.*
"""
    output_path.write_text(card, encoding="utf-8")
    logger.info("Model card written to %s", output_path)


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_experiment(
    data_source: str,
    model_type: str = "rf",
    noise_sigma: float = 0.05,
    random_state: int = 42,
    use_synthetic: bool = False,
) -> tuple:
    """Execute both experiment arms and return (baseline_result, augmented_result)."""

    # ── Prepare data source ──────────────────────────────────────────────────
    if use_synthetic:
        logger.info("Generating synthetic dataset for experiment (--use-synthetic).")
        synthetic_path = EXPERIMENTS_DIR / "_synthetic_data.csv"
        _make_synthetic_dataset(n=2000, seed=random_state).to_csv(
            synthetic_path, index=False
        )
        effective_source = str(synthetic_path)
    else:
        effective_source = data_source

    target_col = "TARGET"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    aug_config = AugmentationConfig(
        enabled=True,
        method="gaussian_noise",
        seed=random_state,
        noise_sigma=noise_sigma,
        noise_fraction=1.0,
    )
    no_aug_config = AugmentationConfig(enabled=False)

    # ── Arm A: Baseline ──────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("ARM A — Baseline (no augmentation)")
    logger.info("=" * 60)
    baseline_path = str(MODELS_DIR / f"baseline_{model_type}_{ts}.joblib")
    baseline_result = train_and_save_model(
        data_source=effective_source,
        target_column=target_col,
        model_path=baseline_path,
        model_type=model_type,
        aug_config=no_aug_config,
        random_state=random_state,
    )

    # ── Arm B: Augmented ─────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("ARM B - Augmented (gaussian_noise, sigma=%.3f)", noise_sigma)
    logger.info("=" * 60)
    augmented_path = str(MODELS_DIR / f"augmented_{model_type}_{ts}.joblib")
    augmented_result = train_and_save_model(
        data_source=effective_source,
        target_column=target_col,
        model_path=augmented_path,
        model_type=model_type,
        aug_config=aug_config,
        random_state=random_state,
    )

    # ── Print comparison ─────────────────────────────────────────────────────
    _print_comparison(baseline_result, augmented_result)

    # ── Save JSON results ────────────────────────────────────────────────────
    results_path = EXPERIMENTS_DIR / f"augmentation_experiment_{ts}.json"
    experiment_record = {
        "timestamp": ts,
        "model_type": model_type,
        "data_source": effective_source,
        "random_state": random_state,
        "augmentation_config": {
            "method": aug_config.method,
            "noise_sigma": aug_config.noise_sigma,
            "noise_fraction": aug_config.noise_fraction,
            "seed": aug_config.seed,
        },
        "baseline": baseline_result.as_dict(),
        "augmented": augmented_result.as_dict(),
    }
    results_path.write_text(
        json.dumps(experiment_record, indent=2), encoding="utf-8"
    )
    logger.info("Results saved to %s", results_path)

    # ── Write model card ─────────────────────────────────────────────────────
    model_card_path = DOCS_DIR / "model_card.md"
    _write_model_card(
        baseline=baseline_result,
        augmented=augmented_result,
        aug_config=aug_config,
        model_type=model_type,
        data_source=effective_source,
        random_state=random_state,
        output_path=model_card_path,
    )

    return baseline_result, augmented_result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run controlled baseline vs augmented experiment."
    )
    parser.add_argument(
        "--data-source",
        type=str,
        default="data/historical_data.csv",
        help="Path to local CSV or Google Drive URL",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["rf", "xgb"],
        default="rf",
    )
    parser.add_argument(
        "--noise-sigma",
        type=float,
        default=0.05,
        help="Gaussian noise sigma (fraction of column std-dev)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed for train/val split — must be the same for both arms",
    )
    parser.add_argument(
        "--use-synthetic",
        action="store_true",
        default=False,
        help="Use a generated synthetic dataset instead of a real CSV (for CI/testing)",
    )

    args = parser.parse_args()

    # Validate data source exists unless synthetic
    if not args.use_synthetic:
        if not os.path.exists(args.data_source) and not args.data_source.startswith(
            "http"
        ):
            logger.error(
                "Data source not found: %s\n"
                "Run scripts/prepare_data.py first, or pass --use-synthetic.",
                args.data_source,
            )
            sys.exit(1)

    run_experiment(
        data_source=args.data_source,
        model_type=args.model_type,
        noise_sigma=args.noise_sigma,
        random_state=args.random_state,
        use_synthetic=args.use_synthetic,
    )
