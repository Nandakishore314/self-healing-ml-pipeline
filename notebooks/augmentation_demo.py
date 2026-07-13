"""
Augmentation Demo Script
========================
A self-contained, runnable walkthrough of the augmentation system.
No real dataset is required — everything runs on synthetic data generated here.

Run from the project root:
    python notebooks/augmentation_demo.py

What this script demonstrates
------------------------------
  Step 1 – Generate a synthetic tabular credit-risk dataset
  Step 2 – Manual split + augmentation via apply_augmentation()
  Step 3 – AugmentationConfig options (enable/disable, methods, sigma)
  Step 4 – Leakage guard: watch AugmentationLeakageError fire
  Step 5 – Post-aug diagnostics (shape, label integrity)
  Step 6 – Full pipeline integration via train_and_save_model()
  Step 7 – Baseline vs augmented side-by-side metric comparison
"""

from __future__ import annotations

import sys
import os
import tempfile

# Make sure src/ is importable when run from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.augmentation import (
    AugmentationConfig,
    AugmentationLeakageError,
    apply_augmentation,
    check_no_leakage,
    gaussian_noise,
)
from src.train_pipeline import train_and_save_model

SEED = 42
SEP = "-" * 64


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def section(title: str) -> None:
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def make_dataset(n: int = 800, seed: int = SEED) -> pd.DataFrame:
    """Generate a synthetic credit-risk-like tabular dataset."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "AMT_INCOME_TOTAL": rng.normal(180_000, 60_000, n).clip(10_000),
        "AMT_CREDIT":       rng.normal(500_000, 200_000, n).clip(50_000),
        "AMT_ANNUITY":      rng.normal(25_000, 8_000, n).clip(2_000),
        "DAYS_BIRTH":       rng.integers(-25_000, -6_000, n),
        "DAYS_EMPLOYED":    rng.integers(-10_000, 0, n),
        "EXT_SOURCE_2":     rng.uniform(0, 1, n),
        "EXT_SOURCE_3":     rng.uniform(0, 1, n),
        "CNT_CHILDREN":     rng.integers(0, 5, n).astype(float),
        "CODE_GENDER":      rng.choice(["M", "F"], n),
        "FLAG_OWN_CAR":     rng.choice(["Y", "N"], n),
        "TARGET":           (rng.uniform(0, 1, n) < 0.15).astype(int),
    })
    return df


# ---------------------------------------------------------------------------
# Step 1: Dataset
# ---------------------------------------------------------------------------

section("STEP 1 — Synthetic dataset")

df = make_dataset()
X = df.drop(columns=["TARGET"])
y = df["TARGET"]

print(f"  Total samples : {len(df)}")
print(f"  Features      : {X.shape[1]}")
print(f"  Target dist   : {y.value_counts().to_dict()}")

# Train / val split — same random_state used throughout the demo
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y
)
print(f"\n  After 80/20 split:")
print(f"    Train : {X_train.shape}   Val : {X_val.shape}")


# ---------------------------------------------------------------------------
# Step 2: Gaussian noise augmentation
# ---------------------------------------------------------------------------

section("STEP 2 — Gaussian-noise augmentation (manual)")

numeric_cols = X_train.select_dtypes(include="number").columns.tolist()
X_train_noisy = gaussian_noise(
    X_train,
    numeric_cols=numeric_cols,
    seed=SEED,
    sigma=0.05,    # 5% of each column's std-dev
    fraction=1.0,  # duplicate every training row with noise
)
print(f"  Original train rows : {len(X_train)}")
print(f"  After noise aug     : {len(X_train_noisy)}  (2x — original + noisy copy)")
print(f"  Columns unchanged   : {X_train_noisy.shape[1]}")

# Spot-check: mean of a numeric column should be nearly identical
col = "AMT_INCOME_TOTAL"
print(f"\n  Column '{col}':")
print(f"    Original mean  : {X_train[col].mean():,.2f}")
print(f"    Augmented mean : {X_train_noisy[col].mean():,.2f}  (noise is zero-mean)")


# ---------------------------------------------------------------------------
# Step 3: AugmentationConfig options
# ---------------------------------------------------------------------------

section("STEP 3 — AugmentationConfig: enable / disable / sigma")

# Enabled — gaussian noise
cfg_on = AugmentationConfig(enabled=True, method="gaussian_noise", seed=SEED, noise_sigma=0.10)
X_aug, y_aug = apply_augmentation(X_train, y_train, cfg_on, val_index=X_val.index)
print(f"  [enabled=True,  sigma=0.10]  train rows: {len(X_train)} -> {len(X_aug)}")

# Disabled — passthrough
cfg_off = AugmentationConfig(enabled=False)
X_pass, y_pass = apply_augmentation(X_train, y_train, cfg_off, val_index=X_val.index)
print(f"  [enabled=False]              train rows: {len(X_train)} -> {len(X_pass)}  (unchanged)")

assert X_pass.equals(X_train), "Passthrough must return original unmodified DataFrame"
assert y_pass.equals(y_train), "Passthrough must return original unmodified Series"
print("  Passthrough assertion passed: original arrays returned unchanged.")

# Different sigma values
for sigma in [0.01, 0.05, 0.20]:
    cfg = AugmentationConfig(enabled=True, method="gaussian_noise", seed=SEED, noise_sigma=sigma)
    Xa, _ = apply_augmentation(X_train, y_train, cfg, val_index=X_val.index)
    col_std_orig = X_train[col].std()
    col_std_aug  = Xa.iloc[len(X_train):][col].std()
    print(f"  sigma={sigma:.2f}  | column '{col}' std  original={col_std_orig:.2f}  "
          f"aug-noise-std~={col_std_orig * sigma:.2f}  actual aug std={col_std_aug:.2f}")


# ---------------------------------------------------------------------------
# Step 4: Leakage guard
# ---------------------------------------------------------------------------

section("STEP 4 — Leakage guard: AugmentationLeakageError")

# Construct overlapping indices deliberately
bad_train_idx = pd.RangeIndex(0, 100)
bad_val_idx   = pd.RangeIndex(50, 150)      # 50 overlapping rows!

bad_X_train = pd.DataFrame({"f": np.arange(100)}, index=bad_train_idx)
bad_y_train = pd.Series(np.zeros(100, dtype=int), index=bad_train_idx)

cfg = AugmentationConfig(enabled=True, method="gaussian_noise", seed=SEED)
try:
    apply_augmentation(bad_X_train, bad_y_train, cfg, val_index=bad_val_idx)
    print("  ERROR: leakage was NOT detected!")
except AugmentationLeakageError as e:
    print(f"  AugmentationLeakageError correctly raised:")
    print(f"    {str(e).splitlines()[0]}")

# Clean split passes silently
try:
    check_no_leakage(X_train, X_val.index)
    print("\n  check_no_leakage(X_train, X_val.index) -> PASSED (no overlap)")
except AugmentationLeakageError:
    print("  ERROR: false positive from check_no_leakage!")


# ---------------------------------------------------------------------------
# Step 5: Post-augmentation diagnostics
# ---------------------------------------------------------------------------

section("STEP 5 — Post-augmentation diagnostics")

cfg = AugmentationConfig(enabled=True, method="gaussian_noise", seed=SEED, noise_sigma=0.05)
X_aug, y_aug = apply_augmentation(X_train, y_train, cfg, val_index=X_val.index)

print(f"  X_aug shape     : {X_aug.shape}   (rows, cols)")
print(f"  y_aug length    : {len(y_aug)}")
print(f"  Shape match     : {len(X_aug) == len(y_aug)}")
print(f"  NaN in X_aug    : {X_aug.isnull().sum().sum()}")
print(f"  NaN in y_aug    : {y_aug.isnull().sum()}")
print(f"  Label dist orig : {y_train.value_counts().to_dict()}")
print(f"  Label dist aug  : {y_aug.value_counts().to_dict()}")


# ---------------------------------------------------------------------------
# Step 6: Full pipeline integration
# ---------------------------------------------------------------------------

section("STEP 6 — Full train_and_save_model() integration")

with tempfile.TemporaryDirectory() as tmp:
    data_path  = os.path.join(tmp, "data.csv")
    model_path = os.path.join(tmp, "model.joblib")
    df.to_csv(data_path, index=False)

    aug_cfg = AugmentationConfig(
        enabled=True, method="gaussian_noise", seed=SEED, noise_sigma=0.05
    )

    result = train_and_save_model(
        data_source   = data_path,
        target_column = "TARGET",
        model_path    = model_path,
        model_type    = "rf",
        aug_config    = aug_cfg,
        random_state  = SEED,
    )

print(f"\n  Training result (augmented):")
print(f"    Accuracy  : {result.accuracy:.4f}")
print(f"    ROC-AUC   : {result.roc_auc:.4f}")
print(f"    F1        : {result.f1:.4f}")
print(f"    Precision : {result.precision:.4f}")
print(f"    Recall    : {result.recall:.4f}")
print(f"    Train rows: {result.train_rows}  Val rows: {result.val_rows}")
print(f"    Augmented : {result.augmented}")


# ---------------------------------------------------------------------------
# Step 7: Baseline vs Augmented side-by-side
# ---------------------------------------------------------------------------

section("STEP 7 — Baseline vs Augmented comparison")

metrics_to_show = ["accuracy", "roc_auc", "f1", "precision", "recall", "train_rows"]
col_w = 12

with tempfile.TemporaryDirectory() as tmp:
    data_path = os.path.join(tmp, "data.csv")
    df.to_csv(data_path, index=False)

    # Arm A: baseline
    baseline = train_and_save_model(
        data_source   = data_path,
        target_column = "TARGET",
        model_path    = os.path.join(tmp, "baseline.joblib"),
        model_type    = "rf",
        aug_config    = AugmentationConfig(enabled=False),
        random_state  = SEED,
    )

    # Arm B: augmented (same split, same seed)
    augmented = train_and_save_model(
        data_source   = data_path,
        target_column = "TARGET",
        model_path    = os.path.join(tmp, "augmented.joblib"),
        model_type    = "rf",
        aug_config    = AugmentationConfig(enabled=True, method="gaussian_noise",
                                           seed=SEED, noise_sigma=0.05),
        random_state  = SEED,
    )

b = baseline.as_dict()
a = augmented.as_dict()

header = f"  {'Metric':<14} {'Baseline':>{col_w}} {'Augmented':>{col_w}} {'Diff':>{col_w}}"
print(header)
print("  " + "-" * (len(header) - 2))

for m in metrics_to_show:
    bv, av = b[m], a[m]
    if isinstance(bv, float):
        diff   = av - bv
        sign   = "+" if diff >= 0 else ""
        better = " <-- better" if diff > 0.001 else (" <-- worse" if diff < -0.001 else "")
        print(f"  {m:<14} {bv:>{col_w}.4f} {av:>{col_w}.4f} "
              f"{sign + f'{diff:.4f}':>{col_w}}{better}")
    else:
        diff = av - bv
        sign = "+" if diff >= 0 else ""
        print(f"  {m:<14} {bv:>{col_w}d} {av:>{col_w}d} {sign + str(diff):>{col_w}}")

print(f"\n  Val rows are identical across arms: {b['val_rows'] == a['val_rows']}")
print(f"  Both arms used random_state={SEED}  (same train/val split guaranteed)")

print(f"\n{SEP}")
print("  Demo complete. All augmentation guards verified.")
print(f"{SEP}\n")
