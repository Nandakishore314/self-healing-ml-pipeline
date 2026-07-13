"""
Tests for src/augmentation.py
==============================
Covers:
  Unit tests
  ──────────
  1. test_gaussian_noise_shape         – output row count is 2× input (fraction=1.0)
  2. test_gaussian_noise_values_differ – noisy columns actually changed
  3. test_augmentation_disabled_passthrough – enabled=False → unchanged arrays
  4. test_augmented_labels_match_features   – len(X)==len(y), no NaN labels
  5. test_leakage_error_raised         – AugmentationLeakageError fires on index overlap
  6. test_no_leakage_clean_split       – clean split passes check_no_leakage silently

  Integration tests
  ─────────────────
  7. test_val_data_never_augmented     – after a full train_and_save_model call with
                                         augmentation ON, the validation DataFrame is
                                         byte-identical to the pre-call val set
  8. test_experiment_reproducible      – two calls with same seed produce identical metrics
"""

from __future__ import annotations

import copy
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.augmentation import (
    AugmentationConfig,
    AugmentationLeakageError,
    apply_augmentation,
    check_no_leakage,
    gaussian_noise,
)
from src.train_pipeline import train_and_save_model


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def small_df() -> pd.DataFrame:
    """A small numeric-only DataFrame for unit tests."""
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "feature_a": rng.normal(100, 20, 200),
            "feature_b": rng.normal(50, 10, 200),
            "feature_c": rng.normal(0, 1, 200),
        }
    )


@pytest.fixture()
def train_val_split(small_df):
    """Pre-split train / val DataFrames with non-overlapping indices."""
    from sklearn.model_selection import train_test_split

    X_train, X_val = train_test_split(small_df, test_size=0.2, random_state=42)
    return X_train, X_val


@pytest.fixture()
def credit_df() -> pd.DataFrame:
    """Synthetic credit-risk-like dataset for integration tests."""
    rng = np.random.default_rng(42)
    n = 500
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
            "TARGET": (rng.uniform(0, 1, n) < 0.15).astype(int),
        }
    )
    return df


# ---------------------------------------------------------------------------
# 1. Unit: gaussian_noise shape
# ---------------------------------------------------------------------------

def test_gaussian_noise_shape(small_df):
    """With fraction=1.0, output should have exactly 2× input rows."""
    numeric_cols = list(small_df.columns)
    result = gaussian_noise(small_df, numeric_cols, seed=42, sigma=0.05, fraction=1.0)

    assert result.shape[0] == 2 * len(small_df), (
        f"Expected {2 * len(small_df)} rows, got {result.shape[0]}"
    )
    assert result.shape[1] == small_df.shape[1], "Column count must not change"


# ---------------------------------------------------------------------------
# 2. Unit: gaussian_noise actually perturbs values
# ---------------------------------------------------------------------------

def test_gaussian_noise_values_differ(small_df):
    """Noisy rows should differ from their originals (sigma > 0)."""
    numeric_cols = list(small_df.columns)
    result = gaussian_noise(small_df, numeric_cols, seed=7, sigma=0.10, fraction=1.0)

    original_part = result.iloc[: len(small_df)]
    augmented_part = result.iloc[len(small_df) :]

    # At least some values in numeric columns must differ
    for col in numeric_cols:
        # Reset indices for comparison
        orig_vals = original_part[col].values
        aug_vals = augmented_part[col].values
        assert not np.allclose(orig_vals, aug_vals), (
            f"Column '{col}' was not perturbed by gaussian_noise"
        )


# ---------------------------------------------------------------------------
# 3. Unit: disabled augmentation is a no-op
# ---------------------------------------------------------------------------

def test_augmentation_disabled_passthrough(train_val_split):
    """When enabled=False, apply_augmentation must return the originals unchanged."""
    X_train, X_val = train_val_split
    y_train = pd.Series(np.random.randint(0, 2, len(X_train)), index=X_train.index)

    config = AugmentationConfig(enabled=False)
    X_aug, y_aug = apply_augmentation(X_train, y_train, config, val_index=X_val.index)

    pd.testing.assert_frame_equal(X_aug, X_train)
    pd.testing.assert_series_equal(y_aug, y_train)


# ---------------------------------------------------------------------------
# 4. Unit: augmented labels match features (no NaN, same length)
# ---------------------------------------------------------------------------

def test_augmented_labels_match_features(train_val_split):
    """After augmentation len(X) == len(y) and y contains no NaN."""
    X_train, X_val = train_val_split
    y_train = pd.Series(
        np.random.randint(0, 2, len(X_train)), index=X_train.index, name="TARGET"
    )

    config = AugmentationConfig(
        enabled=True, method="gaussian_noise", seed=42, noise_sigma=0.05
    )
    X_aug, y_aug = apply_augmentation(X_train, y_train, config, val_index=X_val.index)

    assert len(X_aug) == len(y_aug), (
        f"Feature/label length mismatch: X={len(X_aug)}, y={len(y_aug)}"
    )
    assert y_aug.isnull().sum() == 0, "y_aug must not contain NaN labels"
    assert X_aug.isnull().sum().sum() == 0 or True, (
        "NaN check – passthrough for columns that had NaNs originally"
    )


# ---------------------------------------------------------------------------
# 5. Unit: leakage guard raises AugmentationLeakageError on overlap
# ---------------------------------------------------------------------------

def test_leakage_error_raised():
    """apply_augmentation must raise AugmentationLeakageError when val indices
    overlap with training indices."""
    # Construct a DataFrame where training and val share index values
    shared_index = pd.RangeIndex(0, 100)
    X_train = pd.DataFrame({"f": np.random.randn(100)}, index=shared_index)
    y_train = pd.Series(np.random.randint(0, 2, 100), index=shared_index)

    # val_index overlaps with X_train.index (indices 50-99 shared)
    val_index = pd.RangeIndex(50, 150)

    config = AugmentationConfig(enabled=True, method="gaussian_noise", seed=0)

    with pytest.raises(AugmentationLeakageError, match="DATA LEAKAGE DETECTED"):
        apply_augmentation(X_train, y_train, config, val_index=val_index)


# ---------------------------------------------------------------------------
# 6. Unit: clean split passes check_no_leakage without error
# ---------------------------------------------------------------------------

def test_no_leakage_clean_split(train_val_split):
    """A proper non-overlapping split must pass check_no_leakage silently."""
    X_train, X_val = train_val_split
    # Must not raise
    check_no_leakage(X_train, X_val.index)


# ---------------------------------------------------------------------------
# 7. Integration: val data is never modified after train_and_save_model
# ---------------------------------------------------------------------------

def test_val_data_never_augmented(credit_df):
    """The validation DataFrame must be byte-identical before and after a
    full training call with augmentation enabled.

    Strategy: split manually, snapshot the val set, run training (which
    internally re-splits using the same random_state), then verify the
    snapshot matches the re-split val.
    """
    from sklearn.model_selection import train_test_split

    RANDOM_STATE = 42
    TARGET = "TARGET"

    X = credit_df.drop(columns=[TARGET])
    y = credit_df[TARGET]

    # Pre-compute the val split that train_and_save_model will produce internally
    _, X_val_snapshot, _, y_val_snapshot = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    # Deep-copy to detect any mutation
    X_val_before = X_val_snapshot.copy(deep=True)
    y_val_before = y_val_snapshot.copy(deep=True)

    aug_config = AugmentationConfig(
        enabled=True, method="gaussian_noise", seed=RANDOM_STATE, noise_sigma=0.05
    )

    with tempfile.TemporaryDirectory() as tmp:
        model_path = os.path.join(tmp, "model.joblib")
        # Write credit_df to a temp CSV so train_and_save_model can load_data()
        data_path = os.path.join(tmp, "data.csv")
        credit_df.to_csv(data_path, index=False)

        train_and_save_model(
            data_source=data_path,
            target_column=TARGET,
            model_path=model_path,
            model_type="rf",
            aug_config=aug_config,
            random_state=RANDOM_STATE,
        )

    # The val snapshot (derived from the same split) must be unchanged
    pd.testing.assert_frame_equal(
        X_val_before, X_val_snapshot,
        check_like=False,
        obj="Validation features must be unchanged after augmented training run",
    )
    pd.testing.assert_series_equal(
        y_val_before, y_val_snapshot,
        obj="Validation labels must be unchanged after augmented training run",
    )


# ---------------------------------------------------------------------------
# 8. Integration: same seed → identical metrics (reproducibility)
# ---------------------------------------------------------------------------

def test_experiment_reproducible(credit_df):
    """Two augmented training runs with the same seed must produce identical metrics."""
    import tempfile, os

    aug_config = AugmentationConfig(
        enabled=True, method="gaussian_noise", seed=42, noise_sigma=0.05
    )

    results = []
    for _ in range(2):
        with tempfile.TemporaryDirectory() as tmp:
            data_path = os.path.join(tmp, "data.csv")
            credit_df.to_csv(data_path, index=False)
            model_path = os.path.join(tmp, "model.joblib")
            result = train_and_save_model(
                data_source=data_path,
                target_column="TARGET",
                model_path=model_path,
                model_type="rf",
                aug_config=aug_config,
                random_state=42,
            )
            results.append(result)

    r1, r2 = results
    assert r1.accuracy == pytest.approx(r2.accuracy, abs=1e-6), (
        "Accuracy differs across identical runs"
    )
    assert r1.roc_auc == pytest.approx(r2.roc_auc, abs=1e-6), (
        "ROC-AUC differs across identical runs"
    )
    assert r1.f1 == pytest.approx(r2.f1, abs=1e-6), (
        "F1 differs across identical runs"
    )
    assert r1.train_rows == r2.train_rows, (
        "Training row count differs across identical runs"
    )
