"""
Data Augmentation Module
========================
All augmentation transforms live here and are **training-only by design**.

Public API
----------
    from src.augmentation import AugmentationConfig, apply_augmentation

    config = AugmentationConfig(enabled=True, method="gaussian_noise", seed=42)
    X_aug, y_aug = apply_augmentation(X_train, y_train, config, val_index=X_val.index)

Leakage Guarantees
------------------
* ``apply_augmentation`` accepts only the training split as input.
* A second guard compares ``val_index`` (the held-out indices) against the
  training data's index.  If **any** overlap is found an
  ``AugmentationLeakageError`` is raised immediately – stopping the run before
  any contamination can occur.
* The validation/test splits are never modified by this module.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class AugmentationLeakageError(RuntimeError):
    """Raised when validation/test indices are detected inside the training set.

    This is an unrecoverable error: continuing would contaminate evaluation
    metrics and produce misleading experiment results.
    """


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class AugmentationConfig:
    """Configuration for the augmentation pipeline.

    Parameters
    ----------
    enabled : bool
        Master switch.  When ``False``, ``apply_augmentation`` returns the
        input arrays unchanged and performs no index checks.
    method : str
        Augmentation strategy.  Currently supported:
        * ``"gaussian_noise"`` – adds zero-mean Gaussian noise to every
          numeric column.
        * ``"smote"`` – SMOTE oversampling (requires ``imbalanced-learn``).
        * ``"combined"`` – SMOTE followed by Gaussian noise.
    seed : int
        Random seed used for all stochastic operations, ensuring
        reproducibility across experiment arms.
    noise_sigma : float
        Standard deviation of the Gaussian noise (relative to each column's
        own std-dev).  Only used when ``method`` includes ``"gaussian_noise"``.
    noise_fraction : float
        Fraction of training rows to which noise is applied (0 < f <= 1).
        1.0 = all rows receive a noisy copy appended to the original set.
    smote_k_neighbors : int
        Number of nearest neighbours for SMOTE.  Only used when ``method``
        includes ``"smote"``.
    """

    enabled: bool = True
    method: str = "gaussian_noise"          # "gaussian_noise" | "smote" | "combined"
    seed: int = 42
    noise_sigma: float = 0.05              # 5 % of each column's std-dev
    noise_fraction: float = 1.0            # augment all training rows
    smote_k_neighbors: int = 5
    _numeric_cols: list[str] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        valid_methods = {"gaussian_noise", "smote", "combined"}
        if self.method not in valid_methods:
            raise ValueError(
                f"Unknown augmentation method '{self.method}'. "
                f"Choose from {valid_methods}."
            )
        if not (0.0 < self.noise_fraction <= 1.0):
            raise ValueError("noise_fraction must be in (0, 1].")
        if self.noise_sigma < 0:
            raise ValueError("noise_sigma must be >= 0.")


# ---------------------------------------------------------------------------
# Low-level transform functions
# ---------------------------------------------------------------------------

def gaussian_noise(
    X: pd.DataFrame,
    numeric_cols: list[str],
    *,
    seed: int = 42,
    sigma: float = 0.05,
    fraction: float = 1.0,
) -> pd.DataFrame:
    """Append noisy copies of a subset of rows to *X*.

    Parameters
    ----------
    X : pd.DataFrame
        **Training** data only.
    numeric_cols : list[str]
        Columns to perturb.  Non-numeric columns are copied verbatim.
    seed : int
        RNG seed for reproducibility.
    sigma : float
        Noise magnitude expressed as a fraction of each column's std-dev.
        E.g. 0.05 adds noise ~5 % of the feature's natural spread.
    fraction : float
        Fraction of rows to duplicate with noise (default 1.0 = all rows).

    Returns
    -------
    pd.DataFrame
        Original rows concatenated with noisy copies.  New rows receive a
        reset integer index offset by ``len(X)`` to avoid index collisions.
    """
    rng = np.random.default_rng(seed)
    n_augment = max(1, int(len(X) * fraction))
    aug_rows = X.sample(n=n_augment, random_state=seed, replace=False).copy()

    for col in numeric_cols:
        if col not in aug_rows.columns:
            continue
        col_std = X[col].std(ddof=1)
        if col_std == 0 or np.isnan(col_std):
            continue
        noise = rng.normal(loc=0.0, scale=sigma * col_std, size=len(aug_rows))
        aug_rows[col] = aug_rows[col] + noise

    # Re-index so original indices are preserved without collision
    aug_rows.index = range(len(X), len(X) + len(aug_rows))
    result = pd.concat([X, aug_rows], axis=0)
    logger.debug(
        "gaussian_noise: added %d rows (fraction=%.2f, sigma=%.4f). "
        "New shape: %s",
        len(aug_rows), fraction, sigma, result.shape,
    )
    return result


def smote_oversample(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    seed: int = 42,
    k_neighbors: int = 5,
) -> tuple[pd.DataFrame, pd.Series]:
    """Oversample the minority class using SMOTE.

    Requires ``imbalanced-learn``.  Import is guarded so the rest of the
    module works without it.

    Parameters
    ----------
    X, y : training features and labels.
    seed : int
    k_neighbors : int

    Returns
    -------
    (X_resampled, y_resampled) as DataFrame / Series.
    """
    try:
        from imblearn.over_sampling import SMOTE  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "SMOTE requires 'imbalanced-learn'.  Install it with:\n"
            "    pip install imbalanced-learn"
        ) from exc

    sm = SMOTE(k_neighbors=k_neighbors, random_state=seed)
    X_res, y_res = sm.fit_resample(X, y)
    X_res = pd.DataFrame(X_res, columns=X.columns)
    y_res = pd.Series(y_res, name=y.name)
    logger.debug("SMOTE: shape %s → %s", X.shape, X_res.shape)
    return X_res, y_res


# ---------------------------------------------------------------------------
# Leakage diagnostic helper
# ---------------------------------------------------------------------------

def check_no_leakage(
    X_train: pd.DataFrame,
    val_index: pd.Index,
) -> None:
    """Assert that *val_index* has no overlap with *X_train*'s index.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix (post-split).
    val_index : pd.Index
        Index of the validation (or test) set.

    Raises
    ------
    AugmentationLeakageError
        If any index value appears in both splits.
    """
    overlap = X_train.index.intersection(val_index)
    if len(overlap) > 0:
        raise AugmentationLeakageError(
            f"DATA LEAKAGE DETECTED: {len(overlap)} index value(s) appear in "
            f"both the training split and the validation/test split.\n"
            f"Overlapping indices (first 10): {list(overlap[:10])}\n"
            "Augmentation aborted to protect evaluation integrity."
        )
    logger.debug(
        "Leakage check passed: no overlap between train (%d rows) "
        "and val (%d rows) indices.",
        len(X_train), len(val_index),
    )


# ---------------------------------------------------------------------------
# Main dispatcher
# ---------------------------------------------------------------------------

def apply_augmentation(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: AugmentationConfig,
    val_index: Optional[pd.Index] = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Apply the configured augmentation strategy to the *training* split only.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.  **Must not** include validation/test rows.
    y_train : pd.Series
        Corresponding training labels.
    config : AugmentationConfig
        Augmentation configuration.
    val_index : pd.Index, optional
        Index of the validation set.  When provided, a leakage check is run
        **before** any augmentation takes place.  Strongly recommended.

    Returns
    -------
    (X_aug, y_aug) : tuple[pd.DataFrame, pd.Series]
        Augmented training data.  When ``config.enabled=False``, the original
        arrays are returned unchanged.

    Raises
    ------
    AugmentationLeakageError
        If index overlap with ``val_index`` is detected.
    """
    if not config.enabled:
        logger.info("Augmentation disabled – returning original training data.")
        return X_train, y_train

    # ── Leakage guard ────────────────────────────────────────────────────────
    if val_index is not None:
        check_no_leakage(X_train, val_index)

    logger.info(
        "Applying augmentation: method=%s  seed=%d  train_shape=%s",
        config.method, config.seed, X_train.shape,
    )

    numeric_cols = (
        config._numeric_cols
        if config._numeric_cols
        else X_train.select_dtypes(include="number").columns.tolist()
    )

    X_aug = X_train.copy()
    y_aug = y_train.copy()

    if config.method in ("gaussian_noise", "combined"):
        X_aug = gaussian_noise(
            X_aug,
            numeric_cols=numeric_cols,
            seed=config.seed,
            sigma=config.noise_sigma,
            fraction=config.noise_fraction,
        )
        # Extend y_aug to match the new rows
        extra_rows = len(X_aug) - len(y_aug)
        if extra_rows > 0:
            # Sample labels to match noisy feature rows (same row order)
            y_extra = y_train.sample(
                n=extra_rows, random_state=config.seed, replace=False
            ).copy()
            y_extra.index = range(len(y_train), len(y_train) + extra_rows)
            y_aug = pd.concat([y_aug, y_extra])

    if config.method in ("smote", "combined"):
        X_aug, y_aug = smote_oversample(
            X_aug, y_aug,
            seed=config.seed,
            k_neighbors=config.smote_k_neighbors,
        )

    # ── Post-augmentation sanity diagnostics ─────────────────────────────────
    _run_diagnostics(X_train, y_train, X_aug, y_aug, config)

    return X_aug, y_aug


# ---------------------------------------------------------------------------
# Diagnostic helpers
# ---------------------------------------------------------------------------

def _run_diagnostics(
    X_orig: pd.DataFrame,
    y_orig: pd.Series,
    X_aug: pd.DataFrame,
    y_aug: pd.Series,
    config: AugmentationConfig,
) -> None:
    """Log quick sanity checks after augmentation."""
    issues: list[str] = []

    if len(X_aug) != len(y_aug):
        issues.append(
            f"Shape mismatch after augmentation: X={len(X_aug)}, y={len(y_aug)}"
        )

    nan_X = X_aug.isnull().sum().sum()
    if nan_X > 0:
        issues.append(f"X_aug contains {nan_X} NaN values after augmentation")

    nan_y = y_aug.isnull().sum()
    if nan_y > 0:
        issues.append(f"y_aug contains {nan_y} NaN labels after augmentation")

    if issues:
        for issue in issues:
            logger.error("AUGMENTATION DIAGNOSTIC FAILED: %s", issue)
        raise RuntimeError(
            "Augmentation produced invalid data:\n" + "\n".join(issues)
        )

    logger.info(
        "Augmentation diagnostics passed ✓ | "
        "train: %d→%d rows | method: %s | seed: %d",
        len(X_orig), len(X_aug), config.method, config.seed,
    )
    label_dist_orig = y_orig.value_counts(normalize=True).round(3).to_dict()
    label_dist_aug = y_aug.value_counts(normalize=True).round(3).to_dict()
    logger.info("Label distribution – original: %s | augmented: %s",
                label_dist_orig, label_dist_aug)
