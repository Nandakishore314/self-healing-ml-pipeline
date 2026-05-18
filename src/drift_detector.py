"""
Statistical Data Drift Detector
================================
Compares feature distributions between a historical baseline DataFrame
and an incoming inference batch using the Two-Sample Kolmogorov-Smirnov
test.  When the p-value for any monitored feature falls below the
configured significance level (alpha), drift is flagged.

Usage
-----
    from src.drift_detector import DriftDetector

    detector = DriftDetector(alpha=0.05)
    report   = detector.run(baseline_df, current_df, features=["AMT_INCOME_TOTAL"])

    if report.drift_detected:
        print("Drift found!", report.drifted_features)
"""

import logging
from dataclasses import dataclass, field

import pandas as pd
from scipy.stats import ks_2samp

# ---------------------------------------------------------------------------
# Logging – mirrors the style used in train_pipeline.py
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes for structured results
# ---------------------------------------------------------------------------
@dataclass
class FeatureDriftResult:
    """Result of a single-feature KS test."""

    feature: str
    ks_statistic: float
    p_value: float
    drift_detected: bool


@dataclass
class DriftReport:
    """Aggregated drift report across all monitored features."""

    drift_detected: bool = False
    alpha: float = 0.05
    feature_results: list[FeatureDriftResult] = field(default_factory=list)

    @property
    def drifted_features(self) -> list[str]:
        """Return names of features where drift was detected."""
        return [r.feature for r in self.feature_results if r.drift_detected]

    def summary(self) -> str:
        """Human-readable summary of the drift report."""
        lines = [
            f"Drift Detected : {self.drift_detected}",
            f"Alpha          : {self.alpha}",
            f"Features Tested: {len(self.feature_results)}",
            f"Features Drifted: {len(self.drifted_features)}",
            "",
            f"{'Feature':<25} {'KS Stat':>10} {'p-value':>12} {'Drifted?':>10}",
            "-" * 60,
        ]
        for r in self.feature_results:
            flag = "YES" if r.drift_detected else "no"
            lines.append(
                f"{r.feature:<25} {r.ks_statistic:>10.4f} {r.p_value:>12.6f} {flag:>10}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core detector class
# ---------------------------------------------------------------------------
class DriftDetector:
    """Detects data drift via the Two-Sample Kolmogorov-Smirnov test.

    Parameters
    ----------
    alpha : float, default 0.05
        Significance level.  A feature is flagged as drifted when its
        KS-test p-value is **below** this threshold.
    """

    def __init__(self, alpha: float = 0.05) -> None:
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1 (exclusive).")
        self.alpha = alpha
        logger.info(f"DriftDetector initialised (alpha={self.alpha})")

    # ---- single-feature test ------------------------------------------------
    def test_feature(
        self,
        baseline: pd.Series,
        current: pd.Series,
        feature_name: str,
    ) -> FeatureDriftResult:
        """Run a KS test on one continuous feature.

        Parameters
        ----------
        baseline : pd.Series
            Reference distribution (historical / training data).
        current : pd.Series
            New distribution (inference batch).
        feature_name : str
            Column name – used only for labelling results / logs.

        Returns
        -------
        FeatureDriftResult
        """
        # Drop NaNs so scipy doesn't choke
        clean_baseline = baseline.dropna()
        clean_current = current.dropna()

        statistic, p_value = ks_2samp(clean_baseline, clean_current)
        drifted = p_value < self.alpha

        if drifted:
            logger.warning(
                f"DRIFT detected in '{feature_name}' "
                f"(KS={statistic:.4f}, p={p_value:.6f})"
            )
        else:
            logger.info(
                f"No drift in '{feature_name}' "
                f"(KS={statistic:.4f}, p={p_value:.6f})"
            )

        return FeatureDriftResult(
            feature=feature_name,
            ks_statistic=statistic,
            p_value=p_value,
            drift_detected=drifted,
        )

    # ---- multi-feature orchestrator -----------------------------------------
    def run(
        self,
        baseline_df: pd.DataFrame,
        current_df: pd.DataFrame,
        features: list[str] | None = None,
    ) -> DriftReport:
        """Test multiple features for drift at once.

        Parameters
        ----------
        baseline_df : pd.DataFrame
            Historical / training data.
        current_df : pd.DataFrame
            New inference batch.
        features : list[str] or None
            Columns to monitor.  If ``None``, all shared numeric columns
            are tested automatically.

        Returns
        -------
        DriftReport
        """
        if features is None:
            # Auto-detect: use numeric columns common to both DataFrames
            baseline_num = set(
                baseline_df.select_dtypes(include="number").columns
            )
            current_num = set(
                current_df.select_dtypes(include="number").columns
            )
            features = sorted(baseline_num & current_num)
            logger.info(f"Auto-detected {len(features)} numeric features to monitor.")

        report = DriftReport(alpha=self.alpha)

        for feat in features:
            if feat not in baseline_df.columns or feat not in current_df.columns:
                logger.warning(f"Skipping '{feat}' – not found in both DataFrames.")
                continue

            result = self.test_feature(
                baseline=baseline_df[feat],
                current=current_df[feat],
                feature_name=feat,
            )
            report.feature_results.append(result)

        # Overall flag: True if ANY feature drifted
        report.drift_detected = any(r.drift_detected for r in report.feature_results)

        if report.drift_detected:
            logger.warning(
                f"Overall DRIFT DETECTED across {len(report.drifted_features)} "
                f"feature(s): {report.drifted_features}"
            )
        else:
            logger.info("No drift detected across monitored features.")

        return report
