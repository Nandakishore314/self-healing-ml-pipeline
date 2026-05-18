"""
Inference Simulation Script
============================
Simulates a production environment by reading `current_data.csv` in
small batches and passing each batch through the DriftDetector to check
for distribution shift against the historical baseline.

Usage
-----
    python scripts/simulate_inference.py
    python scripts/simulate_inference.py --batch-size 2000
    python scripts/simulate_inference.py --features AMT_INCOME_TOTAL AMT_CREDIT
"""

import argparse
import logging
import os
import sys
import time

import pandas as pd

# Ensure the project root is on sys.path so `src` imports work
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.drift_detector import DriftDetector

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default paths (relative to project root)
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
HISTORICAL_PATH = os.path.join(DATA_DIR, "historical_data.csv")
CURRENT_PATH = os.path.join(DATA_DIR, "current_data.csv")

# Default features to monitor for drift
DEFAULT_FEATURES = ["AMT_INCOME_TOTAL", "AMT_CREDIT"]


def load_datasets(
    historical_path: str, current_path: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the baseline and current datasets from CSV files."""
    logger.info(f"Loading historical baseline from {historical_path}")
    historical_df = pd.read_csv(historical_path)
    logger.info(f"  Baseline shape: {historical_df.shape}")

    logger.info(f"Loading current inference data from {current_path}")
    current_df = pd.read_csv(current_path)
    logger.info(f"  Current shape: {current_df.shape}")

    return historical_df, current_df


def simulate_inference(
    historical_df: pd.DataFrame,
    current_df: pd.DataFrame,
    features: list[str],
    batch_size: int,
    alpha: float,
) -> None:
    """Stream current_df in batches and run drift detection on each.

    Parameters
    ----------
    historical_df : pd.DataFrame
        The stable reference dataset (training distribution).
    current_df : pd.DataFrame
        The production dataset to stream through in chunks.
    features : list[str]
        Column names to monitor for drift.
    batch_size : int
        Number of rows per simulated inference batch.
    alpha : float
        Significance level for the KS test.
    """
    detector = DriftDetector(alpha=alpha)
    total_rows = len(current_df)
    num_batches = (total_rows + batch_size - 1) // batch_size

    logger.info(
        f"Starting inference simulation: {total_rows} rows "
        f"in {num_batches} batches of ~{batch_size}"
    )
    print("=" * 65)

    drift_triggered_batches = 0

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, total_rows)
        batch_df = current_df.iloc[start:end]

        print(f"\n--- Batch {batch_idx + 1}/{num_batches} "
              f"(rows {start}-{end - 1}) ---")

        report = detector.run(historical_df, batch_df, features=features)

        # Print the tabular summary for this batch
        print(report.summary())

        if report.drift_detected:
            drift_triggered_batches += 1
            print(
                f"\n>> ALERT: Drift detected in batch {batch_idx + 1}! "
                f"Drifted features: {report.drifted_features}"
            )

        # Small delay to simulate real-time processing
        time.sleep(0.3)

    # ---------------------------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("SIMULATION COMPLETE")
    print(f"  Total batches processed : {num_batches}")
    print(f"  Batches with drift      : {drift_triggered_batches}")
    print(
        f"  Drift rate              : "
        f"{drift_triggered_batches / num_batches * 100:.1f}%"
    )
    print("=" * 65)

    if drift_triggered_batches > 0:
        logger.warning(
            f"Drift was detected in {drift_triggered_batches}/{num_batches} "
            f"batches. Consider triggering a model retrain."
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 2: Simulate production inference and detect data drift"
    )
    parser.add_argument(
        "--historical",
        type=str,
        default=HISTORICAL_PATH,
        help="Path to the historical baseline CSV",
    )
    parser.add_argument(
        "--current",
        type=str,
        default=CURRENT_PATH,
        help="Path to the current / production CSV",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5000,
        help="Number of rows per simulated inference batch (default: 5000)",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=DEFAULT_FEATURES,
        help="Feature names to monitor for drift",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for the KS test (default: 0.05)",
    )

    args = parser.parse_args()

    historical_df, current_df = load_datasets(args.historical, args.current)
    simulate_inference(
        historical_df=historical_df,
        current_df=current_df,
        features=args.features,
        batch_size=args.batch_size,
        alpha=args.alpha,
    )
