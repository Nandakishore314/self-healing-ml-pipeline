import numpy as np
import pandas as pd

from src.drift_detector import DriftDetector


def test_identical_arrays_no_drift():
    """Test that two identical arrays do not trigger drift detection."""
    np.random.seed(42)
    data = np.random.normal(0, 1, 1000)

    baseline_df = pd.DataFrame({"feature1": data})
    current_df = pd.DataFrame({"feature1": data})

    detector = DriftDetector(alpha=0.05)
    report = detector.run(baseline_df, current_df, features=["feature1"])

    assert not report.drift_detected, (
        "Identical arrays should not trigger drift detection"
    )


def test_shifted_arrays_with_drift():
    """Test that a normal array and a heavily shifted/multiplied array trigger drift detection."""
    np.random.seed(42)
    data = np.random.normal(0, 1, 1000)

    baseline_df = pd.DataFrame({"feature1": data})

    # Create a heavily shifted array
    shifted_data = data * 5 + 10
    current_df = pd.DataFrame({"feature1": shifted_data})

    detector = DriftDetector(alpha=0.05)
    report = detector.run(baseline_df, current_df, features=["feature1"])

    assert report.drift_detected, "Heavily shifted array should trigger drift detection"
