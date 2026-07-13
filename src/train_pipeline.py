import argparse
import logging
import re
from dataclasses import dataclass, field, asdict
from typing import Optional

import gdown
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import xgboost as xgb

from src.augmentation import AugmentationConfig, apply_augmentation

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structured result dataclass
# ---------------------------------------------------------------------------

@dataclass
class TrainingResult:
    """Structured metrics from a single training run.

    Returned by ``train_and_save_model`` so callers (e.g. the experiment
    runner) can compare baseline vs augmented runs without parsing log lines.
    """

    accuracy: float = 0.0
    roc_auc: float = 0.0
    f1: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    train_rows: int = 0
    val_rows: int = 0
    model_type: str = "rf"
    augmented: bool = False
    augmentation_config: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)

    def log_summary(self, label: str = "") -> None:
        prefix = f"[{label}] " if label else ""
        logger.info(
            "%sAccuracy=%.4f  ROC-AUC=%.4f  F1=%.4f  "
            "Precision=%.4f  Recall=%.4f  "
            "train_rows=%d  val_rows=%d  augmented=%s",
            prefix,
            self.accuracy, self.roc_auc, self.f1,
            self.precision, self.recall,
            self.train_rows, self.val_rows, self.augmented,
        )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(data_source: str) -> pd.DataFrame:
    """
    Loads data from a local file or downwards from Google Drive.
    """
    input_path = data_source
    if data_source.startswith("http"):
        logger.info(f"Downloading dataset from URL: {data_source}")
        output_file = "downloaded_dataset.csv"

        # Extract ID if present in the URL
        match = re.search(r"/d/([a-zA-Z0-9_-]+)", data_source)
        if match:
            file_id = match.group(1)
            gdown.download(id=file_id, output=output_file, quiet=False)
        else:
            gdown.download(url=data_source, output=output_file, quiet=False)

        input_path = output_file

    logger.info(f"Loading dataset from {input_path}")
    df = pd.read_csv(input_path)
    logger.info(f"Dataset loaded with shape: {df.shape}")
    return df


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------

def build_pipeline(numeric_features, categorical_features, model_type="rf") -> Pipeline:
    """Builds the scikit-learn training pipeline."""
    logger.info(f"Building ML pipeline with {model_type}...")

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Choose model
    if model_type.lower() == "xgb":
        classifier = xgb.XGBClassifier(
            n_estimators=100, random_state=42, n_jobs=-1, eval_metric="logloss"
        )
    else:
        classifier = RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        )

    pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", classifier)]
    )

    return pipeline


# ---------------------------------------------------------------------------
# Training orchestrator
# ---------------------------------------------------------------------------

def train_and_save_model(
    data_source: str,
    target_column: str,
    model_path: str,
    model_type: str,
    aug_config: Optional[AugmentationConfig] = None,
    random_state: int = 42,
) -> TrainingResult:
    """Main training orchestrator.

    Parameters
    ----------
    data_source : str
        Local path or Google Drive URL to the CSV dataset.
    target_column : str
        Name of the label column.
    model_path : str
        Where to save the trained joblib pipeline.
    model_type : str
        ``"rf"`` or ``"xgb"``.
    aug_config : AugmentationConfig, optional
        When provided and ``aug_config.enabled=True``, augmentation is applied
        **only** to the training split before ``pipeline.fit``.
        The validation split is **never** modified.
    random_state : int
        Seed for ``train_test_split``.  Fixing this value across runs ensures
        the baseline and augmented experiments see the identical val set.

    Returns
    -------
    TrainingResult
        Structured metrics dict for comparison.
    """
    # 1. Load Data
    df = load_data(data_source)

    # Ensure target column exists
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in the dataset.")

    # Drop rows where target is missing
    df = df.dropna(subset=[target_column])

    # 2. Split features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Using stratify ensures balanced train/test split for imbalanced datasets like Home Credit Risk
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    logger.info(f"Training set shape: {X_train.shape}, Val set shape: {X_val.shape}")

    # 3. Apply augmentation ONLY to training data
    #    The val split (X_val, y_val) is never passed into apply_augmentation.
    augmentation_config_dict: dict = {}
    if aug_config is not None and aug_config.enabled:
        logger.info("Augmentation enabled – applying to training split only.")
        X_train, y_train = apply_augmentation(
            X_train,
            y_train,
            aug_config,
            val_index=X_val.index,  # leakage guard: val indices are checked
        )
        augmentation_config_dict = {
            "enabled": aug_config.enabled,
            "method": aug_config.method,
            "seed": aug_config.seed,
            "noise_sigma": aug_config.noise_sigma,
            "noise_fraction": aug_config.noise_fraction,
        }
        logger.info(f"Post-augmentation training shape: {X_train.shape}")
    else:
        logger.info("Augmentation disabled – training on raw data.")

    # 4. Identify numeric and categorical columns (from training set AFTER augmentation)
    numeric_features = X_train.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()
    categorical_features = X_train.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    logger.info(
        f"Found {len(numeric_features)} numeric and {len(categorical_features)} categorical features."
    )

    # 5. Build Pipeline
    pipeline = build_pipeline(numeric_features, categorical_features, model_type)

    # 6. Train Model  (only X_train / y_train — val never touches .fit())
    logger.info("Training model...")
    pipeline.fit(X_train, y_train)

    # 7. Evaluate on the held-out validation set
    y_pred = pipeline.predict(X_val)
    y_proba = (
        pipeline.predict_proba(X_val)[:, 1]
        if hasattr(pipeline, "predict_proba")
        else None
    )

    result = TrainingResult(
        accuracy=accuracy_score(y_val, y_pred),
        roc_auc=roc_auc_score(y_val, y_proba) if y_proba is not None else float("nan"),
        f1=f1_score(y_val, y_pred, zero_division=0),
        precision=precision_score(y_val, y_pred, zero_division=0),
        recall=recall_score(y_val, y_pred, zero_division=0),
        train_rows=len(X_train),
        val_rows=len(X_val),
        model_type=model_type,
        augmented=aug_config is not None and aug_config.enabled,
        augmentation_config=augmentation_config_dict,
    )
    result.log_summary(label="augmented" if result.augmented else "baseline")

    # 8. Save Model
    logger.info(f"Saving pipeline to {model_path}...")
    joblib.dump(pipeline, model_path)
    logger.info("Pipeline saved successfully.")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1: ML Model Training Pipeline")
    parser.add_argument(
        "--data-source",
        type=str,
        required=True,
        help="Local path or Google Drive URL to the CSV dataset",
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default="TARGET",
        help="Target column name (e.g., TARGET for Home Credit risk)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="pipeline_model.joblib",
        help="Path to save the joblib pipeline",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["rf", "xgb"],
        default="rf",
        help="Model type to train: 'rf' for Random Forest, 'xgb' for XGBoost",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        default=False,
        help="Enable Gaussian-noise augmentation on the training split",
    )
    parser.add_argument(
        "--aug-sigma",
        type=float,
        default=0.05,
        help="Noise sigma (fraction of each column std-dev) when augmentation is on",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for train/val split (keep fixed across experiments)",
    )

    args = parser.parse_args()

    aug_cfg = (
        AugmentationConfig(enabled=True, method="gaussian_noise",
                           seed=args.random_state, noise_sigma=args.aug_sigma)
        if args.augment
        else AugmentationConfig(enabled=False)
    )

    train_and_save_model(
        args.data_source,
        args.target_column,
        args.model_path,
        args.model_type,
        aug_config=aug_cfg,
        random_state=args.random_state,
    )
