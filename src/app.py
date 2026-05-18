"""
LendShield AI — FastAPI Service
===============================
Provides REST API endpoints for real-time credit underwriting scoring,
observability drift telemetry, and self-healing automated retraining.
"""

import logging
import os
import sys
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.drift_detector import DriftDetector
from src.train_pipeline import train_and_save_model

# ---------------------------------------------------------------------------
# Setup logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="LendShield AI API Gateway",
    description="Statistical drift detection & automated retraining service for credit risk scoring",
    version="1.0.0",
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Paths and Defaults
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
HISTORICAL_PATH = os.path.join(DATA_DIR, "historical_data.csv")
CURRENT_PATH = os.path.join(DATA_DIR, "current_data.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "pipeline_model.joblib")

# Ensure directory for model exists
os.makedirs(MODEL_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------
class RetrainRequest(BaseModel):
    model_type: str = Field("rf", description="Model architecture type: 'rf' or 'xgb'")
    data_source: str = Field(HISTORICAL_PATH, description="Path or URL to dataset to train on")


class PredictionRequest(BaseModel):
    # Standard applicant features for default prediction
    AMT_INCOME_TOTAL: float = Field(..., examples=[150000.0])
    AMT_CREDIT: float = Field(..., examples=[500000.0])
    AMT_ANNUITY: float = Field(..., examples=[25000.0])
    CNT_CHILDREN: int = Field(0, examples=[1])
    DAYS_BIRTH: int = Field(-12000, examples=[-12000])
    DAYS_EMPLOYED: int = Field(-1500, examples=[-1500])
    CODE_GENDER: str = Field("F", examples=["F"])
    FLAG_OWN_CAR: str = Field("N", examples=["N"])
    FLAG_OWN_REALTY: str = Field("Y", examples=["Y"])
    NAME_INCOME_TYPE: str = Field("Working", examples=["Working"])
    NAME_EDUCATION_TYPE: str = Field("Secondary / special education", examples=["Secondary / special education"])
    NAME_FAMILY_STATUS: str = Field("Married", examples=["Married"])
    NAME_HOUSING_TYPE: str = Field("House / apartment", examples=["House / apartment"])


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def read_root():
    """Retrieve API status, registered metadata, and model status."""
    model_loaded = os.path.exists(MODEL_PATH)
    historical_exists = os.path.exists(HISTORICAL_PATH)
    current_exists = os.path.exists(CURRENT_PATH)

    return {
        "status": "healthy",
        "app_name": "Self-Healing ML Pipeline",
        "services": {
            "drift_detector": "active",
            "model_trainer": "active",
            "predictor": "active" if model_loaded else "inactive (no trained model found)"
        },
        "artifacts": {
            "model_present": model_loaded,
            "historical_baseline_present": historical_exists,
            "current_inference_present": current_exists,
        }
    }


@app.get("/drift/status")
def get_drift_status(alpha: float = 0.05):
    """Run Kolmogorov-Smirnov statistical test across baseline and batch inference datasets."""
    if not os.path.exists(HISTORICAL_PATH) or not os.path.exists(CURRENT_PATH):
        raise HTTPException(
            status_code=400,
            detail="Requires both baseline (historical_data.csv) and production (current_data.csv) to compute drift.",
        )

    try:
        baseline_df = pd.read_csv(HISTORICAL_PATH)
        current_df = pd.read_csv(CURRENT_PATH)

        # Monitor specific key credit application metrics
        features_to_monitor = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY"]
        features_in_both = [f for f in features_to_monitor if f in baseline_df.columns and f in current_df.columns]

        if not features_in_both:
            raise HTTPException(
                status_code=400,
                detail="Monitored features not found in baseline/production datasets.",
            )

        detector = DriftDetector(alpha=alpha)
        report = detector.run(baseline_df, current_df, features=features_in_both)

        results = []
        for res in report.feature_results:
            results.append({
                "feature": res.feature,
                "ks_statistic": float(res.ks_statistic),
                "p_value": float(res.p_value),
                "drift_detected": bool(res.drift_detected)
            })

        return {
            "drift_detected": bool(report.drift_detected),
            "alpha": float(report.alpha),
            "drifted_features": report.drifted_features,
            "metrics": results
        }

    except Exception as e:
        logger.error(f"Failed to calculate drift status: {e}")
        raise HTTPException(status_code=500, detail=f"Internal metrics extraction failed: {str(e)}")


@app.post("/train")
def trigger_retraining(req: RetrainRequest):
    """Trigger the automated training pipeline to train and serialize a new model version."""
    if not os.path.exists(req.data_source):
        raise HTTPException(
            status_code=400,
            detail=f"Requested data source does not exist: {req.data_source}"
        )

    try:
        logger.info(f"Triggering automated retraining: model_type={req.model_type}")
        train_and_save_model(
            data_source=req.data_source,
            target_column="TARGET",
            model_path=MODEL_PATH,
            model_type=req.model_type,
        )
        return {
            "status": "success",
            "message": "Automated pipeline finished successfully. New model serialised.",
            "model_path": MODEL_PATH,
            "model_type": req.model_type
        }
    except Exception as e:
        logger.error(f"Retraining failed: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline retraining execution error: {str(e)}")


@app.post("/predict")
def predict_risk(req: PredictionRequest):
    """Predict risk level and probability of default for a loan applicant."""
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(
            status_code=404,
            detail="Trained model model registry file not found. Call /train first to serialize a pipeline model."
        )

    try:
        # Load the serialized scikit-learn/XGBoost Pipeline
        pipeline = joblib.load(MODEL_PATH)
        
        # Turn JSON request body into a pandas single-row DataFrame
        applicant_data = pd.DataFrame([req.model_dump()])

        # Run scoring prediction
        prob = pipeline.predict_proba(applicant_data)[0][1]
        prediction = int(pipeline.predict(applicant_data)[0])

        return {
            "default_prediction": prediction,
            "default_probability": float(prob),
            "risk_assessment": "HIGH RISK" if prob > 0.3 else ("MODERATE RISK" if prob > 0.1 else "LOW RISK"),
            "approved": bool(prob < 0.2)
        }
    except Exception as e:
        logger.error(f"Prediction logic error: {e}")
        raise HTTPException(status_code=500, detail=f"Scoring engine execution error: {str(e)}")
