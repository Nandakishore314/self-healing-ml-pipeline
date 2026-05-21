from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import pandas as pd

from src.app import app

client = TestClient(app)


def test_read_root_endpoint():
    """Test health check root response payload metadata."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "services" in data
    assert "artifacts" in data


@patch("os.path.exists")
def test_get_drift_status_missing_files(mock_exists):
    """Test drift route raises 400 error when data files are missing."""
    mock_exists.return_value = False
    response = client.get("/drift/status")
    assert response.status_code == 400
    assert "Requires both baseline" in response.json()["detail"]


@patch("os.path.exists")
@patch("pandas.read_csv")
@patch("src.app.DriftDetector")
def test_get_drift_status_success(mock_detector_class, mock_read_csv, mock_exists):
    """Test successful K-S statistical calculation payload mapping."""
    mock_exists.return_value = True

    mock_df = pd.DataFrame(
        {"AMT_INCOME_TOTAL": [1], "AMT_CREDIT": [1], "AMT_ANNUITY": [1]}
    )
    mock_read_csv.return_value = mock_df

    mock_report = MagicMock()
    mock_report.drift_detected = False
    mock_report.alpha = 0.05
    mock_report.drifted_features = []

    feature_mock = MagicMock()
    feature_mock.feature = "AMT_INCOME_TOTAL"
    feature_mock.ks_statistic = 0.1
    feature_mock.p_value = 0.8
    feature_mock.drift_detected = False
    mock_report.feature_results = [feature_mock]
    
    mock_detector_class.return_value.run.return_value = mock_report

    response = client.get("/drift/status?alpha=0.05")
    assert response.status_code == 200
    assert response.json()["drift_detected"] is False


@patch("os.path.exists")
def test_trigger_retraining_missing_source(mock_exists):
    """Test retraining route raises 400 error when source dataset path is invalid."""
    mock_exists.return_value = False
    response = client.post(
        "/train", json={"model_type": "rf", "data_source": "invalid.csv"}
    )
    assert response.status_code == 400


@patch("os.path.exists")
@patch("src.app.train_and_save_model")
def test_trigger_retraining_success(mock_train_func, mock_exists):
    """Test a valid pipeline training invocation loop maps to success payload."""
    mock_exists.return_value = True
    response = client.post(
        "/train", json={"model_type": "rf", "data_source": "valid.csv"}
    )
    assert response.status_code == 200
    assert response.json()["status"] == "success"


@patch("os.path.exists")
def test_predict_risk_no_model(mock_exists):
    """Test prediction route raises 404 error when serialization joblib asset is missing."""
    mock_exists.return_value = False
    payload = {
        "application_id": "APP-001",
        "AMT_INCOME_TOTAL": 100000.0,
        "AMT_CREDIT": 300000.0,
        "AMT_ANNUITY": 150000.0,
        "CNT_CHILDREN": 0,
        "DAYS_BIRTH": -10000,
        "DAYS_EMPLOYED": -1000,
        "CODE_GENDER": "M",
        "FLAG_OWN_CAR": "Y",
        "FLAG_OWN_REALTY": "Y",
        "NAME_INCOME_TYPE": "Working",
        "NAME_EDUCATION_TYPE": "Higher education",
        "NAME_FAMILY_STATUS": "Single",
        "NAME_HOUSING_TYPE": "House / apartment",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 404


@patch("os.path.exists")
@patch("joblib.load")
def test_predict_risk_success(mock_joblib_load, mock_exists):
    """Test evaluation logic runs inference vectors safely over mapped pipeline mocks."""
    mock_exists.return_value = True

    mock_pipeline = MagicMock()
    mock_pipeline.predict_proba.return_value = [[0.85, 0.15]]
    mock_pipeline.predict.return_value = [0]
    mock_joblib_load.return_value = mock_pipeline

    payload = {
        "application_id": "APP-TEST-999",
        "AMT_INCOME_TOTAL": 100000.0,
        "AMT_CREDIT": 300000.0,
        "AMT_ANNUITY": 150000.0,
        "CNT_CHILDREN": 0,
        "DAYS_BIRTH": -10000,
        "DAYS_EMPLOYED": -1000,
        "CODE_GENDER": "M",
        "FLAG_OWN_CAR": "Y",
        "FLAG_OWN_REALTY": "Y",
        "NAME_INCOME_TYPE": "Working",
        "NAME_EDUCATION_TYPE": "Higher education",
        "NAME_FAMILY_STATUS": "Single",
        "NAME_HOUSING_TYPE": "House / apartment",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "default_prediction" in response.json()
    assert response.json()["default_prediction"] == 0


@patch("os.path.exists")
@patch("joblib.load")
@patch("google.generativeai.GenerativeModel")
def test_chatbot_endpoint_with_memory(mock_gen_model, mock_joblib_load, mock_exists):
    """Test chatbot endpoint initializes memory and parses interactions safely using mocks."""
    mock_exists.return_value = True

    # 1. We must score the application first so it exists in application_registry
    mock_pipeline = MagicMock()
    mock_pipeline.predict_proba.return_value = [[0.85, 0.15]]
    mock_pipeline.predict.return_value = [0]
    mock_joblib_load.return_value = mock_pipeline

    payload = {
        "application_id": "APP-CHAT-123",
        "AMT_INCOME_TOTAL": 100000.0,
        "AMT_CREDIT": 300000.0,
        "AMT_ANNUITY": 150000.0,
        "CNT_CHILDREN": 0,
        "DAYS_BIRTH": -10000,
        "DAYS_EMPLOYED": -1000,
        "CODE_GENDER": "M",
        "FLAG_OWN_CAR": "Y",
        "FLAG_OWN_REALTY": "Y",
        "NAME_INCOME_TYPE": "Working",
        "NAME_EDUCATION_TYPE": "Higher education",
        "NAME_FAMILY_STATUS": "Single",
        "NAME_HOUSING_TYPE": "House / apartment",
    }
    pred_response = client.post("/predict", json=payload)
    assert pred_response.status_code == 200

    # Mock Gemini interaction structures to completely avoid API/Key dependency network loops
    mock_chat_instance = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Mocked LLM Credit Risk Analysis Response Statement."
    mock_chat_instance.send_message.return_value = mock_response
    mock_gen_model.return_value.start_chat.return_value = mock_chat_instance

    # 2. First chat session request
    chat_payload_1 = {
        "application_id": "APP-CHAT-123",
        "message": "Explain my current debt score structure.",
    }
    response_1 = client.post("/chat", json=chat_payload_1)
    assert response_1.status_code == 200
    data_1 = response_1.json()
    assert "response" in data_1
    
    # history_depth starts with context(2) + new turn(2) = 4
    history_depth_1 = len(data_1["history"])

    # 3. Second chat session request
    chat_payload_2 = {
        "application_id": "APP-CHAT-123",
        "message": "What about my income?",
    }
    response_2 = client.post("/chat", json=chat_payload_2)
    assert response_2.status_code == 200
    data_2 = response_2.json()
    history_depth_2 = len(data_2["history"])
    
    # Verify that history_depth increments properly across multiple session requests
    assert history_depth_2 > history_depth_1
