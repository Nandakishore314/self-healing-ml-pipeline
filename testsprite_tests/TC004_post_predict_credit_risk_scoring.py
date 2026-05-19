import requests

BASE_URL = "http://localhost:8000"
TIMEOUT = 30

def test_post_predict_credit_risk_scoring():
    predict_url = f"{BASE_URL}/predict"
    train_url = f"{BASE_URL}/train"

    # Sample applicant features matching request schema
    applicant_features = {
        "AMT_INCOME_TOTAL": 50000.0,
        "AMT_CREDIT": 200000.0,
        "AMT_ANNUITY": 15000.0,
        "CNT_CHILDREN": 1,
        "DAYS_BIRTH": -12000,
        "DAYS_EMPLOYED": 4000,
        "CODE_GENDER": "M",
        "FLAG_OWN_CAR": "Y",
        "FLAG_OWN_REALTY": "N",
        "NAME_INCOME_TYPE": "Working",
        "NAME_EDUCATION_TYPE": "Secondary / secondary special",
        "NAME_FAMILY_STATUS": "Single / not married",
        "NAME_HOUSING_TYPE": "Rented apartment"
    }

    # First, ensure that a trained model exists by triggering training
    train_payload = {
        "model_type": "rf",
        "data_source": "data/historical_data.csv"
    }


    try:
        train_response = requests.post(train_url, json=train_payload, timeout=TIMEOUT)
        assert train_response.status_code == 200, f"Training failed with status code {train_response.status_code}"
        train_json = train_response.json()
        assert "status" in train_json and isinstance(train_json["status"], str)
        assert "model_path" in train_json and isinstance(train_json["model_path"], str)
        assert "model_type" in train_json and isinstance(train_json["model_type"], str)

        # With trained model, test the predict endpoint success case
        predict_response = requests.post(predict_url, json=applicant_features, timeout=TIMEOUT)
        assert predict_response.status_code == 200, f"Predict failed with status code {predict_response.status_code}"
        predict_json = predict_response.json()
        # Validate response keys and types
        assert "default_prediction" in predict_json and isinstance(predict_json["default_prediction"], int)
        assert "default_probability" in predict_json and isinstance(predict_json["default_probability"], float)
        assert "risk_assessment" in predict_json and isinstance(predict_json["risk_assessment"], str)
        assert "approved" in predict_json and isinstance(predict_json["approved"], bool)

    finally:
        # Testing scoring engine execution error (500, 400 or 422 expected)
        invalid_features = applicant_features.copy()
        invalid_features["AMT_INCOME_TOTAL"] = "invalid_float"  # cause server-side scoring error

        predict_500_response = requests.post(predict_url, json=invalid_features, timeout=TIMEOUT)
        assert predict_500_response.status_code in (400, 422, 500), f"Expected error status 400 or 422 or 500, got {predict_500_response.status_code}"

test_post_predict_credit_risk_scoring()
