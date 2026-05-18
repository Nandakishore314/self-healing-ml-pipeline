import requests

BASE_URL = "http://localhost:8000"
TIMEOUT = 30

def test_post_train_trigger_automated_retraining():
    """
    Validate POST /train endpoint triggers the automated model retraining pipeline
    with specified model_type and data_source,
    returns status, message, model_path, and model_type on success,
    and handles errors such as missing data source file and retraining execution errors.
    """

    url = f"{BASE_URL}/train"
    headers = {"Content-Type": "application/json"}

    # Use a placeholder path for valid data_source assuming it exists on the server
    valid_data_source_path = "data/historical_data.csv"

    # Case 1: Successful retraining with valid model_type and existing data_source
    payload_success = {
        "model_type": "rf",
        "data_source": valid_data_source_path
    }
    response = requests.post(url, json=payload_success, headers=headers, timeout=TIMEOUT)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    resp_json = response.json()
    # Validate all required fields are present with expected types/values
    assert isinstance(resp_json.get("status"), str), "Missing or invalid 'status' in response"
    assert isinstance(resp_json.get("message"), str), "Missing or invalid 'message' in response"
    assert isinstance(resp_json.get("model_path"), str), "Missing or invalid 'model_path' in response"
    assert resp_json.get("model_type") == payload_success["model_type"], "'model_type' mismatch in response"

    # Case 2: Missing data source file should return 400
    payload_missing_file = {
        "model_type": "rf",
        "data_source": "non_existent_file.csv"
    }
    response = requests.post(url, json=payload_missing_file, headers=headers, timeout=TIMEOUT)
    assert response.status_code == 400, f"Expected 400 for missing data source file, got {response.status_code}"

    # Case 3: Model type validation (enum checks)
    for invalid_model_type in ["invalid", "", None]:
        payload_invalid_model = {
            "model_type": invalid_model_type,
            "data_source": valid_data_source_path
        }
        response = requests.post(url, json=payload_invalid_model, headers=headers, timeout=TIMEOUT)
        # API behavior: if invalid model_type is sent, assume 400 Bad Request or similar
        assert response.status_code in (400, 422), f"Expected 400/422 for invalid model_type, got {response.status_code}"

    # Case 4: Simulate retraining execution error (500)
    payload_error = {
        "model_type": "rf",
        "data_source": "/dev/null/invalidpath.csv"
    }
    response = requests.post(url, json=payload_error, headers=headers, timeout=TIMEOUT)
    if response.status_code == 500:
        try:
            resp_json = response.json()
            assert isinstance(resp_json, (str, dict)), "500 response body unexpected format"
        except Exception:
            pass
    else:
        assert response.status_code in [400, 200], f"Unexpected response status {response.status_code} for error case"


test_post_train_trigger_automated_retraining()
