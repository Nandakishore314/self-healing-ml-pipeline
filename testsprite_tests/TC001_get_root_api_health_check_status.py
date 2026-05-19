import requests


def test_get_root_api_health_check_status():
    base_url = "http://localhost:8000/"
    timeout = 30
    try:
        response = requests.get(base_url, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as e:
        assert False, f"Request failed: {e}"

    try:
        data = response.json()
    except ValueError:
        assert False, "Response is not valid JSON"

    # Validate keys presence
    assert "status" in data, "Missing 'status' key in response"
    assert "app_name" in data, "Missing 'app_name' key in response"
    assert "services" in data, "Missing 'services' key in response"
    assert "artifacts" in data, "Missing 'artifacts' key in response"

    # Validate types
    assert isinstance(data["status"], str), "'status' should be a string"
    assert isinstance(data["app_name"], str), "'app_name' should be a string"
    assert isinstance(data["services"], dict), "'services' should be an object/dict"
    assert isinstance(data["artifacts"], dict), "'artifacts' should be an object/dict"

    # Validate expected app name and status value
    expected_app_name = "Self-Healing ML Pipeline"
    assert data["app_name"] == expected_app_name, (
        f"app_name expected '{expected_app_name}', got '{data['app_name']}'"
    )
    assert data["status"].lower() == "healthy" or data["status"].lower() == "ok", (
        "status should indicate healthy or ok"
    )


test_get_root_api_health_check_status()
