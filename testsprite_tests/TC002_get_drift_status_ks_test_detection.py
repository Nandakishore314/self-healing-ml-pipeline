import requests

BASE_URL = "http://localhost:8000"
TIMEOUT = 30


def test_get_drift_status_ks_test_detection():
    url = f"{BASE_URL}/drift/status"
    params = {"alpha": 0.05}
    try:
        response = requests.get(url, params=params, timeout=TIMEOUT)
    except requests.RequestException as e:
        assert False, f"Request to {url} failed with exception: {e}"

    assert response.status_code in [200, 400, 500], (
        f"Unexpected status code: {response.status_code}"
    )

    if response.status_code == 200:
        try:
            data = response.json()
        except ValueError:
            assert False, "Response is not valid JSON"

        assert isinstance(data.get("drift_detected"), bool), (
            "'drift_detected' should be a boolean"
        )
        assert isinstance(data.get("alpha"), float), "'alpha' should be a float"
        assert abs(data.get("alpha") - params["alpha"]) < 1e-6, (
            f"'alpha' value mismatch: expected {params['alpha']}, got {data.get('alpha')}"
        )
        assert isinstance(data.get("drifted_features"), list), (
            "'drifted_features' should be a list"
        )
        for feature in data.get("drifted_features"):
            assert isinstance(feature, str), "Each drifted feature should be a string"
        assert isinstance(data.get("metrics"), list), (
            "'metrics' should be a list (array)"
        )
    elif response.status_code == 400:
        # Expect plain text message about missing baseline or production data files
        text = response.text.lower()
        assert "missing" in text and ("baseline" in text or "production" in text), (
            f"Unexpected 400 message: {response.text}"
        )
    elif response.status_code == 500:
        # Expect plain text about internal drift computation error
        text = response.text.lower()
        assert "internal" in text and "error" in text, (
            f"Unexpected 500 message: {response.text}"
        )


test_get_drift_status_ks_test_detection()
