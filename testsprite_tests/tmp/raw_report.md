
# TestSprite AI Testing Report(MCP)

---

## 1️⃣ Document Metadata
- **Project Name:** self-healing-ml-pipeline
- **Date:** 2026-05-18
- **Prepared by:** TestSprite AI Team

---

## 2️⃣ Requirement Validation Summary

#### Test TC001 get root api health check status
- **Test Code:** [TC001_get_root_api_health_check_status.py](./TC001_get_root_api_health_check_status.py)
- **Test Error:** Traceback (most recent call last):
  File "/var/task/handler.py", line 258, in run_with_retry
    exec(code, exec_env)
  File "<string>", line 34, in <module>
  File "<string>", line 31, in test_get_root_api_health_check_status
AssertionError: app_name expected 'Self-Healing ML Pipeline', got 'LendShield AI API Gateway'

- **Test Visualization and Result:** https://www.testsprite.com/dashboard/mcp/tests/adaf7d2c-7121-463e-ba8a-b646d877fe19/ef5f5284-cc10-424e-b167-72c2a520f1c9
- **Status:** ❌ Failed
- **Analysis / Findings:** {{TODO:AI_ANALYSIS}}.
---

#### Test TC002 get drift status ks test detection
- **Test Code:** [TC002_get_drift_status_ks_test_detection.py](./TC002_get_drift_status_ks_test_detection.py)
- **Test Visualization and Result:** https://www.testsprite.com/dashboard/mcp/tests/adaf7d2c-7121-463e-ba8a-b646d877fe19/c5c6e6cb-849d-4523-ab58-a35a13065672
- **Status:** ✅ Passed
- **Analysis / Findings:** {{TODO:AI_ANALYSIS}}.
---

#### Test TC003 post train trigger automated_retraining
- **Test Code:** [TC003_post_train_trigger_automated_retraining.py](./TC003_post_train_trigger_automated_retraining.py)
- **Test Error:** Traceback (most recent call last):
  File "/var/task/handler.py", line 258, in run_with_retry
    exec(code, exec_env)
  File "<string>", line 68, in <module>
  File "<string>", line 26, in test_post_train_trigger_automated_retraining
AssertionError: Expected 200, got 400

- **Test Visualization and Result:** https://www.testsprite.com/dashboard/mcp/tests/adaf7d2c-7121-463e-ba8a-b646d877fe19/05b9f0e5-5cc2-4d5e-9c74-d3b48bf46f68
- **Status:** ❌ Failed
- **Analysis / Findings:** {{TODO:AI_ANALYSIS}}.
---

#### Test TC004 post predict credit risk scoring
- **Test Code:** [TC004_post_predict_credit_risk_scoring.py](./TC004_post_predict_credit_risk_scoring.py)
- **Test Error:** Traceback (most recent call last):
  File "/var/task/handler.py", line 258, in run_with_retry
    exec(code, exec_env)
  File "<string>", line 61, in <module>
  File "<string>", line 36, in test_post_predict_credit_risk_scoring
AssertionError: Training failed with status code 400

- **Test Visualization and Result:** https://www.testsprite.com/dashboard/mcp/tests/adaf7d2c-7121-463e-ba8a-b646d877fe19/65fd1348-b34d-4259-a13b-0ba392c11931
- **Status:** ❌ Failed
- **Analysis / Findings:** {{TODO:AI_ANALYSIS}}.
---


## 3️⃣ Coverage & Matching Metrics

- **25.00** of tests passed

| Requirement        | Total Tests | ✅ Passed | ❌ Failed  |
|--------------------|-------------|-----------|------------|
| ...                | ...         | ...       | ...        |
---


## 4️⃣ Key Gaps / Risks
{AI_GNERATED_KET_GAPS_AND_RISKS}
---