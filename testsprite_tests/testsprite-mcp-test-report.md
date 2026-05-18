# TestSprite AI Testing Report(MCP)

---

## 1️⃣ Document Metadata
- **Project Name:** self-healing-ml-pipeline
- **Date:** 2026-05-18
- **Prepared by:** TestSprite AI Team

---

## 2️⃣ Requirement Validation Summary

### Requirement: Health Check & Service Status

#### Test TC001 get root api health check status
- **Test Code:** [TC001_get_root_api_health_check_status.py](./TC001_get_root_api_health_check_status.py)
- **Test Error:** 
  ```text
  AssertionError: Expected app_name 'Self-Healing ML Pipeline', got 'LendShield AI API Gateway'
  ```
- **Test Visualization and Result:** https://www.testsprite.com/dashboard/mcp/tests/671896f0-b999-40cb-8e1c-ffd860464ccc/ebc8d228-70e0-4d46-8661-8c4f8921d221
- **Status:** ❌ Failed
- **Analysis / Findings:** The test failed due to a simple string mismatch. The FastAPI implementation in `src/app.py` names the application "LendShield AI API Gateway", but the PRD JSON used for test generation expected "Self-Healing ML Pipeline". The endpoint itself is functioning perfectly and returning a 200 OK with correct status fields.

---

### Requirement: Statistical Drift Detection

#### Test TC002 get drift status ks test detection
- **Test Code:** [TC002_get_drift_status_ks_test_detection.py](./TC002_get_drift_status_ks_test_detection.py)
- **Test Visualization and Result:** https://www.testsprite.com/dashboard/mcp/tests/671896f0-b999-40cb-8e1c-ffd860464ccc/d02afc9f-791b-4361-9a6a-072df7e5dbeb
- **Status:** ✅ Passed
- **Analysis / Findings:** The drift detection KS-test algorithm successfully executes on the provided baselines and successfully calculates the p-values across monitored features.

---

### Requirement: Automated Model Retraining (Self-Healing)

#### Test TC003 post train trigger automated_retraining
- **Test Code:** [TC003_post_train_trigger_automated_retraining.py](./TC003_post_train_trigger_automated_retraining.py)
- **Test Visualization and Result:** https://www.testsprite.com/dashboard/mcp/tests/671896f0-b999-40cb-8e1c-ffd860464ccc/3309fb7e-f96e-4ddd-a440-01d9f07cc511
- **Status:** ✅ Passed
- **Analysis / Findings:** The retraining trigger endpoint successfully orchestrated the pipeline re-training and serialized the updated model using the supplied data source.

---

### Requirement: Credit Risk Prediction

#### Test TC004 post predict credit risk scoring
- **Test Code:** [TC004_post_predict_credit_risk_scoring.py](./TC004_post_predict_credit_risk_scoring.py)
- **Test Error:** 
  ```text
  AssertionError: Model training failed or data source not found. Status: 400
  ```
- **Test Visualization and Result:** https://www.testsprite.com/dashboard/mcp/tests/671896f0-b999-40cb-8e1c-ffd860464ccc/7f6f1051-f53d-44cd-bf88-3f87141adfd3
- **Status:** ❌ Failed
- **Analysis / Findings:** The test logic for `/predict` appears to attempt triggering a prerequisite model training (`/train`) using a non-existent or default `data_source` that the server rejects with a 400 Error ("Requested data source does not exist"). The prediction endpoint itself might be fine, but the test setup sequence fails before it can execute the actual prediction.

---

## 3️⃣ Coverage & Matching Metrics

- **50.00%** of tests passed

| Requirement                          | Total Tests | ✅ Passed | ❌ Failed  |
|--------------------------------------|-------------|-----------|------------|
| Health Check & Service Status        | 1           | 0         | 1          |
| Statistical Drift Detection          | 1           | 1         | 0          |
| Automated Model Retraining           | 1           | 1         | 0          |
| Credit Risk Prediction               | 1           | 0         | 1          |

---

## 4️⃣ Key Gaps / Risks
1. **Hardcoded Naming Convention:** The PRD expectations vs. Actual implementation name diverges (`Self-Healing ML Pipeline` vs `LendShield AI API Gateway`). Either update the PRD to match the implementation or change the string in `src/app.py`.
2. **Missing Test Data:** Test TC004 relies on a pre-existing CSV file to run training, or it sends an invalid mock file path that the API appropriately rejects. The API logic is correct to reject missing files (400 Bad Request), but the test suite needs to be modified to either create a temp mock CSV or use the default real CSV (`d:\data\self-healing-ml-pipeline\data\historical_data.csv`).
3. **Synchronous Execution Block:** Both the drift processing and training pipeline currently execute synchronously. On slow hardware, this could lead to API timeouts when hit concurrently.
---
