***

# Product Requirements Document (PRD)

## Project Name: Self-Healing ML Observability & Drift Pipeline

---

## 1. Product Overview & Objective
Production machine learning models suffer from accuracy decay over time due to **data drift** (shifts in the underlying real-world data distributions). 

The objective of this project is to build an automated, closed-loop, **self-healing Machine Learning pipeline** for the **Home Credit Default Risk** dataset. The system must ingest live production streaming batches, mathematically measure distribution shifts against a training baseline using non-parametric statistical tests, alert operators of significant drift, and automatically trigger the model retraining pipeline to self-heal.

---

## 2. Target Features to Monitor
The system will prioritize monitoring continuous, high-impact numerical features relating to credit risk:
*   `AMT_INCOME_TOTAL` (Applicant total annual income)
*   `AMT_CREDIT` (Credit amount of the loan application)
*   `AMT_ANNUITY` (Annuity amount of the loan)

---

## 3. Core Functional Requirements

### FR-1: Data Preparation & Drift Injection
*   **Data Ingestion**: The system must download and cache the raw Home Credit Default Risk dataset (CSV format).
*   **Baseline Splitting**: The dataset must be split into two operational segments:
    1.  `historical_data.csv`: A stable, clean dataset representing the training baseline distribution (80% of rows).
    2.  `current_data.csv`: A dataset representing production runtime inputs (20% of rows).
*   **Controlled Drift Injection**: To validate the detector, the ingestion script must programmatically inject drift into `current_data.csv` (e.g., inflating `AMT_INCOME_TOTAL` by 5% and `AMT_CREDIT` by 3%) so we can verify that the detector performs under a realistic distribution shift.

### FR-2: Statistical Drift Detector
*   **Statistical Algorithm**: The engine must perform the **Two-Sample Kolmogorov-Smirnov (KS) Test** (`scipy.stats.ks_2samp`) on continuous features to compare baseline vs. production batches.
*   **Configurable Significance ($\alpha$)**: The detector must support a configurable significance level $\alpha$ (defaulting to `0.05`).
*   **Structured Reporting**: The output must return a structured report containing:
    - Overall boolean flag `drift_detected` (True if any monitored feature violates $\alpha$).
    - Feature-specific results including computed **KS-statistic**, **p-value**, and a boolean drift flag.
    - A clean, formatted tabular string summary for stdout logging.
*   **Auto Feature Selection**: If no explicit feature list is passed, the engine must automatically identify and monitor all numeric columns (using `select_dtypes(include='number')`).

### FR-3: Production Inference Simulation
*   **Streaming Emulation**: The system must read the active production file (`current_data.csv`) and process it in sequential chunks (defaulting to batches of 5,000 records).
*   **Tabular Telemetry Log**: For each incoming batch, the simulation must print a real-time console layout showing:
    - The batch index and record range.
    - Tabular stats per feature (KS Stat, p-value, Drift Status).
*   **Visual Alerting**: An explicit, eye-catching console warning must trigger if overall drift is flagged for the batch.

### FR-4: Automated Retraining (Self-Healing)
*   **Trigger Mechanism**: The pipeline must integrate a retraining trigger. If a production batch registers significant data drift, the system must automatically execute the training script (`train_pipeline.py`).
*   **Model Update**: The retraining orchestrator must build a fresh model pipeline (supporting Random Forest or XGBoost), impute missing values, scale numeric characteristics, train on the updated baseline data, and overwrite the active serialized model binary (`pipeline_model.joblib`).

---

## 4. Technical Stack & Environment
*   **Language**: Python 3.10+
*   **Libraries**:
    - Data Processing: `pandas`, `numpy`
    - Core ML & Pipeline: `scikit-learn`, `xgboost`, `joblib`
    - Scientific Math: `scipy` (for `stats.ks_2samp`)
    - API Gateway (Optional Wrapper): `FastAPI`, `uvicorn`
*   **Quality Assurance**:
    - Formatting & Style Guidelines: `ruff` (for formatting and lint checks)
    - Execution Engine: Local isolated Virtual Environment (`venv`)

---

## 5. Acceptance Criteria (Definition of Done)
1.  **Preparation validation**: Running `prepare_data.py` generates baseline and drifted CSV files without errors.
2.  **Telemetry verification**: Running `simulate_inference.py` processes the streaming batches, runs the KS-test, and prints the reports cleanly.
3.  **100% Drift Detection**: Given the 5% income and 3% credit inflation, the drift detector must report a 100% drift detection rate over the 11 simulated streaming batches.
4.  **Zero Linting/Formatting issues**: Running `ruff check` and `ruff format --check` must yield no errors.