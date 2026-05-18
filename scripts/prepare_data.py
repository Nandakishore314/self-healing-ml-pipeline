"""
Dataset Selection & EDA – Data Preparation Script
==================================================
Dataset : Home Credit Default Risk (House Crime / Credit Risk)
Task    : Binary classification (TARGET: 0 = repay, 1 = default)
Outputs :
  data/historical_data.csv – 80% of the cleaned data (used for model training)
  data/current_data.csv    – 20% of the cleaned data with slight distribution
                             shifts introduced to simulate real-world data drift
"""

import os
import re
import gdown
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# 1. Load the raw dataset
# ---------------------------------------------------------------------------
# The dataset is hosted on Google Drive based on the previous workflow.
# URL from previous context: https://drive.google.com/file/d/1RckAzPnNiDHeeFJ6PVbZka62-099ZiWd/view?usp=drive_link
DATA_SOURCE = "https://drive.google.com/file/d/1RckAzPnNiDHeeFJ6PVbZka62-099ZiWd/view?usp=drive_link"

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)
RAW_PATH = os.path.join(RAW_DIR, "downloaded_dataset.csv")

print("Loading dataset from Google Drive...")
if not os.path.exists(RAW_PATH):
    match = re.search(r"/d/([a-zA-Z0-9_-]+)", DATA_SOURCE)
    if match:
        file_id = match.group(1)
        gdown.download(id=file_id, output=RAW_PATH, quiet=False)
    else:
        gdown.download(url=DATA_SOURCE, output=RAW_PATH, quiet=False)
else:
    print(f"  File already exists at {RAW_PATH}, skipping download.")

df_raw = pd.read_csv(RAW_PATH)
print(f"  Loaded dataset ({len(df_raw)} rows).")

# ---------------------------------------------------------------------------
# 2. Identify and handle missing values & Preprocessing
# ---------------------------------------------------------------------------
df = df_raw.copy()

# Drop rows where target is missing
if "TARGET" in df.columns:
    df = df.dropna(subset=["TARGET"])

# For demonstration, we'll keep a subset of highly predictive or common features
# to keep processing fast and EDA clean, avoiding loading all 120+ columns.
# We'll keep some demographics and external sources.
columns_to_keep = [
    "TARGET",
    "NAME_CONTRACT_TYPE",
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "CNT_CHILDREN",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
]

# Only keep columns that actually exist in the downloaded dataset
existing_cols = [col for col in columns_to_keep if col in df.columns]
df = df[existing_cols]

# We will let the pipeline handle the heavy imputation, but for the sake of
# drift simulation, we'll drop extreme nulls in the key drift columns to be safe.
if "AMT_INCOME_TOTAL" in df.columns and "AMT_CREDIT" in df.columns:
    df = df.dropna(subset=["AMT_INCOME_TOTAL", "AMT_CREDIT"])

print("\nData subset shape:", df.shape)

# ---------------------------------------------------------------------------
# 3. Train / test split
#    historical_data : 80 % – used for model training / EDA
#    current_data    : 20 % – used for drift detection later
#    Stratify by TARGET so both splits share the same class balance.
# ---------------------------------------------------------------------------
historical_df, current_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["TARGET"] if "TARGET" in df.columns else None,
)

# ---------------------------------------------------------------------------
# 4. Introduce a subtle distribution shift in current_data to simulate
#    production data drift over time (e.g., inflation affecting income/credit).
#    • AMT_INCOME_TOTAL shifted up by ~5 %
#    • AMT_CREDIT shifted up by ~3 %
# ---------------------------------------------------------------------------
DRIFT_ROW_FRACTION = 0.50  # fraction of current_data rows that receive the shift
INCOME_DRIFT_FACTOR = 1.05  # +5 % – simulates economic inflation
CREDIT_DRIFT_FACTOR = 1.03  # +3 % – simulates increased credit limits over time

rng = np.random.default_rng(seed=0)

drift_mask = rng.random(len(current_df)) < DRIFT_ROW_FRACTION
current_df = current_df.copy()

if "AMT_INCOME_TOTAL" in current_df.columns:
    current_df.loc[drift_mask, "AMT_INCOME_TOTAL"] = (
        current_df.loc[drift_mask, "AMT_INCOME_TOTAL"] * INCOME_DRIFT_FACTOR
    )

if "AMT_CREDIT" in current_df.columns:
    current_df.loc[drift_mask, "AMT_CREDIT"] = (
        current_df.loc[drift_mask, "AMT_CREDIT"] * CREDIT_DRIFT_FACTOR
    )

# ---------------------------------------------------------------------------
# 5. Save the output files
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

historical_path = os.path.join(DATA_DIR, "historical_data.csv")
current_path = os.path.join(DATA_DIR, "current_data.csv")

historical_df.to_csv(historical_path, index=False)
current_df.to_csv(current_path, index=False)

print("\nOutput files written:")
print(f"  historical_data.csv : {len(historical_df)} rows -> {historical_path}")
print(f"  current_data.csv    : {len(current_df)} rows -> {current_path}")

# ---------------------------------------------------------------------------
# 6. Quick sanity summary
# ---------------------------------------------------------------------------
print("\n--- historical_data.csv target distribution ---")
if "TARGET" in historical_df.columns:
    print(historical_df["TARGET"].value_counts(normalize=True).round(3).to_string())

print("\nData preparation complete!")
