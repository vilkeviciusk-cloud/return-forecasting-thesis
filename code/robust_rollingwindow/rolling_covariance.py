import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# 1. File paths
# ============================================================

DATA_PATH = r"C:\Users\vilke\OneDrive\Dokumentai\thesis_project\data\processed\dataset_thesis_baseline.csv"

ROBUSTNESS_DIR = Path(
    r"C:\Users\vilke\OneDrive\Dokumentai\thesis_project\output\forecasts\robustness"
)

FORECAST_PATH = ROBUSTNESS_DIR / "combined_forecasts_rolling_240_for_portfolio.csv"

COV_OUTPUT_PATH = ROBUSTNESS_DIR / "covariance_matrices_rolling_240.csv"


# ============================================================
# 2. Load data
# ============================================================

data = pd.read_csv(DATA_PATH)
data["date"] = pd.to_datetime(data["date"])

forecasts = pd.read_csv(FORECAST_PATH)
forecasts["date"] = pd.to_datetime(forecasts["date"])


# ============================================================
# 3. Define assets and settings
# ============================================================

asset_cols = [
    "NoDur_excess", "Durbl_excess", "Manuf_excess", "Enrgy_excess",
    "HiTec_excess", "Telcm_excess", "Shops_excess", "Hlth_excess",
    "Utils_excess", "Other_excess"
]

rolling_window = 240
portfolio_dates = sorted(forecasts["date"].unique())


# ============================================================
# 4. Create covariance matrices
# ============================================================

all_covariances = []

for current_date in portfolio_dates:

    current_index_list = data.index[data["date"] == current_date].tolist()

    if len(current_index_list) == 0:
        continue

    current_index = current_index_list[0]

    if current_index < rolling_window:
        continue

    past_returns = data.loc[
        current_index - rolling_window:current_index - 1,
        asset_cols
    ]

    sigma = past_returns.cov()

    for asset_i in asset_cols:
        for asset_j in asset_cols:

            all_covariances.append({
                "date": current_date,
                "asset_i": asset_i,
                "asset_j": asset_j,
                "covariance": sigma.loc[asset_i, asset_j]
            })


# ============================================================
# 5. Save
# ============================================================

covariances = pd.DataFrame(all_covariances)
covariances = covariances.round(10)

covariances.to_csv(COV_OUTPUT_PATH, index=False)

print("Rolling covariance matrices saved successfully.")
print(covariances.head())
print(covariances.shape)
print(f"\nSaved as {COV_OUTPUT_PATH}")