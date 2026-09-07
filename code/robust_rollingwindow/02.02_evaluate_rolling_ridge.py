import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# 1. File paths
# ============================================================

ROBUSTNESS_DIR = Path(
    r"C:\Users\vilke\OneDrive\Dokumentai\thesis_project\output\forecasts\robustness"
)

FORECAST_PATH = ROBUSTNESS_DIR / "forecasts_ridge_rolling_240.csv"

OUTPUT_PATH = ROBUSTNESS_DIR / "evaluation_ridge_rolling_240.csv"


# ============================================================
# 2. Load forecasts
# ============================================================

data = pd.read_csv(FORECAST_PATH)
data["date"] = pd.to_datetime(data["date"])


# ============================================================
# 3. Forecast errors
# ============================================================

data["error_hist"] = data["realized"] - data["historical_mean"]
data["error_ridge"] = data["realized"] - data["ridge"]
data["error_ridge_restricted"] = data["realized"] - data["ridge_restricted"]

data["sq_error_hist"] = data["error_hist"] ** 2
data["sq_error_ridge"] = data["error_ridge"] ** 2
data["sq_error_ridge_restricted"] = data["error_ridge_restricted"] ** 2


# ============================================================
# 4. Compute MSPE, RMSPE, R²OS
# ============================================================

results = []

for asset in data["asset"].unique():

    df = data[data["asset"] == asset]

    mspe_hist = df["sq_error_hist"].mean()
    mspe_ridge = df["sq_error_ridge"].mean()
    mspe_ridge_restricted = df["sq_error_ridge_restricted"].mean()

    results.append({
        "asset": asset,

        "MSPE_HistoricalMean": mspe_hist,
        "MSPE_Ridge": mspe_ridge,
        "MSPE_RidgeRestricted": mspe_ridge_restricted,

        "RMSPE_HistoricalMean": np.sqrt(mspe_hist),
        "RMSPE_Ridge": np.sqrt(mspe_ridge),
        "RMSPE_RidgeRestricted": np.sqrt(mspe_ridge_restricted),

        "R2OS_Ridge": 1 - (mspe_ridge / mspe_hist),
        "R2OS_RidgeRestricted": 1 - (mspe_ridge_restricted / mspe_hist)
    })


# ============================================================
# 5. Save results
# ============================================================

results_df = pd.DataFrame(results)
results_df = results_df.round(6)

results_df.to_csv(OUTPUT_PATH, index=False)

print("Rolling Ridge evaluation saved successfully.")
print(results_df)
print(f"\nSaved as {OUTPUT_PATH}")