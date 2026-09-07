import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# 1. File paths
# ============================================================

ROBUSTNESS_DIR = Path(
    r"C:\Users\vilke\OneDrive\Dokumentai\thesis_project\output\forecasts\robustness"
)

FORECAST_PATH = ROBUSTNESS_DIR / "forecasts_random_forest_rolling_240.csv"

OUTPUT_PATH = ROBUSTNESS_DIR / "evaluation_random_forest_rolling_240.csv"


# ============================================================
# 2. Load forecasts
# ============================================================

data = pd.read_csv(FORECAST_PATH)
data["date"] = pd.to_datetime(data["date"])


# ============================================================
# 3. Forecast errors
# ============================================================

data["error_hist"] = data["realized"] - data["historical_mean"]
data["error_rf"] = data["realized"] - data["random_forest"]

data["sq_error_hist"] = data["error_hist"] ** 2
data["sq_error_rf"] = data["error_rf"] ** 2


# ============================================================
# 4. Compute MSPE, RMSPE, R²OS
# ============================================================

results = []

for asset in data["asset"].unique():

    df = data[data["asset"] == asset]

    mspe_hist = df["sq_error_hist"].mean()
    mspe_rf = df["sq_error_rf"].mean()

    results.append({
        "asset": asset,

        "MSPE_HistoricalMean": mspe_hist,
        "MSPE_RandomForest": mspe_rf,

        "RMSPE_HistoricalMean": np.sqrt(mspe_hist),
        "RMSPE_RandomForest": np.sqrt(mspe_rf),

        "R2OS_RandomForest": 1 - (mspe_rf / mspe_hist)
    })


# ============================================================
# 5. Save results
# ============================================================

results_df = pd.DataFrame(results)
results_df = results_df.round(6)

results_df.to_csv(OUTPUT_PATH, index=False)

print("Rolling Random Forest evaluation saved successfully.")
print(results_df)
print(f"\nSaved as {OUTPUT_PATH}")