import pandas as pd
import numpy as np

# ============================================================
# 1. Load Random Forest forecasts and historical mean benchmark
# ============================================================

rf = pd.read_csv(
    r"C:\Users\vilke\OneDrive\Dokumentai\thesis_project\output\forecasts\forecasts_random_forest.csv"
)

ols = pd.read_csv(
    r"C:\Users\vilke\OneDrive\Dokumentai\thesis_project\output\forecasts\forecasts_ols.csv"
)

rf["date"] = pd.to_datetime(rf["date"])
ols["date"] = pd.to_datetime(ols["date"])


# ============================================================
# 2. Merge RF forecasts with the historical mean benchmark
#    used in the main evaluation
# ============================================================

data = rf[
    [
        "date",
        "asset",
        "realized",
        "random_forest"
    ]
].copy()

benchmark = ols[
    [
        "date",
        "asset",
        "historical_mean"
    ]
].copy()

data = data.merge(
    benchmark,
    on=["date", "asset"],
    how="inner"
)

data = data.dropna(subset=["realized", "random_forest", "historical_mean"])
data = data.sort_values(["asset", "date"]).reset_index(drop=True)


# ============================================================
# 3. Forecast errors
# ============================================================

data["error_hist"] = data["realized"] - data["historical_mean"]
data["error_rf"] = data["realized"] - data["random_forest"]

data["sq_error_hist"] = data["error_hist"] ** 2
data["sq_error_rf"] = data["error_rf"] ** 2


# ============================================================
# 4. Compute industry-level MSPE, RMSPE, and R²OS
# ============================================================

results = []

for asset in data["asset"].unique():

    df = data[data["asset"] == asset].copy()

    mspe_hist = df["sq_error_hist"].mean()
    mspe_rf = df["sq_error_rf"].mean()

    rmspe_hist = np.sqrt(mspe_hist)
    rmspe_rf = np.sqrt(mspe_rf)

    r2os_rf = 1 - (mspe_rf / mspe_hist)

    results.append({
        "asset": asset,
        "MSPE_HistoricalMean": mspe_hist,
        "MSPE_RandomForest": mspe_rf,
        "RMSPE_HistoricalMean": rmspe_hist,
        "RMSPE_RandomForest": rmspe_rf,
        "R2OS_RandomForest": r2os_rf
    })


# ============================================================
# 5. Results table
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values("asset").reset_index(drop=True)
results_df = results_df.round(6)

print(results_df)


# ============================================================
# 6. Save evaluation table
# ============================================================

results_df.to_csv(
    r"C:\Users\vilke\OneDrive\Dokumentai\thesis_project\output\forecasts\random_forest_evaluation.csv",
    index=False
)

print("\nRandom Forest evaluation saved successfully.")