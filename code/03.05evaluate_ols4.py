import pandas as pd
import numpy as np

# ============================================================
# 1. Load OLS-4 forecasts
# ============================================================

data = pd.read_csv(
    r"C:\Users\vilke\OneDrive\Dokumentai\thesis_project\output\forecasts\forecasts_ols4.csv"
)

data["date"] = pd.to_datetime(data["date"])


# ============================================================
# 2. Forecast errors
# ============================================================

data["error_hist"] = data["realized"] - data["historical_mean"]

data["error_ols4"] = data["realized"] - data["ols4"]

data["error_ols4_restricted"] = (
    data["realized"] - data["ols4_restricted"]
)


# ============================================================
# 3. Squared forecast errors
# ============================================================

data["sq_error_hist"] = data["error_hist"] ** 2

data["sq_error_ols4"] = data["error_ols4"] ** 2

data["sq_error_ols4_restricted"] = (
    data["error_ols4_restricted"] ** 2
)


# ============================================================
# 4. Compute MSPE, RMSPE, and R²OS
# ============================================================

results = []

for asset in data["asset"].unique():

    df = data[data["asset"] == asset]

    mspe_hist = df["sq_error_hist"].mean()
    mspe_ols4 = df["sq_error_ols4"].mean()
    mspe_ols4_restricted = df["sq_error_ols4_restricted"].mean()

    rmspe_hist = np.sqrt(mspe_hist)
    rmspe_ols4 = np.sqrt(mspe_ols4)
    rmspe_ols4_restricted = np.sqrt(mspe_ols4_restricted)

    r2os_ols4 = 1 - (mspe_ols4 / mspe_hist)
    r2os_ols4_restricted = 1 - (mspe_ols4_restricted / mspe_hist)

    results.append({
        "asset": asset,

        "MSPE_HistoricalMean": mspe_hist,
        "MSPE_OLS4": mspe_ols4,
        "MSPE_OLS4_Restricted": mspe_ols4_restricted,

        "RMSPE_HistoricalMean": rmspe_hist,
        "RMSPE_OLS4": rmspe_ols4,
        "RMSPE_OLS4_Restricted": rmspe_ols4_restricted,

        "R2OS_OLS4": r2os_ols4,
        "R2OS_OLS4_Restricted": r2os_ols4_restricted
    })


# ============================================================
# 5. Results table
# ============================================================

results_df = pd.DataFrame(results)
results_df = results_df.round(6)

print(results_df)


# ============================================================
# 6. Save evaluation results
# ============================================================

results_df.to_csv(
    r"C:\Users\vilke\OneDrive\Dokumentai\thesis_project\output\forecasts\ols4_evaluation.csv",
    index=False
)

print("\nOLS-4 evaluation saved successfully.")