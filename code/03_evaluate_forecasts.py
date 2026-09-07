import pandas as pd
import numpy as np

# ============================================================
# 1. Load forecast data
# ============================================================

data = pd.read_csv(
    r"C:\Users\vilke\OneDrive\Dokumentai\thesis_project\output\forecasts\forecasts_ridge.csv"
)

data["date"] = pd.to_datetime(data["date"])


# ============================================================
# 2. Forecast errors
# ============================================================

# Historical mean errors
data["error_hist"] = data["realized"] - data["historical_mean"]

# Ridge errors
data["error_ridge"] = data["realized"] - data["ridge"]

# Restricted Ridge errors
data["error_ridge_restricted"] = (
    data["realized"] - data["ridge_restricted"]
)


# ============================================================
# 3. Squared forecast errors
# ============================================================

data["sq_error_hist"] = data["error_hist"] ** 2

data["sq_error_ridge"] = data["error_ridge"] ** 2

data["sq_error_ridge_restricted"] = (
    data["error_ridge_restricted"] ** 2
)


# ============================================================
# 4. Compute MSPE and Out-of-Sample R²
# ============================================================

results = []

for asset in data["asset"].unique():

    df = data[data["asset"] == asset]

    # Mean squared prediction errors
    mspe_hist = df["sq_error_hist"].mean()

    mspe_ridge = df["sq_error_ridge"].mean()

    mspe_ridge_restricted = (
        df["sq_error_ridge_restricted"].mean()
    )

    # Out-of-sample R²
    r2os_ridge = 1 - (mspe_ridge / mspe_hist)

    r2os_ridge_restricted = (
        1 - (mspe_ridge_restricted / mspe_hist)
    )

    results.append({
        "asset": asset,

        "MSPE_HistoricalMean": mspe_hist,

        "MSPE_Ridge": mspe_ridge,

        "MSPE_RidgeRestricted": mspe_ridge_restricted,

        "R2OS_Ridge": r2os_ridge,

        "R2OS_RidgeRestricted": r2os_ridge_restricted
    })


# ============================================================
# 5. Final results table
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.round(6)

print(results_df)


# ============================================================
# 6. Save results
# ============================================================

results_df.to_csv(
    r"C:\Users\vilke\OneDrive\Dokumentai\thesis_project\output\forecasts\ridge_evaluation_480_5yeargap.csv",
    index=False
)

print("\nEvaluation results saved successfully.")