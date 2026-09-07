import pandas as pd
import numpy as np

# ============================================================
# 1. Load OLS forecasts
# ============================================================

data = pd.read_csv(
    r"C:\Users\vilke\OneDrive\Dokumentai\thesis_project\output\forecasts\forecasts_ols.csv"
)

data["date"] = pd.to_datetime(data["date"])


# ============================================================
# 2. Forecast errors
# ============================================================

# Historical mean errors
data["error_hist"] = (
    data["realized"] - data["historical_mean"]
)

# OLS errors
data["error_ols"] = (
    data["realized"] - data["ols"]
)

# Restricted OLS errors
data["error_ols_restricted"] = (
    data["realized"] - data["ols_restricted"]
)


# ============================================================
# 3. Squared forecast errors
# ============================================================

data["sq_error_hist"] = data["error_hist"] ** 2

data["sq_error_ols"] = data["error_ols"] ** 2

data["sq_error_ols_restricted"] = (
    data["error_ols_restricted"] ** 2
)


# ============================================================
# 4. Compute evaluation metrics
# ============================================================

results = []

for asset in data["asset"].unique():

    df = data[data["asset"] == asset]

    # --------------------------------------------------------
    # MSPE
    # --------------------------------------------------------

    mspe_hist = df["sq_error_hist"].mean()

    mspe_ols = df["sq_error_ols"].mean()

    mspe_ols_restricted = (
        df["sq_error_ols_restricted"].mean()
    )

    # --------------------------------------------------------
    # RMSPE
    # --------------------------------------------------------

    rmspe_hist = np.sqrt(mspe_hist)

    rmspe_ols = np.sqrt(mspe_ols)

    rmspe_ols_restricted = (
        np.sqrt(mspe_ols_restricted)
    )

    # --------------------------------------------------------
    # Out-of-sample R²
    # --------------------------------------------------------

    r2os_ols = 1 - (mspe_ols / mspe_hist)

    r2os_ols_restricted = (
        1 - (mspe_ols_restricted / mspe_hist)
    )

    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    results.append({
        "asset": asset,

        "MSPE_HistoricalMean": mspe_hist,
        "MSPE_OLS": mspe_ols,
        "MSPE_OLS_Restricted": mspe_ols_restricted,

        "RMSPE_HistoricalMean": rmspe_hist,
        "RMSPE_OLS": rmspe_ols,
        "RMSPE_OLS_Restricted": rmspe_ols_restricted,

        "R2OS_OLS": r2os_ols,
        "R2OS_OLS_Restricted": r2os_ols_restricted
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
    r"C:\Users\vilke\OneDrive\Dokumentai\thesis_project\output\forecasts\ols_evaluation.csv",
    index=False
)

print("\nOLS evaluation saved successfully.")