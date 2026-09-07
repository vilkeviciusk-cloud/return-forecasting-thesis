import pandas as pd
import numpy as np

# ============================================================
# 1. Load Elastic Net forecasts
# ============================================================

data = pd.read_csv(
    r"C:\Users\vilke\OneDrive\Dokumentai\thesis_project\output\forecasts\forecasts_elastic_net.csv"
)

data["date"] = pd.to_datetime(data["date"])


# ============================================================
# 2. Forecast errors
# ============================================================

# Historical mean errors
data["error_hist"] = (
    data["realized"] - data["historical_mean"]
)

# Elastic Net errors
data["error_enet"] = (
    data["realized"] - data["elastic_net"]
)

# Restricted Elastic Net errors
data["error_enet_restricted"] = (
    data["realized"] - data["elastic_net_restricted"]
)


# ============================================================
# 3. Squared forecast errors
# ============================================================

data["sq_error_hist"] = data["error_hist"] ** 2

data["sq_error_enet"] = data["error_enet"] ** 2

data["sq_error_enet_restricted"] = (
    data["error_enet_restricted"] ** 2
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

    mspe_enet = df["sq_error_enet"].mean()

    mspe_enet_restricted = (
        df["sq_error_enet_restricted"].mean()
    )

    # --------------------------------------------------------
    # RMSPE
    # --------------------------------------------------------

    rmspe_hist = np.sqrt(mspe_hist)

    rmspe_enet = np.sqrt(mspe_enet)

    rmspe_enet_restricted = (
        np.sqrt(mspe_enet_restricted)
    )

    # --------------------------------------------------------
    # Out-of-sample R²
    # --------------------------------------------------------

    r2os_enet = 1 - (mspe_enet / mspe_hist)

    r2os_enet_restricted = (
        1 - (mspe_enet_restricted / mspe_hist)
    )

    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    results.append({
        "asset": asset,

        "MSPE_HistoricalMean": mspe_hist,
        "MSPE_ElasticNet": mspe_enet,
        "MSPE_ElasticNetRestricted": mspe_enet_restricted,

        "RMSPE_HistoricalMean": rmspe_hist,
        "RMSPE_ElasticNet": rmspe_enet,
        "RMSPE_ElasticNetRestricted": rmspe_enet_restricted,

        "R2OS_ElasticNet": r2os_enet,
        "R2OS_ElasticNetRestricted": r2os_enet_restricted
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
    r"C:\Users\vilke\OneDrive\Dokumentai\thesis_project\output\forecasts\elastic_net_evaluation.csv",
    index=False
)

print("\nElastic Net evaluation saved successfully.")