import pandas as pd
import numpy as np
from pathlib import Path
from scipy.optimize import minimize


# ============================================================
# 1. File paths
# ============================================================

ROBUSTNESS_DIR = Path(
    r"C:\Users\vilke\OneDrive\Dokumentai\thesis_project\output\forecasts\robustness"
)

FORECAST_PATH = ROBUSTNESS_DIR / "combined_forecasts_rolling_240_for_portfolio.csv"
COV_PATH = ROBUSTNESS_DIR / "covariance_matrices_rolling_240.csv"

OUTPUT_PATH = ROBUSTNESS_DIR / "portfolio_weights_rolling_cap20.csv"


# ============================================================
# 2. Load data
# ============================================================

forecasts = pd.read_csv(FORECAST_PATH)
forecasts["date"] = pd.to_datetime(forecasts["date"])

cov_long = pd.read_csv(COV_PATH)
cov_long["date"] = pd.to_datetime(cov_long["date"])


# ============================================================
# 3. Settings
# ============================================================

asset_cols = [
    "NoDur_excess", "Durbl_excess", "Manuf_excess", "Enrgy_excess",
    "HiTec_excess", "Telcm_excess", "Shops_excess", "Hlth_excess",
    "Utils_excess", "Other_excess"
]

models = [
    "historical_mean",
    "ols4_restricted",
    "ridge_restricted"
]

cap = 0.20
risk_aversion = 3


# ============================================================
# 4. Helper functions
# ============================================================

def get_covariance_matrix(cov_long, current_date, asset_cols):

    temp = cov_long[cov_long["date"] == current_date]

    sigma = temp.pivot(
        index="asset_i",
        columns="asset_j",
        values="covariance"
    )

    sigma = sigma.loc[asset_cols, asset_cols]

    return sigma.values


def capped_mean_variance_weights(mu_hat, sigma, cap, risk_aversion=3):

    mu_hat = np.asarray(mu_hat)
    sigma = np.asarray(sigma)

    n_assets = len(mu_hat)
    initial_weights = np.repeat(1 / n_assets, n_assets)

    def objective(weights):
        expected_return = weights @ mu_hat
        variance = weights @ sigma @ weights
        utility = expected_return - (risk_aversion / 2) * variance
        return -utility

    constraints = (
        {
            "type": "eq",
            "fun": lambda weights: np.sum(weights) - 1
        },
    )

    bounds = tuple((0, cap) for _ in range(n_assets))

    result = minimize(
        objective,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    if not result.success:
        return initial_weights

    return result.x


# ============================================================
# 5. Create portfolio weights
# ============================================================

all_weights = []

portfolio_dates = sorted(forecasts["date"].unique())
available_cov_dates = set(cov_long["date"].unique())

for current_date in portfolio_dates:

    if current_date not in available_cov_dates:
        continue

    forecast_month = forecasts[forecasts["date"] == current_date].copy()
    forecast_month = forecast_month.set_index("asset").loc[asset_cols]

    sigma = get_covariance_matrix(cov_long, current_date, asset_cols)

    # 1/N benchmark
    equal_weights = np.repeat(1 / len(asset_cols), len(asset_cols))

    for asset, weight in zip(asset_cols, equal_weights):

        all_weights.append({
            "date": current_date,
            "model": "one_over_n",
            "cap": "none",
            "asset": asset,
            "weight": weight
        })

    # Cap-20 model portfolios
    for model in models:

        mu_hat = forecast_month[model].values.astype(float)

        weights = capped_mean_variance_weights(
            mu_hat=mu_hat,
            sigma=sigma,
            cap=cap,
            risk_aversion=risk_aversion
        )

        model_name = f"{model}_cap_20_rolling"

        for asset, weight in zip(asset_cols, weights):

            all_weights.append({
                "date": current_date,
                "model": model_name,
                "cap": cap,
                "asset": asset,
                "weight": weight
            })


# ============================================================
# 6. Save
# ============================================================

weights_df = pd.DataFrame(all_weights)
weights_df = weights_df.round(8)

weights_df.to_csv(OUTPUT_PATH, index=False)

print("Rolling cap-20 portfolio weights saved successfully.")
print(weights_df.head(30))
print(weights_df.shape)
print(f"\nSaved as {OUTPUT_PATH}")