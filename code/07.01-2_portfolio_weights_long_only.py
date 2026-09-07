import pandas as pd
import numpy as np

from scipy.optimize import minimize


# ============================================================
# 1. File paths
# ============================================================

FORECAST_PATH = r"C:\Users\vilke\OneDrive\Dokumentai\thesis_project\output\forecasts\combined_forecasts_for_portfolio.csv"

COV_PATH = r"C:\Users\vilke\OneDrive\Dokumentai\thesis_project\output\forecasts\covariance_matrices.csv"

OUTPUT_PATH = r"C:\Users\vilke\OneDrive\Dokumentai\thesis_project\output\portfolios\portfolio_weights_long_only.csv"


# ============================================================
# 2. Load data
# ============================================================

forecasts = pd.read_csv(FORECAST_PATH)
forecasts["date"] = pd.to_datetime(forecasts["date"])

cov_long = pd.read_csv(COV_PATH)
cov_long["date"] = pd.to_datetime(cov_long["date"])


# ============================================================
# 3. Define assets and models
# ============================================================

asset_cols = [
    "NoDur_excess", "Durbl_excess", "Manuf_excess", "Enrgy_excess",
    "HiTec_excess", "Telcm_excess", "Shops_excess", "Hlth_excess",
    "Utils_excess", "Other_excess"
]

models = [
    "historical_mean",
    "ridge_restricted",
    "ols4_restricted",
    "random_forest"
]

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


def long_only_mean_variance_weights(mu_hat, sigma, risk_aversion=3):

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

    bounds = tuple((0, 1) for _ in range(n_assets))

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
# 5. Create long-only weights
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
            "asset": asset,
            "weight": weight
        })

    # Long-only model portfolios
    for model in models:

        mu_hat = forecast_month[model].values.astype(float)

        weights = long_only_mean_variance_weights(
            mu_hat=mu_hat,
            sigma=sigma,
            risk_aversion=risk_aversion
        )

        for asset, weight in zip(asset_cols, weights):

            all_weights.append({
                "date": current_date,
                "model": model,
                "asset": asset,
                "weight": weight
            })


# ============================================================
# 6. Save weights
# ============================================================

weights_df = pd.DataFrame(all_weights)
weights_df = weights_df.round(8)

weights_df.to_csv(OUTPUT_PATH, index=False)

print("Long-only portfolio weights saved successfully.")
print(weights_df.head(20))
print(weights_df.shape)