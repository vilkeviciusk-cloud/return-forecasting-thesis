import pandas as pd
import numpy as np
from pathlib import Path
from scipy.optimize import minimize


# ============================================================
# 1. File paths
# ============================================================

FORECAST_PATH = r"C:\Users\vilke\OneDrive\Dokumentai\thesis_project\output\forecasts\combined_forecasts_for_portfolio.csv"

INV_COV_PATH = r"C:\Users\vilke\OneDrive\Dokumentai\thesis_project\output\forecasts\inverse_covariance_matrices.csv"

COV_PATH = r"C:\Users\vilke\OneDrive\Dokumentai\thesis_project\output\forecasts\covariance_matrices.csv"

OUTPUT_DIR = Path(
    r"C:\Users\vilke\OneDrive\Dokumentai\thesis_project\output\portfolios"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Load data
# ============================================================

forecasts = pd.read_csv(FORECAST_PATH)
forecasts["date"] = pd.to_datetime(forecasts["date"])

inv_cov_long = pd.read_csv(INV_COV_PATH)
inv_cov_long["date"] = pd.to_datetime(inv_cov_long["date"])

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
    "ols_restricted",
    "ols4_restricted",
    "ridge_restricted",
    "elastic_net_restricted",
    "random_forest"
]

risk_aversion = 3

caps = [0.20, 0.30, 0.50]
lambdas = [0.25, 0.50, 0.75]

equal_weight = 1 / len(asset_cols)


# ============================================================
# 4. Helper functions
# ============================================================

def get_inverse_covariance_matrix(inv_cov_long, current_date, asset_cols):

    temp = inv_cov_long[inv_cov_long["date"] == current_date]

    inv_sigma = temp.pivot(
        index="asset_i",
        columns="asset_j",
        values="inverse_covariance"
    )

    inv_sigma = inv_sigma.loc[asset_cols, asset_cols]

    return inv_sigma.values


def get_covariance_matrix(cov_long, current_date, asset_cols):

    temp = cov_long[cov_long["date"] == current_date]

    sigma = temp.pivot(
        index="asset_i",
        columns="asset_j",
        values="covariance"
    )

    sigma = sigma.loc[asset_cols, asset_cols]

    return sigma.values


def unrestricted_mean_variance_weights(mu_hat, inv_sigma):

    mu_hat = np.asarray(mu_hat)

    raw_weights = inv_sigma @ mu_hat
    weight_sum = raw_weights.sum()

    if np.abs(weight_sum) < 1e-10:
        return np.repeat(equal_weight, len(mu_hat))

    return raw_weights / weight_sum


def constrained_mean_variance_weights(mu_hat, sigma, lower_bound=0.0, upper_bound=1.0):

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

    bounds = tuple((lower_bound, upper_bound) for _ in range(n_assets))

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


def cap_label(cap):
    return int(cap * 100)


# ============================================================
# 5. Create unrestricted weights
# ============================================================

all_unrestricted = []

portfolio_dates = sorted(forecasts["date"].unique())
available_inv_cov_dates = set(inv_cov_long["date"].unique())

for current_date in portfolio_dates:

    if current_date not in available_inv_cov_dates:
        continue

    forecast_month = forecasts[forecasts["date"] == current_date].copy()
    forecast_month = forecast_month.set_index("asset").loc[asset_cols]

    inv_sigma = get_inverse_covariance_matrix(
        inv_cov_long,
        current_date,
        asset_cols
    )

    # 1/N
    for asset in asset_cols:
        all_unrestricted.append({
            "date": current_date,
            "model": "one_over_n",
            "asset": asset,
            "weight": equal_weight
        })

    # Forecast-based unrestricted portfolios
    for model in models:

        if model not in forecast_month.columns:
            print(f"Skipping {model}: column not found.")
            continue

        mu_hat = forecast_month[model].values.astype(float)

        weights = unrestricted_mean_variance_weights(mu_hat, inv_sigma)

        for asset, weight in zip(asset_cols, weights):
            all_unrestricted.append({
                "date": current_date,
                "model": model,
                "asset": asset,
                "weight": weight
            })


weights_unrestricted = pd.DataFrame(all_unrestricted).round(8)
weights_unrestricted.to_csv(OUTPUT_DIR / "portfolio_weights.csv", index=False)

print("Saved unrestricted weights.")
print(weights_unrestricted.shape)


# ============================================================
# 6. Create long-only weights
# ============================================================

all_long_only = []

available_cov_dates = set(cov_long["date"].unique())

for current_date in portfolio_dates:

    if current_date not in available_cov_dates:
        continue

    forecast_month = forecasts[forecasts["date"] == current_date].copy()
    forecast_month = forecast_month.set_index("asset").loc[asset_cols]

    sigma = get_covariance_matrix(
        cov_long,
        current_date,
        asset_cols
    )

    # 1/N
    for asset in asset_cols:
        all_long_only.append({
            "date": current_date,
            "model": "one_over_n",
            "asset": asset,
            "weight": equal_weight
        })

    # Forecast-based long-only portfolios
    for model in models:

        if model not in forecast_month.columns:
            print(f"Skipping {model}: column not found.")
            continue

        mu_hat = forecast_month[model].values.astype(float)

        weights = constrained_mean_variance_weights(
            mu_hat=mu_hat,
            sigma=sigma,
            lower_bound=0.0,
            upper_bound=1.0
        )

        for asset, weight in zip(asset_cols, weights):
            all_long_only.append({
                "date": current_date,
                "model": model,
                "asset": asset,
                "weight": weight
            })


weights_long_only = pd.DataFrame(all_long_only).round(8)
weights_long_only.to_csv(OUTPUT_DIR / "portfolio_weights_long_only.csv", index=False)

print("Saved long-only weights.")
print(weights_long_only.shape)


# ============================================================
# 7. Create capped weights
# ============================================================

all_capped = []

for current_date in portfolio_dates:

    if current_date not in available_cov_dates:
        continue

    forecast_month = forecasts[forecasts["date"] == current_date].copy()
    forecast_month = forecast_month.set_index("asset").loc[asset_cols]

    sigma = get_covariance_matrix(
        cov_long,
        current_date,
        asset_cols
    )

    # 1/N
    for asset in asset_cols:
        all_capped.append({
            "date": current_date,
            "model": "one_over_n",
            "cap": "none",
            "asset": asset,
            "weight": equal_weight
        })

    # Forecast-based capped portfolios
    for model in models:

        if model not in forecast_month.columns:
            print(f"Skipping {model}: column not found.")
            continue

        mu_hat = forecast_month[model].values.astype(float)

        for cap in caps:

            weights = constrained_mean_variance_weights(
                mu_hat=mu_hat,
                sigma=sigma,
                lower_bound=0.0,
                upper_bound=cap
            )

            model_name = f"{model}_cap_{cap_label(cap)}"

            for asset, weight in zip(asset_cols, weights):
                all_capped.append({
                    "date": current_date,
                    "model": model_name,
                    "cap": cap,
                    "asset": asset,
                    "weight": weight
                })


weights_capped = pd.DataFrame(all_capped).round(8)
weights_capped.to_csv(OUTPUT_DIR / "portfolio_weights_capped.csv", index=False)

print("Saved capped weights.")
print(weights_capped.shape)


# ============================================================
# 8. Create lambda long-only weights
# ============================================================

all_lambda_long = []

one_over_n_long = weights_long_only[weights_long_only["model"] == "one_over_n"].copy()
one_over_n_long["base_model"] = "one_over_n"
one_over_n_long["lambda"] = np.nan
one_over_n_long["model"] = "one_over_n"

all_lambda_long.append(one_over_n_long)

model_weights_long = weights_long_only[weights_long_only["model"] != "one_over_n"].copy()

for lam in lambdas:

    temp = model_weights_long.copy()

    temp["base_model"] = temp["model"]

    temp["weight"] = (
        lam * temp["weight"]
        + (1 - lam) * equal_weight
    )

    temp["lambda"] = lam

    temp["model"] = (
        temp["base_model"]
        + "_lambda_"
        + str(int(lam * 100))
    )

    all_lambda_long.append(temp)


weights_lambda_long = pd.concat(all_lambda_long, ignore_index=True)

weights_lambda_long = weights_lambda_long[
    ["date", "model", "base_model", "lambda", "asset", "weight"]
]

weights_lambda_long = weights_lambda_long.round(8)
weights_lambda_long.to_csv(OUTPUT_DIR / "portfolio_weights_lambda_long_only.csv", index=False)

print("Saved lambda long-only weights.")
print(weights_lambda_long.shape)


# ============================================================
# 9. Create cap + lambda weights
# ============================================================

all_cap_lambda = []

one_over_n_capped = weights_capped[weights_capped["model"] == "one_over_n"].copy()
one_over_n_capped["base_model"] = "one_over_n"
one_over_n_capped["lambda"] = np.nan
one_over_n_capped["model"] = "one_over_n"

all_cap_lambda.append(one_over_n_capped)

model_weights_capped = weights_capped[weights_capped["model"] != "one_over_n"].copy()

for lam in lambdas:

    temp = model_weights_capped.copy()

    temp["base_model"] = temp["model"]

    temp["weight"] = (
        lam * temp["weight"]
        + (1 - lam) * equal_weight
    )

    temp["lambda"] = lam

    temp["model"] = (
        temp["base_model"]
        + "_lambda_"
        + str(int(lam * 100))
    )

    all_cap_lambda.append(temp)


weights_cap_lambda = pd.concat(all_cap_lambda, ignore_index=True)

weights_cap_lambda = weights_cap_lambda[
    ["date", "model", "base_model", "cap", "lambda", "asset", "weight"]
]

weights_cap_lambda = weights_cap_lambda.round(8)
weights_cap_lambda.to_csv(OUTPUT_DIR / "portfolio_weights_cap_lambda.csv", index=False)

print("Saved cap + lambda weights.")
print(weights_cap_lambda.shape)


print("\nAll weight files created successfully.")