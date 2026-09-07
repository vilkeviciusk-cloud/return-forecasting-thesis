import pandas as pd
import numpy as np
from pathlib import Path
from scipy.optimize import minimize


# ============================================================
# 1. File paths
# ============================================================

DATA_PATH = r"C:\Users\vilke\OneDrive\Dokumentai\thesis_project\data\processed\dataset_thesis_baseline.csv"

PORTFOLIO_DIR = Path(
    r"C:\Users\vilke\OneDrive\Dokumentai\thesis_project\output\portfolios"
)

ROBUSTNESS_DIR = Path(
    r"C:\Users\vilke\OneDrive\Dokumentai\thesis_project\output\forecasts\robustness"
)

FORECAST_PATH = ROBUSTNESS_DIR / "combined_forecasts_rolling_240_for_portfolio_all_models.csv"
COV_PATH = ROBUSTNESS_DIR / "covariance_matrices_rolling_240.csv"

# Used only to define the common 696-month sample
MAIN_RETURNS_PATH = PORTFOLIO_DIR / "portfolio_returns_capped.csv"

WEIGHTS_OUTPUT_PATH = ROBUSTNESS_DIR / "portfolio_weights_rolling_best_main_specs_common_sample.csv"
RETURNS_OUTPUT_PATH = ROBUSTNESS_DIR / "portfolio_returns_rolling_best_main_specs_common_sample.csv"
PERFORMANCE_OUTPUT_PATH = ROBUSTNESS_DIR / "portfolio_performance_rolling_best_main_specs_common_sample.csv"


# ============================================================
# 2. Load data
# ============================================================

data = pd.read_csv(DATA_PATH)
data["date"] = pd.to_datetime(data["date"])

forecasts = pd.read_csv(FORECAST_PATH)
forecasts["date"] = pd.to_datetime(forecasts["date"])

cov_long = pd.read_csv(COV_PATH)
cov_long["date"] = pd.to_datetime(cov_long["date"])

main_returns = pd.read_csv(MAIN_RETURNS_PATH, low_memory=False)
main_returns["date"] = pd.to_datetime(main_returns["date"])


# ============================================================
# 3. Restrict rolling analysis to main expanding-window sample
# ============================================================

main_start_date = main_returns["date"].min()
main_end_date = main_returns["date"].max()

forecasts = forecasts[
    (forecasts["date"] >= main_start_date) &
    (forecasts["date"] <= main_end_date)
].copy()

cov_long = cov_long[
    (cov_long["date"] >= main_start_date) &
    (cov_long["date"] <= main_end_date)
].copy()

print("Main sample starts:", main_start_date)
print("Main sample ends:", main_end_date)
print("Rolling forecast months in common sample:", forecasts["date"].nunique())


# ============================================================
# 4. Settings
# ============================================================

asset_cols = [
    "NoDur_excess", "Durbl_excess", "Manuf_excess", "Enrgy_excess",
    "HiTec_excess", "Telcm_excess", "Shops_excess", "Hlth_excess",
    "Utils_excess", "Other_excess"
]

risk_aversion = 3
months_per_year = 12
equal_weight = 1 / len(asset_cols)

# Main-analysis best specifications.
# Lambda means:
# final weights = lambda * model weights + (1 - lambda) * 1/N.
strategy_specs = [
    {
        "strategy": "one_over_n",
        "forecast_col": None,
        "cap": None,
        "lambda": None,
        "type": "benchmark"
    },
    {
        "strategy": "historical_mean_cap_20_lambda_25",
        "forecast_col": "historical_mean",
        "cap": 0.20,
        "lambda": 0.25,
        "type": "cap_lambda"
    },
    {
        "strategy": "ols_restricted_cap_30",
        "forecast_col": "ols_restricted",
        "cap": 0.30,
        "lambda": None,
        "type": "cap"
    },
    {
        "strategy": "ols4_restricted_cap_20",
        "forecast_col": "ols4_restricted",
        "cap": 0.20,
        "lambda": None,
        "type": "cap"
    },
    {
        "strategy": "ridge_restricted_cap_20_lambda_75",
        "forecast_col": "ridge_restricted",
        "cap": 0.20,
        "lambda": 0.75,
        "type": "cap_lambda"
    },
    {
        "strategy": "elastic_net_restricted_cap_30",
        "forecast_col": "elastic_net_restricted",
        "cap": 0.30,
        "lambda": None,
        "type": "cap"
    },
    {
        "strategy": "random_forest_cap_20",
        "forecast_col": "random_forest",
        "cap": 0.20,
        "lambda": None,
        "type": "cap"
    }
]


# ============================================================
# 5. Helper functions
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


def annual_return(monthly_returns):
    return monthly_returns.mean() * months_per_year


def annual_volatility(monthly_returns):
    return monthly_returns.std() * np.sqrt(months_per_year)


def sharpe_ratio(monthly_returns):
    ann_return = annual_return(monthly_returns)
    ann_vol = annual_volatility(monthly_returns)

    if ann_vol == 0:
        return np.nan

    return ann_return / ann_vol


def cer(monthly_returns):
    ann_return = annual_return(monthly_returns)
    ann_vol = annual_volatility(monthly_returns)

    return ann_return - (risk_aversion / 2) * (ann_vol ** 2)


# ============================================================
# 6. Create rolling portfolio weights
# ============================================================

all_weights = []

portfolio_dates = sorted(forecasts["date"].unique())
available_cov_dates = set(cov_long["date"].unique())

for current_date in portfolio_dates:

    if current_date not in available_cov_dates:
        continue

    forecast_month = forecasts[forecasts["date"] == current_date].copy()
    forecast_month = forecast_month.set_index("asset").loc[asset_cols]

    sigma = get_covariance_matrix(
        cov_long=cov_long,
        current_date=current_date,
        asset_cols=asset_cols
    )

    for spec in strategy_specs:

        strategy = spec["strategy"]

        # ----------------------------------------------------
        # 1/N benchmark
        # ----------------------------------------------------
        if spec["type"] == "benchmark":

            weights = np.repeat(equal_weight, len(asset_cols))

        # ----------------------------------------------------
        # Capped or cap-lambda forecast portfolios
        # ----------------------------------------------------
        else:

            forecast_col = spec["forecast_col"]
            cap = spec["cap"]

            if forecast_col not in forecast_month.columns:
                print(f"Skipping {strategy} on {current_date}: missing {forecast_col}")
                continue

            mu_hat = forecast_month[forecast_col].values.astype(float)

            model_weights = capped_mean_variance_weights(
                mu_hat=mu_hat,
                sigma=sigma,
                cap=cap,
                risk_aversion=risk_aversion
            )

            if spec["type"] == "cap_lambda":

                lam = spec["lambda"]

                weights = (
                    lam * model_weights
                    + (1 - lam) * equal_weight
                )

            else:

                weights = model_weights

        for asset, weight in zip(asset_cols, weights):

            all_weights.append({
                "date": current_date,
                "strategy": strategy,
                "asset": asset,
                "weight": weight
            })


weights_df = pd.DataFrame(all_weights)
weights_df = weights_df.round(8)

weights_df.to_csv(WEIGHTS_OUTPUT_PATH, index=False)

print("\nRolling best-spec portfolio weights saved.")
print(weights_df.head(20))
print(weights_df.shape)
print(f"Saved as {WEIGHTS_OUTPUT_PATH}")


# ============================================================
# 7. Construct rolling portfolio returns
# ============================================================

all_returns = []

portfolio_dates = sorted(weights_df["date"].unique())

for current_date in portfolio_dates:

    realized_row = data[data["date"] == current_date]

    if realized_row.empty:
        continue

    realized_vector = realized_row[asset_cols].iloc[0].values.astype(float)

    weights_month = weights_df[weights_df["date"] == current_date].copy()

    for strategy, group in weights_month.groupby("strategy"):

        group = group.set_index("asset").loc[asset_cols]

        weight_vector = group["weight"].values.astype(float)

        portfolio_return = weight_vector @ realized_vector

        all_returns.append({
            "date": current_date,
            "strategy": strategy,
            "portfolio_excess_return": portfolio_return
        })


returns_df = pd.DataFrame(all_returns)

returns_df = returns_df.sort_values(["date", "strategy"]).reset_index(drop=True)
returns_df = returns_df.round(8)

returns_df.to_csv(RETURNS_OUTPUT_PATH, index=False)

print("\nRolling best-spec portfolio returns saved.")
print(returns_df.head(20))
print(returns_df.shape)
print(f"Saved as {RETURNS_OUTPUT_PATH}")


# ============================================================
# 8. Evaluate rolling portfolio returns
# ============================================================

performance_rows = []

for strategy, df in returns_df.groupby("strategy"):

    returns = df["portfolio_excess_return"]

    performance_rows.append({
        "strategy": strategy,
        "mean_monthly_excess_return": returns.mean(),
        "volatility_monthly": returns.std(),
        "mean_annualized_excess_return": annual_return(returns),
        "volatility_annualized": annual_volatility(returns),
        "sharpe_ratio": sharpe_ratio(returns),
        "certainty_equivalent_return_gamma_3": cer(returns),
        "min_monthly_return": returns.min(),
        "max_monthly_return": returns.max(),
        "number_of_months": len(returns)
    })


performance_df = pd.DataFrame(performance_rows)

strategy_order = [
    "one_over_n",
    "historical_mean_cap_20_lambda_25",
    "ols_restricted_cap_30",
    "ols4_restricted_cap_20",
    "ridge_restricted_cap_20_lambda_75",
    "elastic_net_restricted_cap_30",
    "random_forest_cap_20"
]

performance_df["order"] = performance_df["strategy"].apply(
    lambda x: strategy_order.index(x) if x in strategy_order else 999
)

performance_df = performance_df.sort_values("order").drop(columns=["order"])

performance_df = performance_df.round(6)

performance_df.to_csv(PERFORMANCE_OUTPUT_PATH, index=False)

print("\nRolling best-spec portfolio performance saved.")
print(performance_df)
print(f"Saved as {PERFORMANCE_OUTPUT_PATH}")

print("\nNumber of months per strategy:")
print(performance_df[["strategy", "number_of_months"]])