import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# 1. File paths
# ============================================================

DATA_PATH = r"C:\Users\vilke\OneDrive\Dokumentai\thesis_project\data\processed\dataset_thesis_baseline.csv"

PORTFOLIO_DIR = Path(
    r"C:\Users\vilke\OneDrive\Dokumentai\thesis_project\output\portfolios"
)

RESULTS_DIR = Path(
    r"C:\Users\vilke\OneDrive\Dokumentai\thesis_project\output\result tables\tables"
)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

WEIGHTS_CAPPED_PATH = PORTFOLIO_DIR / "portfolio_weights_capped.csv"
WEIGHTS_CAP_LAMBDA_PATH = PORTFOLIO_DIR / "portfolio_weights_cap_lambda.csv"

OUTPUT_MONTHLY_PATH = RESULTS_DIR / "best_portfolios_monthly_returns_with_costs.csv"
OUTPUT_SUMMARY_PATH = RESULTS_DIR / "best_portfolios_turnover_costs_summary.csv"


# ============================================================
# 2. Load data
# ============================================================

data = pd.read_csv(DATA_PATH)
data["date"] = pd.to_datetime(data["date"])

weights_capped = pd.read_csv(WEIGHTS_CAPPED_PATH, low_memory=False)
weights_cap_lambda = pd.read_csv(WEIGHTS_CAP_LAMBDA_PATH, low_memory=False)

weights_capped["date"] = pd.to_datetime(weights_capped["date"])
weights_cap_lambda["date"] = pd.to_datetime(weights_cap_lambda["date"])


# ============================================================
# 3. Define assets and selected strategies
# ============================================================

asset_cols = [
    "NoDur_excess", "Durbl_excess", "Manuf_excess", "Enrgy_excess",
    "HiTec_excess", "Telcm_excess", "Shops_excess", "Hlth_excess",
    "Utils_excess", "Other_excess"
]

selected_from_capped = {
    "one_over_n": "1/N",
    "ols_restricted_cap_30": "OLS cap 30%",
    "ols4_restricted_cap_20": "OLS-4 cap 20%",
    "elastic_net_restricted_cap_30": "Elastic Net cap 30%",
    "random_forest_cap_20": "Random Forest cap 20%"
}

selected_from_cap_lambda = {
    "historical_mean_cap_20_lambda_25": "Historical mean cap 20% + lambda 0.25",
    "ridge_restricted_cap_20_lambda_75": "Ridge cap 20% + lambda 0.75"
}

strategy_family = {
    "1/N": "Benchmark",
    "Historical mean cap 20% + lambda 0.25": "Historical mean",
    "OLS cap 30%": "OLS",
    "OLS-4 cap 20%": "OLS-4",
    "Ridge cap 20% + lambda 0.75": "Ridge",
    "Elastic Net cap 30%": "Elastic Net",
    "Random Forest cap 20%": "Random Forest"
}


# ============================================================
# 4. Settings
# ============================================================

months_per_year = 12
risk_aversion = 3

transaction_costs = {
    "10bps": 0.0010,
    "25bps": 0.0025,
    "50bps": 0.0050
}


# ============================================================
# 5. Helper functions
# ============================================================

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


def get_realized_returns(current_date):
    row = data[data["date"] == current_date]

    if row.empty:
        return None

    return row[asset_cols].iloc[0].values.astype(float)


def calculate_turnover_and_returns(weights):
    """
    Calculates gross portfolio returns and turnover.

    Turnover is calculated as:
        0.5 * sum(abs(current_weights - drifted_previous_weights))

    where previous weights are first allowed to drift with realized returns.
    """

    all_rows = []

    for strategy, group in weights.groupby("strategy"):

        group = group.sort_values("date").copy()
        dates = sorted(group["date"].unique())

        previous_weights = None
        previous_date = None

        for current_date in dates:

            current_group = group[group["date"] == current_date].copy()

            available_assets = set(current_group["asset"])
            missing_assets = [asset for asset in asset_cols if asset not in available_assets]

            if len(missing_assets) > 0:
                print(f"Skipping {strategy} on {current_date}: missing {missing_assets}")
                continue

            current_group = current_group.set_index("asset").loc[asset_cols]
            current_weights = current_group["weight"].values.astype(float)

            realized_returns = get_realized_returns(current_date)

            if realized_returns is None:
                continue

            gross_return = current_weights @ realized_returns

            if previous_weights is None:
                turnover = np.nan

            else:
                previous_returns = get_realized_returns(previous_date)

                if previous_returns is None:
                    turnover = np.nan

                else:
                    previous_portfolio_return = previous_weights @ previous_returns

                    drifted_weights = (
                        previous_weights * (1 + previous_returns)
                        / (1 + previous_portfolio_return)
                    )

                    turnover = 0.5 * np.sum(
                        np.abs(current_weights - drifted_weights)
                    )

            all_rows.append({
                "date": current_date,
                "strategy": strategy,
                "model_family": strategy_family.get(strategy, strategy),
                "gross_return": gross_return,
                "turnover": turnover
            })

            previous_weights = current_weights
            previous_date = current_date

    return pd.DataFrame(all_rows)


# ============================================================
# 6. Select and combine best portfolio weights
# ============================================================

selected_weights = []

# From capped weights
for original_model, strategy_name in selected_from_capped.items():

    temp = weights_capped[weights_capped["model"] == original_model].copy()

    if temp.empty:
        print(f"Warning: {original_model} not found in capped weights.")
        continue

    temp["strategy"] = strategy_name
    selected_weights.append(temp[["date", "strategy", "asset", "weight"]])


# From cap + lambda weights
for original_model, strategy_name in selected_from_cap_lambda.items():

    temp = weights_cap_lambda[weights_cap_lambda["model"] == original_model].copy()

    if temp.empty:
        print(f"Warning: {original_model} not found in cap-lambda weights.")
        continue

    temp["strategy"] = strategy_name
    selected_weights.append(temp[["date", "strategy", "asset", "weight"]])


best_weights = pd.concat(selected_weights, ignore_index=True)

print("Selected strategies:")
print(best_weights["strategy"].unique())


# ============================================================
# 7. Calculate gross returns and turnover
# ============================================================

monthly = calculate_turnover_and_returns(best_weights)

monthly = monthly.sort_values(["date", "strategy"]).reset_index(drop=True)


# ============================================================
# 8. Add net returns after transaction costs
# ============================================================

# First month has no rebalancing turnover; set cost to zero for net returns.
turnover_for_costs = monthly["turnover"].fillna(0)

for label, cost in transaction_costs.items():

    monthly[f"net_return_{label}"] = (
        monthly["gross_return"]
        - cost * turnover_for_costs
    )


# ============================================================
# 9. Save monthly file
# ============================================================

monthly = monthly.round(8)
monthly.to_csv(OUTPUT_MONTHLY_PATH, index=False)

print("\nMonthly best portfolio returns with costs saved:")
print(OUTPUT_MONTHLY_PATH)
print(monthly.head(20))


# ============================================================
# 10. Create summary table
# ============================================================

summary_rows = []

for strategy, df in monthly.groupby("strategy"):

    row = {
        "Strategy": strategy,
        "Model": strategy_family.get(strategy, strategy),
        "Avg. turnover": df["turnover"].mean(),
        "Median turnover": df["turnover"].median(),

        "Gross ann. return": annual_return(df["gross_return"]),
        "Gross Sharpe": sharpe_ratio(df["gross_return"]),
        "Gross CER": cer(df["gross_return"])
    }

    for label in transaction_costs.keys():

        net_col = f"net_return_{label}"

        row[f"Net ann. return {label}"] = annual_return(df[net_col])
        row[f"Net Sharpe {label}"] = sharpe_ratio(df[net_col])
        row[f"Net CER {label}"] = cer(df[net_col])

    summary_rows.append(row)


summary = pd.DataFrame(summary_rows)

# Nice ordering
strategy_order = [
    "1/N",
    "Historical mean cap 20% + lambda 0.25",
    "OLS cap 30%",
    "OLS-4 cap 20%",
    "Ridge cap 20% + lambda 0.75",
    "Elastic Net cap 30%",
    "Random Forest cap 20%"
]

summary["order"] = summary["Strategy"].apply(
    lambda x: strategy_order.index(x) if x in strategy_order else 999
)

summary = summary.sort_values("order").drop(columns=["order"])

summary = summary.round(6)

summary.to_csv(OUTPUT_SUMMARY_PATH, index=False)

print("\nBest portfolio transaction-cost summary saved:")
print(OUTPUT_SUMMARY_PATH)
print(summary)