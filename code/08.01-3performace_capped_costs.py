import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# 1. File paths
# ============================================================

DATA_PATH = r"C:\Users\vilke\OneDrive\Dokumentai\thesis_project\data\processed\dataset_thesis_baseline.csv"

WEIGHTS_PATH = r"C:\Users\vilke\OneDrive\Dokumentai\thesis_project\output\portfolios\portfolio_weights_capped.csv"

PORTFOLIO_RETURNS_PATH = r"C:\Users\vilke\OneDrive\Dokumentai\thesis_project\output\portfolios\portfolio_returns_capped.csv"

OUTPUT_DIR = Path(
    r"C:\Users\vilke\OneDrive\Dokumentai\thesis_project\output\portfolios\turnover_costs_results"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "portfolio_performance_capped_after_costs.csv"


# ============================================================
# 2. Load data
# ============================================================

data = pd.read_csv(DATA_PATH)
data["date"] = pd.to_datetime(data["date"])

weights = pd.read_csv(WEIGHTS_PATH, low_memory=False)
weights["date"] = pd.to_datetime(weights["date"])

returns = pd.read_csv(PORTFOLIO_RETURNS_PATH, low_memory=False)
returns["date"] = pd.to_datetime(returns["date"])


# ============================================================
# 3. Clean cap column
# ============================================================

weights["cap"] = weights["cap"].astype(str)
returns["cap"] = returns["cap"].astype(str)


# ============================================================
# 4. Settings
# ============================================================

asset_cols = [
    "NoDur_excess", "Durbl_excess", "Manuf_excess", "Enrgy_excess",
    "HiTec_excess", "Telcm_excess", "Shops_excess", "Hlth_excess",
    "Utils_excess", "Other_excess"
]

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
    return row[asset_cols].iloc[0].values.astype(float)


# ============================================================
# 6. Calculate turnover
# ============================================================

turnover_rows = []

for (model, cap), group in weights.groupby(["model", "cap"]):

    group = group.sort_values("date").copy()

    previous_weights = None
    previous_date = None

    for current_date in sorted(group["date"].unique()):

        current_group = group[group["date"] == current_date].copy()

        # Check all assets exist
        available_assets = set(current_group["asset"])
        missing_assets = [asset for asset in asset_cols if asset not in available_assets]

        if len(missing_assets) > 0:
            print(
                f"Skipping incomplete group: date={current_date}, "
                f"model={model}, cap={cap}, missing={missing_assets}"
            )
            continue

        current_weights = (
            current_group
            .set_index("asset")
            .loc[asset_cols]["weight"]
            .values
            .astype(float)
        )

        if previous_weights is None:
            turnover = np.nan

        else:
            previous_returns = get_realized_returns(previous_date)

            previous_portfolio_return = previous_weights @ previous_returns

            drifted_weights = (
                previous_weights * (1 + previous_returns)
                / (1 + previous_portfolio_return)
            )

            turnover = 0.5 * np.sum(
                np.abs(current_weights - drifted_weights)
            )

        turnover_rows.append({
            "date": current_date,
            "model": model,
            "cap": cap,
            "turnover": turnover
        })

        previous_weights = current_weights
        previous_date = current_date


turnover = pd.DataFrame(turnover_rows)


# ============================================================
# 7. Merge returns and turnover
# ============================================================

results_data = returns.merge(
    turnover,
    on=["date", "model", "cap"],
    how="left"
)


# ============================================================
# 8. Calculate net returns
# ============================================================

for label, cost in transaction_costs.items():

    results_data[f"net_return_{label}"] = (
        results_data["portfolio_excess_return"]
        - cost * results_data["turnover"]
    )


# ============================================================
# 9. Final after-cost table
# ============================================================

results = []

for (model, cap), df in results_data.groupby(["model", "cap"]):

    row = {
        "model": model,
        "cap": cap,
        "average_turnover": df["turnover"].mean(),
        "median_turnover": df["turnover"].median()
    }

    for label in transaction_costs.keys():

        net_returns = df[f"net_return_{label}"]

        row[f"net_annual_return_{label}"] = annual_return(net_returns)
        row[f"net_sharpe_{label}"] = sharpe_ratio(net_returns)
        row[f"net_CER_{label}"] = cer(net_returns)

    results.append(row)


# ============================================================
# 10. Save results
# ============================================================

results_df = pd.DataFrame(results)
results_df = results_df.round(6)

results_df.to_csv(OUTPUT_PATH, index=False)

print("After-cost capped portfolio performance saved.")
print(results_df)
print(f"\nSaved as {OUTPUT_PATH}")