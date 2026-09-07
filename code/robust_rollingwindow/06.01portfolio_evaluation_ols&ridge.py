import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# 1. File paths
# ============================================================

ROBUSTNESS_DIR = Path(
    r"C:\Users\vilke\OneDrive\Dokumentai\thesis_project\output\forecasts\robustness"
)

PORTFOLIO_RETURNS_PATH = ROBUSTNESS_DIR / "portfolio_returns_rolling_cap20.csv"

OUTPUT_PATH = ROBUSTNESS_DIR / "portfolio_performance_rolling_cap20.csv"


# ============================================================
# 2. Load portfolio returns
# ============================================================

data = pd.read_csv(PORTFOLIO_RETURNS_PATH, low_memory=False)
data["date"] = pd.to_datetime(data["date"])


# ============================================================
# 3. Settings
# ============================================================

months_per_year = 12
risk_aversion = 3


# ============================================================
# 4. Helper functions
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


# ============================================================
# 5. Evaluate performance
# ============================================================

results = []

for (model, cap), df in data.groupby(["model", "cap"]):

    returns = df["portfolio_excess_return"]

    results.append({
        "model": model,
        "cap": cap,
        "mean_annualized_excess_return": annual_return(returns),
        "volatility_annualized": annual_volatility(returns),
        "sharpe_ratio": sharpe_ratio(returns),
        "certainty_equivalent_return_gamma_3": cer(returns),
        "min_monthly_return": returns.min(),
        "max_monthly_return": returns.max(),
        "number_of_months": len(returns)
    })


# ============================================================
# 6. Save
# ============================================================

results_df = pd.DataFrame(results)
results_df = results_df.round(6)

results_df.to_csv(OUTPUT_PATH, index=False)

print("Rolling cap-20 portfolio performance saved successfully.")
print(results_df)
print(f"\nSaved as {OUTPUT_PATH}")