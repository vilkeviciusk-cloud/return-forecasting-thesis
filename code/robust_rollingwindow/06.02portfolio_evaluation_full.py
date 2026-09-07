import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# 1. File paths
# ============================================================

PORTFOLIO_DIR = Path(
    r"C:\Users\vilke\OneDrive\Dokumentai\thesis_project\output\portfolios"
)

ROBUSTNESS_DIR = Path(
    r"C:\Users\vilke\OneDrive\Dokumentai\thesis_project\output\forecasts\robustness"
)

EXPANDING_RETURNS_PATH = PORTFOLIO_DIR / "portfolio_returns_capped.csv"

ROLLING_RETURNS_PATH = ROBUSTNESS_DIR / "portfolio_returns_rolling_cap20_with_rf.csv"

OUTPUT_PATH = ROBUSTNESS_DIR / "portfolio_performance_rolling_cap20_with_rf_common_start.csv"


# ============================================================
# 2. Load data
# ============================================================

expanding = pd.read_csv(EXPANDING_RETURNS_PATH, low_memory=False)
rolling = pd.read_csv(ROLLING_RETURNS_PATH, low_memory=False)

expanding["date"] = pd.to_datetime(expanding["date"])
rolling["date"] = pd.to_datetime(rolling["date"])

rolling["cap"] = rolling["cap"].astype(str)


# ============================================================
# 3. Cut rolling sample to expanding start date
# ============================================================

expanding_start_date = expanding["date"].min()
expanding_end_date = expanding["date"].max()

rolling_common = rolling[
    (rolling["date"] >= expanding_start_date) &
    (rolling["date"] <= expanding_end_date)
].copy()

print("Expanding sample starts:", expanding_start_date)
print("Expanding sample ends:", expanding_end_date)
print("Rolling common sample starts:", rolling_common["date"].min())
print("Rolling common sample ends:", rolling_common["date"].max())
print("Number of common months:", rolling_common["date"].nunique())


# ============================================================
# 4. Performance functions
# ============================================================

months_per_year = 12
risk_aversion = 3

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
# 5. Evaluate rolling portfolios on common start sample
# ============================================================

results = []

for (model, cap), df in rolling_common.groupby(["model", "cap"]):

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
# 6. Save results
# ============================================================

results_df = pd.DataFrame(results)
results_df = results_df.round(6)

results_df.to_csv(OUTPUT_PATH, index=False)

print("\nRolling cap-20 portfolio performance on expanding-window sample:")
print(results_df)
print(f"\nSaved as {OUTPUT_PATH}")