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

EXPANDING_PATH = PORTFOLIO_DIR / "portfolio_returns_capped.csv"

ROLLING_PATH = ROBUSTNESS_DIR / "portfolio_returns_rolling_cap20.csv"

OUTPUT_PATH = ROBUSTNESS_DIR / "compare_expanding_vs_rolling_cap20_common_sample.csv"


# ============================================================
# 2. Load portfolio returns
# ============================================================

expanding = pd.read_csv(EXPANDING_PATH, low_memory=False)
rolling = pd.read_csv(ROLLING_PATH, low_memory=False)

expanding["date"] = pd.to_datetime(expanding["date"])
rolling["date"] = pd.to_datetime(rolling["date"])

expanding["cap"] = expanding["cap"].astype(str)
rolling["cap"] = rolling["cap"].astype(str)


# ============================================================
# 3. Keep only comparable models
# ============================================================

expanding_keep_models = [
    "one_over_n",
    "historical_mean_cap_20",
    "ols4_restricted_cap_20",
    "ridge_restricted_cap_20"
]

rolling_keep_models = [
    "one_over_n",
    "historical_mean_cap_20_rolling",
    "ols4_restricted_cap_20_rolling",
    "ridge_restricted_cap_20_rolling"
]

expanding = expanding[expanding["model"].isin(expanding_keep_models)].copy()
rolling = rolling[rolling["model"].isin(rolling_keep_models)].copy()


# ============================================================
# 4. Rename models so expanding and rolling line up
# ============================================================

expanding_name_map = {
    "one_over_n": "one_over_n",
    "historical_mean_cap_20": "historical_mean_cap_20",
    "ols4_restricted_cap_20": "ols4_restricted_cap_20",
    "ridge_restricted_cap_20": "ridge_restricted_cap_20"
}

rolling_name_map = {
    "one_over_n": "one_over_n",
    "historical_mean_cap_20_rolling": "historical_mean_cap_20",
    "ols4_restricted_cap_20_rolling": "ols4_restricted_cap_20",
    "ridge_restricted_cap_20_rolling": "ridge_restricted_cap_20"
}

expanding["model_clean"] = expanding["model"].map(expanding_name_map)
rolling["model_clean"] = rolling["model"].map(rolling_name_map)


# ============================================================
# 5. Common dates only
# ============================================================

common_dates = sorted(
    set(expanding["date"].unique()).intersection(
        set(rolling["date"].unique())
    )
)

expanding_common = expanding[expanding["date"].isin(common_dates)].copy()
rolling_common = rolling[rolling["date"].isin(common_dates)].copy()

print("Common sample starts:", min(common_dates))
print("Common sample ends:", max(common_dates))
print("Number of common months:", len(common_dates))


# ============================================================
# 6. Performance functions
# ============================================================

months_per_year = 12
risk_aversion = 3

def annual_return(monthly_returns):
    return monthly_returns.mean() * months_per_year

def annual_volatility(monthly_returns):
    return monthly_returns.std() * np.sqrt(months_per_year)

def sharpe_ratio(monthly_returns):
    ann_ret = annual_return(monthly_returns)
    ann_vol = annual_volatility(monthly_returns)

    if ann_vol == 0:
        return np.nan

    return ann_ret / ann_vol

def cer(monthly_returns):
    ann_ret = annual_return(monthly_returns)
    ann_vol = annual_volatility(monthly_returns)

    return ann_ret - (risk_aversion / 2) * (ann_vol ** 2)


# ============================================================
# 7. Evaluate expanding and rolling on same dates
# ============================================================

results = []

for estimation_method, df_all in [
    ("expanding", expanding_common),
    ("rolling_240", rolling_common)
]:

    for model, df in df_all.groupby("model_clean"):

        returns = df["portfolio_excess_return"]

        results.append({
            "estimation_method": estimation_method,
            "model": model,
            "mean_annualized_excess_return": annual_return(returns),
            "volatility_annualized": annual_volatility(returns),
            "sharpe_ratio": sharpe_ratio(returns),
            "certainty_equivalent_return_gamma_3": cer(returns),
            "min_monthly_return": returns.min(),
            "max_monthly_return": returns.max(),
            "number_of_months": len(returns)
        })


# ============================================================
# 8. Save results
# ============================================================

results_df = pd.DataFrame(results)
results_df = results_df.round(6)

results_df.to_csv(OUTPUT_PATH, index=False)

print("\nCommon-sample expanding vs rolling comparison:")
print(results_df)
print(f"\nSaved as {OUTPUT_PATH}")