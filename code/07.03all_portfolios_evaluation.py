import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# 1. File paths
# ============================================================

PORTFOLIO_DIR = Path(
    r"C:\Users\vilke\OneDrive\Dokumentai\thesis_project\output\portfolios"
)

RESULTS_DIR = Path(
    r"C:\Users\vilke\OneDrive\Dokumentai\thesis_project\output\result tables\tables"
)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Settings
# ============================================================

months_per_year = 12
risk_aversion = 3


# ============================================================
# 3. Helper functions
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


def evaluate_portfolio_returns(returns_path, output_path, group_cols):

    data = pd.read_csv(returns_path, low_memory=False)
    data["date"] = pd.to_datetime(data["date"])

    for col in group_cols:
        if col in data.columns:
            data[col] = data[col].astype(str)

    results = []

    for keys, df in data.groupby(group_cols, dropna=False):

        if not isinstance(keys, tuple):
            keys = (keys,)

        returns = df["portfolio_excess_return"]

        row = {}

        for col, key in zip(group_cols, keys):
            row[col] = key

        row.update({
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

        results.append(row)

    results_df = pd.DataFrame(results)
    results_df = results_df.round(6)

    results_df.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")
    print(results_df)


# ============================================================
# 4. Evaluate all return files
# ============================================================

evaluate_portfolio_returns(
    returns_path=PORTFOLIO_DIR / "portfolio_returns.csv",
    output_path=RESULTS_DIR / "portfolio_performance_unrestricted.csv",
    group_cols=["model"]
)

evaluate_portfolio_returns(
    returns_path=PORTFOLIO_DIR / "portfolio_returns_long_only.csv",
    output_path=RESULTS_DIR / "portfolio_performance_long_only.csv",
    group_cols=["model"]
)

evaluate_portfolio_returns(
    returns_path=PORTFOLIO_DIR / "portfolio_returns_capped.csv",
    output_path=RESULTS_DIR / "portfolio_performance_capped.csv",
    group_cols=["model", "cap"]
)

evaluate_portfolio_returns(
    returns_path=PORTFOLIO_DIR / "portfolio_returns_lambda_long_only.csv",
    output_path=RESULTS_DIR / "portfolio_performance_lambda_long_only.csv",
    group_cols=["model", "base_model", "lambda"]
)

evaluate_portfolio_returns(
    returns_path=PORTFOLIO_DIR / "portfolio_returns_cap_lambda.csv",
    output_path=RESULTS_DIR / "portfolio_performance_cap_lambda.csv",
    group_cols=["model", "base_model", "cap", "lambda"]
)

print("\nAll portfolio performance files created successfully.")