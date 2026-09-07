import pandas as pd
import numpy as np

# ============================================================
# 1. File paths
# ============================================================

PORTFOLIO_RETURNS_PATH = r"C:\Users\vilke\OneDrive\Dokumentai\thesis_project\output\portfolios\portfolio_returns_long_only.csv"

OUTPUT_PATH = r"C:\Users\vilke\OneDrive\Dokumentai\thesis_project\output\portfolios\portfolio_performance_long_only.csv"

# ============================================================
# 2. Load portfolio returns
# ============================================================

data = pd.read_csv(PORTFOLIO_RETURNS_PATH)
data["date"] = pd.to_datetime(data["date"])


# ============================================================
# 3. Settings
# ============================================================

months_per_year = 12
risk_aversion = 3  # gamma for certainty-equivalent return


# ============================================================
# 4. Evaluate portfolio performance
# ============================================================

results = []

for model in data["model"].unique():

    df = data[data["model"] == model].copy()

    returns = df["portfolio_excess_return"]

    # Monthly statistics
    mean_monthly = returns.mean()
    volatility_monthly = returns.std()

    # Annualized statistics
    mean_annualized = mean_monthly * months_per_year
    volatility_annualized = volatility_monthly * np.sqrt(months_per_year)

    # Sharpe ratio
    if volatility_annualized == 0:
        sharpe_ratio = np.nan
    else:
        sharpe_ratio = mean_annualized / volatility_annualized

    # Certainty-equivalent return
    # Uses annualized mean and annualized variance
    cer = mean_annualized - (risk_aversion / 2) * (volatility_annualized ** 2)

    results.append({
        "model": model,
        "mean_monthly_excess_return": mean_monthly,
        "volatility_monthly": volatility_monthly,
        "mean_annualized_excess_return": mean_annualized,
        "volatility_annualized": volatility_annualized,
        "sharpe_ratio": sharpe_ratio,
        "certainty_equivalent_return_gamma_3": cer,
        "min_monthly_return": returns.min(),
        "max_monthly_return": returns.max(),
        "number_of_months": len(returns)
    })


# ============================================================
# 5. Save results
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.round(6)

results_df.to_csv(OUTPUT_PATH, index=False)

print("Portfolio performance saved successfully.")
print(results_df)
print("\nSaved as portfolio_performance.csv")