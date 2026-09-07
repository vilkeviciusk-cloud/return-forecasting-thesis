import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import norm


# ============================================================
# 1. File paths
# ============================================================

FORECAST_DIR = Path(
    r"C:\Users\vilke\OneDrive\Dokumentai\thesis_project\output\forecasts"
)

ROBUSTNESS_DIR = FORECAST_DIR / "robustness"

OUTPUT_PATH = ROBUSTNESS_DIR / "dm_rolling_vs_expanding_all_models.csv"
SUMMARY_OUTPUT_PATH = ROBUSTNESS_DIR / "dm_rolling_vs_expanding_all_models_summary.csv"


# ============================================================
# 2. Load forecasts
# ============================================================

# Expanding-window forecasts
ols_expanding = pd.read_csv(FORECAST_DIR / "forecasts_ols.csv")
ols4_expanding = pd.read_csv(FORECAST_DIR / "forecasts_ols4.csv")
ridge_expanding = pd.read_csv(FORECAST_DIR / "forecasts_ridge.csv")
elastic_expanding = pd.read_csv(FORECAST_DIR / "forecasts_elastic_net.csv")
rf_expanding = pd.read_csv(FORECAST_DIR / "forecasts_random_forest_samllergrid.csv")

# Rolling-window forecasts
ols_rolling = pd.read_csv(ROBUSTNESS_DIR / "forecasts_ols_rolling_240.csv")
ols4_rolling = pd.read_csv(ROBUSTNESS_DIR / "forecasts_ols4_rolling_240.csv")
ridge_rolling = pd.read_csv(ROBUSTNESS_DIR / "forecasts_ridge_rolling_240.csv")
elastic_rolling = pd.read_csv(ROBUSTNESS_DIR / "forecasts_elastic_net_rolling_240.csv")
rf_rolling = pd.read_csv(ROBUSTNESS_DIR / "forecasts_random_forest_rolling_240.csv")

for df in [
    ols_expanding, ols_rolling,
    ols4_expanding, ols4_rolling,
    ridge_expanding, ridge_rolling,
    elastic_expanding, elastic_rolling,
    rf_expanding, rf_rolling
]:
    df["date"] = pd.to_datetime(df["date"])


# ============================================================
# 3. Newey-West standard error
# ============================================================

def newey_west_se(x, lag=None):
    """
    Newey-West standard error for the mean of a time series.
    """

    x = np.asarray(x)
    x = x[~np.isnan(x)]

    T = len(x)

    if T <= 1:
        return np.nan

    x_centered = x - x.mean()

    if lag is None:
        lag = int(np.floor(4 * (T / 100) ** (2 / 9)))

    gamma_0 = np.sum(x_centered * x_centered) / T
    var = gamma_0

    for l in range(1, lag + 1):
        weight = 1 - l / (lag + 1)
        gamma_l = np.sum(x_centered[l:] * x_centered[:-l]) / T
        var += 2 * weight * gamma_l

    return np.sqrt(var / T)


# ============================================================
# 4. Diebold-Mariano test
# ============================================================

def dm_test_same_model(expanding_df, rolling_df, expanding_col, rolling_col, model_name):
    """
    Compares expanding-window and rolling-window forecasts
    for the same model.

    Loss difference:
        expanding squared error - rolling squared error

    Positive value means rolling has lower squared error.
    Negative value means expanding has lower squared error.
    """

    expanding_small = expanding_df[
        ["date", "asset", "realized", expanding_col]
    ].copy()

    rolling_small = rolling_df[
        ["date", "asset", rolling_col]
    ].copy()

    expanding_small = expanding_small.rename(
        columns={expanding_col: "forecast_expanding"}
    )

    rolling_small = rolling_small.rename(
        columns={rolling_col: "forecast_rolling"}
    )

    merged = expanding_small.merge(
        rolling_small,
        on=["date", "asset"],
        how="inner"
    )

    results = []

    for asset in merged["asset"].unique():

        df = merged[merged["asset"] == asset].copy()

        error_expanding = df["realized"] - df["forecast_expanding"]
        error_rolling = df["realized"] - df["forecast_rolling"]

        loss_expanding = error_expanding ** 2
        loss_rolling = error_rolling ** 2

        # Positive means rolling has lower squared error
        loss_difference = loss_expanding - loss_rolling

        mean_loss_difference = loss_difference.mean()
        se = newey_west_se(loss_difference)

        if se == 0 or np.isnan(se):
            dm_stat = np.nan
            p_value_two_sided = np.nan
            p_value_one_sided_rolling_better = np.nan
        else:
            dm_stat = mean_loss_difference / se

            # Two-sided: are rolling and expanding significantly different?
            p_value_two_sided = 2 * (1 - norm.cdf(abs(dm_stat)))

            # One-sided: is rolling significantly better than expanding?
            p_value_one_sided_rolling_better = 1 - norm.cdf(dm_stat)

        results.append({
            "model": model_name,
            "asset": asset,
            "forecast_expanding": expanding_col,
            "forecast_rolling": rolling_col,
            "mean_loss_difference_expanding_minus_rolling": mean_loss_difference,
            "DM_statistic": dm_stat,
            "p_value_two_sided_difference": p_value_two_sided,
            "p_value_one_sided_rolling_better": p_value_one_sided_rolling_better,
            "number_of_observations": len(df)
        })

    return results


# ============================================================
# 5. Run comparisons
# ============================================================

all_results = []

# OLS
all_results += dm_test_same_model(
    expanding_df=ols_expanding,
    rolling_df=ols_rolling,
    expanding_col="ols",
    rolling_col="ols",
    model_name="ols"
)

all_results += dm_test_same_model(
    expanding_df=ols_expanding,
    rolling_df=ols_rolling,
    expanding_col="ols_restricted",
    rolling_col="ols_restricted",
    model_name="ols_restricted"
)

# OLS-4
all_results += dm_test_same_model(
    expanding_df=ols4_expanding,
    rolling_df=ols4_rolling,
    expanding_col="ols4",
    rolling_col="ols4",
    model_name="ols4"
)

all_results += dm_test_same_model(
    expanding_df=ols4_expanding,
    rolling_df=ols4_rolling,
    expanding_col="ols4_restricted",
    rolling_col="ols4_restricted",
    model_name="ols4_restricted"
)

# Ridge
all_results += dm_test_same_model(
    expanding_df=ridge_expanding,
    rolling_df=ridge_rolling,
    expanding_col="ridge",
    rolling_col="ridge",
    model_name="ridge"
)

all_results += dm_test_same_model(
    expanding_df=ridge_expanding,
    rolling_df=ridge_rolling,
    expanding_col="ridge_restricted",
    rolling_col="ridge_restricted",
    model_name="ridge_restricted"
)

# Elastic Net
all_results += dm_test_same_model(
    expanding_df=elastic_expanding,
    rolling_df=elastic_rolling,
    expanding_col="elastic_net",
    rolling_col="elastic_net",
    model_name="elastic_net"
)

all_results += dm_test_same_model(
    expanding_df=elastic_expanding,
    rolling_df=elastic_rolling,
    expanding_col="elastic_net_restricted",
    rolling_col="elastic_net_restricted",
    model_name="elastic_net_restricted"
)

# Random Forest
all_results += dm_test_same_model(
    expanding_df=rf_expanding,
    rolling_df=rf_rolling,
    expanding_col="random_forest",
    rolling_col="random_forest",
    model_name="random_forest"
)


# ============================================================
# 6. Save detailed results
# ============================================================

results_df = pd.DataFrame(all_results)
results_df = results_df.round(6)

results_df.to_csv(OUTPUT_PATH, index=False)

print("DM rolling vs expanding detailed results saved successfully.")
print(results_df)
print(f"\nSaved as {OUTPUT_PATH}")


# ============================================================
# 7. Create compact summary table
# ============================================================

summary_rows = []

for model, df in results_df.groupby("model"):

    summary_rows.append({
        "Model": model,
        "Avg. DM statistic": df["DM_statistic"].mean(),
        "Median p-value": df["p_value_two_sided_difference"].median(),
        "Significant at 5%": (df["p_value_two_sided_difference"] < 0.05).sum(),
        "Significant at 10%": (df["p_value_two_sided_difference"] < 0.10).sum(),
        "Rolling better by sign": (df["mean_loss_difference_expanding_minus_rolling"] > 0).sum(),
        "Expanding better by sign": (df["mean_loss_difference_expanding_minus_rolling"] < 0).sum()
    })


summary_df = pd.DataFrame(summary_rows)

model_order = {
    "ols": 1,
    "ols_restricted": 2,
    "ols4": 3,
    "ols4_restricted": 4,
    "ridge": 5,
    "ridge_restricted": 6,
    "elastic_net": 7,
    "elastic_net_restricted": 8,
    "random_forest": 9
}

summary_df["order"] = summary_df["Model"].map(model_order)
summary_df = summary_df.sort_values("order").drop(columns=["order"])

summary_df = summary_df.round(6)

summary_df.to_csv(SUMMARY_OUTPUT_PATH, index=False)

print("\nDM rolling vs expanding summary:")
print(summary_df)
print(f"\nSaved as {SUMMARY_OUTPUT_PATH}")