import pandas as pd
import numpy as np

# ============================================================
# 1. File paths
# ============================================================

FORECAST_PATH = r"C:\Users\vilke\OneDrive\Dokumentai\thesis_project\output\forecasts\combined_forecasts_for_portfolio.csv"

INV_COV_PATH = r"C:\Users\vilke\OneDrive\Dokumentai\thesis_project\output\forecasts\inverse_covariance_matrices.csv"

OUTPUT_PATH = r"C:\Users\vilke\OneDrive\Dokumentai\thesis_project\output\forecasts\portfolio_weights.csv"


# ============================================================
# 2. Load data
# ============================================================

forecasts = pd.read_csv(FORECAST_PATH)
forecasts["date"] = pd.to_datetime(forecasts["date"])

inv_cov_long = pd.read_csv(INV_COV_PATH)
inv_cov_long["date"] = pd.to_datetime(inv_cov_long["date"])


# ============================================================
# 3. Define assets and models
# ============================================================

asset_cols = [
    "NoDur_excess", "Durbl_excess", "Manuf_excess", "Enrgy_excess",
    "HiTec_excess", "Telcm_excess", "Shops_excess", "Hlth_excess",
    "Utils_excess", "Other_excess"
]

models = [
    "historical_mean",
    "ridge_restricted",
    "ols4_restricted",
    "random_forest"
]


# ============================================================
# 4. Helper functions
# ============================================================

def get_inverse_covariance_matrix(inv_cov_long, current_date, asset_cols):
    """
    Converts inverse covariance matrix from long format into 10x10 matrix.
    """

    temp = inv_cov_long[inv_cov_long["date"] == current_date]

    inv_sigma = temp.pivot(
        index="asset_i",
        columns="asset_j",
        values="inverse_covariance"
    )

    inv_sigma = inv_sigma.loc[asset_cols, asset_cols]

    return inv_sigma.values


def mean_variance_weights(mu_hat, inv_sigma):
    """
    Computes normalized mean-variance weights:

    raw weights = inverse covariance matrix × expected return vector

    final weights are normalized to sum to one.
    """

    mu_hat = np.asarray(mu_hat)

    raw_weights = inv_sigma @ mu_hat

    weight_sum = raw_weights.sum()

    # Safety fallback if denominator is too close to zero
    if np.abs(weight_sum) < 1e-10:
        return np.repeat(1 / len(mu_hat), len(mu_hat))

    weights = raw_weights / weight_sum

    return weights


# ============================================================
# 5. Create portfolio weights
# ============================================================

all_weights = []

portfolio_dates = sorted(forecasts["date"].unique())

for current_date in portfolio_dates:

    # Skip date if no inverse covariance matrix exists
    if current_date not in set(inv_cov_long["date"].unique()):
        continue

    # Get forecast rows for this month
    forecast_month = forecasts[forecasts["date"] == current_date].copy()

    # Ensure assets are in correct order
    forecast_month = forecast_month.set_index("asset").loc[asset_cols]

    # Get inverse covariance matrix
    inv_sigma = get_inverse_covariance_matrix(
        inv_cov_long,
        current_date,
        asset_cols
    )

    # --------------------------------------------------------
    # Equal-weight benchmark
    # --------------------------------------------------------

    equal_weights = np.repeat(1 / len(asset_cols), len(asset_cols))

    for asset, weight in zip(asset_cols, equal_weights):

        all_weights.append({
            "date": current_date,
            "model": "one_over_n",
            "asset": asset,
            "weight": weight
        })

    # --------------------------------------------------------
    # Forecast-based models
    # --------------------------------------------------------

    for model in models:

        mu_hat = forecast_month[model].values.astype(float)

        weights = mean_variance_weights(mu_hat, inv_sigma)

        for asset, weight in zip(asset_cols, weights):

            all_weights.append({
                "date": current_date,
                "model": model,
                "asset": asset,
                "weight": weight
            })


# ============================================================
# 6. Save weights
# ============================================================

weights_df = pd.DataFrame(all_weights)

weights_df = weights_df.round(8)

weights_df.to_csv(OUTPUT_PATH, index=False)

print("Portfolio weights saved successfully.")
print(weights_df.head(20))
print(weights_df.shape)