import pandas as pd
import numpy as np

# ============================================================
# 1. File paths
# ============================================================

DATA_PATH = r"C:\Users\vilke\OneDrive\Dokumentai\thesis_project\data\processed\dataset_thesis_baseline.csv"

FORECAST_PATH = r"C:\Users\vilke\OneDrive\Dokumentai\thesis_project\output\forecasts\combined_forecasts_for_portfolio.csv"

COV_OUTPUT_PATH = r"C:\Users\vilke\OneDrive\Dokumentai\thesis_project\output\forecasts\covariance_matrices.csv"

INV_COV_OUTPUT_PATH = r"C:\Users\vilke\OneDrive\Dokumentai\thesis_project\output\forecasts\inverse_covariance_matrices.csv"


# ============================================================
# 2. Load data
# ============================================================

data = pd.read_csv(DATA_PATH)
data["date"] = pd.to_datetime(data["date"])

forecasts = pd.read_csv(FORECAST_PATH)
forecasts["date"] = pd.to_datetime(forecasts["date"])


# ============================================================
# 3. Define assets and settings
# ============================================================

asset_cols = [
    "NoDur_excess", "Durbl_excess", "Manuf_excess", "Enrgy_excess",
    "HiTec_excess", "Telcm_excess", "Shops_excess", "Hlth_excess",
    "Utils_excess", "Other_excess"
]

portfolio_dates = sorted(forecasts["date"].unique())

cov_window = 480  # previous 40 years


# ============================================================
# 4. Create covariance and inverse covariance matrices
# ============================================================

all_covariances = []
all_inverse_covariances = []

for current_date in portfolio_dates:

    current_index_list = data.index[data["date"] == current_date].tolist()

    if len(current_index_list) == 0:
        continue

    current_index = current_index_list[0]

    # Need 480 past observations before this date
    if current_index < cov_window:
        continue

    # Use only past returns, not current-month returns
    past_returns = data.loc[
        current_index - cov_window:current_index - 1,
        asset_cols
    ]

    # Historical covariance matrix
    sigma = past_returns.cov()

    # Pseudo-inverse for numerical stability
    inv_sigma = pd.DataFrame(
        np.linalg.pinv(sigma.values),
        index=asset_cols,
        columns=asset_cols
    )

    # Save both matrices in long format
    for asset_i in asset_cols:
        for asset_j in asset_cols:

            all_covariances.append({
                "date": current_date,
                "asset_i": asset_i,
                "asset_j": asset_j,
                "covariance": sigma.loc[asset_i, asset_j]
            })

            all_inverse_covariances.append({
                "date": current_date,
                "asset_i": asset_i,
                "asset_j": asset_j,
                "inverse_covariance": inv_sigma.loc[asset_i, asset_j]
            })


# ============================================================
# 5. Save results
# ============================================================

covariances = pd.DataFrame(all_covariances)
inverse_covariances = pd.DataFrame(all_inverse_covariances)

covariances = covariances.round(10)
inverse_covariances = inverse_covariances.round(10)

covariances.to_csv(COV_OUTPUT_PATH, index=False)
inverse_covariances.to_csv(INV_COV_OUTPUT_PATH, index=False)


# ============================================================
# 6. Final checks
# ============================================================

print("Covariance matrices saved successfully.")
print("Inverse covariance matrices saved successfully.")

print("\nCovariance file shape:")
print(covariances.shape)

print("\nInverse covariance file shape:")
print(inverse_covariances.shape)

print("\nFirst rows of covariance file:")
print(covariances.head())

print("\nFirst rows of inverse covariance file:")
print(inverse_covariances.head())