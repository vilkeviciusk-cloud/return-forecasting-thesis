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


# ============================================================
# 2. Load data
# ============================================================

data = pd.read_csv(DATA_PATH)
data["date"] = pd.to_datetime(data["date"])


# ============================================================
# 3. Settings
# ============================================================

asset_cols = [
    "NoDur_excess", "Durbl_excess", "Manuf_excess", "Enrgy_excess",
    "HiTec_excess", "Telcm_excess", "Shops_excess", "Hlth_excess",
    "Utils_excess", "Other_excess"
]


# ============================================================
# 4. Helper function
# ============================================================

def create_portfolio_returns(weights_path, output_path, group_cols):

    weights = pd.read_csv(weights_path, low_memory=False)
    weights["date"] = pd.to_datetime(weights["date"])

    for col in group_cols:
        if col in weights.columns:
            weights[col] = weights[col].astype(str)

    all_returns = []

    portfolio_dates = sorted(weights["date"].unique())

    for current_date in portfolio_dates:

        realized_row = data[data["date"] == current_date]

        if realized_row.empty:
            continue

        realized_vector = realized_row[asset_cols].iloc[0].values.astype(float)

        weights_month = weights[weights["date"] == current_date].copy()

        for keys, group in weights_month.groupby(group_cols, dropna=False):

            if not isinstance(keys, tuple):
                keys = (keys,)

            group = group.set_index("asset").loc[asset_cols]

            weight_vector = group["weight"].values.astype(float)

            portfolio_return = weight_vector @ realized_vector

            row = {
                "date": current_date,
                "portfolio_excess_return": portfolio_return
            }

            for col, key in zip(group_cols, keys):
                row[col] = key

            all_returns.append(row)

    returns_df = pd.DataFrame(all_returns)

    ordered_cols = ["date"] + group_cols + ["portfolio_excess_return"]
    returns_df = returns_df[ordered_cols]

    returns_df = returns_df.sort_values(
        ["date"] + group_cols
    ).reset_index(drop=True)

    returns_df = returns_df.round(8)

    returns_df.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")
    print(returns_df.shape)


# ============================================================
# 5. Create returns for all weight files
# ============================================================

create_portfolio_returns(
    weights_path=PORTFOLIO_DIR / "portfolio_weights.csv",
    output_path=PORTFOLIO_DIR / "portfolio_returns.csv",
    group_cols=["model"]
)

create_portfolio_returns(
    weights_path=PORTFOLIO_DIR / "portfolio_weights_long_only.csv",
    output_path=PORTFOLIO_DIR / "portfolio_returns_long_only.csv",
    group_cols=["model"]
)

create_portfolio_returns(
    weights_path=PORTFOLIO_DIR / "portfolio_weights_capped.csv",
    output_path=PORTFOLIO_DIR / "portfolio_returns_capped.csv",
    group_cols=["model", "cap"]
)

create_portfolio_returns(
    weights_path=PORTFOLIO_DIR / "portfolio_weights_lambda_long_only.csv",
    output_path=PORTFOLIO_DIR / "portfolio_returns_lambda_long_only.csv",
    group_cols=["model", "base_model", "lambda"]
)

create_portfolio_returns(
    weights_path=PORTFOLIO_DIR / "portfolio_weights_cap_lambda.csv",
    output_path=PORTFOLIO_DIR / "portfolio_returns_cap_lambda.csv",
    group_cols=["model", "base_model", "cap", "lambda"]
)

print("\nAll portfolio return files created successfully.")