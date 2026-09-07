import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# 1. File paths
# ============================================================

DATA_PATH = r"C:\Users\vilke\OneDrive\Dokumentai\thesis_project\data\processed\dataset_thesis_baseline.csv"

ROBUSTNESS_DIR = Path(
    r"C:\Users\vilke\OneDrive\Dokumentai\thesis_project\output\forecasts\robustness"
)

WEIGHTS_PATH = ROBUSTNESS_DIR / "portfolio_weights_rolling_cap20.csv"

OUTPUT_PATH = ROBUSTNESS_DIR / "portfolio_returns_rolling_cap20.csv"


# ============================================================
# 2. Load data
# ============================================================

data = pd.read_csv(DATA_PATH)
data["date"] = pd.to_datetime(data["date"])

weights = pd.read_csv(WEIGHTS_PATH, low_memory=False)
weights["date"] = pd.to_datetime(weights["date"])


# ============================================================
# 3. Define assets
# ============================================================

asset_cols = [
    "NoDur_excess", "Durbl_excess", "Manuf_excess", "Enrgy_excess",
    "HiTec_excess", "Telcm_excess", "Shops_excess", "Hlth_excess",
    "Utils_excess", "Other_excess"
]


# ============================================================
# 4. Create portfolio returns
# ============================================================

all_portfolio_returns = []

portfolio_dates = sorted(weights["date"].unique())

for current_date in portfolio_dates:

    realized_row = data[data["date"] == current_date]

    if realized_row.empty:
        continue

    realized_vector = realized_row[asset_cols].iloc[0].values.astype(float)

    weights_month = weights[weights["date"] == current_date].copy()

    for (model, cap), group in weights_month.groupby(["model", "cap"]):

        group = group.set_index("asset").loc[asset_cols]

        weight_vector = group["weight"].values.astype(float)

        portfolio_return = weight_vector @ realized_vector

        all_portfolio_returns.append({
            "date": current_date,
            "model": model,
            "cap": cap,
            "portfolio_excess_return": portfolio_return
        })


# ============================================================
# 5. Save
# ============================================================

portfolio_returns = pd.DataFrame(all_portfolio_returns)

portfolio_returns = portfolio_returns.sort_values(
    ["date", "model", "cap"]
).reset_index(drop=True)

portfolio_returns = portfolio_returns.round(8)

portfolio_returns.to_csv(OUTPUT_PATH, index=False)

print("Rolling cap-20 portfolio returns saved successfully.")
print(portfolio_returns.head(30))
print(portfolio_returns.shape)
print(f"\nSaved as {OUTPUT_PATH}")