import pandas as pd
import numpy as np

# ============================================================
# 1. File paths
# ============================================================

DATA_PATH = r"C:\Users\vilke\OneDrive\Dokumentai\thesis_project\data\processed\dataset_thesis_baseline.csv"

WEIGHTS_PATH = r"C:\Users\vilke\OneDrive\Dokumentai\thesis_project\output\portfolios\portfolio_weights_long_only.csv"

OUTPUT_PATH = r"C:\Users\vilke\OneDrive\Dokumentai\thesis_project\output\portfolios\portfolio_returns_long_only.csv"


# ============================================================
# 2. Load data
# ============================================================

data = pd.read_csv(DATA_PATH)
data["date"] = pd.to_datetime(data["date"])

weights = pd.read_csv(WEIGHTS_PATH)
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
models = sorted(weights["model"].unique())

for current_date in portfolio_dates:

    # Realized excess returns for this month
    realized_row = data[data["date"] == current_date]

    if realized_row.empty:
        continue

    realized_vector = realized_row[asset_cols].iloc[0].values.astype(float)

    for model in models:

        model_weights = weights[
            (weights["date"] == current_date) &
            (weights["model"] == model)
        ].copy()

        # Make sure weights follow the correct asset order
        model_weights = model_weights.set_index("asset").loc[asset_cols]

        weight_vector = model_weights["weight"].values.astype(float)

        portfolio_return = weight_vector @ realized_vector

        all_portfolio_returns.append({
            "date": current_date,
            "model": model,
            "portfolio_excess_return": portfolio_return
        })


# ============================================================
# 5. Save portfolio returns
# ============================================================

portfolio_returns = pd.DataFrame(all_portfolio_returns)

portfolio_returns = portfolio_returns.sort_values(
    ["date", "model"]
).reset_index(drop=True)

portfolio_returns = portfolio_returns.round(8)

portfolio_returns.to_csv(OUTPUT_PATH, index=False)


# ============================================================
# 6. Final checks
# ============================================================

print("Portfolio returns saved successfully.")
print(portfolio_returns.head(20))
print(portfolio_returns.shape)
print("\nSaved as portfolio_returns.csv")