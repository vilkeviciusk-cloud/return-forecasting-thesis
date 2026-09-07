import pandas as pd
from pathlib import Path


# ============================================================
# 1. File paths
# ============================================================

ROBUSTNESS_DIR = Path(
    r"C:\Users\vilke\OneDrive\Dokumentai\thesis_project\output\forecasts\robustness"
)

ols4_path = ROBUSTNESS_DIR / "forecasts_ols4_rolling_240.csv"
ridge_path = ROBUSTNESS_DIR / "forecasts_ridge_rolling_240.csv"

output_path = ROBUSTNESS_DIR / "combined_forecasts_rolling_240_for_portfolio.csv"


# ============================================================
# 2. Load forecast files
# ============================================================

ols4 = pd.read_csv(ols4_path)
ridge = pd.read_csv(ridge_path)

ols4["date"] = pd.to_datetime(ols4["date"])
ridge["date"] = pd.to_datetime(ridge["date"])


# ============================================================
# 3. Keep required columns
# ============================================================

ols4_keep = ols4[
    [
        "date",
        "asset",
        "realized",
        "historical_mean",
        "ols4_restricted"
    ]
].copy()

ridge_keep = ridge[
    [
        "date",
        "asset",
        "ridge_restricted"
    ]
].copy()


# ============================================================
# 4. Merge
# ============================================================

combined = ols4_keep.merge(
    ridge_keep,
    on=["date", "asset"],
    how="inner"
)

combined = combined.sort_values(["date", "asset"]).reset_index(drop=True)
combined = combined.round(8)

combined.to_csv(output_path, index=False)

print("Combined rolling forecast file saved successfully.")
print(combined.head())
print(combined.shape)
print(f"\nSaved as {output_path}")