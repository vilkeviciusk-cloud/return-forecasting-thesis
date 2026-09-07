import pandas as pd

# ============================================================
# 1. File paths
# ============================================================

FORECAST_DIR = r"C:\Users\vilke\OneDrive\Dokumentai\thesis_project\output\forecasts"

ridge_path = FORECAST_DIR + r"\forecasts_ridge.csv"
ols4_path = FORECAST_DIR + r"\forecasts_ols4.csv"
rf_path = FORECAST_DIR + r"\forecasts_random_forest_samllergrid.csv"
elastic_path = FORECAST_DIR + r"\forecasts_elastic_net.csv"
ols_path = FORECAST_DIR + r"\forecasts_ols.csv"

output_path = FORECAST_DIR + r"\combined_forecasts_for_portfolio.csv"


# ============================================================
# 2. Load forecast files
# ============================================================

ridge = pd.read_csv(ridge_path)
ols4 = pd.read_csv(ols4_path)
rf = pd.read_csv(rf_path)
elastic = pd.read_csv(elastic_path)
ols = pd.read_csv(ols_path)

for df in [ridge, ols4, rf, elastic, ols]:
    df["date"] = pd.to_datetime(df["date"])


# ============================================================
# 3. Keep only useful columns
# ============================================================

ridge_keep = ridge[
    [
        "date",
        "asset",
        "realized",
        "historical_mean",
        "ridge_restricted"
    ]
].copy()

ols4_keep = ols4[
    [
        "date",
        "asset",
        "ols4_restricted"
    ]
].copy()

rf_keep = rf[
    [
        "date",
        "asset",
        "random_forest"
    ]
].copy()

elastic_keep = elastic[
    [
        "date",
        "asset",
        "elastic_net_restricted"
    ]
].copy()

ols_keep = ols[
    [
        "date",
        "asset",
        "ols_restricted"
    ]
].copy()


# ============================================================
# 4. Merge forecasts
# ============================================================

combined = ridge_keep.merge(
    ols4_keep,
    on=["date", "asset"],
    how="inner"
)

combined = combined.merge(
    rf_keep,
    on=["date", "asset"],
    how="inner"
)

combined = combined.merge(
    elastic_keep,
    on=["date", "asset"],
    how="inner"
)

combined = combined.merge(
    ols_keep,
    on=["date", "asset"],
    how="inner"
)


# ============================================================
# 5. Sort and save
# ============================================================

combined = combined.sort_values(["date", "asset"]).reset_index(drop=True)
combined = combined.round(8)

combined.to_csv(output_path, index=False)

print("Combined forecast file saved successfully.")
print(combined.head())
print(combined.shape)
print(combined.columns.tolist())
print("Saved as combined_forecasts_for_portfolio.csv")