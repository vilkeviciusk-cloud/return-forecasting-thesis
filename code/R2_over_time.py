import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# 1. File paths
# ============================================================

forecast_dir = Path(r"C:\Users\vilke\OneDrive\Dokumentai\thesis_project\output\forecasts")

files = {
    "ols_restricted": forecast_dir / "forecasts_ols.csv",
    "ols4_restricted": forecast_dir / "forecasts_ols4.csv",
    "ridge_restricted": forecast_dir / "forecasts_ridge.csv",
    "elastic_net_restricted": forecast_dir / "forecasts_elastic_net.csv",
    "random_forest": forecast_dir / "forecasts_random_forest_samllergrid.csv",
}

# ============================================================
# 2. Read and keep only needed columns
# ============================================================

ols = pd.read_csv(files["ols_restricted"])[
    ["date", "asset", "realized", "historical_mean", "ols_restricted"]
]

ols4 = pd.read_csv(files["ols4_restricted"])[
    ["date", "asset", "realized", "historical_mean", "ols4_restricted"]
]

ridge = pd.read_csv(files["ridge_restricted"])[
    ["date", "asset", "realized", "historical_mean", "ridge_restricted"]
]

elastic = pd.read_csv(files["elastic_net_restricted"])[
    ["date", "asset", "realized", "historical_mean", "elastic_net_restricted"]
]

rf = pd.read_csv(files["random_forest"])[
    ["date", "asset", "realized", "random_forest"]
]

# ============================================================
# 3. Merge all forecasts into one dataframe
# ============================================================

df = ols.merge(
    ols4[["date", "asset", "ols4_restricted"]],
    on=["date", "asset"],
    how="inner"
)

df = df.merge(
    ridge[["date", "asset", "ridge_restricted"]],
    on=["date", "asset"],
    how="inner"
)

df = df.merge(
    elastic[["date", "asset", "elastic_net_restricted"]],
    on=["date", "asset"],
    how="inner"
)

df = df.merge(
    rf[["date", "asset", "random_forest"]],
    on=["date", "asset"],
    how="inner"
)

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["asset", "date"]).reset_index(drop=True)

merged_path = forecast_dir / "forecasts_restricted_models_merged.csv"
df.to_csv(merged_path, index=False)

print("Merged file saved to:")
print(merged_path)
print("\nMerged dataframe shape:", df.shape)

# ============================================================
# 4. Calculate cumulative R2_OS over time by asset
# ============================================================

models = [
    "ols_restricted",
    "ols4_restricted",
    "ridge_restricted",
    "elastic_net_restricted",
    "random_forest"
]

label_map = {
    "ols_restricted": "OLS restricted",
    "ols4_restricted": "OLS-4 restricted",
    "ridge_restricted": "Ridge restricted",
    "elastic_net_restricted": "Elastic Net restricted",
    "random_forest": "Random Forest"
}

results = []

for model in models:
    temp_list = []

    for asset, g in df.groupby("asset"):
        g = g.sort_values("date").copy()

        model_se = (g["realized"] - g[model]) ** 2
        benchmark_se = (g["realized"] - g["historical_mean"]) ** 2

        cum_model_se = model_se.cumsum()
        cum_benchmark_se = benchmark_se.cumsum()

        g["cum_r2_os"] = 1 - (cum_model_se / cum_benchmark_se)
        g["model"] = model

        temp_list.append(g[["date", "asset", "model", "cum_r2_os"]])

    results.append(pd.concat(temp_list, ignore_index=True))

r2_time_asset = pd.concat(results, ignore_index=True)

# ============================================================
# 5. Average cumulative R2_OS across industries
# ============================================================

r2_avg = (
    r2_time_asset
    .groupby(["date", "model"], as_index=False)["cum_r2_os"]
    .mean()
)

r2_path = forecast_dir / "cumulative_r2_os_restricted_models.csv"
r2_avg.to_csv(r2_path, index=False)

print("\nCumulative R2 file saved to:")
print(r2_path)

# ============================================================
# 6. Final average cumulative R2_OS by model
# ============================================================

final_r2 = (
    r2_avg
    .sort_values("date")
    .groupby("model")
    .tail(1)
    .sort_values("cum_r2_os", ascending=False)
)

print("\nFinal average cumulative R2_OS:")
print(final_r2[["model", "cum_r2_os"]])

# ============================================================
# 7. Plot cumulative R2_OS from 1970 onward
# ============================================================

plot_start = pd.Timestamp("1970-01-01")
r2_plot = r2_avg[r2_avg["date"] >= plot_start].copy()

plt.figure(figsize=(11, 6))

for model in models:
    temp = r2_plot[r2_plot["model"] == model]
    plt.plot(
        temp["date"],
        temp["cum_r2_os"],
        label=label_map[model],
        linewidth=1.8
    )

plt.axhline(0, linestyle="--", linewidth=1)

plt.title("Cumulative Out-of-Sample $R^2$ Over Time")
plt.xlabel("Date")
plt.ylabel("Average cumulative $R^2_{OS}$")
plt.legend(loc="best")
plt.tight_layout()

fig_path = forecast_dir / "cumulative_r2_os_restricted_models_1970_onward.png"
plt.savefig(fig_path, dpi=300, bbox_inches="tight")
plt.show()

print("\nFigure saved to:")
print(fig_path)

# ============================================================
# 8. Identify largest monthly changes in cumulative R2_OS
# ============================================================

r2_avg = r2_avg.sort_values(["model", "date"]).copy()

r2_avg["r2_change"] = (
    r2_avg
    .groupby("model")["cum_r2_os"]
    .diff()
)

r2_avg["abs_change"] = r2_avg["r2_change"].abs()

largest_jumps = (
    r2_avg[r2_avg["date"] >= plot_start]
    .sort_values("abs_change", ascending=False)
    .head(30)
)

print("\nLargest monthly changes in average cumulative R2_OS from 1970 onward:")
print(
    largest_jumps[
        ["date", "model", "cum_r2_os", "r2_change"]
    ].to_string(index=False)
)

# ============================================================
# 9. Specifically inspect early/mid-1970s shock periods
# ============================================================

shock_period = r2_avg[
    (
        (r2_avg["date"] >= "1970-01-01") &
        (r2_avg["date"] <= "1972-12-31")
    )
    |
    (
        (r2_avg["date"] >= "1975-01-01") &
        (r2_avg["date"] <= "1977-12-31")
    )
].copy()

shock_period = shock_period.sort_values("abs_change", ascending=False)

print("\nLargest changes during 1970-1972 and 1975-1977:")
print(
    shock_period[
        ["date", "model", "cum_r2_os", "r2_change"]
    ].head(30).to_string(index=False)
)

# Optional: save shock table
shock_path = forecast_dir / "largest_r2_os_changes_1970s.csv"
shock_period.to_csv(shock_path, index=False)

print("\nShock-period table saved to:")
print(shock_path)

# ============================================================
# 10. Asset-level drivers of the shocks
# ============================================================

r2_time_asset = r2_time_asset.sort_values(["asset", "model", "date"]).copy()

r2_time_asset["r2_change"] = (
    r2_time_asset
    .groupby(["asset", "model"])["cum_r2_os"]
    .diff()
)

r2_time_asset["abs_change"] = r2_time_asset["r2_change"].abs()

asset_shocks = r2_time_asset[
    (
        (r2_time_asset["date"] >= "1970-01-01") &
        (r2_time_asset["date"] <= "1972-12-31")
    )
    |
    (
        (r2_time_asset["date"] >= "1975-01-01") &
        (r2_time_asset["date"] <= "1977-12-31")
    )
].copy()

asset_shocks = asset_shocks.sort_values("abs_change", ascending=False)

print("\nLargest asset-level changes during 1970-1972 and 1975-1977:")
print(
    asset_shocks[
        ["date", "asset", "model", "cum_r2_os", "r2_change"]
    ].head(40).to_string(index=False)
)

# Optional: save asset-level shock table
asset_shock_path = forecast_dir / "largest_asset_level_r2_os_changes_1970s.csv"
asset_shocks.to_csv(asset_shock_path, index=False)

print("\nAsset-level shock table saved to:")
print(asset_shock_path)