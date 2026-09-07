import pandas as pd
from pathlib import Path


# ============================================================
# 1. File paths
# ============================================================

ROBUSTNESS_DIR = Path(
    r"C:\Users\vilke\OneDrive\Dokumentai\thesis_project\output\forecasts\robustness"
)

ols_path = ROBUSTNESS_DIR / "forecasts_ols_rolling_240.csv"
ols4_path = ROBUSTNESS_DIR / "forecasts_ols4_rolling_240.csv"
ridge_path = ROBUSTNESS_DIR / "forecasts_ridge_rolling_240.csv"
elastic_path = ROBUSTNESS_DIR / "forecasts_elastic_net_rolling_240.csv"
rf_path = ROBUSTNESS_DIR / "forecasts_random_forest_rolling_240.csv"

combined_output_path = ROBUSTNESS_DIR / "combined_forecasts_rolling_240_for_portfolio_all_models.csv"
summary_output_path = ROBUSTNESS_DIR / "forecast_summary_rolling_240_all_models.csv"


# ============================================================
# 2. Load forecast files
# ============================================================

ols = pd.read_csv(ols_path)
ols4 = pd.read_csv(ols4_path)
ridge = pd.read_csv(ridge_path)
elastic = pd.read_csv(elastic_path)
rf = pd.read_csv(rf_path)

for df in [ols, ols4, ridge, elastic, rf]:
    df["date"] = pd.to_datetime(df["date"])


# ============================================================
# 3. Keep required columns
# ============================================================

ols_keep = ols[
    [
        "date",
        "asset",
        "realized",
        "historical_mean",
        "ols",
        "ols_restricted"
    ]
].copy()

ols4_keep = ols4[
    [
        "date",
        "asset",
        "ols4",
        "ols4_restricted"
    ]
].copy()

ridge_keep = ridge[
    [
        "date",
        "asset",
        "ridge",
        "ridge_restricted"
    ]
].copy()

elastic_keep = elastic[
    [
        "date",
        "asset",
        "elastic_net",
        "elastic_net_restricted"
    ]
].copy()

rf_keep = rf[
    [
        "date",
        "asset",
        "random_forest"
    ]
].copy()


# ============================================================
# 4. Merge all forecasts
# ============================================================

combined = ols_keep.merge(
    ols4_keep,
    on=["date", "asset"],
    how="inner"
)

combined = combined.merge(
    ridge_keep,
    on=["date", "asset"],
    how="inner"
)

combined = combined.merge(
    elastic_keep,
    on=["date", "asset"],
    how="inner"
)

combined = combined.merge(
    rf_keep,
    on=["date", "asset"],
    how="inner"
)

combined = combined.sort_values(["date", "asset"]).reset_index(drop=True)
combined = combined.round(8)


# ============================================================
# 4.1. Restrict to main-analysis start date
# ============================================================

START_DATE = pd.Timestamp("1967-01-01")

combined = combined[combined["date"] >= START_DATE].copy()
combined = combined.reset_index(drop=True)

print("Rolling sample starts:", combined["date"].min())
print("Rolling sample ends:", combined["date"].max())
print("Number of months:", combined["date"].nunique())


# ============================================================
# 4.2. Save combined rolling forecast file
# ============================================================

combined.to_csv(combined_output_path, index=False)

print("\nCombined rolling forecast file saved successfully.")
print(combined.head())
print(combined.shape)
print(f"\nSaved as {combined_output_path}")


# ============================================================
# 5. Helper function for summary evaluation
# ============================================================

def add_summary_row(results, df, model_name, restriction, forecast_col):

    df = df.copy()

    df["error_hist"] = df["realized"] - df["historical_mean"]
    df["error_model"] = df["realized"] - df[forecast_col]

    df["sq_error_hist"] = df["error_hist"] ** 2
    df["sq_error_model"] = df["error_model"] ** 2

    industry_rows = []

    for asset in df["asset"].unique():

        temp = df[df["asset"] == asset]

        mspe_hist = temp["sq_error_hist"].mean()
        mspe_model = temp["sq_error_model"].mean()

        r2os = 1 - (mspe_model / mspe_hist)

        industry_rows.append({
            "asset": asset,
            "MSPE_Model": mspe_model,
            "R2OS": r2os
        })

    industry_df = pd.DataFrame(industry_rows)

    results.append({
        "Model": model_name,
        "Restriction": restriction,
        "Average MSPE": industry_df["MSPE_Model"].mean(),
        "Average R2OS": industry_df["R2OS"].mean(),
        "No. positive R2OS": (industry_df["R2OS"] > 0).sum()
    })


# ============================================================
# 6. Create rolling forecast summary table
# ============================================================

summary_rows = []

add_summary_row(
    results=summary_rows,
    df=combined,
    model_name="OLS",
    restriction="No",
    forecast_col="ols"
)

add_summary_row(
    results=summary_rows,
    df=combined,
    model_name="OLS",
    restriction="Yes",
    forecast_col="ols_restricted"
)

add_summary_row(
    results=summary_rows,
    df=combined,
    model_name="OLS-4",
    restriction="No",
    forecast_col="ols4"
)

add_summary_row(
    results=summary_rows,
    df=combined,
    model_name="OLS-4",
    restriction="Yes",
    forecast_col="ols4_restricted"
)

add_summary_row(
    results=summary_rows,
    df=combined,
    model_name="Ridge",
    restriction="No",
    forecast_col="ridge"
)

add_summary_row(
    results=summary_rows,
    df=combined,
    model_name="Ridge",
    restriction="Yes",
    forecast_col="ridge_restricted"
)

add_summary_row(
    results=summary_rows,
    df=combined,
    model_name="Elastic Net",
    restriction="No",
    forecast_col="elastic_net"
)

add_summary_row(
    results=summary_rows,
    df=combined,
    model_name="Elastic Net",
    restriction="Yes",
    forecast_col="elastic_net_restricted"
)

add_summary_row(
    results=summary_rows,
    df=combined,
    model_name="Random Forest",
    restriction="No",
    forecast_col="random_forest"
)


summary = pd.DataFrame(summary_rows)

summary = summary[
    [
        "Model",
        "Restriction",
        "Average MSPE",
        "Average R2OS",
        "No. positive R2OS"
    ]
]

summary = summary.round(6)

summary.to_csv(summary_output_path, index=False)

print("\nRolling forecast summary table:")
print(summary)
print(f"\nSaved as {summary_output_path}")