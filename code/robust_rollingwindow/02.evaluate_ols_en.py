import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# 1. File paths
# ============================================================

ROBUSTNESS_DIR = Path(
    r"C:\Users\vilke\OneDrive\Dokumentai\thesis_project\output\forecasts\robustness"
)

OLS_FORECAST_PATH = ROBUSTNESS_DIR / "forecasts_ols_rolling_240.csv"
ELASTIC_FORECAST_PATH = ROBUSTNESS_DIR / "forecasts_elastic_net_rolling_240.csv"

OLS_OUTPUT_PATH = ROBUSTNESS_DIR / "evaluation_ols_rolling_240.csv"
ELASTIC_OUTPUT_PATH = ROBUSTNESS_DIR / "evaluation_elastic_net_rolling_240.csv"

SUMMARY_OUTPUT_PATH = ROBUSTNESS_DIR / "evaluation_ols_elastic_net_rolling_240_summary.csv"


# ============================================================
# 2. Evaluation helper
# ============================================================

def evaluate_forecasts(forecast_path, output_path, model_name, forecast_columns):

    data = pd.read_csv(forecast_path)
    data["date"] = pd.to_datetime(data["date"])

    # Historical mean benchmark
    data["error_historical_mean"] = data["realized"] - data["historical_mean"]
    data["sq_error_historical_mean"] = data["error_historical_mean"] ** 2

    results = []

    for asset in data["asset"].unique():

        df = data[data["asset"] == asset].copy()

        mspe_hist = df["sq_error_historical_mean"].mean()

        row = {
            "asset": asset,
            "MSPE_HistoricalMean": mspe_hist,
            "RMSPE_HistoricalMean": np.sqrt(mspe_hist)
        }

        for forecast_col in forecast_columns:

            error_col = f"error_{forecast_col}"
            sq_error_col = f"sq_error_{forecast_col}"

            df[error_col] = df["realized"] - df[forecast_col]
            df[sq_error_col] = df[error_col] ** 2

            mspe_model = df[sq_error_col].mean()

            if forecast_col.endswith("_restricted"):
                clean_name = model_name + "Restricted"
            else:
                clean_name = model_name

            row[f"MSPE_{clean_name}"] = mspe_model
            row[f"RMSPE_{clean_name}"] = np.sqrt(mspe_model)
            row[f"R2OS_{clean_name}"] = 1 - (mspe_model / mspe_hist)

        results.append(row)

    results_df = pd.DataFrame(results)
    results_df = results_df.round(6)

    results_df.to_csv(output_path, index=False)

    print(f"\nSaved evaluation: {output_path}")
    print(results_df)

    return results_df


# ============================================================
# 3. Run separate evaluations
# ============================================================

ols_eval = evaluate_forecasts(
    forecast_path=OLS_FORECAST_PATH,
    output_path=OLS_OUTPUT_PATH,
    model_name="OLS",
    forecast_columns=["ols", "ols_restricted"]
)

elastic_eval = evaluate_forecasts(
    forecast_path=ELASTIC_FORECAST_PATH,
    output_path=ELASTIC_OUTPUT_PATH,
    model_name="ElasticNet",
    forecast_columns=["elastic_net", "elastic_net_restricted"]
)


# ============================================================
# 4. Create compact summary table
# ============================================================

summary_rows = []

summary_specs = [
    {
        "Model": "OLS",
        "Restriction": "No",
        "df": ols_eval,
        "mspe_col": "MSPE_OLS",
        "r2_col": "R2OS_OLS"
    },
    {
        "Model": "OLS",
        "Restriction": "Yes",
        "df": ols_eval,
        "mspe_col": "MSPE_OLSRestricted",
        "r2_col": "R2OS_OLSRestricted"
    },
    {
        "Model": "Elastic Net",
        "Restriction": "No",
        "df": elastic_eval,
        "mspe_col": "MSPE_ElasticNet",
        "r2_col": "R2OS_ElasticNet"
    },
    {
        "Model": "Elastic Net",
        "Restriction": "Yes",
        "df": elastic_eval,
        "mspe_col": "MSPE_ElasticNetRestricted",
        "r2_col": "R2OS_ElasticNetRestricted"
    }
]

for spec in summary_specs:

    df = spec["df"]

    summary_rows.append({
        "Model": spec["Model"],
        "Restriction": spec["Restriction"],
        "Average MSPE": df[spec["mspe_col"]].mean(),
        "Average R2OS": df[spec["r2_col"]].mean(),
        "No. positive R2OS": (df[spec["r2_col"]] > 0).sum()
    })


summary_df = pd.DataFrame(summary_rows)
summary_df = summary_df.round(6)

summary_df.to_csv(SUMMARY_OUTPUT_PATH, index=False)

print("\nRolling OLS and Elastic Net summary:")
print(summary_df)
print(f"\nSaved summary as: {SUMMARY_OUTPUT_PATH}")