import pandas as pd
from pathlib import Path


# ============================================================
# 1. File paths
# ============================================================

EVAL_DIR = Path(
    r"C:\Users\vilke\OneDrive\Dokumentai\thesis_project\output\forecasts"
)

OUTPUT_DIR = Path(
    r"C:\Users\vilke\OneDrive\Dokumentai\thesis_project\output\result tables"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "forecast_summary_table.csv"


# ============================================================
# 2. Evaluation files
# ============================================================
# Change file names here if yours are slightly different.

files = {
    "OLS": EVAL_DIR / "ols_evaluation.csv",
    "OLS-4": EVAL_DIR / "ols4_evaluation.csv",
    "Ridge": EVAL_DIR / "ridge_evaluation_480_5yeargap.csv",
    "Elastic Net": EVAL_DIR / "elastic_net_evaluation.csv",
    "Random Forest": EVAL_DIR / "random_forest_evaluation.csv"
}


# ============================================================
# 3. Helper function
# ============================================================

def add_model_summary(results, df, model_name, restriction, mspe_col, r2_col):
    """
    Adds one row to the final summary table.
    """

    if mspe_col not in df.columns:
        print(f"Skipping {model_name} {restriction}: missing {mspe_col}")
        return

    if r2_col not in df.columns:
        print(f"Skipping {model_name} {restriction}: missing {r2_col}")
        return

    average_mspe = df[mspe_col].mean()
    average_r2 = df[r2_col].mean()
    positive_r2_count = (df[r2_col] > 0).sum()

    results.append({
        "Model": model_name,
        "Restriction": restriction,
        "Average MSPE": average_mspe,
        "Average R2OS": average_r2,
        "No. positive R2OS": positive_r2_count
    })


# ============================================================
# 4. Build summary table
# ============================================================

results = []

# ------------------------------------------------------------
# OLS
# ------------------------------------------------------------

ols = pd.read_csv(files["OLS"])

add_model_summary(
    results=results,
    df=ols,
    model_name="OLS",
    restriction="No",
    mspe_col="MSPE_OLS",
    r2_col="R2OS_OLS"
)

add_model_summary(
    results=results,
    df=ols,
    model_name="OLS",
    restriction="Yes",
    mspe_col="MSPE_OLS_Restricted",
    r2_col="R2OS_OLS_Restricted"
)


# ------------------------------------------------------------
# OLS-4
# ------------------------------------------------------------

ols4 = pd.read_csv(files["OLS-4"])

add_model_summary(
    results=results,
    df=ols4,
    model_name="OLS-4",
    restriction="No",
    mspe_col="MSPE_OLS4",
    r2_col="R2OS_OLS4"
)

add_model_summary(
    results=results,
    df=ols4,
    model_name="OLS-4",
    restriction="Yes",
    mspe_col="MSPE_OLS4_Restricted",
    r2_col="R2OS_OLS4_Restricted"
)


# ------------------------------------------------------------
# Ridge
# ------------------------------------------------------------

ridge = pd.read_csv(files["Ridge"])

add_model_summary(
    results=results,
    df=ridge,
    model_name="Ridge",
    restriction="No",
    mspe_col="MSPE_Ridge",
    r2_col="R2OS_Ridge"
)

add_model_summary(
    results=results,
    df=ridge,
    model_name="Ridge",
    restriction="Yes",
    mspe_col="MSPE_RidgeRestricted",
    r2_col="R2OS_RidgeRestricted"
)


# ------------------------------------------------------------
# Elastic Net
# ------------------------------------------------------------

elastic = pd.read_csv(files["Elastic Net"])

add_model_summary(
    results=results,
    df=elastic,
    model_name="Elastic Net",
    restriction="No",
    mspe_col="MSPE_ElasticNet",
    r2_col="R2OS_ElasticNet"
)

add_model_summary(
    results=results,
    df=elastic,
    model_name="Elastic Net",
    restriction="Yes",
    mspe_col="MSPE_ElasticNetRestricted",
    r2_col="R2OS_ElasticNetRestricted"
)


# ------------------------------------------------------------
# Random Forest
# ------------------------------------------------------------
# RF has no restricted version unless you created one.
# So we only add unrestricted RF.

rf = pd.read_csv(files["Random Forest"])

add_model_summary(
    results=results,
    df=rf,
    model_name="Random Forest",
    restriction="No",
    mspe_col="MSPE_RandomForest",
    r2_col="R2OS_RandomForest"
)


# ============================================================
# 5. Save final table
# ============================================================

summary = pd.DataFrame(results)

summary = summary[
    [
        "Model",
        "Restriction",
        "Average MSPE",
        "Average R2OS",
        "No. positive R2OS"
    ]
]

summary["Average MSPE"] = summary["Average MSPE"].round(6)
summary["Average R2OS"] = summary["Average R2OS"].round(6)

summary.to_csv(OUTPUT_PATH, index=False)

print("Forecast summary table saved successfully.")
print(summary)
print(f"\nSaved as: {OUTPUT_PATH}")