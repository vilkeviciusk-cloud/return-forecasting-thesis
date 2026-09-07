import pandas as pd
import numpy as np

# ============================================================
# File paths
# ============================================================

RAW_DIR = r"C:\Users\vilke\OneDrive\Dokumentai\thesis_project\data\raw"
PROCESSED_DIR = r"C:\Users\vilke\OneDrive\Dokumentai\thesis_project\data\processed"


# ============================================================
# 1. Load Goyal-Welch predictor data
# ============================================================

gw = pd.read_excel(RAW_DIR + r"\PredictorData2024.xlsx")
gw["date"] = pd.to_datetime(gw["yyyymm"], format="%Y%m")


# ============================================================
# 2. Load 10 Industry Portfolio data
# ============================================================

industry_cols = [
    "NoDur", "Durbl", "Manuf", "Enrgy", "HiTec",
    "Telcm", "Shops", "Hlth", "Utils", "Other"
]

port = pd.read_csv(
    RAW_DIR + r"\10_Industry_Portfolios.csv",
    skiprows=12,
    nrows=1197,
    names=["yyyymm"] + industry_cols
)

port["yyyymm"] = port["yyyymm"].astype(str).str.strip()
port["date"] = pd.to_datetime(port["yyyymm"], format="%Y%m")


# ============================================================
# 3. Merge both datasets by date
# ============================================================

data = pd.merge(gw, port, on="date", how="inner")

data = data.rename(columns={"yyyymm_x": "yyyymm"})
data = data.drop(columns=["yyyymm_y"], errors="ignore")

data = data.sort_values("date").reset_index(drop=True)


# ============================================================
# 4. Clean and convert industry returns
# ============================================================

data[industry_cols] = data[industry_cols].replace([-99.99, -999], np.nan)
data[industry_cols] = data[industry_cols] / 100


# ============================================================
# 5. Create excess returns
# ============================================================

excess_cols = []

for col in industry_cols:
    excess_col = col + "_excess"
    data[excess_col] = data[col] - data["Rfree"]
    excess_cols.append(excess_col)


# ============================================================
# 6. Create additional predictors
# ============================================================

data["dp"] = np.log(data["D12"] / data["Index"])
data["ep"] = np.log(data["E12"] / data["Index"])

data["tms"] = data["lty"] - data["tbl"]
data["dfy"] = data["BAA"] - data["AAA"]
data["dfr"] = data["corpr"] - data["ltr"]


# ============================================================
# 7. Define predictor sets
# ============================================================

baseline_predictors = [
    "dp", "ep", "b/m", "tbl", "lty", "tms",
    "dfy", "ntis", "infl", "ltr", "dfr", "svar"
]

extended_predictors = baseline_predictors + [
    "CRSP_SPvw", "CRSP_SPvwx"
]


# ============================================================
# 8. Create lagged predictor columns
# ============================================================

baseline_predictors_lag1 = []
extended_predictors_lag1 = []

for col in extended_predictors:
    lag_col = col + "_lag1"
    data[lag_col] = data[col].shift(1)

    extended_predictors_lag1.append(lag_col)

    if col in baseline_predictors:
        baseline_predictors_lag1.append(lag_col)


# ============================================================
# 9. Define targets
# ============================================================

target_cols = excess_cols


# ============================================================
# 10. Create clean datasets
# ============================================================

data_baseline = data.dropna(
    subset=baseline_predictors_lag1 + target_cols + ["Rfree"]
).reset_index(drop=True)

data_extended = data.dropna(
    subset=extended_predictors_lag1 + target_cols + ["Rfree"]
).reset_index(drop=True)


# ============================================================
# 11. Keep only relevant columns
# ============================================================

baseline_keep_cols = (
    ["date", "yyyymm", "Rfree"]
    + industry_cols
    + excess_cols
    + baseline_predictors
    + baseline_predictors_lag1
)

extended_keep_cols = (
    ["date", "yyyymm", "Rfree"]
    + industry_cols
    + excess_cols
    + extended_predictors
    + extended_predictors_lag1
)

data_baseline = data_baseline[baseline_keep_cols]
data_extended = data_extended[extended_keep_cols]


# ============================================================
# 12. Save datasets
# ============================================================

data_baseline = data_baseline.round(6)
data_extended = data_extended.round(6)

data_baseline.to_csv(PROCESSED_DIR + r"\dataset_thesis_baseline.csv", index=False)
data_extended.to_csv(PROCESSED_DIR + r"\dataset_thesis_extended.csv", index=False)

data_baseline.to_excel(PROCESSED_DIR + r"\dataset_thesis_baseline.xlsx", index=False)
data_extended.to_excel(PROCESSED_DIR + r"\dataset_thesis_extended.xlsx", index=False)


# ============================================================
# 13. Final checks
# ============================================================

print("Datasets saved successfully.")
print("Baseline shape:", data_baseline.shape)
print("Extended shape:", data_extended.shape)
print("Targets:", target_cols)