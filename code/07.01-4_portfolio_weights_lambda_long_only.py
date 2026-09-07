import pandas as pd
import numpy as np


# ============================================================
# 1. File paths
# ============================================================

WEIGHTS_PATH = r"C:\Users\vilke\OneDrive\Dokumentai\thesis_project\output\portfolios\portfolio_weights_long_only.csv"

OUTPUT_PATH = r"C:\Users\vilke\OneDrive\Dokumentai\thesis_project\output\portfolios\portfolio_weights_lambda_long_only.csv"


# ============================================================
# 2. Load long-only weights
# ============================================================

weights = pd.read_csv(WEIGHTS_PATH)
weights["date"] = pd.to_datetime(weights["date"])


# ============================================================
# 3. Settings
# ============================================================

asset_cols = [
    "NoDur_excess", "Durbl_excess", "Manuf_excess", "Enrgy_excess",
    "HiTec_excess", "Telcm_excess", "Shops_excess", "Hlth_excess",
    "Utils_excess", "Other_excess"
]

lambdas = [0.25, 0.50, 0.75]

equal_weight = 1 / len(asset_cols)


# ============================================================
# 4. Create lambda-combined weights
# ============================================================

all_weights = []

# Keep original 1/N benchmark
one_over_n = weights[weights["model"] == "one_over_n"].copy()
one_over_n["base_model"] = "one_over_n"
one_over_n["lambda"] = np.nan
one_over_n["model"] = "one_over_n"

all_weights.append(one_over_n)

# Combine only model-based long-only portfolios with 1/N
model_weights = weights[weights["model"] != "one_over_n"].copy()

for lam in lambdas:

    temp = model_weights.copy()

    temp["base_model"] = temp["model"]

    temp["weight"] = (
        lam * temp["weight"]
        + (1 - lam) * equal_weight
    )

    temp["lambda"] = lam

    temp["model"] = (
        temp["base_model"]
        + "_lambda_"
        + str(int(lam * 100))
    )

    all_weights.append(temp)


# ============================================================
# 5. Save lambda-combined weights
# ============================================================

lambda_weights = pd.concat(all_weights, ignore_index=True)

lambda_weights = lambda_weights[
    ["date", "model", "base_model", "lambda", "asset", "weight"]
]

lambda_weights = lambda_weights.round(8)

lambda_weights.to_csv(OUTPUT_PATH, index=False)

print("Lambda long-only weights saved successfully.")
print(lambda_weights.head(30))
print(lambda_weights.shape)