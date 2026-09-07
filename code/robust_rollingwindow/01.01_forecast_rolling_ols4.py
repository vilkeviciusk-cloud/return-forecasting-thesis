import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.linear_model import LinearRegression


# ============================================================
# 1. File paths
# ============================================================

DATA_PATH = r"C:\Users\vilke\OneDrive\Dokumentai\thesis_project\data\processed\dataset_thesis_baseline.csv"

OUTPUT_DIR = Path(
    r"C:\Users\vilke\OneDrive\Dokumentai\thesis_project\output\forecasts\robustness"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "forecasts_ols4_rolling_240.csv"


# ============================================================
# 2. Load baseline dataset
# ============================================================

data = pd.read_csv(DATA_PATH)
data["date"] = pd.to_datetime(data["date"])


# ============================================================
# 3. Define OLS-4 predictors and targets
# ============================================================

ols4_predictors = [
    "dp_lag1",
    "tbl_lag1",
    "tms_lag1",
    "dfy_lag1"
]

targets = [col for col in data.columns if col.endswith("_excess")]


# ============================================================
# 4. Rolling-window settings
# ============================================================

rolling_window = 240  # 20 years

all_forecasts = []


# ============================================================
# 5. Rolling-window OLS-4 forecasting loop
# ============================================================

for t in range(rolling_window, len(data)):

    print("Forecast step:", t)

    train = data.iloc[t - rolling_window:t]
    test = data.iloc[t:t + 1]

    forecast_date = test["date"].iloc[0]

    X_train = train[ols4_predictors]
    X_test = test[ols4_predictors]

    for target in targets:

        y_train = train[target]
        realized = test[target].iloc[0]

        historical_mean = y_train.mean()

        ols4_model = LinearRegression()
        ols4_model.fit(X_train, y_train)

        ols4_forecast = ols4_model.predict(X_test)[0]
        ols4_restricted = max(ols4_forecast, 0)

        all_forecasts.append({
            "date": forecast_date,
            "asset": target,
            "realized": realized,
            "historical_mean": historical_mean,
            "ols4": ols4_forecast,
            "ols4_restricted": ols4_restricted
        })


# ============================================================
# 6. Save forecasts
# ============================================================

forecasts = pd.DataFrame(all_forecasts)
forecasts = forecasts.round(8)

forecasts.to_csv(OUTPUT_PATH, index=False)

print("\nRolling OLS-4 forecasts saved successfully.")
print(forecasts.head())
print(forecasts.shape)
print(f"\nSaved as {OUTPUT_PATH}")