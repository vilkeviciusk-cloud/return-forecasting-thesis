import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression


# ============================================================
# 1. Load baseline dataset
# ============================================================

data = pd.read_csv(
    r"C:\Users\vilke\OneDrive\Dokumentai\thesis_project\data\processed\dataset_thesis_baseline.csv"
)

data["date"] = pd.to_datetime(data["date"])


# ============================================================
# 2. Define OLS-4 predictors and targets
# ============================================================

ols4_predictors = [
    "dp_lag1",     # valuation
    "tbl_lag1",    # short-term interest rate
    "tms_lag1",    # term spread / business cycle
    "dfy_lag1"     # default yield spread / credit risk
]

targets = [col for col in data.columns if col.endswith("_excess")]

print("OLS-4 predictors:")
print(ols4_predictors)

print("\nTargets:")
print(targets)


# ============================================================
# 3. Forecast settings
# ============================================================

initial_window = 480  # 40 years

all_forecasts = []


# ============================================================
# 4. Expanding-window OLS-4 forecasting loop
# ============================================================

for t in range(initial_window, len(data)):

    print("Forecast step:", t)

    train = data.iloc[:t]
    test = data.iloc[t:t+1]

    forecast_date = test["date"].iloc[0]

    X_train = train[ols4_predictors]
    X_test = test[ols4_predictors]

    for target in targets:

        y_train = train[target]
        realized = test[target].iloc[0]

        # Historical mean benchmark
        historical_mean = y_train.mean()

        # OLS-4 model
        ols4_model = LinearRegression()
        ols4_model.fit(X_train, y_train)

        ols4_forecast = ols4_model.predict(X_test)[0]

        # Restricted OLS-4 forecast
        ols4_restricted = max(ols4_forecast, 0)

        # Store forecasts
        all_forecasts.append({
            "date": forecast_date,
            "asset": target,
            "realized": realized,
            "historical_mean": historical_mean,
            "ols4": ols4_forecast,
            "ols4_restricted": ols4_restricted
        })


# ============================================================
# 5. Save forecasts
# ============================================================

forecasts = pd.DataFrame(all_forecasts)
forecasts = forecasts.round(8)

forecasts.to_csv(
    r"C:\Users\vilke\OneDrive\Dokumentai\thesis_project\output\forecasts\forecasts_ols4.csv",
    index=False
)

print("\nOLS-4 forecasts saved successfully.")
print(forecasts.head())
print(forecasts.shape)
print("\nSaved as forecasts_ols4.csv")