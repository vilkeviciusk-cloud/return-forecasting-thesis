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
# 2. Define predictors and targets
# ============================================================

predictors = [col for col in data.columns if col.endswith("_lag1")]
targets = [col for col in data.columns if col.endswith("_excess")]

print("Predictors:")
print(predictors)

print("\nTargets:")
print(targets)


# ============================================================
# 3. Forecast settings
# ============================================================

initial_window = 480  # 40 years

all_forecasts = []


# ============================================================
# 4. Expanding-window OLS forecasting loop
# ============================================================

for t in range(initial_window, len(data)):

    print("Forecast step:", t)

    train = data.iloc[:t]
    test = data.iloc[t:t+1]

    forecast_date = test["date"].iloc[0]

    X_train = train[predictors]
    X_test = test[predictors]

    for target in targets:

        y_train = train[target]
        realized = test[target].iloc[0]

        # ----------------------------------------------------
        # Historical mean benchmark
        # ----------------------------------------------------

        historical_mean = y_train.mean()

        # ----------------------------------------------------
        # OLS model
        # ----------------------------------------------------

        ols_model = LinearRegression()

        ols_model.fit(X_train, y_train)

        ols_forecast = ols_model.predict(X_test)[0]

        # ----------------------------------------------------
        # Restricted OLS forecast
        # Campbell-Thompson restriction:
        # negative excess return forecasts set to zero
        # ----------------------------------------------------

        ols_restricted = max(ols_forecast, 0)

        # ----------------------------------------------------
        # Store forecasts
        # ----------------------------------------------------

        all_forecasts.append({
            "date": forecast_date,
            "asset": target,
            "realized": realized,
            "historical_mean": historical_mean,
            "ols": ols_forecast,
            "ols_restricted": ols_restricted
        })


# ============================================================
# 5. Save forecasts
# ============================================================

forecasts = pd.DataFrame(all_forecasts)

forecasts = forecasts.round(8)

forecasts.to_csv(
    r"C:\Users\vilke\OneDrive\Dokumentai\thesis_project\output\forecasts\forecasts_ols.csv",
    index=False
)

print("\nOLS forecasts saved successfully.")
print(forecasts.head())
print(forecasts.shape)
print("\nSaved as forecasts_ols.csv")