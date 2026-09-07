import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV


# ============================================================
# 1. File paths
# ============================================================

DATA_PATH = r"C:\Users\vilke\OneDrive\Dokumentai\thesis_project\data\processed\dataset_thesis_baseline.csv"

OUTPUT_DIR = Path(
    r"C:\Users\vilke\OneDrive\Dokumentai\thesis_project\output\forecasts\robustness"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "forecasts_random_forest_rolling_240.csv"


# ============================================================
# 2. Load baseline dataset
# ============================================================

data = pd.read_csv(DATA_PATH)
data["date"] = pd.to_datetime(data["date"])


# ============================================================
# 3. Define predictors and targets
# ============================================================

predictors = [col for col in data.columns if col.endswith("_lag1")]
targets = [col for col in data.columns if col.endswith("_excess")]

print("Predictors:")
print(predictors)

print("\nTargets:")
print(targets)


# ============================================================
# 4. Rolling-window settings
# ============================================================

rolling_window = 240  # 20 years

# Conservative RF grid — same idea as your better original version
rf_grid = {
    "n_estimators": [300, 500],
    "max_depth": [3, 5],
    "min_samples_leaf": [5, 10, 20],
    "min_samples_split": [10, 20],
    "max_features": [0.3, "sqrt"]
}

all_forecasts = []

# Store selected hyperparameters for each 5-year tuning period and asset
best_params_by_period_asset = {}


# ============================================================
# 5. Rolling-window forecasting loop
# ============================================================

for t in range(rolling_window, len(data)):

    print("Forecast step:", t)

    # Use only previous 240 months
    train = data.iloc[t - rolling_window:t]
    test = data.iloc[t:t + 1]

    forecast_date = test["date"].iloc[0]
    forecast_year = forecast_date.year

    # 5-year hyperparameter block
    tuning_period = forecast_year - (forecast_year % 5)

    X_train = train[predictors]
    X_test = test[predictors]

    for target in targets:

        y_train = train[target]
        realized = test[target].iloc[0]

        # Rolling historical mean benchmark
        historical_mean = y_train.mean()

        param_key = (tuning_period, target)

        # ----------------------------------------------------
        # Tune RF hyperparameters once per 5-year period
        # for each asset
        # ----------------------------------------------------
        if param_key not in best_params_by_period_asset:

            rf_model = RandomForestRegressor(
                random_state=42,
                n_jobs=-1
            )

            rf_cv = GridSearchCV(
                estimator=rf_model,
                param_grid=rf_grid,
                cv=TimeSeriesSplit(n_splits=5),
                scoring="neg_mean_squared_error",
                n_jobs=-1
            )

            rf_cv.fit(X_train, y_train)

            best_params = rf_cv.best_params_
            best_params_by_period_asset[param_key] = best_params

        else:
            best_params = best_params_by_period_asset[param_key]

        # ----------------------------------------------------
        # Refit RF each month using selected 5-year parameters
        # ----------------------------------------------------
        rf_final = RandomForestRegressor(
            **best_params,
            random_state=42,
            n_jobs=-1
        )

        rf_final.fit(X_train, y_train)

        rf_forecast = rf_final.predict(X_test)[0]

        all_forecasts.append({
            "date": forecast_date,
            "asset": target,
            "realized": realized,
            "historical_mean": historical_mean,
            "random_forest": rf_forecast,

            "rf_n_estimators": best_params["n_estimators"],
            "rf_max_depth": best_params["max_depth"],
            "rf_min_samples_leaf": best_params["min_samples_leaf"],
            "rf_min_samples_split": best_params["min_samples_split"],
            "rf_max_features": best_params["max_features"],
            "tuning_period": tuning_period
        })


# ============================================================
# 6. Save forecasts
# ============================================================

forecasts = pd.DataFrame(all_forecasts)
forecasts = forecasts.round(8)

forecasts.to_csv(OUTPUT_PATH, index=False)

print("\nRolling Random Forest forecasts saved successfully.")
print(forecasts.head())
print(forecasts.shape)
print(f"\nSaved as {OUTPUT_PATH}")

print("\nSelected RF hyperparameter summary:")
print(forecasts[
    [
        "rf_n_estimators",
        "rf_max_depth",
        "rf_min_samples_leaf",
        "rf_min_samples_split",
        "rf_max_features"
    ]
].describe(include="all"))