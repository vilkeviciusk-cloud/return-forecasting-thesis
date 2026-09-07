import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV


# ============================================================
# 1. File paths
# ============================================================

DATA_PATH = r"C:\Users\vilke\OneDrive\Dokumentai\thesis_project\data\processed\dataset_thesis_baseline.csv"

OUTPUT_DIR = Path(
    r"C:\Users\vilke\OneDrive\Dokumentai\thesis_project\output\forecasts\robustness"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "forecasts_ridge_rolling_240.csv"


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

alpha_grid = np.array([
    0.0001, 0.001, 0.01, 0.1, 1,
    10, 30, 100, 300, 1000,
    3000, 5000, 8000, 10000, 30000,
    100000, 300000, 1000000
])

all_forecasts = []

# Store selected alpha for each 5-year tuning period and asset
best_alpha_by_period_asset = {}


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
    # Example: 1967,1968,1969 -> 1965 block
    #          1970-1974 -> 1970 block
    tuning_period = forecast_year - (forecast_year % 5)

    X_train = train[predictors]
    X_test = test[predictors]

    for target in targets:

        y_train = train[target]
        realized = test[target].iloc[0]

        # Rolling historical mean benchmark
        historical_mean = y_train.mean()

        alpha_key = (tuning_period, target)

        # ----------------------------------------------------
        # Tune alpha once per 5-year period for each asset
        # ----------------------------------------------------
        if alpha_key not in best_alpha_by_period_asset:

            ridge_pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("ridge", Ridge())
            ])

            ridge_grid = {
                "ridge__alpha": alpha_grid
            }

            ridge_cv = GridSearchCV(
                estimator=ridge_pipeline,
                param_grid=ridge_grid,
                cv=TimeSeriesSplit(n_splits=5),
                scoring="neg_mean_squared_error"
            )

            ridge_cv.fit(X_train, y_train)

            best_alpha = ridge_cv.best_params_["ridge__alpha"]
            best_alpha_by_period_asset[alpha_key] = best_alpha

        else:
            best_alpha = best_alpha_by_period_asset[alpha_key]

        # ----------------------------------------------------
        # Refit Ridge each month using selected alpha
        # ----------------------------------------------------
        ridge_model = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=best_alpha))
        ])

        ridge_model.fit(X_train, y_train)

        ridge_forecast = ridge_model.predict(X_test)[0]
        ridge_restricted = max(ridge_forecast, 0)

        all_forecasts.append({
            "date": forecast_date,
            "asset": target,
            "realized": realized,
            "historical_mean": historical_mean,
            "ridge": ridge_forecast,
            "ridge_restricted": ridge_restricted,
            "ridge_alpha": best_alpha,
            "tuning_period": tuning_period
        })


# ============================================================
# 6. Save forecasts
# ============================================================

forecasts = pd.DataFrame(all_forecasts)
forecasts = forecasts.round(8)

forecasts.to_csv(OUTPUT_PATH, index=False)

print("\nRolling Ridge forecasts saved successfully.")
print(forecasts.head())
print(forecasts.shape)
print(f"\nSaved as {OUTPUT_PATH}")

print("\nSelected alpha summary:")
print(forecasts["ridge_alpha"].describe())