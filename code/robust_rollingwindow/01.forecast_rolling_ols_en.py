import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.linear_model import LinearRegression, ElasticNet
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

OLS_OUTPUT_PATH = OUTPUT_DIR / "forecasts_ols_rolling_240.csv"
ELASTIC_OUTPUT_PATH = OUTPUT_DIR / "forecasts_elastic_net_rolling_240.csv"


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

print("Number of predictors:", len(predictors))
print("Targets:")
print(targets)


# ============================================================
# 4. Rolling-window settings
# ============================================================

rolling_window = 240  # 20 years

# Use same-type grid as before. Adjust if your original grid was different.
elastic_grid = {
    "elastic__alpha": np.array([
        0.0001, 0.001, 0.01, 0.1, 1,
        10, 30, 100, 300, 1000,
        3000, 5000, 8000, 10000, 30000,
        100000, 300000, 1000000
    ]),
    "elastic__l1_ratio": [0.1, 0.5, 0.9]
}

best_params_by_period_asset = {}

ols_forecasts = []
elastic_forecasts = []


# ============================================================
# 5. Rolling-window forecasting loop
# ============================================================

for t in range(rolling_window, len(data)):

    print("Forecast step:", t)

    train = data.iloc[t - rolling_window:t]
    test = data.iloc[t:t + 1]

    forecast_date = test["date"].iloc[0]
    forecast_year = forecast_date.year

    # Recalculate Elastic Net hyperparameters every 5 years
    tuning_period = forecast_year - (forecast_year % 5)

    X_train = train[predictors]
    X_test = test[predictors]

    for target in targets:

        y_train = train[target]
        realized = test[target].iloc[0]

        # Rolling historical mean benchmark
        historical_mean = y_train.mean()

        # ----------------------------------------------------
        # OLS rolling forecast
        # ----------------------------------------------------

        ols_model = LinearRegression()
        ols_model.fit(X_train, y_train)

        ols_forecast = ols_model.predict(X_test)[0]
        ols_restricted = max(ols_forecast, 0)

        ols_forecasts.append({
            "date": forecast_date,
            "asset": target,
            "realized": realized,
            "historical_mean": historical_mean,
            "ols": ols_forecast,
            "ols_restricted": ols_restricted
        })

        # ----------------------------------------------------
        # Elastic Net rolling forecast
        # ----------------------------------------------------

        param_key = (tuning_period, target)

        if param_key not in best_params_by_period_asset:

            elastic_pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("elastic", ElasticNet(
                    max_iter=100000,
                    random_state=42
                ))
            ])

            elastic_cv = GridSearchCV(
                estimator=elastic_pipeline,
                param_grid=elastic_grid,
                cv=TimeSeriesSplit(n_splits=5),
                scoring="neg_mean_squared_error",
                n_jobs=-1
            )

            elastic_cv.fit(X_train, y_train)

            best_params = elastic_cv.best_params_
            best_params_by_period_asset[param_key] = best_params

        else:
            best_params = best_params_by_period_asset[param_key]

        elastic_model = Pipeline([
            ("scaler", StandardScaler()),
            ("elastic", ElasticNet(
                alpha=best_params["elastic__alpha"],
                l1_ratio=best_params["elastic__l1_ratio"],
                max_iter=100000,
                random_state=42
            ))
        ])

        elastic_model.fit(X_train, y_train)

        elastic_forecast = elastic_model.predict(X_test)[0]
        elastic_restricted = max(elastic_forecast, 0)

        elastic_forecasts.append({
            "date": forecast_date,
            "asset": target,
            "realized": realized,
            "historical_mean": historical_mean,
            "elastic_net": elastic_forecast,
            "elastic_net_restricted": elastic_restricted,
            "elastic_alpha": best_params["elastic__alpha"],
            "elastic_l1_ratio": best_params["elastic__l1_ratio"],
            "tuning_period": tuning_period
        })


# ============================================================
# 6. Save OLS forecasts
# ============================================================

ols_df = pd.DataFrame(ols_forecasts)
ols_df = ols_df.round(8)

ols_df.to_csv(OLS_OUTPUT_PATH, index=False)

print("\nRolling OLS forecasts saved successfully.")
print(ols_df.head())
print(ols_df.shape)
print(f"Saved as {OLS_OUTPUT_PATH}")


# ============================================================
# 7. Save Elastic Net forecasts
# ============================================================

elastic_df = pd.DataFrame(elastic_forecasts)
elastic_df = elastic_df.round(8)

elastic_df.to_csv(ELASTIC_OUTPUT_PATH, index=False)

print("\nRolling Elastic Net forecasts saved successfully.")
print(elastic_df.head())
print(elastic_df.shape)
print(f"Saved as {ELASTIC_OUTPUT_PATH}")

print("\nSelected Elastic Net alpha summary:")
print(elastic_df["elastic_alpha"].describe())

print("\nSelected Elastic Net l1_ratio counts:")
print(elastic_df["elastic_l1_ratio"].value_counts())