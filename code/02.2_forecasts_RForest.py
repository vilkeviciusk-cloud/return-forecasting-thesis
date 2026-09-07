import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV


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


# ============================================================
# 3. Forecast settings
# ============================================================

initial_window = 480  # 40 years of monthly data

'''rf_param_grid = {
    "n_estimators": [300, 500],
    "max_depth": [3, 5, 7],
    "min_samples_leaf": [5, 10, 20],
    "min_samples_split": [10, 20],
    "max_features": [0.3, 0.5, "sqrt"]
}
try these values broader grid'''
rf_param_grid = {
    "n_estimators": [300, 500, 800],
    "max_depth": [3, 5, 7, 10],
    "min_samples_leaf": [2, 5, 10, 20],
    "min_samples_split": [5, 10, 20],
    "max_features": [0.3, 0.5, 0.7, "sqrt"]
}

all_forecasts = []

best_params_by_year_asset = {}


# ============================================================
# 4. Expanding-window Random Forest forecasts
# ============================================================

for t in range(initial_window, len(data)):
    print(t)
    train = data.iloc[:t]
    test = data.iloc[t:t+1]

    forecast_date = test["date"].iloc[0]
    forecast_year = forecast_date.year

    X_train = train[predictors]
    X_test = test[predictors]

    for target in targets:
        y_train = train[target]
        realized = test[target].iloc[0]

        # ----------------------------------------------------
        # Tune RF hyperparameters once per 5 years per asset
        # ----------------------------------------------------
        tuning_year = (forecast_year // 5) * 5
        param_key = (tuning_year, target)

        if param_key not in best_params_by_year_asset:

            rf = RandomForestRegressor(
                random_state=42,
                n_jobs=-1
            )

            rf_cv = GridSearchCV(
                estimator=rf,
                param_grid=rf_param_grid,
                cv=TimeSeriesSplit(n_splits=5),
                scoring="neg_mean_squared_error",
                n_jobs=-1
            )

            rf_cv.fit(X_train, y_train)

            best_params = rf_cv.best_params_
            best_params_by_year_asset[param_key] = best_params

        else:
            best_params = best_params_by_year_asset[param_key]

        # ----------------------------------------------------
        # Refit RF each month using selected yearly parameters
        # ----------------------------------------------------
        rf_model = RandomForestRegressor(
            random_state=42,
            n_jobs=-1,
            **best_params
        )

        rf_model.fit(X_train, y_train)
        rf_forecast = rf_model.predict(X_test)[0]

        all_forecasts.append({
            "date": forecast_date,
            "asset": target,
            "realized": realized,
            "random_forest": rf_forecast,
            "rf_n_estimators": best_params["n_estimators"],
            "rf_max_depth": best_params["max_depth"],
            "rf_min_samples_leaf": best_params["min_samples_leaf"],
            "rf_min_samples_split": best_params["min_samples_split"],
            "rf_max_features": best_params["max_features"]
        })


# ============================================================
# 5. Save forecasts
# ============================================================

forecasts_rf = pd.DataFrame(all_forecasts)
forecasts_rf = forecasts_rf.round(8)

forecasts_rf.to_csv(
    r"C:\Users\vilke\OneDrive\Dokumentai\thesis_project\output\forecasts\forecasts_random_forest.csv",
    index=False
)

print("\nRandom Forest forecasts saved successfully.")
print(forecasts_rf.head())
print(forecasts_rf.shape)