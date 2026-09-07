import pandas as pd
import numpy as np

from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
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

print("Predictors:")
print(predictors)

print("\nTargets:")
print(targets)


# ============================================================
# 3. Forecast settings
# ============================================================

initial_window = 480  # 40 years of monthly data

# Elastic Net tuning:
# alpha = total penalty strength
# l1_ratio = mix between Ridge and Lasso
# l1_ratio = 0 would be Ridge-like, l1_ratio = 1 is Lasso
alpha_grid = np.logspace(-8, -3, 25)
l1_ratio_grid = [0.05, 0.1, 0.25, 0.5]

all_forecasts = []

# Store best parameters for each 5-year block and asset
best_params_by_period_asset = {}


# ============================================================
# 4. Expanding-window forecasting loop
# ============================================================

for t in range(initial_window, len(data)):

    print("Forecast step:", t)

    train = data.iloc[:t]
    test = data.iloc[t:t+1]

    forecast_date = test["date"].iloc[0]
    forecast_year = forecast_date.year

    # Tune only every 5 years
    tuning_year = (forecast_year // 5) * 5

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
        # Tune Elastic Net once every 5 years for each asset
        # ----------------------------------------------------
        param_key = (tuning_year, target)

        if param_key not in best_params_by_period_asset:

            elastic_net_pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("elastic_net", ElasticNet(
                    max_iter=100000,
                    random_state=42
                ))
            ])

            elastic_net_grid = {
                "elastic_net__alpha": alpha_grid,
                "elastic_net__l1_ratio": l1_ratio_grid
            }

            elastic_net_cv = GridSearchCV(
                estimator=elastic_net_pipeline,
                param_grid=elastic_net_grid,
                cv=TimeSeriesSplit(n_splits=5),
                scoring="neg_mean_squared_error",
                n_jobs=-1
            )

            elastic_net_cv.fit(X_train, y_train)

            best_params = elastic_net_cv.best_params_
            best_params_by_period_asset[param_key] = best_params

        else:
            best_params = best_params_by_period_asset[param_key]

        best_alpha = best_params["elastic_net__alpha"]
        best_l1_ratio = best_params["elastic_net__l1_ratio"]

        # ----------------------------------------------------
        # Refit Elastic Net each month using selected parameters
        # ----------------------------------------------------
        elastic_net_model = Pipeline([
            ("scaler", StandardScaler()),
            ("elastic_net", ElasticNet(
                alpha=best_alpha,
                l1_ratio=best_l1_ratio,
                max_iter=100000,
                random_state=42
            ))
        ])

        elastic_net_model.fit(X_train, y_train)

        elastic_net_forecast = elastic_net_model.predict(X_test)[0]

        # ----------------------------------------------------
        # Restricted Elastic Net forecast
        # Campbell-Thompson-style restriction:
        # negative excess return forecasts are set to zero
        # ----------------------------------------------------
        elastic_net_restricted = max(elastic_net_forecast, 0)

        # ----------------------------------------------------
        # Store forecasts
        # ----------------------------------------------------
        all_forecasts.append({
            "date": forecast_date,
            "asset": target,
            "realized": realized,
            "historical_mean": historical_mean,
            "elastic_net": elastic_net_forecast,
            "elastic_net_restricted": elastic_net_restricted,
            "elastic_net_alpha": best_alpha,
            "elastic_net_l1_ratio": best_l1_ratio
        })


# ============================================================
# 5. Save forecasts
# ============================================================

forecasts = pd.DataFrame(all_forecasts)
forecasts = forecasts.round(8)

forecasts.to_csv(
    r"C:\Users\vilke\OneDrive\Dokumentai\thesis_project\output\forecasts\forecasts_elastic_net.csv",
    index=False
)

print("\nElastic Net forecasts saved successfully.")
print(forecasts.head())
print(forecasts.shape)
print("\nSaved as forecasts_elastic_net.csv")