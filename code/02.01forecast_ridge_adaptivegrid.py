import pandas as pd
import numpy as np

from sklearn.linear_model import Ridge
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

initial_window = 480  # 20 years of monthly data

all_forecasts = []

# Stores selected alpha for each year and asset
best_alpha_by_year_asset = {}


# ============================================================
# 4. Adaptive Ridge alpha tuning function
# ============================================================

def tune_ridge_alpha(X_train, y_train):
    """
    Tune Ridge alpha using an adaptive grid.
    If the best alpha is at the lower or upper boundary,
    the grid expands and searches again.
    """

    lower_exp = -4     # 10^-4
    upper_exp = 8      # 10^8

    best_alpha = None

    for _ in range(5):  # allow up to 5 grid expansions

        alpha_grid = np.logspace(lower_exp, upper_exp, 40)

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

        # If best alpha is at lower boundary, expand downward
        if best_alpha == alpha_grid[0]:
            lower_exp -= 2

        # If best alpha is at upper boundary, expand upward
        elif best_alpha == alpha_grid[-1]:
            upper_exp += 2

        # Otherwise, the best alpha is inside the grid
        else:
            break

    return best_alpha


# ============================================================
# 5. Expanding-window forecasting loop
# ============================================================

for t in range(initial_window, len(data)):

    print("Forecast step:", t)

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
        # Historical mean benchmark
        # ----------------------------------------------------
        historical_mean = y_train.mean()

        # ----------------------------------------------------
        # Tune Ridge alpha once per year for each asset
        # ----------------------------------------------------
        alpha_key = (forecast_year, target)

        if alpha_key not in best_alpha_by_year_asset:

            best_alpha = tune_ridge_alpha(X_train, y_train)
            best_alpha_by_year_asset[alpha_key] = best_alpha

        else:
            best_alpha = best_alpha_by_year_asset[alpha_key]

        # ----------------------------------------------------
        # Refit Ridge each month using selected yearly alpha
        # ----------------------------------------------------
        ridge_model = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=best_alpha))
        ])

        ridge_model.fit(X_train, y_train)

        ridge_forecast = ridge_model.predict(X_test)[0]

        # ----------------------------------------------------
        # Restricted Ridge forecast
        # Campbell-Thompson restriction:
        # negative excess return forecasts are set to zero
        # ----------------------------------------------------
        ridge_restricted = max(ridge_forecast, 0)

        # ----------------------------------------------------
        # Store results
        # ----------------------------------------------------
        all_forecasts.append({
            "date": forecast_date,
            "asset": target,
            "realized": realized,
            "historical_mean": historical_mean,
            "ridge": ridge_forecast,
            "ridge_restricted": ridge_restricted,
            "ridge_alpha": best_alpha
        })


# ============================================================
# 6. Save forecasts
# ============================================================

forecasts = pd.DataFrame(all_forecasts)
forecasts = forecasts.round(8)

forecasts.to_csv(
    r"C:\Users\vilke\OneDrive\Dokumentai\thesis_project\output\forecasts\forecasts_ridge_adap_grid.csv",
    index=False
)

print("\nForecasts saved successfully.")
print(forecasts.head())
print(forecasts.shape)
print("\nSaved as forecasts_ridge.csv")