import pandas as pd
data = pd.read_csv(
    r"C:\Users\vilke\OneDrive\Dokumentai\thesis_project\output\forecasts\forecasts_ridge.csv"
)

data["date"] = pd.to_datetime(data["date"])
print(data["ridge_alpha"].describe())
print((data["ridge"] < 0).sum())

print(data["ridge"].describe())
print(data["ridge_restricted"].describe())