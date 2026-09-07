Machine Learning for Out-of-Sample Return Forecasting and Portfolio Allocation

Bachelor thesis project (University of Amsterdam, BSc Econometrics and Data Science). Examines whether machine learning and regularized linear models can outperform traditional benchmarks in monthly out-of-sample return forecasting for U.S. industry portfolios, and whether any forecasting gains translate into economically valuable portfolio strategies.

Overview

Using monthly data on 10 U.S. industry portfolios combined with macro-financial predictors (Goyal–Welch dataset, 1927–2024), this project builds an expanding-window out-of-sample forecasting framework comparing:

Historical mean benchmark
OLS predictive regression (unrestricted and restricted)
OLS-4 (parsimonious 4-predictor regression)
Ridge regression
Elastic Net
Random Forest

Forecasts are evaluated using MSPE, out-of-sample R², Clark-West tests, and Diebold-Mariano tests, with a 20-year rolling-window specification as a robustness check.

Predicted returns are then converted into mean-variance portfolios under long-only, weight-cap, and shrinkage constraints, and evaluated on Sharpe ratio, certainty-equivalent return, turnover, and net performance after transaction costs.

Key Findings
Ridge regression provides the most consistent forecasting gains, outperforming the historical mean in 7 of 10 industries.
Random Forest does not reliably outperform simpler linear models — model complexity alone does not overcome the noise in monthly returns.
Capped portfolios (e.g. OLS, Random Forest with maximum weight constraints) can beat the naive 1/N benchmark before transaction costs.
Under realistic transaction costs, the 1/N benchmark remains the most robust strategy overall.
Repository Structure
├── processed/     # Cleaned and merged industry portfolio + macro-financial predictor data
├── code/          # Forecasting models, portfolio construction, and evaluation scripts
└── output/forecasts/ # Model forecast outputs and portfolio results
Methods
Forecasting: expanding-window (main) and 20-year rolling-window (robustness check) out-of-sample forecasting
Models: historical mean, OLS, OLS-4, Ridge, Elastic Net, Random Forest
Evaluation: MSPE, out-of-sample R², Clark-West test, Diebold-Mariano test
Portfolio construction: mean-variance optimization under long-only, weight-cap, and 1/N-shrinkage constraints
Portfolio evaluation: annualized return, volatility, Sharpe ratio, certainty-equivalent return, turnover, net returns after transaction costs (10/25/50 bps)
Tools

Python (pandas, NumPy, statsmodels, scikit-learn)

Note

This repository reflects an ongoing student research project; some files in code/ are exploratory or unused. The full written thesis with detailed methodology, tables, and discussion is available as Thesis.pdf .
