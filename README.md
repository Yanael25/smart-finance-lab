# Smart Portfolio Lab

Developed during an M1 in Financial Markets Engineering, this project goes beyond the standard Markowitz framework to reflect how portfolio optimization is actually approached in practice.

The empirical failure of mean-variance optimization is well documented: covariance matrices
estimated on historical data are unstable, optimal weights are hypersensitive to input errors,
and out-of-sample performance consistently disappoints. Rather than ignoring this, the project
takes it as its central thread, demonstrating the failure quantitatively, then implementing
the fixes used in practice: Ledoit-Wolf shrinkage, Black-Litterman, and walk-forward
validation across multiple strategies.

The codebase is structured as a proper Python package, with modular src/ components,
type hints throughout, and notebooks serving as documented execution layers on top of the library.

## Status

This project is actively under development. The full scope covers 12 modules
spanning risk metrics, portfolio optimization, backtesting and factor models.

Completed so far:
- Data pipeline: adjusted prices, log-returns, risk-free rate alignment
- Descriptive statistics and normality testing across the full universe
- Four exploratory visualizations (normalized prices, return distributions,
  risk-return scatter, correlation heatmap)

---

## What this covers

**Data** : Adjusted price pipeline via yfinance, daily and monthly log-returns,
risk-free rate alignment via FRED (DGS1). 10-asset equity universe + S&P 500 benchmark,
January 2013 to present.

**Risk** : VaR via three methods (historical, Gaussian, Cornish-Fisher expansion),
Expected Shortfall, semi-deviation, maximum drawdown. Cornish-Fisher is motivated
empirically: all 10 assets reject normality under Jarque-Bera, with significant
negative skewness and excess kurtosis.

**Performance** : Sharpe, Treynor, Jensen alpha, Information Ratio, Tracking Error,
bull/bear beta decomposition.

**Optimization** : Markowitz mean-variance, Minimum Variance, Maximum Sharpe, Equal
Risk Contribution, CPPI, Black-Litterman. All strategies compared on the same
efficient frontier.

**Robustness** : Empirical proof of covariance instability across sub-periods.
Ledoit-Wolf shrinkage as the industry-standard correction. Black-Litterman as the
practitioner's alternative to unconstrained mean-variance.

**Validation** : Walk-forward backtesting engine: 60-month estimation window,
1-month out-of-sample test, annual rebalancing. Out-of-sample Sharpe, realized VaR
and maximum drawdown reported for all strategies.

**Stress testing** : Four historical scenarios (2008, Covid-19, Fed tightening 2022,
Tech bubble 2000), hypothetical shocks, and correlation breakdown analysis via
regime-dependent covariance matrices (HMM).

**Factor models** : CAPM, Fama-French-Carhart 4-factor, APT via PCA and macro factors
(CPI, industrial production). Portfolio alpha decomposition: how much of the Sharpe
is genuine alpha vs hidden factor exposure.

---

## Structure

smart-finance-lab/
├── src/
│   ├── data/               
│   ├── risk/               
│   ├── optimization/       
│   ├── backtesting/        
│   ├── covariance/         
│   └── factor_models/      
├── notebooks/              
├── figures/                
└── README.md


---

## Stack

Python 3.11, yfinance, FRED API, pandas, numpy, cvxpy, plotly,
statsmodels, scikit-learn, hmmlearn

---

## Setup
bash
git clone https://github.com/Yanael25/smart-finance-lab.git
cd smart-finance-lab
pip install -r requirements.txt