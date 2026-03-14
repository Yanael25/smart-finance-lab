"""
preprocessing.py
----------------
Computes log-returns and aligns the risk-free rate for a given price DataFrame.
"""

import numpy as np
import pandas as pd
from fredapi import Fred
from typing import Optional


TRADING_DAYS = 252


class Preprocessor:
    """
    Computes log-returns and retrieves the risk-free rate from FRED.

    Parameters
    ----------
    prices : pd.DataFrame
        Adjusted closing prices, shape (n_days, n_assets).
    tickers : list[str]
        Asset tickers — used to separate assets from benchmark.
    benchmark : str
        Benchmark ticker.
    fred_api_key : str
        FRED API key for risk-free rate download.
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        tickers: list[str],
        benchmark: str,
        fred_api_key: str
    ):
        self.prices = prices
        self.tickers = tickers
        self.benchmark = benchmark
        self.fred_api_key = fred_api_key

    def compute_log_returns(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Computes daily and monthly log-returns.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            Daily log-returns, Monthly log-returns.
        """
        log_returns_daily = np.log(
            self.prices / self.prices.shift(1)
        ).dropna()

        log_returns_monthly = np.log(
            self.prices.resample("ME").last() /
            self.prices.resample("ME").last().shift(1)
        ).dropna()

        return log_returns_daily, log_returns_monthly

    def get_risk_free_rate(
        self,
        index_daily: pd.Index,
        index_monthly: pd.Index
    ) -> tuple[pd.Series, pd.Series]:
        """
        Downloads DGS1 from FRED and aligns it with returns index.

        Parameters
        ----------
        index_daily : pd.Index
            Index of daily returns — used for alignment.
        index_monthly : pd.Index
            Index of monthly returns — used for alignment.

        Returns
        -------
        tuple[pd.Series, pd.Series]
            Daily risk-free rate, Monthly risk-free rate.
        """
        fred = Fred(api_key=self.fred_api_key)
        rf_raw = fred.get_series(
            "DGS1",
            observation_start=self.prices.index[0]
        ).dropna()

        # Annualized % → daily decimal
        rf_daily = rf_raw / 100 / TRADING_DAYS
        rf_daily = rf_daily.reindex(index_daily, method="ffill")

        # Monthly = sum of daily rates within the month
        rf_monthly = rf_daily.resample("ME").sum()
        rf_monthly = rf_monthly.reindex(index_monthly, method="ffill")

        return rf_daily, rf_monthly

    def split_assets_benchmark(
        self,
        log_returns: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Splits a returns DataFrame into assets and benchmark.

        Parameters
        ----------
        log_returns : pd.DataFrame
            Full returns DataFrame including benchmark.

        Returns
        -------
        tuple[pd.DataFrame, pd.Series]
            Asset returns, Benchmark returns.
        """
        return log_returns[self.tickers], log_returns[self.benchmark]