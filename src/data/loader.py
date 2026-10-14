"""
loader.py
---------
Downloads and stores adjusted price data from Yahoo Finance.
"""

import yfinance as yf
import pandas as pd
from typing import Optional


class DataLoader:
    """
    Downloads adjusted closing prices for a given universe of assets.

    Parameters
    ----------
    tickers : list[str]
        List of asset tickers (e.g. ['AAPL', 'MSFT']).
    benchmark : str
        Benchmark ticker (e.g. '^GSPC').
    start_date : str
        Start date in 'YYYY-MM-DD' format.
    end_date : Optional[str]
        End date in 'YYYY-MM-DD' format. Defaults to today if None.
    """

    def __init__(
        self,
        tickers: list[str],
        benchmark: str,
        start_date: str,
        end_date: Optional[str] = None
    ):
        self.tickers = tickers
        self.benchmark = benchmark
        self.start_date = start_date
        self.end_date = end_date
        self._prices: Optional[pd.DataFrame] = None

    def download(self) -> pd.DataFrame:
        """
        Downloads adjusted prices for all tickers + benchmark.

        Returns
        -------
        pd.DataFrame
            DataFrame of adjusted closing prices, shape (n_days, n_assets + 1).
        """
        all_tickers = self.tickers + [self.benchmark]

        raw = yf.download(
            tickers=all_tickers,
            start=self.start_date,
            end=self.end_date,
            auto_adjust=True,
            progress=False
        )

        prices = raw["Close"].copy()

        if prices.isnull().sum().sum() > 0:
            print(f"Warning: {prices.isnull().sum().sum()} NaN values detected.")

        self._prices = prices
        return prices

    @property
    def prices(self) -> pd.DataFrame:
        """Returns cached prices. Raises an error if download() was not called."""
        if self._prices is None:
            raise ValueError("No data loaded yet. Call download() first.")
        return self._prices

    def get_asset_prices(self) -> pd.DataFrame:
        """Returns prices for assets only, excluding benchmark."""
        return self.prices[self.tickers]

    def get_benchmark_prices(self) -> pd.Series:
        """Returns benchmark price series."""
        return self.prices[self.benchmark]
