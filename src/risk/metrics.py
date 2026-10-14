"""
metrics.py
----------
Risk metrics for daily return series.

Every function accepts either a pd.Series (one asset or one portfolio) or a
pd.DataFrame (one column per asset), unless stated otherwise.

Sign convention: VaR, ES and maximum drawdown are returned as positive numbers,
i.e. as losses. A 99% VaR of 0.03 means "we expect to lose more than 3% on
1 day out of 100".
"""

from typing import Union

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm
from scipy.special import xlogy

Returns = Union[pd.Series, pd.DataFrame]
Scalar = Union[float, pd.Series]


# -----------------------------------------------------------------------------
# Input checks
# -----------------------------------------------------------------------------

def _check_returns(r: Returns) -> None:
    if not isinstance(r, (pd.Series, pd.DataFrame)):
        raise TypeError(
            f"Expected a pandas Series or DataFrame, got {type(r).__name__}."
        )
    if len(r) == 0:
        raise ValueError("The return series is empty.")


def _check_level(level: float) -> None:
    if not 0 < level < 100:
        raise ValueError(f"level must be strictly between 0 and 100, got {level}.")


# -----------------------------------------------------------------------------
# Value at Risk
# -----------------------------------------------------------------------------

def var_historic(r: Returns, level: float = 5) -> Scalar:
    """
    Historical VaR: empirical quantile of past returns.

    Parameters
    ----------
    r : pd.Series or pd.DataFrame
        Periodic simple returns.
    level : float
        Tail probability in percent (5 for a 95% VaR, 1 for a 99% VaR).

    Returns
    -------
    float or pd.Series
        VaR as a positive loss.
    """
    _check_returns(r)
    _check_level(level)
    return -r.quantile(level / 100)


def var_gaussian(r: Returns, level: float = 5, modified: bool = False) -> Scalar:
    """
    Parametric VaR under a Normal assumption, with an optional
    Cornish-Fisher adjustment for skewness and excess kurtosis.

    Parameters
    ----------
    r : pd.Series or pd.DataFrame
        Periodic simple returns.
    level : float
        Tail probability in percent.
    modified : bool
        If True, the Normal quantile is replaced by the Cornish-Fisher quantile.

    Returns
    -------
    float or pd.Series
        VaR as a positive loss.
    """
    _check_returns(r)
    _check_level(level)

    z = norm.ppf(level / 100)

    if modified:
        s = r.skew()
        k = r.kurt()   # pandas returns excess kurtosis
        z = (
            z
            + (z ** 2 - 1) * s / 6
            + (z ** 3 - 3 * z) * k / 24
            - (2 * z ** 3 - 5 * z) * s ** 2 / 36
        )

    return -(r.mean() + z * r.std(ddof=1))


def var_cornish_fisher(r: Returns, level: float = 5) -> Scalar:
    """Cornish-Fisher VaR. Shortcut for var_gaussian(r, level, modified=True)."""
    return var_gaussian(r, level, modified=True)


# -----------------------------------------------------------------------------
# Expected Shortfall
# -----------------------------------------------------------------------------

def es_historic(r: Returns, level: float = 5) -> Scalar:
    """
    Historical Expected Shortfall: average return on the days where the
    loss is beyond the historical VaR.

    Parameters
    ----------
    r : pd.Series or pd.DataFrame
        Periodic simple returns.
    level : float
        Tail probability in percent.

    Returns
    -------
    float or pd.Series
        ES as a positive loss.
    """
    _check_returns(r)
    _check_level(level)

    threshold = r.quantile(level / 100)
    tail = r.where(r <= threshold)   # keeps tail days, NaN elsewhere
    return -tail.mean()


def es_gaussian(r: Returns, level: float = 5) -> Scalar:
    """
    Expected Shortfall under a Normal assumption (closed form).

    ES = -(mu - sigma * phi(z) / alpha)

    Parameters
    ----------
    r : pd.Series or pd.DataFrame
        Periodic simple returns.
    level : float
        Tail probability in percent.

    Returns
    -------
    float or pd.Series
        ES as a positive loss.
    """
    _check_returns(r)
    _check_level(level)

    alpha = level / 100
    z = norm.ppf(alpha)
    return -(r.mean() - r.std(ddof=1) * norm.pdf(z) / alpha)


# -----------------------------------------------------------------------------
# Downside risk and drawdowns
# -----------------------------------------------------------------------------

def semi_deviation(r: Returns, target: float = 0.0) -> Scalar:
    """
    Downside deviation below a target return.

    Only returns below the target contribute, so upside volatility
    is not penalised.

    Parameters
    ----------
    r : pd.Series or pd.DataFrame
        Periodic simple returns.
    target : float
        Minimum acceptable return per period (0 by default).

    Returns
    -------
    float or pd.Series
        Periodic downside deviation (not annualised).
    """
    _check_returns(r)
    shortfall = (r - target).clip(upper=0)
    return np.sqrt((shortfall ** 2).mean())


def drawdown(r: Returns) -> Returns:
    """
    Drawdown path: percentage distance of the wealth index from its
    running maximum.

    Parameters
    ----------
    r : pd.Series or pd.DataFrame
        Periodic simple returns.

    Returns
    -------
    pd.Series or pd.DataFrame
        Drawdown at each date, as a negative number (0 at a new high).
    """
    _check_returns(r)
    wealth = (1 + r.fillna(0)).cumprod()
    peaks = wealth.cummax()
    return wealth / peaks - 1


def max_drawdown(r: Returns) -> Scalar:
    """Largest peak-to-trough loss, as a positive number."""
    return -drawdown(r).min()


def max_drawdown_date(r: Returns) -> Union[pd.Timestamp, pd.Series]:
    """Date of the trough of the maximum drawdown."""
    return drawdown(r).idxmin()


# -----------------------------------------------------------------------------
# Summary table
# -----------------------------------------------------------------------------

def risk_summary(
    r: pd.DataFrame,
    level: float = 5,
    periods_per_year: int = 252
) -> pd.DataFrame:
    """
    One-line risk profile per column.

    Parameters
    ----------
    r : pd.DataFrame
        Periodic simple returns, one column per asset.
    level : float
        Tail probability in percent used for VaR and ES.
    periods_per_year : int
        252 for daily data, 12 for monthly data.

    Returns
    -------
    pd.DataFrame
        Assets in rows, risk metrics in columns.
    """
    if not isinstance(r, pd.DataFrame):
        raise TypeError("risk_summary expects a DataFrame with one column per asset.")

    ann = np.sqrt(periods_per_year)
    conf = 100 - level

    return pd.DataFrame({
        "Volatility (ann.)": r.std() * ann,
        "Semi-deviation (ann.)": semi_deviation(r) * ann,
        "Skewness": r.skew(),
        "Excess Kurtosis": r.kurt(),
        f"VaR Hist {conf:g}%": var_historic(r, level),
        f"VaR Gauss {conf:g}%": var_gaussian(r, level),
        f"VaR CF {conf:g}%": var_cornish_fisher(r, level),
        f"ES Hist {conf:g}%": es_historic(r, level),
        "Max Drawdown": max_drawdown(r),
    })


# -----------------------------------------------------------------------------
# Rolling VaR and backtesting
# -----------------------------------------------------------------------------

def rolling_var(
    r: Returns,
    window: int = 252,
    level: float = 1,
    method: str = "historic"
) -> Returns:
    """
    Out-of-sample VaR forecast estimated on a rolling window.

    The forecast for day t only uses data up to day t-1 (shift of one day),
    so there is no look-ahead bias.

    Parameters
    ----------
    r : pd.Series or pd.DataFrame
        Periodic simple returns.
    window : int
        Estimation window length in periods.
    level : float
        Tail probability in percent.
    method : str
        'historic', 'gaussian' or 'cornish-fisher'.

    Returns
    -------
    pd.Series or pd.DataFrame
        VaR forecast as a positive loss, NaN during the first `window` days.
    """
    _check_returns(r)
    _check_level(level)

    p = level / 100
    roll = r.rolling(window)

    if method == "historic":
        var = -roll.quantile(p)
    elif method in ("gaussian", "cornish-fisher"):
        z = norm.ppf(p)
        if method == "cornish-fisher":
            s = roll.skew()
            k = roll.kurt()
            z = (
                z
                + (z ** 2 - 1) * s / 6
                + (z ** 3 - 3 * z) * k / 24
                - (2 * z ** 3 - 5 * z) * s ** 2 / 36
            )
        var = -(roll.mean() + z * roll.std())
    else:
        raise ValueError(
            f"Unknown method '{method}'. Use 'historic', 'gaussian' or 'cornish-fisher'."
        )

    return var.shift(1)


def kupiec_test(r: pd.Series, var_forecast: pd.Series, level: float = 1) -> dict:
    """
    Kupiec (1995) proportion-of-failures test.

    H0: the observed exception rate equals the VaR tail probability.
    A p-value below 5% means the model gets the frequency of losses wrong.

    Parameters
    ----------
    r : pd.Series
        Realised returns.
    var_forecast : pd.Series
        VaR forecasts aligned on the same dates, as positive losses.
    level : float
        Tail probability in percent used to build the forecasts.

    Returns
    -------
    dict
        Observations, exceptions, expected exceptions, LR statistic, p-value.
    """
    if not isinstance(r, pd.Series) or not isinstance(var_forecast, pd.Series):
        raise TypeError("kupiec_test expects two pandas Series.")
    _check_level(level)

    data = pd.concat([r, var_forecast], axis=1, keys=["ret", "var"]).dropna()
    if data.empty:
        raise ValueError("No overlapping dates between returns and VaR forecasts.")

    exceptions = data["ret"] < -data["var"]
    t = len(data)
    x = int(exceptions.sum())
    p = level / 100
    p_hat = x / t

    # xlogy handles the 0 * log(0) = 0 case when there are no exceptions
    log_l0 = xlogy(t - x, 1 - p) + xlogy(x, p)
    log_l1 = xlogy(t - x, 1 - p_hat) + xlogy(x, p_hat)
    lr = -2 * (log_l0 - log_l1)

    return {
        "Observations": t,
        "Exceptions": x,
        "Expected": round(t * p, 1),
        "Exception rate": p_hat,
        "LR stat": lr,
        "p-value": chi2.sf(lr, df=1),
    }


def basel_zone(n_exceptions: int) -> str:
    """
    Basel traffic light for a 99% VaR backtested over 250 days.

    Green: 0 to 4 exceptions, Yellow: 5 to 9, Red: 10 or more.
    """
    if n_exceptions <= 4:
        return "Green"
    if n_exceptions <= 9:
        return "Yellow"
    return "Red"
