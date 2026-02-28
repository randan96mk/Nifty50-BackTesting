"""
Pure indicator functions: ATR, StdDev, LinReg slope, Pivot detection.
All functions operate on numpy arrays and return numpy arrays.
"""

import numpy as np


def calc_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate ATR using Wilder's smoothing method.
    First value = SMA of true range, subsequent = EMA-style smoothing.
    """
    n = len(highs)
    tr = np.empty(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )

    atr = np.full(n, np.nan)
    if n < period:
        return atr

    # First ATR value is SMA of first `period` true ranges
    atr[period - 1] = np.mean(tr[:period])

    # Subsequent values use Wilder's smoothing
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    return atr


def calc_stddev(values: np.ndarray, period: int) -> np.ndarray:
    """Rolling standard deviation (population) matching Pine Script ta.stdev."""
    n = len(values)
    result = np.full(n, np.nan)
    for i in range(period - 1, n):
        window = values[i - period + 1: i + 1]
        result[i] = np.std(window, ddof=0)
    return result


def calc_sma(values: np.ndarray, period: int) -> np.ndarray:
    """Simple moving average."""
    n = len(values)
    result = np.full(n, np.nan)
    cumsum = np.cumsum(values)
    result[period - 1:] = (cumsum[period - 1:] - np.concatenate(([0], cumsum[:-period]))) / period
    return result


def calc_variance(values: np.ndarray, period: int) -> np.ndarray:
    """Rolling variance (population) matching Pine Script ta.variance."""
    n = len(values)
    result = np.full(n, np.nan)
    for i in range(period - 1, n):
        window = values[i - period + 1: i + 1]
        result[i] = np.var(window, ddof=0)
    return result


def calc_linreg_slope(closes: np.ndarray, period: int) -> np.ndarray:
    """
    Linear regression slope per Pine Script formula:
    |SMA(close * barIndex, period) - SMA(close, period) * SMA(barIndex, period)| / Var(barIndex, period) / 2
    """
    n = len(closes)
    bar_indices = np.arange(n, dtype=float)

    sma_cn = calc_sma(closes * bar_indices, period)
    sma_c = calc_sma(closes, period)
    sma_n = calc_sma(bar_indices, period)
    var_n = calc_variance(bar_indices, period)

    result = np.full(n, np.nan)
    for i in range(period - 1, n):
        if var_n[i] != 0 and not np.isnan(var_n[i]):
            result[i] = abs(sma_cn[i] - sma_c[i] * sma_n[i]) / var_n[i] / 2
    return result


def detect_pivot_highs(highs: np.ndarray, length: int) -> list[dict]:
    """
    Detect pivot highs with symmetric lookback.
    A pivot high at bar i-length is confirmed at bar i if:
    highs[i-length] > all highs in [i-2*length ... i-length-1] AND [i-length+1 ... i]

    Returns list of {index: int, value: float} where index is the bar where
    the pivot is confirmed (not the actual pivot bar).
    """
    n = len(highs)
    pivots = []
    for i in range(length * 2, n):
        pivot_bar = i - length
        pivot_val = highs[pivot_bar]
        is_pivot = True

        # Check left side
        for j in range(pivot_bar - length, pivot_bar):
            if highs[j] >= pivot_val:
                is_pivot = False
                break

        if is_pivot:
            # Check right side
            for j in range(pivot_bar + 1, pivot_bar + length + 1):
                if highs[j] >= pivot_val:
                    is_pivot = False
                    break

        if is_pivot:
            pivots.append({"index": i, "value": pivot_val, "pivot_bar": pivot_bar})

    return pivots


def detect_pivot_lows(lows: np.ndarray, length: int) -> list[dict]:
    """
    Detect pivot lows with symmetric lookback.
    Same logic as pivot highs but for lows (strictly less than).
    """
    n = len(lows)
    pivots = []
    for i in range(length * 2, n):
        pivot_bar = i - length
        pivot_val = lows[pivot_bar]
        is_pivot = True

        for j in range(pivot_bar - length, pivot_bar):
            if lows[j] <= pivot_val:
                is_pivot = False
                break

        if is_pivot:
            for j in range(pivot_bar + 1, pivot_bar + length + 1):
                if lows[j] <= pivot_val:
                    is_pivot = False
                    break

        if is_pivot:
            pivots.append({"index": i, "value": pivot_val, "pivot_bar": pivot_bar})

    return pivots
