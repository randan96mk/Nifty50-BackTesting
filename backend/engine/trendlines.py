"""
Trendline calculation and breakout detection engine.
Replicates the exact Pine Script v5 logic from "Trendlines with Breaks [LuxAlgo]".
"""

import numpy as np
from .indicators import (
    calc_atr,
    calc_stddev,
    calc_linreg_slope,
    detect_pivot_highs,
    detect_pivot_lows,
)


def calculate_trendlines(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    length: int = 14,
    mult: float = 1.0,
    calc_method: str = "Atr",
) -> dict:
    """
    Full trendline + breakout calculation matching the Pine Script indicator.

    Returns dict with:
        upper_line: float[] - upper trendline value at each bar
        lower_line: float[] - lower trendline value at each bar
        pivot_highs: list of {index, value, pivot_bar}
        pivot_lows: list of {index, value, pivot_bar}
        buy_signals: list of bar indices where BUY fires
        sell_signals: list of bar indices where SELL fires
        slope_values: float[] - raw slope at each bar
    """
    n = len(closes)

    # Step 1: Detect pivots
    pivot_highs = detect_pivot_highs(highs, length)
    pivot_lows = detect_pivot_lows(lows, length)

    # Build lookup sets for fast pivot checking at each bar
    ph_at_bar = {}
    for p in pivot_highs:
        ph_at_bar[p["index"]] = p["value"]

    pl_at_bar = {}
    for p in pivot_lows:
        pl_at_bar[p["index"]] = p["value"]

    # Step 2: Slope calculation
    if calc_method == "Atr":
        atr = calc_atr(highs, lows, closes, length)
        slope_raw = atr / length * mult
    elif calc_method == "Stdev":
        std = calc_stddev(closes, length)
        slope_raw = std / length * mult
    elif calc_method == "Linreg":
        linreg = calc_linreg_slope(closes, length)
        slope_raw = linreg * mult
    else:
        raise ValueError(f"Unknown calc_method: {calc_method}")

    # Replace NaN with 0 for early bars
    slope_raw = np.nan_to_num(slope_raw, nan=0.0)

    # Step 3-5: Iterate through bars computing trendlines and breakouts
    upper_line = np.full(n, np.nan)
    lower_line = np.full(n, np.nan)

    slope_ph = 0.0  # frozen slope for pivot highs
    slope_pl = 0.0  # frozen slope for pivot lows
    upper = 0.0
    lower = 0.0
    upos = 0
    dnos = 0
    prev_upos = 0
    prev_dnos = 0

    buy_signals = []
    sell_signals = []

    for i in range(n):
        slope = slope_raw[i]
        ph = ph_at_bar.get(i)
        pl = pl_at_bar.get(i)

        # Step 3: Slope persistence - freeze slope at pivot detection
        if ph is not None:
            slope_ph = slope
        if pl is not None:
            slope_pl = slope

        # Step 4: Trendline projection
        if ph is not None:
            upper = ph
        else:
            upper = upper - slope_ph

        if pl is not None:
            lower = pl
        else:
            lower = lower + slope_pl

        upper_line[i] = upper
        lower_line[i] = lower

        # Step 5: Breakout detection
        prev_upos = upos
        prev_dnos = dnos

        if ph is not None:
            upos = 0
        elif closes[i] > upper - slope_ph * length:
            upos = 1
        # else upos stays the same

        if pl is not None:
            dnos = 0
        elif closes[i] < lower + slope_pl * length:
            dnos = 1
        # else dnos stays the same

        # Signal on state transition 0 -> 1
        if upos == 1 and prev_upos == 0:
            buy_signals.append(i)
        if dnos == 1 and prev_dnos == 0:
            sell_signals.append(i)

    return {
        "upper_line": upper_line,
        "lower_line": lower_line,
        "pivot_highs": pivot_highs,
        "pivot_lows": pivot_lows,
        "buy_signals": buy_signals,
        "sell_signals": sell_signals,
    }
