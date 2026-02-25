"""
Chart Pattern Detection and Backtesting for Nifty 50 Intraday Data.

Detects and backtests three categories of breakout patterns using OHLC
price data only (no volume required), tuned for 3-min / 5-min candles:

1. Triple Tops and Triple Bottoms  – reversal patterns
2. Triangle Breakouts              – ascending, descending, symmetrical
3. Breakout Buildups               – flags and rectangles
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple


# ═══════════════════════════════════════════════════════════════════════════════
# ATR (Average True Range) — adaptive volatility measurement
# ═══════════════════════════════════════════════════════════════════════════════

def compute_atr(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """
    Compute ATR using Wilder's smoothing.

    Returns an array of the same length as inputs.  The first *period* values
    are NaN.  ATR adapts to the current volatility regime and is used
    throughout the pattern detector for:
      - Noise-adaptive tolerance when comparing peak/trough levels
      - Dynamic stop-loss placement (ATR multiples)
      - Minimum pattern height thresholds
    """
    n = len(closes)
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

    atr[period - 1] = tr[:period].mean()
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    return atr


# ═══════════════════════════════════════════════════════════════════════════════
# ZigZag noise filter — identifies significant turning points
# ═══════════════════════════════════════════════════════════════════════════════

def zigzag_pivots(
    highs: np.ndarray,
    lows: np.ndarray,
    atr: np.ndarray,
    atr_threshold: float = 1.0,
) -> Tuple[List[int], List[int]]:
    """
    ZigZag-based pivot detection that filters out noise below *atr_threshold*
    multiples of ATR.

    A swing is only confirmed when price reverses by at least
    atr_threshold * ATR from the current extreme.  This produces fewer
    but more significant peaks and troughs than the fixed-order method,
    especially on noisy intraday data.

    Returns (peak_indices, trough_indices).
    """
    n = len(highs)
    peaks: List[int] = []
    troughs: List[int] = []

    # Find the first valid ATR index
    start = 0
    for i in range(n):
        if not np.isnan(atr[i]):
            start = i
            break

    if start >= n - 1:
        return peaks, troughs

    # Initialise: direction = +1 means looking for peak, -1 for trough
    direction = 1 if highs[start] >= lows[start] else -1
    last_high_idx = start
    last_low_idx = start

    for i in range(start + 1, n):
        threshold = atr[i] * atr_threshold if not np.isnan(atr[i]) else 0

        if direction == 1:  # trending up, looking for peak
            if highs[i] > highs[last_high_idx]:
                last_high_idx = i  # extend the swing
            elif highs[last_high_idx] - lows[i] >= threshold:
                # Confirmed peak
                peaks.append(last_high_idx)
                direction = -1
                last_low_idx = i
        else:  # trending down, looking for trough
            if lows[i] < lows[last_low_idx]:
                last_low_idx = i  # extend the swing
            elif highs[i] - lows[last_low_idx] >= threshold:
                # Confirmed trough
                troughs.append(last_low_idx)
                direction = 1
                last_high_idx = i

    return peaks, troughs


# ═══════════════════════════════════════════════════════════════════════════════
# Local extrema detection (original fixed-order method, kept as fallback)
# ═══════════════════════════════════════════════════════════════════════════════

def find_peaks(series: np.ndarray, order: int = 5) -> List[int]:
    """
    Find local maxima.  A point at index *i* is a peak if it is strictly
    greater than all points within *order* positions on both sides.
    """
    peaks = []
    for i in range(order, len(series) - order):
        window_left = series[i - order:i]
        window_right = series[i + 1:i + order + 1]
        if series[i] > window_left.max() and series[i] > window_right.max():
            peaks.append(i)
    return peaks


def find_troughs(series: np.ndarray, order: int = 5) -> List[int]:
    """
    Find local minima.  A point at index *i* is a trough if it is strictly
    less than all points within *order* positions on both sides.
    """
    troughs = []
    for i in range(order, len(series) - order):
        window_left = series[i - order:i]
        window_right = series[i + 1:i + order + 1]
        if series[i] < window_left.min() and series[i] < window_right.min():
            troughs.append(i)
    return troughs


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Triple Top / Triple Bottom
# ═══════════════════════════════════════════════════════════════════════════════

def detect_triple_tops(
    df: pd.DataFrame,
    peak_order: int = 5,
    tolerance_pct: float = 0.3,
    min_spacing: int = 8,
    max_pattern_bars: int = 60,
    max_breakout_wait: int = 10,
    min_pattern_height_pct: float = 0.3,
    use_atr: bool = True,
    atr_period: int = 14,
    atr_tolerance_mult: float = 1.5,
    sl_atr_mult: float = 1.5,
    zigzag_threshold: float = 1.0,
) -> List[Dict]:
    """
    Detect Triple Top (bearish reversal) patterns on intraday candles.

    When *use_atr* is True (default), thresholds adapt to current volatility:
    - Peak tolerance = atr_tolerance_mult * ATR (instead of fixed %)
    - SL = resistance + sl_atr_mult * ATR (instead of fixed 0.3%)
    - Peaks found via ZigZag noise filter (instead of fixed-order)

    Trade setup
    -----------
    * Entry  : close of the breakout bar
    * SL     : resistance + sl_atr_mult * ATR
    * Target : neckline - pattern_height (measured move)
    """
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values

    if use_atr:
        atr = compute_atr(highs, lows, closes, period=atr_period)
        zz_peaks, _ = zigzag_pivots(highs, lows, atr, atr_threshold=zigzag_threshold)
        peaks = zz_peaks if len(zz_peaks) >= 3 else find_peaks(highs, order=peak_order)
    else:
        atr = None
        peaks = find_peaks(highs, order=peak_order)

    patterns: List[Dict] = []
    used: set = set()

    for i in range(len(peaks)):
        p1 = peaks[i]
        if p1 in used:
            continue
        for j in range(i + 1, len(peaks)):
            p2 = peaks[j]
            if p2 in used:
                continue
            if p2 - p1 < min_spacing:
                continue
            if p2 - p1 > max_pattern_bars // 2:
                break
            for k in range(j + 1, len(peaks)):
                p3 = peaks[k]
                if p3 in used:
                    continue
                if p3 - p2 < min_spacing:
                    continue
                if p3 - p1 > max_pattern_bars:
                    break

                h1, h2, h3 = highs[p1], highs[p2], highs[p3]
                avg_h = (h1 + h2 + h3) / 3.0

                # Adaptive tolerance: ATR-based or percentage-based
                if use_atr and atr is not None and not np.isnan(atr[p3]):
                    tol = atr[p3] * atr_tolerance_mult
                    if any(abs(h - avg_h) > tol for h in (h1, h2, h3)):
                        continue
                else:
                    if any(abs(h - avg_h) / avg_h * 100 > tolerance_pct
                           for h in (h1, h2, h3)):
                        continue

                # Neckline
                seg = lows[p1:p3 + 1]
                neckline = float(seg.min())
                neckline_idx = int(p1 + seg.argmin())

                # Minimum height
                pattern_height = avg_h - neckline
                if use_atr and atr is not None and not np.isnan(atr[p3]):
                    if pattern_height < atr[p3] * 1.5:
                        continue
                else:
                    if pattern_height / neckline * 100 < min_pattern_height_pct:
                        continue

                # Breakout scan
                end = min(p3 + max_breakout_wait + 1, len(df))
                for b in range(p3 + 1, end):
                    if closes[b] < neckline:
                        entry = closes[b]

                        # ATR-based SL or fixed
                        if use_atr and atr is not None and not np.isnan(atr[b]):
                            sl = avg_h + atr[b] * sl_atr_mult
                        else:
                            sl = avg_h * 1.003

                        target = neckline - pattern_height
                        risk = sl - entry
                        reward = entry - target

                        dt_label = df.iloc[b].get("dt_label", df.iloc[b].get("date", ""))
                        pat_start_dt = df.iloc[p1].get("dt_label", df.iloc[p1].get("date", ""))
                        pat_end_dt = df.iloc[p3].get("dt_label", df.iloc[p3].get("date", ""))

                        patterns.append({
                            "pattern": "Triple Top",
                            "direction": "SHORT",
                            "peak1_idx": p1,
                            "peak2_idx": p2,
                            "peak3_idx": p3,
                            "neckline_idx": neckline_idx,
                            "breakout_idx": b,
                            "resistance": round(avg_h, 2),
                            "neckline": round(neckline, 2),
                            "pattern_height": round(pattern_height, 2),
                            "entry_price": round(entry, 2),
                            "sl": round(sl, 2),
                            "target": round(target, 2),
                            "risk": round(risk, 2),
                            "reward": round(reward, 2),
                            "planned_rr": round(reward / risk, 2) if risk > 0 else 0,
                            "entry_dt": dt_label,
                            "pattern_start_dt": pat_start_dt,
                            "pattern_end_dt": pat_end_dt,
                        })
                        used.update([p1, p2, p3])
                        break

                if p1 in used:
                    break
            if p1 in used:
                break

    return patterns


def detect_triple_bottoms(
    df: pd.DataFrame,
    trough_order: int = 5,
    tolerance_pct: float = 0.3,
    min_spacing: int = 8,
    max_pattern_bars: int = 60,
    max_breakout_wait: int = 10,
    min_pattern_height_pct: float = 0.3,
    use_atr: bool = True,
    atr_period: int = 14,
    atr_tolerance_mult: float = 1.5,
    sl_atr_mult: float = 1.5,
    zigzag_threshold: float = 1.0,
) -> List[Dict]:
    """
    Detect Triple Bottom (bullish reversal) patterns on intraday candles.

    When *use_atr* is True (default), thresholds adapt to current volatility:
    - Trough tolerance = atr_tolerance_mult * ATR
    - SL = support - sl_atr_mult * ATR
    - Troughs found via ZigZag noise filter

    Trade setup
    -----------
    * Entry  : close of the breakout bar
    * SL     : support - sl_atr_mult * ATR
    * Target : neckline + pattern_height (measured move)
    """
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values

    if use_atr:
        atr = compute_atr(highs, lows, closes, period=atr_period)
        _, zz_troughs = zigzag_pivots(highs, lows, atr, atr_threshold=zigzag_threshold)
        troughs = zz_troughs if len(zz_troughs) >= 3 else find_troughs(lows, order=trough_order)
    else:
        atr = None
        troughs = find_troughs(lows, order=trough_order)

    patterns: List[Dict] = []
    used: set = set()

    for i in range(len(troughs)):
        t1 = troughs[i]
        if t1 in used:
            continue
        for j in range(i + 1, len(troughs)):
            t2 = troughs[j]
            if t2 in used:
                continue
            if t2 - t1 < min_spacing:
                continue
            if t2 - t1 > max_pattern_bars // 2:
                break
            for k in range(j + 1, len(troughs)):
                t3 = troughs[k]
                if t3 in used:
                    continue
                if t3 - t2 < min_spacing:
                    continue
                if t3 - t1 > max_pattern_bars:
                    break

                l1, l2, l3 = lows[t1], lows[t2], lows[t3]
                avg_l = (l1 + l2 + l3) / 3.0

                # Adaptive tolerance
                if use_atr and atr is not None and not np.isnan(atr[t3]):
                    tol = atr[t3] * atr_tolerance_mult
                    if any(abs(l - avg_l) > tol for l in (l1, l2, l3)):
                        continue
                else:
                    if any(abs(l - avg_l) / avg_l * 100 > tolerance_pct
                           for l in (l1, l2, l3)):
                        continue

                # Neckline (resistance) = highest high between t1 and t3
                seg = highs[t1:t3 + 1]
                neckline = float(seg.max())
                neckline_idx = int(t1 + seg.argmax())

                pattern_height = neckline - avg_l
                if use_atr and atr is not None and not np.isnan(atr[t3]):
                    if pattern_height < atr[t3] * 1.5:
                        continue
                else:
                    if pattern_height / avg_l * 100 < min_pattern_height_pct:
                        continue

                end = min(t3 + max_breakout_wait + 1, len(df))
                for b in range(t3 + 1, end):
                    if closes[b] > neckline:
                        entry = closes[b]

                        # ATR-based SL
                        if use_atr and atr is not None and not np.isnan(atr[b]):
                            sl = avg_l - atr[b] * sl_atr_mult
                        else:
                            sl = avg_l * 0.997

                        target = neckline + pattern_height
                        risk = entry - sl
                        reward = target - entry

                        dt_label = df.iloc[b].get("dt_label", df.iloc[b].get("date", ""))
                        pat_start_dt = df.iloc[t1].get("dt_label", df.iloc[t1].get("date", ""))
                        pat_end_dt = df.iloc[t3].get("dt_label", df.iloc[t3].get("date", ""))

                        patterns.append({
                            "pattern": "Triple Bottom",
                            "direction": "LONG",
                            "trough1_idx": t1,
                            "trough2_idx": t2,
                            "trough3_idx": t3,
                            "neckline_idx": neckline_idx,
                            "breakout_idx": b,
                            "support": round(avg_l, 2),
                            "neckline": round(neckline, 2),
                            "pattern_height": round(pattern_height, 2),
                            "entry_price": round(entry, 2),
                            "sl": round(sl, 2),
                            "target": round(target, 2),
                            "risk": round(risk, 2),
                            "reward": round(reward, 2),
                            "planned_rr": round(reward / risk, 2) if risk > 0 else 0,
                            "entry_dt": dt_label,
                            "pattern_start_dt": pat_start_dt,
                            "pattern_end_dt": pat_end_dt,
                        })
                        used.update([t1, t2, t3])
                        break

                if t1 in used:
                    break
            if t1 in used:
                break

    return patterns


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Triangle Breakouts (ascending / descending / symmetrical)
# ═══════════════════════════════════════════════════════════════════════════════

def _classify_triangle(
    upper_slope: float,
    lower_slope: float,
    avg_price: float,
    window_size: int,
) -> str | None:
    """
    Classify a triangle based on trendline slopes.

    Slopes are in price-per-bar.  We normalise to a per-window percentage
    change so the threshold is scale-independent.

    * Ascending   : flat upper (~0 %) + rising lower (> +0.5 %)
    * Descending  : falling upper (< -0.5 %) + flat lower (~0 %)
    * Symmetrical : falling upper (< -0.3 %) + rising lower (> +0.3 %)
    """
    upper_pct = upper_slope * window_size / avg_price * 100
    lower_pct = lower_slope * window_size / avg_price * 100

    flat = 0.5  # threshold % — tighter for intraday

    if abs(upper_pct) <= flat and lower_pct > flat:
        return "Ascending Triangle"
    if upper_pct < -flat and abs(lower_pct) <= flat:
        return "Descending Triangle"
    if upper_pct < -0.3 and lower_pct > 0.3:
        return "Symmetrical Triangle"
    return None


def detect_triangles(
    df: pd.DataFrame,
    peak_order: int = 3,
    min_window: int = 15,
    max_window: int = 50,
    window_step: int = 3,
    min_touches: int = 2,
    max_breakout_wait: int = 8,
    min_height_pct: float = 0.2,
) -> List[Dict]:
    """
    Detect triangle consolidation patterns (ascending, descending, symmetrical)
    on intraday candles.

    Quantitative rules
    ------------------
    1. Within a rolling window, find >= *min_touches* peaks and troughs.
    2. Fit linear regression to peaks (upper trendline) and troughs (lower).
    3. Trendlines must converge (width at end < width at start).
    4. Triangle height at the start >= *min_height_pct* % of average price.
    5. Classify type via slope analysis.
    6. Breakout = close outside the projected trendline within
       *max_breakout_wait* bars.

    Trade setup
    -----------
    * Ascending   -> expect bullish breakout above upper trendline
    * Descending  -> expect bearish breakout below lower trendline
    * Symmetrical -> trade whichever side breaks
    * SL: opposite trendline at breakout bar
    * Target: triangle height projected from breakout level
    """
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    n = len(df)

    all_peaks = find_peaks(highs, order=peak_order)
    all_troughs = find_troughs(lows, order=peak_order)

    patterns: List[Dict] = []
    used_windows: List[Tuple[int, int]] = []

    for win_size in range(min_window, max_window + 1, window_step):
        for end in range(win_size, n - max_breakout_wait, window_step):
            start = end - win_size

            # Skip if overlapping with already detected pattern
            if any(s <= start <= e or s <= end <= e for s, e in used_windows):
                continue

            # Peaks / troughs inside window
            pk_idx = [p for p in all_peaks if start <= p <= end]
            tr_idx = [t for t in all_troughs if start <= t <= end]

            if len(pk_idx) < min_touches or len(tr_idx) < min_touches:
                continue

            pk_x = np.array(pk_idx, dtype=float)
            pk_y = np.array([highs[p] for p in pk_idx])
            tr_x = np.array(tr_idx, dtype=float)
            tr_y = np.array([lows[t] for t in tr_idx])

            # Fit trendlines
            upper_slope, upper_intercept = np.polyfit(pk_x, pk_y, 1)
            lower_slope, lower_intercept = np.polyfit(tr_x, tr_y, 1)

            # Convergence check
            width_start = (upper_intercept + upper_slope * start) - \
                          (lower_intercept + lower_slope * start)
            width_end = (upper_intercept + upper_slope * end) - \
                        (lower_intercept + lower_slope * end)

            if width_end >= width_start * 0.85:
                continue  # not converging enough

            avg_price = (pk_y.mean() + tr_y.mean()) / 2
            if width_start / avg_price * 100 < min_height_pct:
                continue

            tri_type = _classify_triangle(
                upper_slope, lower_slope, avg_price, win_size)
            if tri_type is None:
                continue

            # Breakout scan
            for b in range(end + 1, min(end + max_breakout_wait + 1, n)):
                upper_at_b = upper_intercept + upper_slope * b
                lower_at_b = lower_intercept + lower_slope * b

                direction = None
                if closes[b] > upper_at_b:
                    direction = "LONG"
                elif closes[b] < lower_at_b:
                    direction = "SHORT"

                if direction is None:
                    continue

                entry = closes[b]
                tri_height = upper_at_b - lower_at_b
                if tri_height <= 0:
                    tri_height = width_start  # fallback

                if direction == "LONG":
                    sl = lower_at_b
                    target = entry + tri_height
                else:
                    sl = upper_at_b
                    target = entry - tri_height

                risk = abs(entry - sl)
                reward = abs(target - entry)
                if risk <= 0:
                    continue

                dt_label = df.iloc[b].get("dt_label", df.iloc[b].get("date", ""))
                pat_start_dt = df.iloc[start].get("dt_label", df.iloc[start].get("date", ""))
                pat_end_dt = df.iloc[end].get("dt_label", df.iloc[end].get("date", ""))

                patterns.append({
                    "pattern": tri_type,
                    "direction": direction,
                    "window_start_idx": start,
                    "window_end_idx": end,
                    "breakout_idx": b,
                    "upper_slope": round(upper_slope, 4),
                    "upper_intercept": round(upper_intercept, 2),
                    "lower_slope": round(lower_slope, 4),
                    "lower_intercept": round(lower_intercept, 2),
                    "triangle_height": round(tri_height, 2),
                    "entry_price": round(entry, 2),
                    "sl": round(sl, 2),
                    "target": round(target, 2),
                    "risk": round(risk, 2),
                    "reward": round(reward, 2),
                    "planned_rr": round(reward / risk, 2) if risk > 0 else 0,
                    "entry_dt": dt_label,
                    "pattern_start_dt": pat_start_dt,
                    "pattern_end_dt": pat_end_dt,
                })
                used_windows.append((start, end))
                break

    return patterns


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Breakout Buildups  (flags and rectangles)
# ═══════════════════════════════════════════════════════════════════════════════

def detect_flags_rectangles(
    df: pd.DataFrame,
    pole_lookback: int = 8,
    pole_min_pct: float = 0.8,
    consol_min_bars: int = 5,
    consol_max_bars: int = 20,
    consol_max_range_ratio: float = 0.50,
    flag_slope_threshold: float = 0.3,
    max_breakout_wait: int = 5,
) -> List[Dict]:
    """
    Detect flag and rectangle breakout-buildup patterns on intraday candles.

    Quantitative rules
    ------------------
    1. **Pole**: a strong directional move of >= *pole_min_pct* % over
       *pole_lookback* bars.
    2. **Consolidation**: after the pole, price forms a tight range for
       *consol_min_bars* to *consol_max_bars* bars whose range is
       <= *consol_max_range_ratio* x pole height.
    3. **Classification**:
       - *Flag*: consolidation has a slight counter-trend slope
         (slope x bars / price > *flag_slope_threshold* % against pole dir).
       - *Rectangle*: consolidation slope is near zero.
    4. **Breakout**: close beyond the consolidation boundary in the pole
       direction within *max_breakout_wait* bars.

    Trade setup
    -----------
    * Entry  : close of breakout bar
    * SL     : opposite end of consolidation range
    * Target : pole height projected from breakout level
    """
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    n = len(df)

    patterns: List[Dict] = []
    used_ranges: List[Tuple[int, int]] = []

    # Scan for poles
    for pole_end in range(pole_lookback, n - consol_min_bars - max_breakout_wait):
        pole_start = pole_end - pole_lookback

        # Skip if overlapping
        if any(s <= pole_start <= e for s, e in used_ranges):
            continue

        pole_move = closes[pole_end] - closes[pole_start]
        pole_pct = pole_move / closes[pole_start] * 100
        pole_dir = "BULL" if pole_pct > 0 else "BEAR"

        if abs(pole_pct) < pole_min_pct:
            continue

        pole_height = abs(pole_move)

        # Scan for consolidation after the pole
        best_consol = None
        for consol_len in range(consol_min_bars, min(consol_max_bars + 1,
                                                     n - pole_end - max_breakout_wait)):
            c_start = pole_end
            c_end = pole_end + consol_len

            c_highs = highs[c_start:c_end + 1]
            c_lows = lows[c_start:c_end + 1]
            c_range = c_highs.max() - c_lows.min()

            if c_range > consol_max_range_ratio * pole_height:
                continue

            # Slope of closes in consolidation
            c_closes = closes[c_start:c_end + 1]
            x = np.arange(len(c_closes), dtype=float)
            slope, intercept = np.polyfit(x, c_closes, 1)
            slope_pct = slope * consol_len / np.mean(c_closes) * 100

            # Classify
            if pole_dir == "BULL" and slope_pct < -flag_slope_threshold:
                ptype = "Bull Flag"
            elif pole_dir == "BEAR" and slope_pct > flag_slope_threshold:
                ptype = "Bear Flag"
            elif abs(slope_pct) <= flag_slope_threshold:
                ptype = "Rectangle"
            else:
                continue

            best_consol = {
                "c_start": c_start,
                "c_end": c_end,
                "c_high": float(c_highs.max()),
                "c_low": float(c_lows.min()),
                "slope_pct": slope_pct,
                "ptype": ptype,
            }
            # Take the longest valid consolidation
        if best_consol is None:
            continue

        c_end = best_consol["c_end"]
        c_high = best_consol["c_high"]
        c_low = best_consol["c_low"]

        # Breakout scan
        for b in range(c_end + 1, min(c_end + max_breakout_wait + 1, n)):
            direction = None
            if pole_dir == "BULL" and closes[b] > c_high:
                direction = "LONG"
            elif pole_dir == "BEAR" and closes[b] < c_low:
                direction = "SHORT"

            if direction is None:
                continue

            entry = closes[b]
            if direction == "LONG":
                sl = c_low
                target = entry + pole_height
            else:
                sl = c_high
                target = entry - pole_height

            risk = abs(entry - sl)
            reward = abs(target - entry)
            if risk <= 0:
                continue

            dt_label = df.iloc[b].get("dt_label", df.iloc[b].get("date", ""))
            pat_start_dt = df.iloc[pole_start].get("dt_label", df.iloc[pole_start].get("date", ""))
            pat_end_dt = df.iloc[c_end].get("dt_label", df.iloc[c_end].get("date", ""))

            patterns.append({
                "pattern": best_consol["ptype"],
                "direction": direction,
                "pole_start_idx": pole_start,
                "pole_end_idx": pole_end,
                "consol_start_idx": best_consol["c_start"],
                "consol_end_idx": c_end,
                "breakout_idx": b,
                "pole_height": round(pole_height, 2),
                "consol_high": round(c_high, 2),
                "consol_low": round(c_low, 2),
                "consol_range": round(c_high - c_low, 2),
                "entry_price": round(entry, 2),
                "sl": round(sl, 2),
                "target": round(target, 2),
                "risk": round(risk, 2),
                "reward": round(reward, 2),
                "planned_rr": round(reward / risk, 2) if risk > 0 else 0,
                "entry_dt": dt_label,
                "pattern_start_dt": pat_start_dt,
                "pattern_end_dt": pat_end_dt,
            })
            used_ranges.append((pole_start, c_end))
            break

    return patterns


# ═══════════════════════════════════════════════════════════════════════════════
# Backtesting engine
# ═══════════════════════════════════════════════════════════════════════════════

def backtest_patterns(
    df: pd.DataFrame,
    patterns: List[Dict],
    max_hold_bars: int = 20,
) -> pd.DataFrame:
    """
    Simulate trades for each detected pattern on intraday candles.

    For each pattern, enters at the breakout bar close and exits when one of
    the following occurs (checked bar-by-bar):
      1. Target price hit
      2. Stop-loss hit
      3. Max holding period reached -> exit at close
      4. End of trading day -> exit at close (no overnight holding)

    Returns a DataFrame with one row per trade, including realised P&L,
    exit reason, and holding period in bars.
    """
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    n = len(df)

    # Build date array for intraday session tracking
    has_date = "date" in df.columns
    dates = df["date"].values if has_date else None

    trades = []

    for pat in patterns:
        b_idx = pat["breakout_idx"]
        direction = pat["direction"]
        entry = pat["entry_price"]
        sl = pat["sl"]
        target = pat["target"]

        entry_date = dates[b_idx] if dates is not None else None

        exit_price = None
        exit_reason = None
        exit_idx = None

        for bar in range(b_idx + 1, min(b_idx + max_hold_bars + 1, n)):
            # End of day check — exit before overnight gap
            if dates is not None and dates[bar] != entry_date:
                exit_idx = bar - 1  # last bar of entry day
                if exit_idx <= b_idx:
                    exit_idx = bar
                exit_price = closes[min(exit_idx, n - 1)]
                exit_reason = "EOD Exit"
                break

            if direction == "LONG":
                if lows[bar] <= sl:
                    exit_price = sl
                    exit_reason = "SL Hit"
                    exit_idx = bar
                    break
                if highs[bar] >= target:
                    exit_price = target
                    exit_reason = "Target Hit"
                    exit_idx = bar
                    break
            else:  # SHORT
                if highs[bar] >= sl:
                    exit_price = sl
                    exit_reason = "SL Hit"
                    exit_idx = bar
                    break
                if lows[bar] <= target:
                    exit_price = target
                    exit_reason = "Target Hit"
                    exit_idx = bar
                    break

        # Time exit
        if exit_price is None:
            exit_idx = min(b_idx + max_hold_bars, n - 1)
            exit_price = closes[exit_idx]
            exit_reason = "Time Exit"

        if direction == "LONG":
            pnl = exit_price - entry
        else:
            pnl = entry - exit_price

        pnl_pct = pnl / entry * 100
        risk = pat.get("risk", abs(entry - sl))
        actual_rr = pnl / risk if risk > 0 else 0

        # Max adverse excursion
        if direction == "LONG":
            mae_prices = lows[b_idx + 1:exit_idx + 1] if exit_idx > b_idx else [entry]
            mae = entry - min(mae_prices) if len(mae_prices) > 0 else 0
        else:
            mae_prices = highs[b_idx + 1:exit_idx + 1] if exit_idx > b_idx else [entry]
            mae = max(mae_prices) - entry if len(mae_prices) > 0 else 0

        exit_dt = df.iloc[exit_idx].get("dt_label", df.iloc[exit_idx].get("date", ""))

        trades.append({
            "pattern": pat["pattern"],
            "direction": direction,
            "entry_dt": pat["entry_dt"],
            "entry_price": round(entry, 2),
            "sl": round(sl, 2),
            "target": round(target, 2),
            "exit_price": round(exit_price, 2),
            "exit_dt": exit_dt,
            "exit_reason": exit_reason,
            "hold_bars": exit_idx - b_idx,
            "pnl_points": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
            "risk_points": round(risk, 2),
            "actual_rr": round(actual_rr, 2),
            "max_adverse_excursion": round(mae, 2),
            "planned_rr": pat.get("planned_rr", 0),
            "pattern_start_dt": pat.get("pattern_start_dt", ""),
            "pattern_end_dt": pat.get("pattern_end_dt", ""),
            "breakout_idx": pat["breakout_idx"],
            # Carry forward pattern-specific keys for charting
            **{k: v for k, v in pat.items() if k not in [
                "pattern", "direction", "entry_price", "sl", "target",
                "entry_dt", "pattern_start_dt", "pattern_end_dt",
                "risk", "reward", "planned_rr", "breakout_idx",
            ]},
        })

    return pd.DataFrame(trades)


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics computation
# ═══════════════════════════════════════════════════════════════════════════════

def compute_pattern_metrics(trades_df: pd.DataFrame) -> Dict:
    """
    Compute summary statistics for a set of pattern trades.

    Returns: total_trades, winners, losers, win_rate, avg_return_pct,
    total_pnl, max_win, max_loss, profit_factor, max_drawdown,
    avg_risk_reward, avg_hold_bars, target_hits, sl_hits, time_exits.
    """
    if trades_df.empty:
        return {k: 0 for k in [
            "total_trades", "winners", "losers", "win_rate",
            "avg_return_pct", "total_pnl", "max_win", "max_loss",
            "avg_winner_pct", "avg_loser_pct",
            "profit_factor", "max_drawdown", "avg_risk_reward",
            "avg_hold_bars", "target_hits", "sl_hits", "time_exits",
            "eod_exits",
        ]}

    winners = trades_df[trades_df["pnl_points"] > 0]
    losers = trades_df[trades_df["pnl_points"] <= 0]
    gross_profit = winners["pnl_points"].sum() if len(winners) else 0
    gross_loss = abs(losers["pnl_points"].sum()) if len(losers) else 0

    # Drawdown
    cum = trades_df["pnl_points"].cumsum()
    running_max = cum.cummax()
    dd = cum - running_max

    return {
        "total_trades": len(trades_df),
        "winners": len(winners),
        "losers": len(losers),
        "win_rate": round(len(winners) / len(trades_df) * 100, 1),
        "avg_return_pct": round(trades_df["pnl_pct"].mean(), 2),
        "total_pnl": round(trades_df["pnl_points"].sum(), 2),
        "max_win": round(trades_df["pnl_points"].max(), 2),
        "max_loss": round(trades_df["pnl_points"].min(), 2),
        "avg_winner_pct": round(winners["pnl_pct"].mean(), 2) if len(winners) else 0,
        "avg_loser_pct": round(losers["pnl_pct"].mean(), 2) if len(losers) else 0,
        "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss > 0 else float("inf"),
        "max_drawdown": round(dd.min(), 2),
        "avg_risk_reward": round(trades_df["actual_rr"].mean(), 2),
        "avg_hold_bars": round(trades_df["hold_bars"].mean(), 1),
        "target_hits": int((trades_df["exit_reason"] == "Target Hit").sum()),
        "sl_hits": int((trades_df["exit_reason"] == "SL Hit").sum()),
        "time_exits": int((trades_df["exit_reason"] == "Time Exit").sum()),
        "eod_exits": int((trades_df["exit_reason"] == "EOD Exit").sum()),
    }


def run_all_patterns(
    df: pd.DataFrame,
    triple_top_params: Dict | None = None,
    triple_bottom_params: Dict | None = None,
    triangle_params: Dict | None = None,
    flag_rect_params: Dict | None = None,
    max_hold_bars: int = 20,
    use_atr: bool = True,
    atr_period: int = 14,
    sl_atr_mult: float = 1.5,
    zigzag_threshold: float = 1.0,
    earliest_entry: str = "09:30",
    latest_entry: str = "14:00",
    enabled_patterns: set | None = None,
) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    """
    Convenience function: detect all pattern types, backtest, and return
    a combined trades DataFrame plus per-pattern-type metrics.

    *enabled_patterns* is a set of pattern name strings to detect.
    If None, all patterns are enabled.  Disabled detectors are skipped
    entirely (saves compute time).

    ATR-related params are forwarded to all detectors.
    Time-of-day filter removes entries before *earliest_entry* or after
    *latest_entry* (IST, HH:MM) — entries too late in the day don't have
    sufficient runway for the pattern target to be reached.
    """
    if enabled_patterns is None:
        enabled_patterns = {
            "Triple Top", "Triple Bottom",
            "Ascending Triangle", "Descending Triangle", "Symmetrical Triangle",
            "Bull Flag", "Bear Flag", "Rectangle",
        }

    atr_shared = dict(
        use_atr=use_atr,
        atr_period=atr_period,
        sl_atr_mult=sl_atr_mult,
        zigzag_threshold=zigzag_threshold,
    )

    all_patterns = []

    if "Triple Top" in enabled_patterns:
        tt_params = {**atr_shared, **(triple_top_params or {})}
        all_patterns.extend(detect_triple_tops(df, **tt_params))

    if "Triple Bottom" in enabled_patterns:
        tb_params = {**atr_shared, **(triple_bottom_params or {})}
        all_patterns.extend(detect_triple_bottoms(df, **tb_params))

    # Triangle detector finds all three types; filter after detection
    _any_triangle = enabled_patterns & {
        "Ascending Triangle", "Descending Triangle", "Symmetrical Triangle"
    }
    if _any_triangle:
        tri_params = triangle_params or {}
        tri_results = detect_triangles(df, **tri_params)
        all_patterns.extend(
            p for p in tri_results if p["pattern"] in enabled_patterns
        )

    # Flag/Rectangle detector finds all three types; filter after detection
    _any_flag_rect = enabled_patterns & {"Bull Flag", "Bear Flag", "Rectangle"}
    if _any_flag_rect:
        fr_params = flag_rect_params or {}
        fr_results = detect_flags_rectangles(df, **fr_params)
        all_patterns.extend(
            p for p in fr_results if p["pattern"] in enabled_patterns
        )

    if not all_patterns:
        return pd.DataFrame(), {}

    # Time-of-day filter: skip entries outside the trading window
    has_time = "time" in df.columns
    if has_time and earliest_entry and latest_entry:
        times = df["time"].values
        filtered = []
        for pat in all_patterns:
            b_idx = pat["breakout_idx"]
            entry_time = times[b_idx]
            if earliest_entry <= entry_time <= latest_entry:
                filtered.append(pat)
        all_patterns = filtered

    if not all_patterns:
        return pd.DataFrame(), {}

    trades_df = backtest_patterns(df, all_patterns, max_hold_bars=max_hold_bars)

    # Per-pattern metrics
    per_pattern = {}
    for ptype in trades_df["pattern"].unique():
        subset = trades_df[trades_df["pattern"] == ptype]
        per_pattern[ptype] = compute_pattern_metrics(subset)

    return trades_df, per_pattern
