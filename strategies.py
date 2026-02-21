"""
Intraday backtesting strategies for Nifty50 5-minute data.

Strategies implemented:
  1. ORB  – Opening Range Breakout (9:15–9:30 range)
  2. VWAP Pullback – Mean-reversion entries off VWAP
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """Cumulative intraday VWAP from (H+L+C)/3 * volume."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    cum_tp_vol = (tp * df["volume"]).cumsum()
    cum_vol = df["volume"].cumsum()
    return cum_tp_vol / cum_vol


def _candle_body_above(row: pd.Series, level: float) -> bool:
    return min(row["open"], row["close"]) > level


def _candle_body_below(row: pd.Series, level: float) -> bool:
    return max(row["open"], row["close"]) < level


# ---------------------------------------------------------------------------
# ORB Strategy
# ---------------------------------------------------------------------------

def run_orb_day(
    day_df: pd.DataFrame,
    volume_mult: float = 1.5,
    risk_reward: float = 1.5,
    exit_time: str = "12:00",
) -> dict | None:
    """
    Run ORB strategy on a single day's 5-min data.

    Steps (matching the 8-step user flow):
      1. 09:15 candle forms (index 0–2 depending on data).
      2. 09:20 candle closes → opening range high/low.
         (We use the first candle: 09:15–09:20 as the opening range.)
         Actually per spec the range is 09:15–09:30 (3 candles).
      3. Wait for breakout candle 09:30–10:15 with filters.
      4. SL = opposite end of range, target = 1.5× risk.
      5. If nothing hit by exit_time, exit at market.

    Returns trade dict or None if no entry.
    """
    day_df = day_df.copy().reset_index(drop=True)
    if len(day_df) < 6:
        return None

    # --- Step 1-2: Identify Opening Range (09:15 – 09:30, first 3 candles) ---
    or_candles = day_df[day_df["time"] < "09:30"]
    if len(or_candles) == 0:
        # fallback: use first 3 candles
        or_candles = day_df.iloc[:3]

    or_high = or_candles["high"].max()
    or_low = or_candles["low"].min()
    or_range = or_high - or_low

    if or_range < 1:  # avoid degenerate ranges
        return None

    # VWAP at end of opening range
    vwap_vals = compute_vwap(day_df)
    or_end_idx = or_candles.index[-1]
    vwap_at_or = vwap_vals.iloc[or_end_idx]

    # Average volume of opening range candles
    avg_vol = or_candles["volume"].mean()

    # --- Step 3: Scan for breakout (09:30 – 10:15) ---
    scan_candles = day_df[(day_df["time"] >= "09:30") & (day_df["time"] <= "10:15")]

    entry_price = None
    direction = None  # "CE" (long/call) or "PE" (short/put)
    entry_time = None
    entry_idx = None

    for i, row in scan_candles.iterrows():
        current_vwap = vwap_vals.iloc[i]

        # Filter 1: Volume >= volume_mult × average
        vol_ok = row["volume"] >= volume_mult * avg_vol

        # Bullish breakout: body closes above OR high, VWAP below price (bullish)
        if _candle_body_above(row, or_high) and vol_ok and row["close"] > current_vwap:
            entry_price = row["close"]
            direction = "CE"
            entry_time = row["time"]
            entry_idx = i
            break

        # Bearish breakout: body closes below OR low, VWAP above price (bearish)
        if _candle_body_below(row, or_low) and vol_ok and row["close"] < current_vwap:
            entry_price = row["close"]
            direction = "PE"
            entry_time = row["time"]
            entry_idx = i
            break

    if entry_price is None:
        return None

    # --- Step 4: SL and Target ---
    if direction == "CE":
        sl = or_low
        risk = entry_price - sl
        target = entry_price + risk_reward * risk
    else:
        sl = or_high
        risk = sl - entry_price
        target = entry_price - risk_reward * risk

    if risk <= 0:
        return None

    # --- Step 5: Simulate candle-by-candle ---
    exit_price = None
    exit_reason = None
    exit_time_val = None

    for i, row in day_df.iloc[entry_idx + 1 :].iterrows():
        if row["time"] > exit_time:
            exit_price = row["open"]  # exit at market
            exit_reason = "Time Exit"
            exit_time_val = row["time"]
            break

        if direction == "CE":
            if row["low"] <= sl:
                exit_price = sl
                exit_reason = "SL Hit"
                exit_time_val = row["time"]
                break
            if row["high"] >= target:
                exit_price = target
                exit_reason = "Target Hit"
                exit_time_val = row["time"]
                break
        else:
            if row["high"] >= sl:
                exit_price = sl
                exit_reason = "SL Hit"
                exit_time_val = row["time"]
                break
            if row["low"] <= target:
                exit_price = target
                exit_reason = "Target Hit"
                exit_time_val = row["time"]
                break

    # If we never exited (data ends), close at last candle
    if exit_price is None:
        last = day_df.iloc[-1]
        exit_price = last["close"]
        exit_reason = "EOD Exit"
        exit_time_val = last["time"]

    pnl = (exit_price - entry_price) if direction == "CE" else (entry_price - exit_price)

    return {
        "date": day_df.iloc[0]["date"],
        "direction": direction,
        "or_high": round(or_high, 2),
        "or_low": round(or_low, 2),
        "or_range": round(or_range, 2),
        "vwap_at_or": round(vwap_at_or, 2),
        "entry_price": round(entry_price, 2),
        "entry_time": entry_time,
        "sl": round(sl, 2),
        "target": round(target, 2),
        "exit_price": round(exit_price, 2),
        "exit_time": exit_time_val,
        "exit_reason": exit_reason,
        "pnl_points": round(pnl, 2),
        "risk_points": round(risk, 2),
        "reward_ratio": round(pnl / risk, 2) if risk > 0 else 0,
    }


# ---------------------------------------------------------------------------
# VWAP Pullback Strategy
# ---------------------------------------------------------------------------

def run_vwap_pullback_day(
    day_df: pd.DataFrame,
    pullback_pct: float = 0.1,
    risk_reward: float = 1.5,
    exit_time: str = "14:30",
) -> dict | None:
    """
    VWAP Pullback strategy on a single day's 5-min data.

    Logic:
      - After 09:45, compute VWAP.
      - If price touches VWAP (within pullback_pct%) and bounces with
        a confirming candle (close beyond VWAP in trend direction):
          Long if prior trend was up (close > open on majority of prev 3 candles)
          Short if prior trend was down
      - SL: Recent swing low/high (3-candle low/high before entry)
      - Target: risk_reward × risk
      - Exit by exit_time if neither SL nor target hit.
    """
    day_df = day_df.copy().reset_index(drop=True)
    if len(day_df) < 10:
        return None

    vwap_vals = compute_vwap(day_df)

    scan = day_df[day_df["time"] >= "09:45"]
    entry_price = None
    direction = None
    entry_time = None
    entry_idx = None
    sl = None

    for i, row in scan.iterrows():
        if i < 4:
            continue
        current_vwap = vwap_vals.iloc[i]

        # Check proximity to VWAP
        dist_pct = abs(row["close"] - current_vwap) / current_vwap * 100
        if dist_pct > pullback_pct:
            continue

        # Determine prior trend from last 3 candles
        prev3 = day_df.iloc[i - 3 : i]
        bullish_count = (prev3["close"] > prev3["open"]).sum()

        if bullish_count >= 2 and row["close"] > current_vwap:
            # Bullish pullback bounce
            direction = "LONG"
            entry_price = row["close"]
            entry_time = row["time"]
            entry_idx = i
            sl = prev3["low"].min()
            break
        elif bullish_count <= 1 and row["close"] < current_vwap:
            # Bearish pullback bounce
            direction = "SHORT"
            entry_price = row["close"]
            entry_time = row["time"]
            entry_idx = i
            sl = prev3["high"].max()
            break

    if entry_price is None:
        return None

    risk = abs(entry_price - sl)
    if risk <= 0:
        return None

    if direction == "LONG":
        target = entry_price + risk_reward * risk
    else:
        target = entry_price - risk_reward * risk

    exit_price = None
    exit_reason = None
    exit_time_val = None

    for i, row in day_df.iloc[entry_idx + 1 :].iterrows():
        if row["time"] > exit_time:
            exit_price = row["open"]
            exit_reason = "Time Exit"
            exit_time_val = row["time"]
            break

        if direction == "LONG":
            if row["low"] <= sl:
                exit_price = sl
                exit_reason = "SL Hit"
                exit_time_val = row["time"]
                break
            if row["high"] >= target:
                exit_price = target
                exit_reason = "Target Hit"
                exit_time_val = row["time"]
                break
        else:
            if row["high"] >= sl:
                exit_price = sl
                exit_reason = "SL Hit"
                exit_time_val = row["time"]
                break
            if row["low"] <= target:
                exit_price = target
                exit_reason = "Target Hit"
                exit_time_val = row["time"]
                break

    if exit_price is None:
        last = day_df.iloc[-1]
        exit_price = last["close"]
        exit_reason = "EOD Exit"
        exit_time_val = last["time"]

    pnl = (exit_price - entry_price) if direction == "LONG" else (entry_price - exit_price)

    return {
        "date": day_df.iloc[0]["date"],
        "direction": direction,
        "vwap_at_entry": round(vwap_vals.iloc[entry_idx], 2),
        "entry_price": round(entry_price, 2),
        "entry_time": entry_time,
        "sl": round(sl, 2),
        "target": round(target, 2),
        "exit_price": round(exit_price, 2),
        "exit_time": exit_time_val,
        "exit_reason": exit_reason,
        "pnl_points": round(pnl, 2),
        "risk_points": round(risk, 2),
        "reward_ratio": round(pnl / risk, 2) if risk > 0 else 0,
    }


# ---------------------------------------------------------------------------
# Backtest runner
# ---------------------------------------------------------------------------

def backtest(
    df: pd.DataFrame,
    strategy: str = "ORB",
    start_date: str | None = None,
    end_date: str | None = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Run strategy across all trading days in df.
    Returns DataFrame of trades.
    """
    if start_date:
        df = df[df["date"] >= start_date]
    if end_date:
        df = df[df["date"] <= end_date]

    grouped = df.groupby("date")
    trades = []

    for date, day_df in grouped:
        if strategy == "ORB":
            result = run_orb_day(day_df, **kwargs)
        elif strategy == "VWAP Pullback":
            result = run_vwap_pullback_day(day_df, **kwargs)
        else:
            continue
        if result:
            trades.append(result)

    return pd.DataFrame(trades)


def compute_metrics(trades_df: pd.DataFrame) -> dict:
    """Compute summary metrics from a trades DataFrame."""
    if trades_df.empty:
        return {
            "total_trades": 0,
            "winners": 0,
            "losers": 0,
            "win_rate": 0,
            "total_pnl": 0,
            "avg_pnl": 0,
            "max_win": 0,
            "max_loss": 0,
            "avg_winner": 0,
            "avg_loser": 0,
            "profit_factor": 0,
            "max_drawdown": 0,
            "avg_risk_reward": 0,
            "target_hits": 0,
            "sl_hits": 0,
            "time_exits": 0,
        }

    winners = trades_df[trades_df["pnl_points"] > 0]
    losers = trades_df[trades_df["pnl_points"] <= 0]
    gross_profit = winners["pnl_points"].sum() if len(winners) > 0 else 0
    gross_loss = abs(losers["pnl_points"].sum()) if len(losers) > 0 else 0

    # Drawdown
    cumulative = trades_df["pnl_points"].cumsum()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max)
    max_dd = drawdown.min()

    return {
        "total_trades": len(trades_df),
        "winners": len(winners),
        "losers": len(losers),
        "win_rate": round(len(winners) / len(trades_df) * 100, 1),
        "total_pnl": round(trades_df["pnl_points"].sum(), 2),
        "avg_pnl": round(trades_df["pnl_points"].mean(), 2),
        "max_win": round(trades_df["pnl_points"].max(), 2),
        "max_loss": round(trades_df["pnl_points"].min(), 2),
        "avg_winner": round(winners["pnl_points"].mean(), 2) if len(winners) > 0 else 0,
        "avg_loser": round(losers["pnl_points"].mean(), 2) if len(losers) > 0 else 0,
        "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss > 0 else float("inf"),
        "max_drawdown": round(max_dd, 2),
        "avg_risk_reward": round(trades_df["reward_ratio"].mean(), 2),
        "target_hits": int((trades_df["exit_reason"] == "Target Hit").sum()),
        "sl_hits": int((trades_df["exit_reason"] == "SL Hit").sum()),
        "time_exits": int((trades_df["exit_reason"].isin(["Time Exit", "EOD Exit"])).sum()),
    }
