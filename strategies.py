"""
Intraday backtesting strategies for Nifty50 5-minute data.
No volume data required — all filters are price-action based.

Strategies implemented:
  1. ORB          – Opening Range Breakout (9:15–9:30 range)
  2. EMA Pullback – Trend-following entries off EMA bounce
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _candle_body_above(row: pd.Series, level: float) -> bool:
    """True if the entire candle body (open & close) is above the level."""
    return min(row["open"], row["close"]) > level


def _candle_body_below(row: pd.Series, level: float) -> bool:
    """True if the entire candle body (open & close) is below the level."""
    return max(row["open"], row["close"]) < level


def _is_bullish(row: pd.Series) -> bool:
    return row["close"] > row["open"]


def _is_bearish(row: pd.Series) -> bool:
    return row["close"] < row["open"]


# ---------------------------------------------------------------------------
# ORB Strategy  (no volume / no VWAP)
# ---------------------------------------------------------------------------

def run_orb_day(
    day_df: pd.DataFrame,
    risk_reward: float = 1.5,
    exit_time: str = "12:00",
) -> dict | None:
    """
    Run ORB strategy on a single day's 5-min data.

    Flow:
      1. 09:15-09:30 → first 3 candles form the Opening Range (OR High / OR Low).
      2. 09:30-10:15 → scan for a breakout candle whose body closes
         beyond the OR level. No volume or VWAP filter — pure price action.
      3. Entry at close of breakout candle.
         CE (long) if body closes above OR High.
         PE (short) if body closes below OR Low.
      4. SL  = opposite end of the range.
         Target = entry ± risk_reward × risk.
      5. If neither SL nor target hit by exit_time → exit at market.

    Returns trade dict or None if no entry.
    """
    day_df = day_df.copy().reset_index(drop=True)
    if len(day_df) < 6:
        return None

    # --- Step 1: Identify Opening Range (09:15 – 09:30) ---
    or_candles = day_df[day_df["time"] < "09:30"]
    if len(or_candles) == 0:
        or_candles = day_df.iloc[:3]

    or_high = or_candles["high"].max()
    or_low = or_candles["low"].min()
    or_range = or_high - or_low

    if or_range < 1:  # skip degenerate ranges
        return None

    # --- Step 2: Scan for breakout (09:30 – 10:15) ---
    scan_candles = day_df[(day_df["time"] >= "09:30") & (day_df["time"] <= "10:15")]

    entry_price = None
    direction = None  # "CE" (long/call) or "PE" (short/put)
    entry_time = None
    entry_idx = None

    for i, row in scan_candles.iterrows():
        # Bullish breakout: candle body closes above OR high
        if _candle_body_above(row, or_high):
            entry_price = row["close"]
            direction = "CE"
            entry_time = row["time"]
            entry_idx = i
            break

        # Bearish breakout: candle body closes below OR low
        if _candle_body_below(row, or_low):
            entry_price = row["close"]
            direction = "PE"
            entry_time = row["time"]
            entry_idx = i
            break

    if entry_price is None:
        return None

    # --- Step 3: SL and Target ---
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

    # --- Step 4: Simulate candle-by-candle ---
    exit_price = None
    exit_reason = None
    exit_time_val = None

    for i, row in day_df.iloc[entry_idx + 1 :].iterrows():
        if row["time"] > exit_time:
            exit_price = row["open"]
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
# EMA Pullback Strategy  (no volume required)
# ---------------------------------------------------------------------------

def _compute_ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=period, adjust=False).mean()


def run_ema_pullback_day(
    day_df: pd.DataFrame,
    ema_period: int = 20,
    risk_reward: float = 1.5,
    exit_time: str = "14:30",
) -> dict | None:
    """
    EMA Pullback strategy on a single day's 5-min data.

    Logic (no volume needed):
      - Compute intraday EMA on close prices.
      - After 09:45, wait for price to pull back to EMA and bounce:
          LONG  → prior trend bullish (2 of last 3 candles green),
                   candle low touches/crosses EMA, candle closes above EMA.
          SHORT → prior trend bearish (2 of last 3 candles red),
                   candle high touches/crosses EMA, candle closes below EMA.
      - SL: 3-candle swing low (long) or swing high (short) before entry.
      - Target: risk_reward × risk.
      - Exit by exit_time if neither hit.
    """
    day_df = day_df.copy().reset_index(drop=True)
    if len(day_df) < 10:
        return None

    ema = _compute_ema(day_df["close"], ema_period)

    scan = day_df[day_df["time"] >= "09:45"]
    entry_price = None
    direction = None
    entry_time = None
    entry_idx = None
    sl = None

    for i, row in scan.iterrows():
        if i < 4:
            continue
        current_ema = ema.iloc[i]

        prev3 = day_df.iloc[i - 3 : i]
        bullish_count = (prev3["close"] > prev3["open"]).sum()

        # LONG: uptrend + price pulled back to EMA and bounced above
        if bullish_count >= 2 and row["low"] <= current_ema and row["close"] > current_ema:
            direction = "LONG"
            entry_price = row["close"]
            entry_time = row["time"]
            entry_idx = i
            sl = prev3["low"].min()
            break

        # SHORT: downtrend + price pulled up to EMA and rejected below
        if bullish_count <= 1 and row["high"] >= current_ema and row["close"] < current_ema:
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
        "ema_at_entry": round(ema.iloc[entry_idx], 2),
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
        elif strategy == "EMA Pullback":
            result = run_ema_pullback_day(day_df, **kwargs)
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
    drawdown = cumulative - running_max
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
