"""
Trade simulation engine.
Processes OHLC data + signals to generate trade records with P&L.
"""

import numpy as np
import pandas as pd


def run_backtest(
    datetimes: list,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    buy_signals: list[int],
    sell_signals: list[int],
    target_points: float = 50.0,
    stop_loss_points: float = 25.0,
    sl_mode: str = "fixed",
    trailing_stop: bool = False,
    trail_activation: float = 30.0,
    trail_distance: float = 15.0,
    entry_start_time: str = "09:20",
    entry_end_time: str = "15:00",
    exit_time: str = "15:10",
) -> dict:
    """
    Run trade simulation on OHLC data with given signals.

    sl_mode:
        "fixed"        - Use stop_loss_points from entry price
        "break_candle" - Buy SL = previous candle low, Sell SL = previous candle high

    Returns:
        trades: list of trade records
        equity_curve: list of {datetime, cumulative_pnl}
    """
    n = len(closes)

    # Parse time constraints
    entry_start = _parse_time(entry_start_time)
    entry_end = _parse_time(entry_end_time)
    force_exit = _parse_time(exit_time)

    # Build signal lookup: bar_index -> "BUY" or "SELL"
    signal_at = {}
    for idx in buy_signals:
        signal_at[idx] = "BUY"
    for idx in sell_signals:
        # If both BUY and SELL at same bar, SELL overwrites (last wins)
        if idx in signal_at:
            signal_at[idx] = "BOTH"
        else:
            signal_at[idx] = "SELL"

    trades = []
    equity_curve = []
    trade_id = 0
    cumulative_pnl = 0.0

    # Active trade state
    active_trade = None

    for i in range(n):
        dt = datetimes[i]
        bar_time = dt.time() if hasattr(dt, "time") else None

        # 1. Check active trade exits
        if active_trade is not None:
            trade_result = _evaluate_exits(
                active_trade, i, highs[i], lows[i], closes[i],
                dt, bar_time, force_exit, trailing_stop, trail_activation, trail_distance
            )

            if trade_result is not None:
                trade_result["trade_id"] = trade_id
                trade_result["holding_bars"] = i - active_trade["entry_bar"]
                trades.append(trade_result)
                cumulative_pnl += trade_result["pnl_points"]
                equity_curve.append({
                    "datetime": str(dt),
                    "cumulative_pnl": round(cumulative_pnl, 2),
                    "trade_id": trade_id,
                })
                trade_id += 1
                active_trade = None

        # 2. Check for new signals
        sig = signal_at.get(i)
        if sig and bar_time is not None:
            in_time_window = entry_start <= bar_time <= entry_end

            # Compute SL based on mode
            def _get_sl(direction):
                if sl_mode == "break_candle" and i > 0:
                    if direction == "LONG":
                        # SL = previous candle's low
                        return max(closes[i] - lows[i - 1], 1.0)
                    else:
                        # SL = previous candle's high
                        return max(highs[i - 1] - closes[i], 1.0)
                return stop_loss_points

            if sig == "BOTH":
                # If we had a position, it was already closed above by reverse logic
                # Open based on the last signal direction - prefer BUY
                if in_time_window and active_trade is None:
                    active_trade = _open_trade("LONG", i, closes[i], dt, target_points, _get_sl("LONG"))
            elif sig == "BUY":
                # Close short if active, open long
                if active_trade is not None and active_trade["direction"] == "SHORT":
                    result = _close_trade(active_trade, i, closes[i], dt, "REVERSE_SIGNAL")
                    result["trade_id"] = trade_id
                    result["holding_bars"] = i - active_trade["entry_bar"]
                    trades.append(result)
                    cumulative_pnl += result["pnl_points"]
                    equity_curve.append({
                        "datetime": str(dt),
                        "cumulative_pnl": round(cumulative_pnl, 2),
                        "trade_id": trade_id,
                    })
                    trade_id += 1
                    active_trade = None

                if in_time_window and active_trade is None:
                    active_trade = _open_trade("LONG", i, closes[i], dt, target_points, _get_sl("LONG"))

            elif sig == "SELL":
                if active_trade is not None and active_trade["direction"] == "LONG":
                    result = _close_trade(active_trade, i, closes[i], dt, "REVERSE_SIGNAL")
                    result["trade_id"] = trade_id
                    result["holding_bars"] = i - active_trade["entry_bar"]
                    trades.append(result)
                    cumulative_pnl += result["pnl_points"]
                    equity_curve.append({
                        "datetime": str(dt),
                        "cumulative_pnl": round(cumulative_pnl, 2),
                        "trade_id": trade_id,
                    })
                    trade_id += 1
                    active_trade = None

                if in_time_window and active_trade is None:
                    active_trade = _open_trade("SHORT", i, closes[i], dt, target_points, _get_sl("SHORT"))

    # Close any remaining trade at last bar
    if active_trade is not None:
        result = _close_trade(active_trade, n - 1, closes[n - 1], datetimes[n - 1], "TIME_EXIT")
        result["trade_id"] = trade_id
        result["holding_bars"] = (n - 1) - active_trade["entry_bar"]
        trades.append(result)
        cumulative_pnl += result["pnl_points"]
        equity_curve.append({
            "datetime": str(datetimes[n - 1]),
            "cumulative_pnl": round(cumulative_pnl, 2),
            "trade_id": trade_id,
        })

    return {
        "trades": trades,
        "equity_curve": equity_curve,
    }


def _parse_time(time_str: str):
    """Parse HH:MM string to time object."""
    parts = time_str.split(":")
    h, m = int(parts[0]), int(parts[1])
    from datetime import time
    return time(h, m)


def _open_trade(direction: str, bar: int, price: float, dt, target_pts: float, sl_pts: float) -> dict:
    """Create a new trade entry."""
    if direction == "LONG":
        tp = price + target_pts
        sl = price - sl_pts
    else:
        tp = price - target_pts
        sl = price + sl_pts

    return {
        "direction": direction,
        "entry_bar": bar,
        "entry_price": price,
        "entry_time": dt,
        "tp_level": tp,
        "sl_level": sl,
        "max_favorable": 0.0,
        "max_adverse": 0.0,
        "trail_active": False,
        "trail_sl": sl,
    }


def _evaluate_exits(
    trade: dict, bar: int, high: float, low: float, close: float,
    dt, bar_time, force_exit_time, trailing_stop: bool,
    trail_activation: float, trail_distance: float
) -> dict | None:
    """
    Check exit conditions in priority order:
    1. Stop Loss
    2. Target
    3. Time exit
    4. (Reverse signal handled in main loop)
    Returns trade result dict or None.
    """
    direction = trade["direction"]
    entry_price = trade["entry_price"]

    # Track excursions
    if direction == "LONG":
        unrealized_profit = high - entry_price
        unrealized_loss = entry_price - low
    else:
        unrealized_profit = entry_price - low
        unrealized_loss = high - entry_price

    trade["max_favorable"] = max(trade["max_favorable"], unrealized_profit)
    trade["max_adverse"] = max(trade["max_adverse"], unrealized_loss)

    # Trailing stop update
    if trailing_stop and not trade["trail_active"]:
        if direction == "LONG" and (high - entry_price) >= trail_activation:
            trade["trail_active"] = True
            trade["trail_sl"] = high - trail_distance
        elif direction == "SHORT" and (entry_price - low) >= trail_activation:
            trade["trail_active"] = True
            trade["trail_sl"] = low + trail_distance

    if trailing_stop and trade["trail_active"]:
        if direction == "LONG":
            trade["trail_sl"] = max(trade["trail_sl"], high - trail_distance)
        else:
            trade["trail_sl"] = min(trade["trail_sl"], low + trail_distance)

    # 1. Check Stop Loss
    sl_level = trade["trail_sl"] if (trailing_stop and trade["trail_active"]) else trade["sl_level"]
    if direction == "LONG" and low <= sl_level:
        exit_price = sl_level
        reason = "TRAILING_SL" if (trailing_stop and trade["trail_active"]) else "STOPLOSS"
        return _close_trade(trade, bar, exit_price, dt, reason)
    elif direction == "SHORT" and high >= sl_level:
        exit_price = sl_level
        reason = "TRAILING_SL" if (trailing_stop and trade["trail_active"]) else "STOPLOSS"
        return _close_trade(trade, bar, exit_price, dt, reason)

    # 2. Check Target
    if direction == "LONG" and high >= trade["tp_level"]:
        return _close_trade(trade, bar, trade["tp_level"], dt, "TARGET")
    elif direction == "SHORT" and low <= trade["tp_level"]:
        return _close_trade(trade, bar, trade["tp_level"], dt, "TARGET")

    # 3. Check Time exit
    if bar_time is not None and bar_time >= force_exit_time:
        return _close_trade(trade, bar, close, dt, "TIME_EXIT")

    return None


def _close_trade(trade: dict, bar: int, exit_price: float, dt, reason: str) -> dict:
    """Close a trade and return the trade record."""
    direction = trade["direction"]
    entry_price = trade["entry_price"]

    if direction == "LONG":
        pnl = exit_price - entry_price
    else:
        pnl = entry_price - exit_price

    return {
        "direction": direction,
        "entry_time": str(trade["entry_time"]),
        "entry_price": round(entry_price, 2),
        "exit_time": str(dt),
        "exit_price": round(exit_price, 2),
        "exit_reason": reason,
        "pnl_points": round(pnl, 2),
        "max_favorable_excursion": round(trade["max_favorable"], 2),
        "max_adverse_excursion": round(trade["max_adverse"], 2),
    }
