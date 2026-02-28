"""
Performance metric calculations from trade records.
"""

import numpy as np
from collections import defaultdict


def calculate_metrics(trades: list[dict]) -> dict:
    """
    Compute comprehensive performance metrics from trade records.
    """
    if not trades:
        return _empty_metrics()

    pnls = np.array([t["pnl_points"] for t in trades])
    total = len(pnls)
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    win_count = len(wins)
    loss_count = len(losses)

    gross_profit = float(np.sum(wins)) if win_count > 0 else 0.0
    gross_loss = float(np.sum(losses)) if loss_count > 0 else 0.0
    net_pnl = float(np.sum(pnls))

    win_rate = (win_count / total * 100) if total > 0 else 0.0
    avg_win = float(np.mean(wins)) if win_count > 0 else 0.0
    avg_loss = float(np.mean(losses)) if loss_count > 0 else 0.0
    profit_factor = (gross_profit / abs(gross_loss)) if gross_loss != 0 else float("inf")

    # Max drawdown from cumulative equity
    cum_pnl = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum_pnl)
    drawdown = peak - cum_pnl
    max_drawdown = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0

    # Sharpe ratio (trade-level, annualized assuming ~250 trades/year as proxy)
    if len(pnls) > 1 and np.std(pnls) > 0:
        sharpe = float(np.mean(pnls) / np.std(pnls) * np.sqrt(252))
    else:
        sharpe = 0.0

    # Expectancy
    loss_rate = loss_count / total if total > 0 else 0.0
    expectancy = (win_rate / 100 * avg_win) + (loss_rate * avg_loss)

    # Max consecutive wins/losses
    max_consec_wins = _max_consecutive(pnls, positive=True)
    max_consec_losses = _max_consecutive(pnls, positive=False)

    # Average holding bars
    holding_bars = [t.get("holding_bars", 0) for t in trades]
    avg_holding = float(np.mean(holding_bars)) if holding_bars else 0.0

    # Payoff ratio
    payoff_ratio = (avg_win / abs(avg_loss)) if avg_loss != 0 else float("inf")

    # Recovery factor
    recovery_factor = (net_pnl / max_drawdown) if max_drawdown > 0 else float("inf")

    # Long/Short breakdown
    longs = [t for t in trades if t["direction"] == "LONG"]
    shorts = [t for t in trades if t["direction"] == "SHORT"]
    long_wins = [t for t in longs if t["pnl_points"] > 0]
    short_wins = [t for t in shorts if t["pnl_points"] > 0]

    return {
        "total_trades": total,
        "wins": win_count,
        "losses": loss_count,
        "win_rate": round(win_rate, 2),
        "net_pnl": round(net_pnl, 2),
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 4) if profit_factor != float("inf") else "Inf",
        "max_drawdown": round(max_drawdown, 2),
        "sharpe_ratio": round(sharpe, 4),
        "expectancy": round(expectancy, 2),
        "max_consecutive_wins": max_consec_wins,
        "max_consecutive_losses": max_consec_losses,
        "avg_holding_bars": round(avg_holding, 1),
        "payoff_ratio": round(payoff_ratio, 4) if payoff_ratio != float("inf") else "Inf",
        "recovery_factor": round(recovery_factor, 4) if recovery_factor != float("inf") else "Inf",
        "longs_count": len(longs),
        "shorts_count": len(shorts),
        "long_win_rate": round(len(long_wins) / len(longs) * 100, 2) if longs else 0.0,
        "short_win_rate": round(len(short_wins) / len(shorts) * 100, 2) if shorts else 0.0,
    }


def _max_consecutive(pnls: np.ndarray, positive: bool) -> int:
    """Find the maximum consecutive wins (positive=True) or losses (positive=False)."""
    max_count = 0
    current = 0
    for pnl in pnls:
        if (positive and pnl > 0) or (not positive and pnl <= 0):
            current += 1
            max_count = max(max_count, current)
        else:
            current = 0
    return max_count


def _empty_metrics() -> dict:
    return {
        "total_trades": 0, "wins": 0, "losses": 0, "win_rate": 0,
        "net_pnl": 0, "gross_profit": 0, "gross_loss": 0,
        "avg_win": 0, "avg_loss": 0, "profit_factor": 0,
        "max_drawdown": 0, "sharpe_ratio": 0, "expectancy": 0,
        "max_consecutive_wins": 0, "max_consecutive_losses": 0,
        "avg_holding_bars": 0, "payoff_ratio": 0, "recovery_factor": 0,
        "longs_count": 0, "shorts_count": 0,
        "long_win_rate": 0, "short_win_rate": 0,
    }
