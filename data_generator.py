"""
Data loading utilities for Nifty50 intraday backtesting.

Supports:
  - Loading 1-minute CSV data and resampling to 5-minute candles
  - Loading pre-aggregated 5-minute CSV data
  - Generating synthetic 5-minute OHLC data (no volume) for testing
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def _trading_minutes() -> list[str]:
    """Return 5-min interval timestamps from 09:15 to 15:25 (75 candles/day)."""
    base = datetime(2020, 1, 1, 9, 15)
    slots = []
    for i in range(75):
        t = base + timedelta(minutes=5 * i)
        if t.hour >= 15 and t.minute > 25:
            break
        slots.append(t.strftime("%H:%M"))
    return slots


def _business_days(start: str, end: str) -> list[datetime]:
    """Return business days (Mon-Fri) between start and end date strings."""
    dates = pd.bdate_range(start, end)
    return [d.to_pydatetime() for d in dates]


def resample_1min_to_5min(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample 1-minute OHLC data to 5-minute candles.

    Expects columns: datetime, open, high, low, close
    The datetime column must already be parsed as datetime.
    Groups by date first, then resamples within each day so overnight
    gaps don't merge candles across days.
    """
    df = df.copy().sort_values("datetime").reset_index(drop=True)
    df = df.set_index("datetime")

    agg_rules = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }

    resampled = (
        df.resample("5min", offset="0min")
        .agg(agg_rules)
        .dropna(subset=["open"])
        .reset_index()
    )

    resampled["date"] = resampled["datetime"].dt.strftime("%Y-%m-%d")
    resampled["time"] = resampled["datetime"].dt.strftime("%H:%M")

    # Keep only market hours (09:15 – 15:29)
    resampled = resampled[
        (resampled["time"] >= "09:15") & (resampled["time"] <= "15:25")
    ].reset_index(drop=True)

    return resampled


def generate_intraday_data(
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
    base_price: float = 22000.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic 5-min OHLC data for Nifty50 spot (no volume).

    Returns a DataFrame with columns:
        datetime, date, time, open, high, low, close
    """
    rng = np.random.RandomState(seed)
    days = _business_days(start_date, end_date)
    time_slots = _trading_minutes()
    rows = []
    price = base_price

    for day in days:
        daily_drift = rng.normal(0, 0.003)
        daily_vol = rng.uniform(0.001, 0.004)

        for idx, ts in enumerate(time_slots):
            candle_return = daily_drift / len(time_slots) + rng.normal(0, daily_vol)
            o = price
            c = o * (1 + candle_return)

            wick_up = abs(rng.normal(0, daily_vol * 0.5))
            wick_dn = abs(rng.normal(0, daily_vol * 0.5))
            h = max(o, c) * (1 + wick_up)
            l = min(o, c) * (1 - wick_dn)

            dt = day.replace(
                hour=int(ts.split(":")[0]),
                minute=int(ts.split(":")[1]),
                second=0,
            )
            rows.append(
                {
                    "datetime": dt,
                    "date": day.strftime("%Y-%m-%d"),
                    "time": ts,
                    "open": round(o, 2),
                    "high": round(h, 2),
                    "low": round(l, 2),
                    "close": round(c, 2),
                }
            )
            price = c

        # Overnight gap
        price *= 1 + rng.normal(0, 0.005)

    df = pd.DataFrame(rows)
    return df
