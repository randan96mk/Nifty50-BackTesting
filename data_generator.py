"""
Generates realistic synthetic Nifty50 5-minute intraday OHLCV data.
Used when no real historical data CSV is available.
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


def generate_intraday_data(
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
    base_price: float = 22000.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic 5-min OHLCV data for Nifty50 spot.

    Returns a DataFrame with columns:
        datetime, date, time, open, high, low, close, volume
    """
    rng = np.random.RandomState(seed)
    days = _business_days(start_date, end_date)
    time_slots = _trading_minutes()
    rows = []
    price = base_price

    for day in days:
        daily_drift = rng.normal(0, 0.003)  # daily bias
        daily_vol = rng.uniform(0.001, 0.004)  # intraday vol per candle
        day_open = price

        for idx, ts in enumerate(time_slots):
            candle_return = daily_drift / len(time_slots) + rng.normal(0, daily_vol)
            o = price
            c = o * (1 + candle_return)

            wick_up = abs(rng.normal(0, daily_vol * 0.5))
            wick_dn = abs(rng.normal(0, daily_vol * 0.5))
            h = max(o, c) * (1 + wick_up)
            l = min(o, c) * (1 - wick_dn)

            # Volume pattern: higher at open/close, lower midday
            hour = int(ts.split(":")[0])
            minute = int(ts.split(":")[1])
            if hour == 9 and minute < 45:
                vol_mult = rng.uniform(1.5, 3.0)
            elif hour >= 14 and minute >= 30:
                vol_mult = rng.uniform(1.2, 2.5)
            else:
                vol_mult = rng.uniform(0.5, 1.2)
            volume = int(rng.uniform(50000, 200000) * vol_mult)

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
                    "volume": volume,
                }
            )
            price = c

        # Overnight gap
        price *= 1 + rng.normal(0, 0.005)

    df = pd.DataFrame(rows)
    return df


def load_or_generate(
    csv_path: str | None = None,
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
) -> pd.DataFrame:
    """
    Load data from CSV if available, otherwise generate synthetic data.

    Expected CSV columns: datetime, open, high, low, close, volume
    """
    if csv_path:
        df = pd.read_csv(csv_path, parse_dates=["datetime"])
        if "date" not in df.columns:
            df["date"] = df["datetime"].dt.strftime("%Y-%m-%d")
        if "time" not in df.columns:
            df["time"] = df["datetime"].dt.strftime("%H:%M")
        return df

    return generate_intraday_data(start_date=start_date, end_date=end_date)
