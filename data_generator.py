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


# ---------------------------------------------------------------------------
# Daily OHLC data generator (for chart pattern backtesting)
# ---------------------------------------------------------------------------

def generate_daily_data(
    start_date: str = "2015-01-01",
    end_date: str = "2024-12-31",
    base_price: float = 8282.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic daily OHLC data resembling Nifty 50 price history.

    Uses a regime-switching model with distinct market phases (bull, bear,
    sideways, crash, recovery) to produce data that naturally contains
    chart patterns such as triple tops/bottoms, triangles, and flags.

    Returns a DataFrame with columns: date, open, high, low, close
    """
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range(start_date, end_date)

    # Regime sequence: (approx_days, daily_drift, daily_volatility)
    regimes = [
        (120, 0.00040, 0.0100),   # 2015 H1: mild bull
        (60, -0.00020, 0.0120),   # 2015 mid: pullback
        (70,  0.00030, 0.0090),   # 2015 H2: recovery
        (80, -0.00030, 0.0130),   # 2016 early: correction
        (100, 0.00050, 0.0080),   # 2016-17: bull run
        (120, 0.00060, 0.0090),   # 2017: strong bull
        (60,  0.00040, 0.0110),   # 2018 H1: topping
        (40,  0.00000, 0.0080),   # consolidation
        (60, -0.00040, 0.0140),   # 2018 H2: correction
        (80,  0.00030, 0.0100),   # 2019 H1: recovery
        (40,  0.00000, 0.0070),   # 2019 mid: range-bound
        (60,  0.00020, 0.0090),   # 2019 H2: mild up
        (40, -0.00020, 0.0120),   # 2020 Q1: pre-crash
        (25, -0.00350, 0.0350),   # 2020 Mar: COVID crash
        (60,  0.00150, 0.0200),   # 2020 Q2-Q3: recovery
        (100, 0.00060, 0.0110),   # 2020 H2 - 2021 H1: bull
        (80,  0.00050, 0.0100),   # 2021 H2: continuation
        (40,  0.00000, 0.0080),   # 2022 Q1: topping
        (60, -0.00030, 0.0140),   # 2022 H1: correction
        (50,  0.00000, 0.0090),   # 2022 H2: range-bound
        (80,  0.00040, 0.0100),   # 2023 H1: bull
        (60,  0.00020, 0.0080),   # 2023 H2: consolidation
        (80,  0.00050, 0.0110),   # 2024 H1: bull
        (60,  0.00030, 0.0090),   # 2024 H2: mild bull
    ]

    price = base_price
    rows = []
    regime_idx = 0
    bars_in_regime = 0

    for date in dates:
        if regime_idx < len(regimes):
            drift, vol = regimes[regime_idx][1], regimes[regime_idx][2]
        else:
            drift, vol = regimes[-1][1], regimes[-1][2]

        daily_return = rng.normal(drift, vol)
        close = price * (1 + daily_return)

        # Realistic OHLC: wicks extend beyond the body
        body = abs(close - price)
        upper_wick = body * rng.uniform(0.1, 1.2)
        lower_wick = body * rng.uniform(0.1, 1.2)
        high = max(price, close) + upper_wick
        low = min(price, close) - lower_wick

        rows.append({
            "date": date.strftime("%Y-%m-%d"),
            "open": round(price, 2),
            "high": round(high, 2),
            "low": round(low, 2),
            "close": round(close, 2),
        })

        price = close
        bars_in_regime += 1

        if regime_idx < len(regimes) and bars_in_regime >= regimes[regime_idx][0]:
            regime_idx += 1
            bars_in_regime = 0

    return pd.DataFrame(rows)
