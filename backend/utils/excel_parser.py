"""
Excel/CSV file parsing with fuzzy column matching and validation.
"""

import pandas as pd
import numpy as np
import io
from datetime import datetime


# Fuzzy column name mapping
COLUMN_ALIASES = {
    "datetime": ["datetime", "date", "time", "timestamp", "date/time", "date_time", "dt"],
    "open": ["open", "o", "open_price", "opening"],
    "high": ["high", "h", "high_price", "highest"],
    "low": ["low", "l", "low_price", "lowest"],
    "close": ["close", "c", "close_price", "closing", "ltp", "last"],
    "volume": ["volume", "vol", "v", "qty", "quantity"],
}


def parse_file(file_bytes: bytes, filename: str) -> dict:
    """
    Parse uploaded Excel or CSV file into structured OHLC data.
    Returns dict with arrays and metadata.
    """
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""

    if ext in ("xlsx", "xls"):
        df = pd.read_excel(io.BytesIO(file_bytes))
    elif ext == "csv":
        df = pd.read_csv(io.BytesIO(file_bytes))
    else:
        raise ValueError(f"Unsupported file type: .{ext}. Use .xlsx, .xls, or .csv")

    # Fuzzy match columns
    col_map = _match_columns(df.columns.tolist())

    if "datetime" not in col_map:
        raise ValueError("Could not find a date/datetime column. Expected: date, datetime, timestamp, etc.")
    for req in ["open", "high", "low", "close"]:
        if req not in col_map:
            raise ValueError(f"Could not find '{req}' column. Check your file headers.")

    # Rename to standard names
    rename = {col_map[k]: k for k in col_map}
    df = df.rename(columns=rename)

    # Parse datetime
    df["datetime"] = pd.to_datetime(df["datetime"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["datetime"])

    # Drop rows with NaN OHLC
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])

    # Sort by datetime
    df = df.sort_values("datetime").reset_index(drop=True)

    # Convert to serializable format
    datetimes = df["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()

    return {
        "datetimes": datetimes,
        "open": df["open"].values.tolist(),
        "high": df["high"].values.tolist(),
        "low": df["low"].values.tolist(),
        "close": df["close"].values.tolist(),
        "volume": df["volume"].values.tolist() if "volume" in df.columns else [],
        "total_rows": len(df),
        "date_range": f"{datetimes[0]} to {datetimes[-1]}" if datetimes else "",
    }


def _match_columns(columns: list[str]) -> dict:
    """Fuzzy match DataFrame columns to standard OHLC names."""
    matched = {}
    lower_cols = {c.strip().lower(): c for c in columns}

    for std_name, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in lower_cols:
                matched[std_name] = lower_cols[alias]
                break

    return matched
