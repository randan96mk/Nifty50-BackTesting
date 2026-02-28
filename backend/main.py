"""
FastAPI backend for Nifty 50 Trendline Breakout Backtester.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd

from engine.trendlines import calculate_trendlines
from engine.backtester import run_backtest
from engine.metrics import calculate_metrics
from utils.excel_parser import parse_file

app = FastAPI(title="Nifty 50 Trendline Breakout Backtester")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory data store
uploaded_data = {}


class BacktestParams(BaseModel):
    # Algo parameters
    length: int = 14
    mult: float = 1.0
    calc_method: str = "Atr"

    # Date range filter (empty string = no filter)
    date_start: str = ""
    date_end: str = ""

    # Candle timeframe resampling
    timeframe: str = "1m"

    # Trade parameters
    target_points: float = 50.0
    stop_loss_points: float = 25.0
    sl_mode: str = "fixed"  # "fixed" or "break_candle"
    trailing_stop: bool = False
    trail_activation: float = 30.0
    trail_distance: float = 15.0
    entry_start_time: str = "09:20"
    entry_end_time: str = "15:00"
    exit_time: str = "15:10"


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload and parse OHLC data file (xlsx, xls, csv)."""
    if not file.filename:
        raise HTTPException(400, "No file provided")

    contents = await file.read()
    try:
        data = parse_file(contents, file.filename)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(400, f"Failed to parse file: {str(e)}")

    uploaded_data["current"] = data

    return {
        "status": "ok",
        "total_rows": data["total_rows"],
        "date_range": data["date_range"],
        "sample": {
            "datetimes": data["datetimes"][:5],
            "open": data["open"][:5],
            "high": data["high"][:5],
            "low": data["low"][:5],
            "close": data["close"][:5],
        },
    }


@app.post("/api/backtest")
async def run_backtest_endpoint(params: BacktestParams):
    """Run backtest on uploaded data with given parameters."""
    if "current" not in uploaded_data:
        raise HTTPException(400, "No data uploaded. Please upload a file first.")

    raw = uploaded_data["current"]

    # Build a DataFrame for filtering and resampling
    df = pd.DataFrame({
        "datetime": pd.to_datetime(raw["datetimes"]),
        "open": np.array(raw["open"], dtype=float),
        "high": np.array(raw["high"], dtype=float),
        "low": np.array(raw["low"], dtype=float),
        "close": np.array(raw["close"], dtype=float),
    })

    # --- Date range filter ---
    if params.date_start:
        df = df[df["datetime"] >= pd.to_datetime(params.date_start)]
    if params.date_end:
        df = df[df["datetime"] <= pd.to_datetime(params.date_end) + pd.Timedelta(days=1)]

    if len(df) == 0:
        raise HTTPException(400, "No data in selected date range.")

    df = df.sort_values("datetime").reset_index(drop=True)

    # --- Timeframe resampling ---
    TIMEFRAME_MAP = {
        "1m": "1min", "2m": "2min", "3m": "3min", "5m": "5min",
        "10m": "10min", "15m": "15min", "30m": "30min", "1h": "1h",
    }
    tf_rule = TIMEFRAME_MAP.get(params.timeframe)
    if tf_rule and params.timeframe != "1m":
        df = df.set_index("datetime")
        resampled = df.resample(tf_rule).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
        }).dropna()
        resampled = resampled.reset_index()
        df = resampled

    if len(df) == 0:
        raise HTTPException(400, "No data after resampling. Check timeframe and date range.")

    # Extract arrays for engine
    datetimes_list = df["datetime"].tolist()
    dt_strings = df["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values

    # Step 1: Calculate trendlines and signals
    trendline_result = calculate_trendlines(
        highs=highs,
        lows=lows,
        closes=closes,
        length=params.length,
        mult=params.mult,
        calc_method=params.calc_method,
    )

    # Step 2: Run trade simulation
    backtest_result = run_backtest(
        datetimes=datetimes_list,
        opens=opens,
        highs=highs,
        lows=lows,
        closes=closes,
        buy_signals=trendline_result["buy_signals"],
        sell_signals=trendline_result["sell_signals"],
        target_points=params.target_points,
        stop_loss_points=params.stop_loss_points,
        sl_mode=params.sl_mode,
        trailing_stop=params.trailing_stop,
        trail_activation=params.trail_activation,
        trail_distance=params.trail_distance,
        entry_start_time=params.entry_start_time,
        entry_end_time=params.entry_end_time,
        exit_time=params.exit_time,
    )

    # Step 3: Calculate metrics
    metrics = calculate_metrics(backtest_result["trades"])

    # Step 4: Prepare chart data
    n = len(closes)
    upper_line = trendline_result["upper_line"]
    lower_line = trendline_result["lower_line"]

    chart_data = []
    for i in range(n):
        entry = {
            "time": dt_strings[i],
            "open": float(opens[i]),
            "high": float(highs[i]),
            "low": float(lows[i]),
            "close": float(closes[i]),
        }
        if not np.isnan(upper_line[i]):
            entry["upper"] = round(float(upper_line[i]), 2)
        if not np.isnan(lower_line[i]):
            entry["lower"] = round(float(lower_line[i]), 2)
        chart_data.append(entry)

    buy_markers = [
        {"time": dt_strings[i], "position": "belowBar", "color": "#26a69a",
         "shape": "arrowUp", "text": "B", "price": float(lows[i])}
        for i in trendline_result["buy_signals"]
    ]
    sell_markers = [
        {"time": dt_strings[i], "position": "aboveBar", "color": "#ef5350",
         "shape": "arrowDown", "text": "S", "price": float(highs[i])}
        for i in trendline_result["sell_signals"]
    ]

    pivot_high_markers = [
        {"time": dt_strings[p["pivot_bar"]], "value": p["value"], "type": "high"}
        for p in trendline_result["pivot_highs"]
    ]
    pivot_low_markers = [
        {"time": dt_strings[p["pivot_bar"]], "value": p["value"], "type": "low"}
        for p in trendline_result["pivot_lows"]
    ]

    return {
        "chart_data": chart_data,
        "buy_markers": buy_markers,
        "sell_markers": sell_markers,
        "pivot_highs": pivot_high_markers,
        "pivot_lows": pivot_low_markers,
        "trades": backtest_result["trades"],
        "equity_curve": backtest_result["equity_curve"],
        "metrics": metrics,
        "signal_count": {
            "buy": len(trendline_result["buy_signals"]),
            "sell": len(trendline_result["sell_signals"]),
        },
        "candle_count": n,
        "timeframe": params.timeframe,
    }


@app.get("/api/health")
async def health():
    return {"status": "ok"}
