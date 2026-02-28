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

    # Trade parameters
    target_points: float = 50.0
    stop_loss_points: float = 25.0
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

    data = uploaded_data["current"]

    datetimes = pd.to_datetime(data["datetimes"]).tolist()
    opens = np.array(data["open"], dtype=float)
    highs = np.array(data["high"], dtype=float)
    lows = np.array(data["low"], dtype=float)
    closes = np.array(data["close"], dtype=float)

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
        datetimes=datetimes,
        opens=opens,
        highs=highs,
        lows=lows,
        closes=closes,
        buy_signals=trendline_result["buy_signals"],
        sell_signals=trendline_result["sell_signals"],
        target_points=params.target_points,
        stop_loss_points=params.stop_loss_points,
        trailing_stop=params.trailing_stop,
        trail_activation=params.trail_activation,
        trail_distance=params.trail_distance,
        entry_start_time=params.entry_start_time,
        entry_end_time=params.entry_end_time,
        exit_time=params.exit_time,
    )

    # Step 3: Calculate metrics
    metrics = calculate_metrics(backtest_result["trades"])

    # Step 4: Prepare chart data (downsample trendlines for JSON transfer)
    n = len(closes)
    upper_line = trendline_result["upper_line"]
    lower_line = trendline_result["lower_line"]

    # Build OHLC data with trendlines for chart
    chart_data = []
    for i in range(n):
        entry = {
            "time": data["datetimes"][i],
            "open": data["open"][i],
            "high": data["high"][i],
            "low": data["low"][i],
            "close": data["close"][i],
        }
        if not np.isnan(upper_line[i]):
            entry["upper"] = round(float(upper_line[i]), 2)
        if not np.isnan(lower_line[i]):
            entry["lower"] = round(float(lower_line[i]), 2)
        chart_data.append(entry)

    # Signal markers
    buy_markers = [
        {"time": data["datetimes"][i], "position": "belowBar", "color": "#26a69a",
         "shape": "arrowUp", "text": "B", "price": data["low"][i]}
        for i in trendline_result["buy_signals"]
    ]
    sell_markers = [
        {"time": data["datetimes"][i], "position": "aboveBar", "color": "#ef5350",
         "shape": "arrowDown", "text": "S", "price": data["high"][i]}
        for i in trendline_result["sell_signals"]
    ]

    # Pivot markers
    pivot_high_markers = [
        {"time": data["datetimes"][p["pivot_bar"]], "value": p["value"], "type": "high"}
        for p in trendline_result["pivot_highs"]
    ]
    pivot_low_markers = [
        {"time": data["datetimes"][p["pivot_bar"]], "value": p["value"], "type": "low"}
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
    }


@app.get("/api/health")
async def health():
    return {"status": "ok"}
