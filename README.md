# Nifty 50 Trendline Breakout Backtester

A web-based backtesting tool implementing the **"Trendlines with Breaks [LuxAlgo]"** indicator logic on Nifty 50 OHLC candlestick data.

## Architecture

- **Backend:** Python (FastAPI) — backtesting engine, indicators, trade simulation, metrics
- **Frontend:** React (Vite) + TradingView Lightweight Charts + Tailwind CSS

## Quick Start

### 1. Install backend dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Install frontend dependencies

```bash
cd frontend
npm install
```

### 3. Run both servers

Terminal 1 — Backend (port 8000):
```bash
cd backend
uvicorn main:app --reload --port 8000
```

Terminal 2 — Frontend (port 5173):
```bash
cd frontend
npm run dev
```

Open http://localhost:5173 in your browser.

## Usage

1. Upload your Nifty 50 OHLC data file (.xlsx or .csv)
   - Required columns: Date/Datetime, Open, High, Low, Close
   - Supports 1-minute candle data (up to 140K+ rows)
2. Configure algorithm parameters (swing lookback, slope multiplier, slope method)
3. Configure trade parameters (target, stop loss, trailing stop, time filters)
4. Click "Run Backtest"
5. View results: candlestick chart with trendlines and signals, equity curve, trade log, performance metrics

## Algorithm

Replicates the exact Pine Script v5 logic:
- Symmetric pivot detection (lookback both sides)
- Slope calculation via ATR, StdDev, or LinReg
- Trendline projection with slope persistence
- Breakout detection on state transitions

## Trade Simulation

- One position at a time
- Exit priority: Stop Loss > Target > Time Exit > Reverse Signal
- Configurable trailing stop with activation threshold
- Time-based entry/exit filters (market hours)
