import { useState, useCallback } from 'react'
import FileUpload from './components/Sidebar/FileUpload'
import AlgoParams from './components/Sidebar/AlgoParams'
import TradeParams from './components/Sidebar/TradeParams'
import QuickStats from './components/Sidebar/QuickStats'
import CandlestickChart from './components/Charts/CandlestickChart'
import EquityCurveChart from './components/Charts/EquityCurveChart'
import TradeLogTable from './components/TradeLog/TradeLogTable'
import PerformanceDashboard from './components/Dashboard/PerformanceDashboard'
import ErrorBoundary from './components/ErrorBoundary'

const DEFAULT_ALGO_PARAMS = {
  length: 14,
  mult: 1.0,
  calc_method: 'Atr',
}

const DEFAULT_TRADE_PARAMS = {
  target_points: 50,
  stop_loss_points: 25,
  trailing_stop: false,
  trail_activation: 30,
  trail_distance: 15,
  entry_start_time: '09:20',
  entry_end_time: '15:00',
  exit_time: '15:10',
}

export default function App() {
  const [dataLoaded, setDataLoaded] = useState(false)
  const [algoParams, setAlgoParams] = useState(DEFAULT_ALGO_PARAMS)
  const [tradeParams, setTradeParams] = useState(DEFAULT_TRADE_PARAMS)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState(null)

  // Backtest results
  const [chartData, setChartData] = useState(null)
  const [buyMarkers, setBuyMarkers] = useState(null)
  const [sellMarkers, setSellMarkers] = useState(null)
  const [trades, setTrades] = useState(null)
  const [equityCurve, setEquityCurve] = useState(null)
  const [metrics, setMetrics] = useState(null)
  const [signalCount, setSignalCount] = useState(null)

  const onDataLoaded = useCallback((data) => {
    setDataLoaded(true)
    // Clear previous results
    setChartData(null)
    setTrades(null)
    setEquityCurve(null)
    setMetrics(null)
  }, [])

  const runBacktest = useCallback(async () => {
    setRunning(true)
    setError(null)
    try {
      const res = await fetch('/api/backtest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...algoParams, ...tradeParams }),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Backtest failed')

      setChartData(data.chart_data)
      setBuyMarkers(data.buy_markers)
      setSellMarkers(data.sell_markers)
      setTrades(data.trades)
      setEquityCurve(data.equity_curve)
      setMetrics(data.metrics)
      setSignalCount(data.signal_count)
    } catch (err) {
      setError(err.message)
    } finally {
      setRunning(false)
    }
  }, [algoParams, tradeParams])

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="bg-[#1a1d29] border-b border-[#2d3040] px-4 py-3 flex items-center justify-between">
        <div>
          <h1 className="text-base font-semibold text-gray-100">
            Nifty 50 Trendline Breakout Backtester
          </h1>
          <p className="text-[10px] text-gray-500">
            LuxAlgo Trendlines with Breaks — Spot-level backtesting
          </p>
        </div>
        {signalCount && (
          <div className="flex gap-4 text-xs">
            <span className="text-teal-400">{signalCount.buy} Buy Signals</span>
            <span className="text-red-400">{signalCount.sell} Sell Signals</span>
          </div>
        )}
      </header>

      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar */}
        <aside className="w-64 min-w-64 bg-[#1a1d29] border-r border-[#2d3040] p-3 overflow-y-auto flex flex-col">
          <FileUpload onDataLoaded={onDataLoaded} />

          <div className="border-t border-[#2d3040] my-2"></div>
          <AlgoParams params={algoParams} onChange={setAlgoParams} />

          <div className="border-t border-[#2d3040] my-2"></div>
          <TradeParams params={tradeParams} onChange={setTradeParams} />

          <div className="border-t border-[#2d3040] my-2"></div>

          <button
            onClick={runBacktest}
            disabled={!dataLoaded || running}
            className={`w-full py-2.5 rounded-lg text-sm font-medium transition-all ${
              !dataLoaded || running
                ? 'bg-gray-700 text-gray-500 cursor-not-allowed'
                : 'bg-teal-500 hover:bg-teal-400 text-white shadow-lg shadow-teal-500/20'
            }`}
          >
            {running ? (
              <span className="flex items-center justify-center gap-2">
                <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"/>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
                </svg>
                Running...
              </span>
            ) : (
              'Run Backtest'
            )}
          </button>

          {error && (
            <div className="mt-2 text-xs text-red-400 bg-red-400/10 p-2 rounded">{error}</div>
          )}

          <div className="border-t border-[#2d3040] my-3"></div>
          <QuickStats metrics={metrics} />

          <div className="mt-auto pt-4 text-[10px] text-gray-600 text-center">
            Trendlines with Breaks [LuxAlgo]
          </div>
        </aside>

        {/* Main Content */}
        <main className="flex-1 overflow-y-auto p-4 space-y-4">
          {!chartData ? (
            <div className="flex items-center justify-center h-full">
              <div className="text-center">
                <div className="text-6xl mb-4 opacity-20">&#x1F4C8;</div>
                <h2 className="text-lg text-gray-400 mb-2">No backtest results yet</h2>
                <p className="text-sm text-gray-600">
                  Upload your Nifty 50 OHLC data file (.xlsx or .csv), configure parameters, and click "Run Backtest"
                </p>
              </div>
            </div>
          ) : (
            <>
              <ErrorBoundary>
                <CandlestickChart
                  chartData={chartData}
                  buyMarkers={buyMarkers}
                  sellMarkers={sellMarkers}
                  trades={trades}
                />
              </ErrorBoundary>
              <ErrorBoundary>
                <EquityCurveChart equityCurve={equityCurve} />
              </ErrorBoundary>
              <TradeLogTable trades={trades} />
              <PerformanceDashboard metrics={metrics} />
            </>
          )}
        </main>
      </div>
    </div>
  )
}
