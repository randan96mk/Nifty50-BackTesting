import { useEffect, useRef, useCallback } from 'react'
import { createChart, CrosshairMode } from 'lightweight-charts'

export default function CandlestickChart({ chartData, buyMarkers, sellMarkers, trades, onTradeClick }) {
  const containerRef = useRef(null)
  const chartRef = useRef(null)
  const candleSeriesRef = useRef(null)
  const upperLineRef = useRef(null)
  const lowerLineRef = useRef(null)

  // Convert datetime string to Unix timestamp
  const toTimestamp = useCallback((dateStr) => {
    return Math.floor(new Date(dateStr).getTime() / 1000)
  }, [])

  useEffect(() => {
    if (!containerRef.current || !chartData || chartData.length === 0) return

    // Clean up previous chart
    if (chartRef.current) {
      chartRef.current.remove()
      chartRef.current = null
    }

    const chart = createChart(containerRef.current, {
      layout: {
        background: { color: '#1a1d29' },
        textColor: '#8b8fa3',
        fontSize: 11,
      },
      grid: {
        vertLines: { color: '#242736' },
        horzLines: { color: '#242736' },
      },
      crosshair: {
        mode: CrosshairMode.Normal,
      },
      rightPriceScale: {
        borderColor: '#2d3040',
      },
      timeScale: {
        borderColor: '#2d3040',
        timeVisible: true,
        secondsVisible: false,
      },
      width: containerRef.current.clientWidth,
      height: 450,
    })

    chartRef.current = chart

    // Candlestick series
    const candleSeries = chart.addCandlestickSeries({
      upColor: '#26a69a',
      downColor: '#ef5350',
      borderUpColor: '#26a69a',
      borderDownColor: '#ef5350',
      wickUpColor: '#26a69a',
      wickDownColor: '#ef5350',
    })

    const candleData = chartData.map((d) => ({
      time: toTimestamp(d.time),
      open: d.open,
      high: d.high,
      low: d.low,
      close: d.close,
    }))
    candleSeries.setData(candleData)
    candleSeriesRef.current = candleSeries

    // Upper trendline (teal)
    const upperData = chartData
      .filter((d) => d.upper !== undefined)
      .map((d) => ({ time: toTimestamp(d.time), value: d.upper }))

    if (upperData.length > 0) {
      const upperLine = chart.addLineSeries({
        color: '#26a69a',
        lineWidth: 1,
        lineStyle: 2,
        crosshairMarkerVisible: false,
        priceLineVisible: false,
        lastValueVisible: false,
      })
      upperLine.setData(upperData)
      upperLineRef.current = upperLine
    }

    // Lower trendline (red)
    const lowerData = chartData
      .filter((d) => d.lower !== undefined)
      .map((d) => ({ time: toTimestamp(d.time), value: d.lower }))

    if (lowerData.length > 0) {
      const lowerLine = chart.addLineSeries({
        color: '#ef5350',
        lineWidth: 1,
        lineStyle: 2,
        crosshairMarkerVisible: false,
        priceLineVisible: false,
        lastValueVisible: false,
      })
      lowerLine.setData(lowerData)
      lowerLineRef.current = lowerLine
    }

    // Buy/Sell markers on candlestick series
    const markers = []
    if (buyMarkers) {
      buyMarkers.forEach((m) => {
        markers.push({
          time: toTimestamp(m.time),
          position: 'belowBar',
          color: '#26a69a',
          shape: 'arrowUp',
          text: 'B',
        })
      })
    }
    if (sellMarkers) {
      sellMarkers.forEach((m) => {
        markers.push({
          time: toTimestamp(m.time),
          position: 'aboveBar',
          color: '#ef5350',
          shape: 'arrowDown',
          text: 'S',
        })
      })
    }

    // Add exit markers from trades
    if (trades) {
      trades.forEach((t) => {
        markers.push({
          time: toTimestamp(t.exit_time),
          position: t.direction === 'LONG' ? 'aboveBar' : 'belowBar',
          color: '#9e9e9e',
          shape: 'circle',
          text: t.exit_reason === 'TARGET' ? 'TP' : t.exit_reason === 'STOPLOSS' ? 'SL' : 'X',
        })
      })
    }

    markers.sort((a, b) => a.time - b.time)
    candleSeries.setMarkers(markers)

    // Fit content
    chart.timeScale().fitContent()

    // Resize handler
    const handleResize = () => {
      if (containerRef.current) {
        chart.applyOptions({ width: containerRef.current.clientWidth })
      }
    }
    window.addEventListener('resize', handleResize)

    return () => {
      window.removeEventListener('resize', handleResize)
      chart.remove()
      chartRef.current = null
    }
  }, [chartData, buyMarkers, sellMarkers, trades, toTimestamp])

  // Scroll to trade when clicked in table
  useEffect(() => {
    if (onTradeClick && chartRef.current) {
      // exposed via parent
    }
  }, [onTradeClick])

  return (
    <div className="bg-[#1a1d29] rounded-lg border border-[#2d3040] overflow-hidden">
      <div className="px-3 py-2 border-b border-[#2d3040] flex items-center justify-between">
        <span className="text-xs font-medium text-gray-300">Price Chart</span>
        <div className="flex gap-3 text-[10px]">
          <span className="flex items-center gap-1">
            <span className="w-3 h-0.5 bg-teal-400 inline-block"></span>
            Upper Trendline
          </span>
          <span className="flex items-center gap-1">
            <span className="w-3 h-0.5 bg-red-400 inline-block"></span>
            Lower Trendline
          </span>
        </div>
      </div>
      <div ref={containerRef} />
    </div>
  )
}

export function scrollChartToTrade(chartRef, trade, toTimestamp) {
  if (chartRef.current && trade) {
    const ts = toTimestamp(trade.entry_time)
    chartRef.current.timeScale().scrollToPosition(-10, false)
  }
}
