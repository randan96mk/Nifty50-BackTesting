import { useEffect, useRef, useCallback } from 'react'
import {
  createChart,
  CrosshairMode,
  CandlestickSeries,
  LineSeries,
  createSeriesMarkers,
} from 'lightweight-charts'

export default function CandlestickChart({ chartData, buyMarkers, sellMarkers, trades }) {
  const containerRef = useRef(null)
  const chartRef = useRef(null)

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

    // Candlestick series (v5 API)
    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor: '#26a69a',
      downColor: '#ef5350',
      borderUpColor: '#26a69a',
      borderDownColor: '#ef5350',
      wickUpColor: '#26a69a',
      wickDownColor: '#ef5350',
    })

    // Deduplicate timestamps (keep last occurrence for each second)
    const seen = new Map()
    chartData.forEach((d) => {
      const ts = toTimestamp(d.time)
      seen.set(ts, d)
    })
    const uniqueData = Array.from(seen.entries())
      .sort((a, b) => a[0] - b[0])
      .map(([ts, d]) => ({
        time: ts,
        open: d.open,
        high: d.high,
        low: d.low,
        close: d.close,
      }))

    candleSeries.setData(uniqueData)

    // Upper trendline (v5 API)
    const upperEntries = chartData.filter((d) => d.upper !== undefined)
    if (upperEntries.length > 0) {
      const upperSeen = new Map()
      upperEntries.forEach((d) => {
        upperSeen.set(toTimestamp(d.time), d.upper)
      })
      const upperData = Array.from(upperSeen.entries())
        .sort((a, b) => a[0] - b[0])
        .map(([ts, val]) => ({ time: ts, value: val }))

      const upperLine = chart.addSeries(LineSeries, {
        color: '#26a69a',
        lineWidth: 1,
        lineStyle: 2,
        crosshairMarkerVisible: false,
        priceLineVisible: false,
        lastValueVisible: false,
      })
      upperLine.setData(upperData)
    }

    // Lower trendline (v5 API)
    const lowerEntries = chartData.filter((d) => d.lower !== undefined)
    if (lowerEntries.length > 0) {
      const lowerSeen = new Map()
      lowerEntries.forEach((d) => {
        lowerSeen.set(toTimestamp(d.time), d.lower)
      })
      const lowerData = Array.from(lowerSeen.entries())
        .sort((a, b) => a[0] - b[0])
        .map(([ts, val]) => ({ time: ts, value: val }))

      const lowerLine = chart.addSeries(LineSeries, {
        color: '#ef5350',
        lineWidth: 1,
        lineStyle: 2,
        crosshairMarkerVisible: false,
        priceLineVisible: false,
        lastValueVisible: false,
      })
      lowerLine.setData(lowerData)
    }

    // Build markers
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

    // Deduplicate markers at same timestamp — keep all but sort
    markers.sort((a, b) => a.time - b.time)

    // Use v5 createSeriesMarkers API
    if (markers.length > 0) {
      createSeriesMarkers(candleSeries, markers)
    }

    chart.timeScale().fitContent()

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
