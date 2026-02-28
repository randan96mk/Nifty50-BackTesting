import { useEffect, useRef } from 'react'
import { createChart } from 'lightweight-charts'

export default function EquityCurveChart({ equityCurve }) {
  const containerRef = useRef(null)
  const chartRef = useRef(null)

  useEffect(() => {
    if (!containerRef.current || !equityCurve || equityCurve.length === 0) return

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
      rightPriceScale: {
        borderColor: '#2d3040',
      },
      timeScale: {
        borderColor: '#2d3040',
        timeVisible: true,
        secondsVisible: false,
      },
      width: containerRef.current.clientWidth,
      height: 200,
    })

    chartRef.current = chart

    // Area series for equity curve
    const areaSeries = chart.addAreaSeries({
      topColor: 'rgba(38, 166, 154, 0.3)',
      bottomColor: 'rgba(38, 166, 154, 0.02)',
      lineColor: '#26a69a',
      lineWidth: 2,
      crosshairMarkerVisible: true,
      crosshairMarkerRadius: 3,
    })

    const data = equityCurve.map((d) => ({
      time: Math.floor(new Date(d.datetime).getTime() / 1000),
      value: d.cumulative_pnl,
    }))

    areaSeries.setData(data)
    chart.timeScale().fitContent()

    // Draw zero line
    areaSeries.createPriceLine({
      price: 0,
      color: '#8b8fa3',
      lineWidth: 1,
      lineStyle: 2,
      axisLabelVisible: false,
    })

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
  }, [equityCurve])

  return (
    <div className="bg-[#1a1d29] rounded-lg border border-[#2d3040] overflow-hidden">
      <div className="px-3 py-2 border-b border-[#2d3040]">
        <span className="text-xs font-medium text-gray-300">Equity Curve (Cumulative P&L in Points)</span>
      </div>
      <div ref={containerRef} />
    </div>
  )
}
