const TIMEFRAMES = [
  { value: '1m', label: '1 min' },
  { value: '2m', label: '2 min' },
  { value: '3m', label: '3 min' },
  { value: '5m', label: '5 min' },
  { value: '10m', label: '10 min' },
  { value: '15m', label: '15 min' },
  { value: '30m', label: '30 min' },
  { value: '1h', label: '1 hour' },
]

export default function TimeframeSelect({ value, onChange }) {
  return (
    <div className="mb-4">
      <label className="text-xs font-semibold uppercase tracking-wider text-gray-400 mb-2 block">
        Candle Timeframe
      </label>
      <div className="flex flex-wrap gap-1">
        {TIMEFRAMES.map((tf) => (
          <button
            key={tf.value}
            onClick={() => onChange(tf.value)}
            className={`text-[10px] py-1 px-2 rounded border transition-colors ${
              value === tf.value
                ? 'bg-teal-500/20 border-teal-400/50 text-teal-400'
                : 'bg-[#242736] border-[#2d3040] text-gray-400 hover:border-gray-500 hover:text-gray-300'
            }`}
          >
            {tf.label}
          </button>
        ))}
      </div>
    </div>
  )
}
