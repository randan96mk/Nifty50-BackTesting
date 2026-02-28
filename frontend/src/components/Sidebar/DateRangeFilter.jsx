import { useMemo } from 'react'

const QUICK_PRESETS = [
  { label: '6M', months: 6 },
  { label: '1Y', months: 12 },
  { label: '2Y', months: 24 },
  { label: 'All', months: 0 },
]

export default function DateRangeFilter({ params, onChange, dataRange }) {
  const update = (key, value) => onChange({ ...params, [key]: value })

  const applyPreset = (months) => {
    if (months === 0) {
      onChange({ ...params, date_start: '', date_end: '' })
      return
    }
    const end = new Date()
    const start = new Date()
    start.setMonth(start.getMonth() - months)
    onChange({
      ...params,
      date_start: formatDate(start),
      date_end: formatDate(end),
    })
  }

  return (
    <div className="mb-4">
      <label className="text-xs font-semibold uppercase tracking-wider text-gray-400 mb-2 block">
        Date Range
      </label>
      <div className="space-y-2">
        <div className="flex gap-1">
          {QUICK_PRESETS.map((p) => (
            <button
              key={p.label}
              onClick={() => applyPreset(p.months)}
              className="flex-1 text-[10px] py-1 px-1 rounded bg-[#242736] border border-[#2d3040] text-gray-400 hover:border-teal-400/50 hover:text-gray-200 transition-colors"
            >
              {p.label}
            </button>
          ))}
        </div>
        <div className="grid grid-cols-2 gap-2">
          <div>
            <label className="text-xs text-gray-400 block mb-1">From</label>
            <input
              type="date"
              value={params.date_start}
              onChange={(e) => update('date_start', e.target.value)}
            />
          </div>
          <div>
            <label className="text-xs text-gray-400 block mb-1">To</label>
            <input
              type="date"
              value={params.date_end}
              onChange={(e) => update('date_end', e.target.value)}
            />
          </div>
        </div>
        {dataRange && (
          <div className="text-[10px] text-gray-500">
            Data: {dataRange}
          </div>
        )}
      </div>
    </div>
  )
}

function formatDate(d) {
  return d.toISOString().split('T')[0]
}
