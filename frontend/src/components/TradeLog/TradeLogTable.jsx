import { useState, useMemo } from 'react'

export default function TradeLogTable({ trades, onTradeSelect }) {
  const [sortKey, setSortKey] = useState('trade_id')
  const [sortDir, setSortDir] = useState('asc')
  const [filterDir, setFilterDir] = useState('ALL')

  const filteredTrades = useMemo(() => {
    let list = trades || []
    if (filterDir !== 'ALL') {
      list = list.filter((t) => t.direction === filterDir)
    }
    return [...list].sort((a, b) => {
      const aVal = a[sortKey]
      const bVal = b[sortKey]
      const cmp = typeof aVal === 'string' ? aVal.localeCompare(bVal) : aVal - bVal
      return sortDir === 'asc' ? cmp : -cmp
    })
  }, [trades, sortKey, sortDir, filterDir])

  const handleSort = (key) => {
    if (sortKey === key) {
      setSortDir(sortDir === 'asc' ? 'desc' : 'asc')
    } else {
      setSortKey(key)
      setSortDir('asc')
    }
  }

  const exportCsv = () => {
    if (!trades || trades.length === 0) return
    const headers = ['#', 'Direction', 'Entry Time', 'Entry Price', 'Exit Time', 'Exit Price', 'Reason', 'P&L', 'Bars Held']
    const rows = trades.map((t) => [
      t.trade_id, t.direction, t.entry_time, t.entry_price,
      t.exit_time, t.exit_price, t.exit_reason, t.pnl_points, t.holding_bars,
    ])
    const csv = [headers, ...rows].map((r) => r.join(',')).join('\n')
    const blob = new Blob([csv], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'trade_log.csv'
    a.click()
    URL.revokeObjectURL(url)
  }

  const columns = [
    { key: 'trade_id', label: '#', width: 'w-10' },
    { key: 'direction', label: 'Dir', width: 'w-14' },
    { key: 'entry_time', label: 'Entry Time', width: 'w-36' },
    { key: 'entry_price', label: 'Entry', width: 'w-20' },
    { key: 'exit_time', label: 'Exit Time', width: 'w-36' },
    { key: 'exit_price', label: 'Exit', width: 'w-20' },
    { key: 'exit_reason', label: 'Reason', width: 'w-24' },
    { key: 'pnl_points', label: 'P&L', width: 'w-20' },
    { key: 'holding_bars', label: 'Bars', width: 'w-14' },
  ]

  const SortIcon = ({ col }) => {
    if (sortKey !== col) return <span className="text-gray-600 ml-0.5">&#8597;</span>
    return <span className="text-teal-400 ml-0.5">{sortDir === 'asc' ? '&#8593;' : '&#8595;'}</span>
  }

  return (
    <div className="bg-[#1a1d29] rounded-lg border border-[#2d3040] overflow-hidden">
      <div className="px-3 py-2 border-b border-[#2d3040] flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className="text-xs font-medium text-gray-300">Trade Log</span>
          <select
            className="text-[10px] bg-[#242736] border border-[#2d3040] rounded px-1.5 py-0.5"
            value={filterDir}
            onChange={(e) => setFilterDir(e.target.value)}
          >
            <option value="ALL">All</option>
            <option value="LONG">Long</option>
            <option value="SHORT">Short</option>
          </select>
          <span className="text-[10px] text-gray-500">
            {filteredTrades.length} trade{filteredTrades.length !== 1 ? 's' : ''}
          </span>
        </div>
        <button
          onClick={exportCsv}
          className="text-[10px] px-2 py-1 bg-[#242736] border border-[#2d3040] rounded hover:border-teal-400/50 text-gray-400 hover:text-gray-200 transition-colors"
        >
          Export CSV
        </button>
      </div>
      <div className="overflow-auto max-h-72">
        <table className="w-full text-xs">
          <thead className="sticky top-0 bg-[#242736]">
            <tr>
              {columns.map((col) => (
                <th
                  key={col.key}
                  className={`text-left px-2 py-1.5 text-gray-400 font-medium cursor-pointer select-none hover:text-gray-200 ${col.width}`}
                  onClick={() => handleSort(col.key)}
                >
                  {col.label}
                  <SortIcon col={col.key} />
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {filteredTrades.map((t) => (
              <tr
                key={t.trade_id}
                className={`border-t border-[#2d3040]/50 cursor-pointer hover:bg-[#242736]/80 transition-colors ${
                  t.pnl_points > 0 ? 'bg-teal-400/[0.03]' : t.pnl_points < 0 ? 'bg-red-400/[0.03]' : ''
                }`}
                onClick={() => onTradeSelect?.(t)}
              >
                <td className="px-2 py-1.5 text-gray-500">{t.trade_id}</td>
                <td className={`px-2 py-1.5 font-medium ${t.direction === 'LONG' ? 'text-teal-400' : 'text-red-400'}`}>
                  {t.direction}
                </td>
                <td className="px-2 py-1.5 text-gray-300">{formatTime(t.entry_time)}</td>
                <td className="px-2 py-1.5 text-gray-300">{t.entry_price}</td>
                <td className="px-2 py-1.5 text-gray-300">{formatTime(t.exit_time)}</td>
                <td className="px-2 py-1.5 text-gray-300">{t.exit_price}</td>
                <td className="px-2 py-1.5">
                  <span className={`px-1.5 py-0.5 rounded text-[10px] font-medium ${reasonColor(t.exit_reason)}`}>
                    {t.exit_reason}
                  </span>
                </td>
                <td className={`px-2 py-1.5 font-medium ${t.pnl_points > 0 ? 'text-teal-400' : t.pnl_points < 0 ? 'text-red-400' : 'text-gray-400'}`}>
                  {t.pnl_points > 0 ? '+' : ''}{t.pnl_points}
                </td>
                <td className="px-2 py-1.5 text-gray-400">{t.holding_bars}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {filteredTrades.length === 0 && (
          <div className="text-center py-8 text-gray-500 text-xs">No trades to display</div>
        )}
      </div>
    </div>
  )
}

function formatTime(dtStr) {
  if (!dtStr) return ''
  return dtStr.replace('T', ' ').slice(0, 19)
}

function reasonColor(reason) {
  switch (reason) {
    case 'TARGET': return 'bg-teal-400/20 text-teal-400'
    case 'STOPLOSS': return 'bg-red-400/20 text-red-400'
    case 'TRAILING_SL': return 'bg-orange-400/20 text-orange-400'
    case 'TIME_EXIT': return 'bg-blue-400/20 text-blue-400'
    case 'REVERSE_SIGNAL': return 'bg-purple-400/20 text-purple-400'
    default: return 'bg-gray-400/20 text-gray-400'
  }
}
