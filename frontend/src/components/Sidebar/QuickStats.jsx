export default function QuickStats({ metrics }) {
  if (!metrics) return null

  const stats = [
    { label: 'Total Trades', value: metrics.total_trades },
    { label: 'Win Rate', value: `${metrics.win_rate}%`, color: metrics.win_rate >= 50 ? 'text-teal-400' : 'text-red-400' },
    { label: 'Net P&L', value: `${metrics.net_pnl > 0 ? '+' : ''}${metrics.net_pnl} pts`, color: metrics.net_pnl >= 0 ? 'text-teal-400' : 'text-red-400' },
    { label: 'Profit Factor', value: metrics.profit_factor },
    { label: 'Max DD', value: `${metrics.max_drawdown} pts`, color: 'text-red-400' },
  ]

  return (
    <div className="mb-4">
      <label className="text-xs font-semibold uppercase tracking-wider text-gray-400 mb-2 block">
        Quick Stats
      </label>
      <div className="space-y-1.5">
        {stats.map((s) => (
          <div key={s.label} className="flex justify-between items-center text-xs">
            <span className="text-gray-400">{s.label}</span>
            <span className={`font-medium ${s.color || 'text-gray-200'}`}>{s.value}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
