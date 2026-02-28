import MetricCard from './MetricCard'

export default function PerformanceDashboard({ metrics }) {
  if (!metrics) return null

  const cards = [
    { label: 'Total Trades', value: metrics.total_trades, subtitle: `${metrics.longs_count}L / ${metrics.shorts_count}S` },
    { label: 'Win Rate', value: `${metrics.win_rate}%`, color: metrics.win_rate >= 50 ? 'text-teal-400' : 'text-red-400', subtitle: `${metrics.wins}W / ${metrics.losses}L` },
    { label: 'Net P&L', value: `${metrics.net_pnl > 0 ? '+' : ''}${metrics.net_pnl} pts`, color: metrics.net_pnl >= 0 ? 'text-teal-400' : 'text-red-400' },
    { label: 'Avg Win', value: `+${metrics.avg_win} pts`, color: 'text-teal-400' },
    { label: 'Avg Loss', value: `${metrics.avg_loss} pts`, color: 'text-red-400' },
    { label: 'Profit Factor', value: metrics.profit_factor, color: metrics.profit_factor >= 1 ? 'text-teal-400' : 'text-red-400' },
    { label: 'Max Drawdown', value: `${metrics.max_drawdown} pts`, color: 'text-red-400' },
    { label: 'Sharpe Ratio', value: metrics.sharpe_ratio, color: metrics.sharpe_ratio >= 1 ? 'text-teal-400' : 'text-gray-200' },
    { label: 'Max Consec Wins', value: metrics.max_consecutive_wins, color: 'text-teal-400' },
    { label: 'Max Consec Losses', value: metrics.max_consecutive_losses, color: 'text-red-400' },
    { label: 'Avg Holding', value: `${metrics.avg_holding_bars} bars` },
    { label: 'Expectancy', value: `${metrics.expectancy > 0 ? '+' : ''}${metrics.expectancy} pts`, color: metrics.expectancy >= 0 ? 'text-teal-400' : 'text-red-400' },
    { label: 'Payoff Ratio', value: metrics.payoff_ratio },
    { label: 'Recovery Factor', value: metrics.recovery_factor },
    { label: 'Long Win Rate', value: `${metrics.long_win_rate}%`, subtitle: `${metrics.longs_count} trades` },
    { label: 'Short Win Rate', value: `${metrics.short_win_rate}%`, subtitle: `${metrics.shorts_count} trades` },
  ]

  return (
    <div className="bg-[#1a1d29] rounded-lg border border-[#2d3040] overflow-hidden">
      <div className="px-3 py-2 border-b border-[#2d3040]">
        <span className="text-xs font-medium text-gray-300">Performance Metrics</span>
      </div>
      <div className="p-3 grid grid-cols-4 gap-2">
        {cards.map((c) => (
          <MetricCard key={c.label} {...c} />
        ))}
      </div>
    </div>
  )
}
