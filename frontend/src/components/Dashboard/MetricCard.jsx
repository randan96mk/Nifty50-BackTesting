export default function MetricCard({ label, value, color, subtitle }) {
  return (
    <div className="bg-[#242736] rounded-lg p-3 border border-[#2d3040]">
      <div className="text-[10px] text-gray-500 uppercase tracking-wider mb-1">{label}</div>
      <div className={`text-lg font-semibold ${color || 'text-gray-200'}`}>{value}</div>
      {subtitle && <div className="text-[10px] text-gray-500 mt-0.5">{subtitle}</div>}
    </div>
  )
}
