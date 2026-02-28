export default function AlgoParams({ params, onChange }) {
  const update = (key, value) => onChange({ ...params, [key]: value })

  return (
    <div className="mb-4">
      <label className="text-xs font-semibold uppercase tracking-wider text-gray-400 mb-2 block">
        Algorithm Parameters
      </label>
      <div className="space-y-2">
        <div>
          <label className="text-xs text-gray-400 block mb-1">Swing Lookback</label>
          <input
            type="number"
            min={1}
            max={50}
            value={params.length}
            onChange={(e) => update('length', parseInt(e.target.value) || 14)}
          />
        </div>
        <div>
          <label className="text-xs text-gray-400 block mb-1">Slope Multiplier</label>
          <input
            type="number"
            min={0.1}
            max={5.0}
            step={0.1}
            value={params.mult}
            onChange={(e) => update('mult', parseFloat(e.target.value) || 1.0)}
          />
        </div>
        <div>
          <label className="text-xs text-gray-400 block mb-1">Slope Method</label>
          <select
            value={params.calc_method}
            onChange={(e) => update('calc_method', e.target.value)}
          >
            <option value="Atr">ATR</option>
            <option value="Stdev">StdDev</option>
            <option value="Linreg">LinReg</option>
          </select>
        </div>
      </div>
    </div>
  )
}
