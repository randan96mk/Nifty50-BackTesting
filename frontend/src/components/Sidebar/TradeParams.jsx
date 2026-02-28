export default function TradeParams({ params, onChange }) {
  const update = (key, value) => onChange({ ...params, [key]: value })

  return (
    <div className="mb-4">
      <label className="text-xs font-semibold uppercase tracking-wider text-gray-400 mb-2 block">
        Trade Parameters
      </label>
      <div className="space-y-2">
        <div>
          <label className="text-xs text-gray-400 block mb-1">Target (pts)</label>
          <input
            type="number"
            min={1}
            step={5}
            value={params.target_points}
            onChange={(e) => update('target_points', parseFloat(e.target.value) || 50)}
          />
        </div>

        <div>
          <label className="text-xs text-gray-400 block mb-1">Stop Loss Mode</label>
          <select
            value={params.sl_mode}
            onChange={(e) => update('sl_mode', e.target.value)}
          >
            <option value="fixed">Fixed Points</option>
            <option value="break_candle">Break Candle</option>
          </select>
        </div>

        {params.sl_mode === 'fixed' && (
          <div>
            <label className="text-xs text-gray-400 block mb-1">Stop Loss (pts)</label>
            <input
              type="number"
              min={1}
              step={5}
              value={params.stop_loss_points}
              onChange={(e) => update('stop_loss_points', parseFloat(e.target.value) || 25)}
            />
          </div>
        )}

        {params.sl_mode === 'break_candle' && (
          <div className="text-[10px] text-gray-500 bg-[#242736] rounded p-2 border border-[#2d3040]">
            Buy SL = previous candle low<br />
            Sell SL = previous candle high
          </div>
        )}

        <div className="flex items-center gap-2">
          <input
            type="checkbox"
            id="trailing-stop"
            checked={params.trailing_stop}
            onChange={(e) => update('trailing_stop', e.target.checked)}
            className="accent-teal-400"
          />
          <label htmlFor="trailing-stop" className="text-xs text-gray-400">
            Trailing Stop Loss
          </label>
        </div>

        {params.trailing_stop && (
          <div className="grid grid-cols-2 gap-2">
            <div>
              <label className="text-xs text-gray-400 block mb-1">Activation (pts)</label>
              <input
                type="number"
                min={1}
                value={params.trail_activation}
                onChange={(e) => update('trail_activation', parseFloat(e.target.value) || 30)}
              />
            </div>
            <div>
              <label className="text-xs text-gray-400 block mb-1">Distance (pts)</label>
              <input
                type="number"
                min={1}
                value={params.trail_distance}
                onChange={(e) => update('trail_distance', parseFloat(e.target.value) || 15)}
              />
            </div>
          </div>
        )}

        <div className="grid grid-cols-2 gap-2">
          <div>
            <label className="text-xs text-gray-400 block mb-1">Entry Start</label>
            <input
              type="time"
              value={params.entry_start_time}
              onChange={(e) => update('entry_start_time', e.target.value)}
            />
          </div>
          <div>
            <label className="text-xs text-gray-400 block mb-1">Entry End</label>
            <input
              type="time"
              value={params.entry_end_time}
              onChange={(e) => update('entry_end_time', e.target.value)}
            />
          </div>
        </div>
        <div>
          <label className="text-xs text-gray-400 block mb-1">Force Exit Time</label>
          <input
            type="time"
            value={params.exit_time}
            onChange={(e) => update('exit_time', e.target.value)}
          />
        </div>
      </div>
    </div>
  )
}
