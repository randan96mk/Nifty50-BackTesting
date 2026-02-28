import { useCallback, useState } from 'react'

export default function FileUpload({ onDataLoaded }) {
  const [dragging, setDragging] = useState(false)
  const [loading, setLoading] = useState(false)
  const [fileInfo, setFileInfo] = useState(null)
  const [error, setError] = useState(null)

  const handleFile = useCallback(async (file) => {
    if (!file) return
    setLoading(true)
    setError(null)

    const formData = new FormData()
    formData.append('file', file)

    try {
      const res = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Upload failed')

      setFileInfo({
        name: file.name,
        rows: data.total_rows,
        range: data.date_range,
      })
      onDataLoaded(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }, [onDataLoaded])

  const onDrop = useCallback((e) => {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer.files[0]
    handleFile(file)
  }, [handleFile])

  const onDragOver = useCallback((e) => {
    e.preventDefault()
    setDragging(true)
  }, [])

  const onDragLeave = useCallback(() => setDragging(false), [])

  return (
    <div className="mb-4">
      <label className="text-xs font-semibold uppercase tracking-wider text-gray-400 mb-2 block">
        Data File
      </label>
      <div
        onDrop={onDrop}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        className={`border-2 border-dashed rounded-lg p-4 text-center cursor-pointer transition-colors ${
          dragging ? 'border-teal-400 bg-teal-400/10' : 'border-gray-600 hover:border-gray-400'
        }`}
        onClick={() => document.getElementById('file-input').click()}
      >
        <input
          id="file-input"
          type="file"
          accept=".xlsx,.xls,.csv"
          className="hidden"
          onChange={(e) => handleFile(e.target.files[0])}
        />
        {loading ? (
          <div className="text-teal-400 text-sm">Processing...</div>
        ) : fileInfo ? (
          <div className="text-xs">
            <div className="text-teal-400 font-medium truncate">{fileInfo.name}</div>
            <div className="text-gray-400 mt-1">{fileInfo.rows.toLocaleString()} candles</div>
            <div className="text-gray-500 mt-0.5 text-[11px]">{fileInfo.range}</div>
          </div>
        ) : (
          <div className="text-gray-400 text-xs">
            Drop .xlsx / .csv here<br />or click to browse
          </div>
        )}
      </div>
      {error && (
        <div className="mt-2 text-xs text-red-400 bg-red-400/10 p-2 rounded">{error}</div>
      )}
    </div>
  )
}
