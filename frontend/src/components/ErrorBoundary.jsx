import { Component } from 'react'

export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props)
    this.state = { hasError: false, error: null }
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error }
  }

  componentDidCatch(error, errorInfo) {
    console.error('Chart error:', error, errorInfo)
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="bg-[#1a1d29] rounded-lg border border-red-500/30 p-4 text-center">
          <div className="text-red-400 text-sm font-medium mb-1">Chart rendering error</div>
          <div className="text-xs text-gray-500 mb-2">{this.state.error?.message}</div>
          <button
            onClick={() => this.setState({ hasError: false, error: null })}
            className="text-xs px-3 py-1 bg-[#242736] border border-[#2d3040] rounded hover:border-teal-400/50 text-gray-400"
          >
            Retry
          </button>
        </div>
      )
    }
    return this.props.children
  }
}
