interface WinProbabilityProps {
  blue: number | null
  red: number | null
  isPending: boolean
}

export function WinProbability({ blue, red, isPending }: WinProbabilityProps) {
  if (isPending) {
    return (
      <div className="h-10 w-full rounded-full bg-panel-light flex items-center justify-center">
        <div className="w-6 h-6 border-2 border-gold border-t-transparent rounded-full animate-spin" />
      </div>
    )
  }

  if (blue === null || red === null) {
    return (
      <div className="h-10 w-full rounded-full bg-panel-light flex items-center justify-center">
        <span className="text-text-secondary text-sm">Complete draft to predict</span>
      </div>
    )
  }

  const bluePercent = Math.round(blue * 100)
  const redPercent = Math.round(red * 100)

  return (
    <div className="h-10 w-full rounded-full overflow-hidden flex">
      <div
        className="bg-blue-team flex items-center justify-center transition-all duration-500"
        style={{ width: `${bluePercent}%` }}
      >
        <span className="text-white text-sm font-bold">{bluePercent}%</span>
      </div>
      <div
        className="bg-red-team flex items-center justify-center transition-all duration-500"
        style={{ width: `${redPercent}%` }}
      >
        <span className="text-white text-sm font-bold">{redPercent}%</span>
      </div>
    </div>
  )
}
