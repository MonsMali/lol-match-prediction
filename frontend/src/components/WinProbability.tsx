interface WinProbabilityProps {
  blue: number | null
  red: number | null
  isPending: boolean
  blueTeam?: string | null
  redTeam?: string | null
}

export function WinProbability({ blue, red, isPending, blueTeam, redTeam }: WinProbabilityProps) {
  if (isPending) {
    return (
      <div className="w-full flex flex-col items-center gap-2">
        <div className="h-14 w-full rounded-full bg-panel-light flex items-center justify-center">
          <div className="w-6 h-6 border-2 border-gold border-t-transparent rounded-full animate-spin" />
        </div>
      </div>
    )
  }

  if (blue === null || red === null) {
    return (
      <div className="w-full flex flex-col items-center gap-2">
        <div className="h-14 w-full rounded-full bg-panel-light flex items-center justify-center">
          <span className="text-text-secondary text-sm">Complete draft to predict</span>
        </div>
      </div>
    )
  }

  const bluePercent = Math.round(blue * 100)
  const redPercent = Math.round(red * 100)
  const blueLabel = blueTeam ?? 'Blue'
  const redLabel = redTeam ?? 'Red'

  return (
    <div className="w-full flex flex-col gap-1.5">
      <div className="flex justify-between text-xs font-semibold px-1">
        <span className="text-blue-team">{blueLabel}</span>
        <span className="text-red-team">{redLabel}</span>
      </div>
      <div className="h-14 w-full rounded-full overflow-hidden flex shadow-lg">
        <div
          className="bg-blue-team flex items-center justify-center transition-all duration-700 ease-out"
          style={{ width: `${bluePercent}%` }}
        >
          <span className="text-white text-lg font-bold drop-shadow">{bluePercent}%</span>
        </div>
        <div
          className="bg-red-team flex items-center justify-center transition-all duration-700 ease-out"
          style={{ width: `${redPercent}%` }}
        >
          <span className="text-white text-lg font-bold drop-shadow">{redPercent}%</span>
        </div>
      </div>
    </div>
  )
}
