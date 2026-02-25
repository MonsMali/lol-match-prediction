import { useDraftStore } from '../store/draftStore'
import type { SeriesFormat } from '../types'

const FORMAT_OPTIONS: { value: SeriesFormat; label: string }[] = [
  { value: 'single', label: 'Single' },
  { value: 'bo3', label: 'BO3' },
  { value: 'bo5', label: 'BO5' },
]

function winThreshold(format: SeriesFormat): number {
  if (format === 'bo3') return 2
  if (format === 'bo5') return 3
  return 1
}

export function SeriesTracker() {
  const seriesFormat = useDraftStore((s) => s.seriesFormat)
  const seriesScore = useDraftStore((s) => s.seriesScore)
  const currentGame = useDraftStore((s) => s.currentGame)
  const setSeriesFormat = useDraftStore((s) => s.setSeriesFormat)
  const blueTeam = useDraftStore((s) => s.blueTeam)
  const redTeam = useDraftStore((s) => s.redTeam)
  const isSeriesComplete = useDraftStore((s) => s.isSeriesComplete)

  const threshold = winThreshold(seriesFormat)
  const totalGames = seriesFormat === 'bo3' ? 3 : seriesFormat === 'bo5' ? 5 : 1
  const complete = isSeriesComplete()

  const winner =
    complete && seriesScore.blue >= threshold
      ? 'blue'
      : complete && seriesScore.red >= threshold
        ? 'red'
        : null

  return (
    <div className="flex items-center gap-3">
      {/* Format selector -- subtle segmented control */}
      <div className="flex rounded bg-panel-light text-xs overflow-hidden">
        {FORMAT_OPTIONS.map((opt) => (
          <button
            key={opt.value}
            type="button"
            onClick={() => setSeriesFormat(opt.value)}
            className={`px-2.5 py-1 transition-colors ${
              seriesFormat === opt.value
                ? 'bg-gold text-background font-semibold'
                : 'text-text-secondary hover:text-text-primary'
            }`}
          >
            {opt.label}
          </button>
        ))}
      </div>

      {/* Series info -- only when BO3/BO5 active */}
      {seriesFormat !== 'single' && (
        <div className="flex items-center gap-3 text-sm">
          {/* Score */}
          <div className="flex items-center gap-1.5 font-mono font-semibold">
            <span className="text-blue-team">{seriesScore.blue}</span>
            <span className="text-text-secondary">-</span>
            <span className="text-red-team">{seriesScore.red}</span>
          </div>

          {/* Game indicators */}
          <div className="flex items-center gap-1">
            {Array.from({ length: totalGames }, (_, i) => {
              const gameNum = i + 1
              const isBlueWin = gameNum <= seriesScore.blue
              const isRedWin =
                gameNum > seriesScore.blue &&
                gameNum <= seriesScore.blue + seriesScore.red
              const isCurrent = gameNum === currentGame && !complete

              return (
                <div
                  key={i}
                  className={`w-2.5 h-2.5 rounded-full border transition-colors ${
                    isBlueWin
                      ? 'bg-blue-team border-blue-team'
                      : isRedWin
                        ? 'bg-red-team border-red-team'
                        : isCurrent
                          ? 'border-gold bg-gold/30'
                          : 'border-text-secondary/40 bg-transparent'
                  }`}
                  title={
                    isBlueWin
                      ? `Game ${gameNum}: ${blueTeam ?? 'Blue'} win`
                      : isRedWin
                        ? `Game ${gameNum}: ${redTeam ?? 'Red'} win`
                        : isCurrent
                          ? `Game ${gameNum} (current)`
                          : `Game ${gameNum}`
                  }
                />
              )
            })}
          </div>

          {/* Current game or series winner */}
          {winner ? (
            <span className="text-gold-light font-semibold text-xs">
              Series Winner: {winner === 'blue' ? (blueTeam ?? 'Blue') : (redTeam ?? 'Red')}
            </span>
          ) : (
            <span className="text-text-secondary text-xs">
              Game {currentGame} of {totalGames}
            </span>
          )}
        </div>
      )}
    </div>
  )
}
