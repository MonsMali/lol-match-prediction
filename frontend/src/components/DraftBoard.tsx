import { useState, useEffect, useCallback } from 'react'
import { useDraftStore, countFilledSlots } from '../store/draftStore'
import { ModeToggle } from './ModeToggle'
import { SeriesTracker } from './SeriesTracker'
import { TeamPanel } from './TeamPanel'
import { RoleAssignment } from './RoleAssignment'
import { WinProbability } from './WinProbability'
import { DraftControls } from './DraftControls'
import { ChampionGrid } from './ChampionGrid'
import { InsightPanel } from './InsightPanel'
import { SuggestionPanel } from './SuggestionPanel'
import { ModelBanner } from './ModelBanner'
import type { InsightFactor, ChampionSuggestion, ModelMeta } from '../types'

interface PredictionEntry {
  blue: number
  red: number
  blueTeam: string
  redTeam: string
  timestamp: number
}

export interface FullPrediction {
  blue: number
  red: number
  blueInsights: InsightFactor[]
  redInsights: InsightFactor[]
  blueSuggestions: ChampionSuggestion[]
  redSuggestions: ChampionSuggestion[]
  model: ModelMeta
}

export function DraftBoard() {
  const [prediction, setPrediction] = useState<FullPrediction | null>(null)
  const [isPredicting, setIsPredicting] = useState(false)
  const [history, setHistory] = useState<PredictionEntry[]>([])
  const [showHistory, setShowHistory] = useState(false)

  const mode = useDraftStore((s) => s.mode)
  const currentDraftStep = useDraftStore((s) => s.currentDraftStep)
  const currentPhaseLabel = useDraftStore((s) => s.currentPhaseLabel)
  const undoLastStep = useDraftStore((s) => s.undoLastStep)
  const currentStep = useDraftStore((s) => s.currentStep)
  const blueBans = useDraftStore((s) => s.blueBans)
  const redBans = useDraftStore((s) => s.redBans)
  const bluePicks = useDraftStore((s) => s.bluePicks)
  const redPicks = useDraftStore((s) => s.redPicks)
  const blueTeam = useDraftStore((s) => s.blueTeam)
  const redTeam = useDraftStore((s) => s.redTeam)
  const selectChampion = useDraftStore((s) => s.selectChampion)

  const step = currentDraftStep()
  const phaseLabel = currentPhaseLabel()
  const isDraftComplete = phaseLabel === 'Draft Complete'

  const filledCount = countFilledSlots({ blueBans, redBans, bluePicks, redPicks })

  const handlePrediction = useCallback((data: FullPrediction | null) => {
    setPrediction(data)
    setIsPredicting(false)
    if (data && (data.blue > 0 || data.red > 0)) {
      setHistory((prev) => [
        {
          blue: data.blue,
          red: data.red,
          blueTeam: blueTeam ?? 'Blue',
          redTeam: redTeam ?? 'Red',
          timestamp: Date.now(),
        },
        ...prev.slice(0, 9),
      ])
    }
  }, [blueTeam, redTeam])

  const handleSuggestions = useCallback((blue: ChampionSuggestion[], red: ChampionSuggestion[]) => {
    setPrediction((prev) => prev ? { ...prev, blueSuggestions: blue, redSuggestions: red } : prev)
  }, [])

  // Keyboard shortcuts
  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if ((e.ctrlKey || e.metaKey) && e.key === 'z' && !e.shiftKey) {
        const target = e.target as HTMLElement
        if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA') return
        e.preventDefault()
        undoLastStep()
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [undoLastStep])

  return (
    <div className="flex flex-col gap-3">
      {/* Top bar: mode toggle + series */}
      <div className="flex items-center justify-center gap-4 flex-wrap">
        <ModeToggle />
        <SeriesTracker />
      </div>

      {/* Phase indicator */}
      {mode === 'live' && (
        <div className="flex items-center justify-center gap-3">
          <div className="text-center">
            <span className="text-sm font-semibold text-gold-light tracking-wide uppercase">
              {phaseLabel}
            </span>
            {step && (
              <span className="text-sm ml-2">
                {'-- '}
                <span className={step.team === 'blue' ? 'text-blue-team font-semibold' : 'text-red-team font-semibold'}>
                  {step.team === 'blue' ? 'Blue' : 'Red'} Team
                </span>
                {step.action === 'ban' ? ' Banning' : ' Picking'}
              </span>
            )}
            {isDraftComplete && (
              <span className="text-sm text-text-secondary ml-2">-- Assign Roles &amp; Edit Slots</span>
            )}
          </div>
          {currentStep > 0 && (
            <button
              type="button"
              onClick={undoLastStep}
              className="text-xs px-2 py-1 rounded bg-panel-light text-text-secondary hover:text-gold transition-colors"
              title="Undo (Ctrl+Z)"
            >
              Undo
            </button>
          )}
        </div>
      )}

      {mode === 'bulk' && (
        <div className="text-center text-sm text-text-secondary">
          <span className="font-semibold text-gold-light">Quick Entry</span>
          {' -- Click a slot then select a champion'}
          <span className="ml-2 text-text-primary font-mono">{filledCount}/20</span>
          <span className="text-text-secondary"> selections made</span>
        </div>
      )}

      {/* Desktop: 3-column compact layout */}
      <div className="hidden md:grid grid-cols-[minmax(200px,280px)_1fr_minmax(200px,280px)] gap-3 items-start">
        {/* Blue side */}
        <div className="flex flex-col gap-2">
          <TeamPanel side="blue" />
          <RoleAssignment side="blue" />
        </div>

        {/* Center: champion grid + prediction + controls */}
        <div className="flex flex-col gap-3">
          <ModelBanner meta={prediction?.model ?? null} />
          <WinProbability
            blue={prediction?.blue ?? null}
            red={prediction?.red ?? null}
            isPending={isPredicting}
            blueTeam={blueTeam}
            redTeam={redTeam}
          />
          {prediction && (
            <>
              <InsightPanel
                blueInsights={prediction.blueInsights}
                redInsights={prediction.redInsights}
                blueTeam={blueTeam ?? 'Blue'}
                redTeam={redTeam ?? 'Red'}
              />
              <SuggestionPanel
                blueSuggestions={prediction.blueSuggestions}
                redSuggestions={prediction.redSuggestions}
                blueTeam={blueTeam ?? 'Blue'}
                redTeam={redTeam ?? 'Red'}
              />
            </>
          )}
          <DraftControls onPrediction={handlePrediction} onSuggestions={handleSuggestions} />
          <ChampionGrid onSelect={selectChampion} />
        </div>

        {/* Red side */}
        <div className="flex flex-col gap-2">
          <TeamPanel side="red" />
          <RoleAssignment side="red" />
        </div>
      </div>

      {/* Mobile: stacked layout */}
      <div className="flex flex-col gap-3 md:hidden">
        <TeamPanel side="blue" />
        <TeamPanel side="red" />

        <ChampionGrid onSelect={selectChampion} />

        <RoleAssignment side="blue" />
        <RoleAssignment side="red" />

        <ModelBanner meta={prediction?.model ?? null} />
        <WinProbability
          blue={prediction?.blue ?? null}
          red={prediction?.red ?? null}
          isPending={isPredicting}
          blueTeam={blueTeam}
          redTeam={redTeam}
        />
        {prediction && (
          <>
            <InsightPanel
              blueInsights={prediction.blueInsights}
              redInsights={prediction.redInsights}
              blueTeam={blueTeam ?? 'Blue'}
              redTeam={redTeam ?? 'Red'}
            />
            <SuggestionPanel
              blueSuggestions={prediction.blueSuggestions}
              redSuggestions={prediction.redSuggestions}
              blueTeam={blueTeam ?? 'Blue'}
              redTeam={redTeam ?? 'Red'}
            />
          </>
        )}
        <DraftControls onPrediction={handlePrediction} />
      </div>

      {/* Prediction history */}
      {history.length > 0 && (
        <div className="mt-1">
          <button
            type="button"
            onClick={() => setShowHistory(!showHistory)}
            className="text-xs text-text-secondary hover:text-gold transition-colors mx-auto block"
          >
            {showHistory ? 'Hide' : 'Show'} Prediction History ({history.length})
          </button>
          {showHistory && (
            <div className="mt-2 bg-panel rounded p-3 max-h-[200px] overflow-y-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-text-secondary border-b border-panel-light">
                    <th className="text-left py-1 font-medium">Match</th>
                    <th className="text-center py-1 font-medium">Blue %</th>
                    <th className="text-center py-1 font-medium">Red %</th>
                    <th className="text-right py-1 font-medium">Time</th>
                  </tr>
                </thead>
                <tbody>
                  {history.map((entry, i) => (
                    <tr key={entry.timestamp} className={i === 0 ? 'text-text-primary' : 'text-text-secondary'}>
                      <td className="py-1">
                        <span className="text-blue-team">{entry.blueTeam}</span>
                        {' vs '}
                        <span className="text-red-team">{entry.redTeam}</span>
                      </td>
                      <td className="text-center py-1 text-blue-team font-mono">{Math.round(entry.blue * 100)}%</td>
                      <td className="text-center py-1 text-red-team font-mono">{Math.round(entry.red * 100)}%</td>
                      <td className="text-right py-1 text-text-secondary">
                        {new Date(entry.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
