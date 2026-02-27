import { useEffect, useCallback } from 'react'
import { useDraftStore, buildPredictRequest } from '../store/draftStore'
import { usePrediction, useSuggestions } from '../api/predict'
import type { InsightFactor, PickImpact, ChampionSuggestion, ModelMeta } from '../types'

export interface FullPrediction {
  blue: number
  red: number
  blueInsights: InsightFactor[]
  redInsights: InsightFactor[]
  bluePickImpacts: PickImpact[]
  redPickImpacts: PickImpact[]
  blueSuggestions: ChampionSuggestion[]
  redSuggestions: ChampionSuggestion[]
  model: ModelMeta
}

interface DraftControlsProps {
  onPrediction: (data: FullPrediction | null) => void
  onSuggestions: (blue: ChampionSuggestion[], red: ChampionSuggestion[]) => void
}

export function DraftControls({ onPrediction, onSuggestions }: DraftControlsProps) {
  const resetDraft = useDraftStore((s) => s.resetDraft)
  const resetAll = useDraftStore((s) => s.resetAll)
  const seriesFormat = useDraftStore((s) => s.seriesFormat)
  const isSeriesComplete = useDraftStore((s) => s.isSeriesComplete)
  const recordGameResult = useDraftStore((s) => s.recordGameResult)
  const blueTeam = useDraftStore((s) => s.blueTeam)
  const redTeam = useDraftStore((s) => s.redTeam)
  const blueBans = useDraftStore((s) => s.blueBans)
  const redBans = useDraftStore((s) => s.redBans)
  const bluePicks = useDraftStore((s) => s.bluePicks)
  const redPicks = useDraftStore((s) => s.redPicks)
  const blueRoles = useDraftStore((s) => s.blueRoles)
  const redRoles = useDraftStore((s) => s.redRoles)
  const prediction = usePrediction()
  const suggestions = useSuggestions()

  const allSlotsFilled = [...blueBans, ...redBans, ...bluePicks, ...redPicks].every((s) => s !== null)
  const teamsSelected = blueTeam !== null && redTeam !== null
  const allRolesAssigned =
    Object.values(blueRoles).every((v) => v !== null) &&
    Object.values(redRoles).every((v) => v !== null)
  const ready = allSlotsFilled && teamsSelected && allRolesAssigned
  const seriesComplete = isSeriesComplete()
  const isSeries = seriesFormat !== 'single'
  const hasPrediction = prediction.isSuccess

  const handlePredict = useCallback(() => {
    if (!ready || prediction.isPending) return
    const request = buildPredictRequest()

    // Fast path: prediction + insights (~15ms)
    prediction.mutate(request, {
      onSuccess: (data) => {
        onPrediction({
          blue: data.blue_win_probability,
          red: data.red_win_probability,
          blueInsights: data.blue_insights ?? [],
          redInsights: data.red_insights ?? [],
          bluePickImpacts: data.blue_pick_impacts ?? [],
          redPickImpacts: data.red_pick_impacts ?? [],
          blueSuggestions: [],
          redSuggestions: [],
          model: data.model,
        })
      },
    })

    // Async path: suggestions (~100ms), streams in separately
    suggestions.mutate(request, {
      onSuccess: (data) => {
        onSuggestions(
          data.blue_suggestions ?? [],
          data.red_suggestions ?? [],
        )
      },
    })
  }, [ready, prediction, suggestions, onPrediction, onSuggestions])

  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if (e.key === 'Enter' && !e.ctrlKey && !e.metaKey) {
        const target = e.target as HTMLElement
        if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.tagName === 'SELECT') return
        handlePredict()
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [handlePredict])

  function handleRecordResult(winner: 'blue' | 'red') {
    recordGameResult(winner)
    prediction.reset()
    suggestions.reset()
    onPrediction(null)
  }

  function handleResetDraft() {
    resetDraft()
    prediction.reset()
    suggestions.reset()
  }

  function handleResetAll() {
    resetAll()
    prediction.reset()
    suggestions.reset()
    onPrediction(null)
  }

  return (
    <div className="flex flex-col gap-2 items-center">
      <div className="flex gap-2 justify-center flex-wrap">
        <button
          type="button"
          onClick={handlePredict}
          disabled={!ready || prediction.isPending}
          className={`
            px-6 py-2 rounded font-semibold text-sm transition-colors
            ${ready
              ? 'bg-gold text-background hover:bg-gold-light cursor-pointer'
              : 'bg-disabled text-text-secondary cursor-not-allowed'
            }
          `}
        >
          {prediction.isPending ? 'Predicting...' : 'Get Prediction'}
        </button>
        <button
          type="button"
          onClick={handleResetDraft}
          className="px-4 py-2 rounded text-sm bg-panel-light text-text-secondary hover:text-text-primary transition-colors"
        >
          Reset Draft
        </button>
        <button
          type="button"
          onClick={handleResetAll}
          className="px-4 py-2 rounded text-sm bg-panel-light text-text-secondary hover:text-text-primary transition-colors"
        >
          Reset All
        </button>
      </div>

      {isSeries && hasPrediction && !seriesComplete && (
        <div className="flex flex-col items-center gap-1.5 mt-1">
          <span className="text-xs text-text-secondary">Record Game Result</span>
          <div className="flex gap-2">
            <button
              type="button"
              onClick={() => handleRecordResult('blue')}
              className="px-4 py-1.5 rounded text-xs font-semibold bg-blue-team/20 text-blue-team border border-blue-team/40 hover:bg-blue-team/30 transition-colors"
            >
              {blueTeam ?? 'Blue'} Wins
            </button>
            <button
              type="button"
              onClick={() => handleRecordResult('red')}
              className="px-4 py-1.5 rounded text-xs font-semibold bg-red-team/20 text-red-team border border-red-team/40 hover:bg-red-team/30 transition-colors"
            >
              {redTeam ?? 'Red'} Wins
            </button>
          </div>
        </div>
      )}

      {isSeries && seriesComplete && (
        <div className="flex flex-col items-center gap-1.5 mt-1">
          <button
            type="button"
            onClick={handleResetAll}
            className="px-5 py-2 rounded text-sm font-semibold bg-gold/20 text-gold-light border border-gold/40 hover:bg-gold/30 transition-colors"
          >
            New Series
          </button>
        </div>
      )}

      {prediction.isError && (
        <p className="text-red-400 text-xs text-center max-w-xs">
          {prediction.error instanceof Error ? prediction.error.message : 'Prediction failed'}
        </p>
      )}
    </div>
  )
}
