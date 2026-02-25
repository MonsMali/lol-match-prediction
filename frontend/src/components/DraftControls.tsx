import { useDraftStore, buildPredictRequest } from '../store/draftStore'
import { usePrediction } from '../api/predict'

export function DraftControls({ onPrediction }: { onPrediction: (blue: number, red: number) => void }) {
  const isDraftReady = useDraftStore((s) => s.isDraftReady)
  const resetDraft = useDraftStore((s) => s.resetDraft)
  const resetAll = useDraftStore((s) => s.resetAll)
  const seriesFormat = useDraftStore((s) => s.seriesFormat)
  const isSeriesComplete = useDraftStore((s) => s.isSeriesComplete)
  const recordGameResult = useDraftStore((s) => s.recordGameResult)
  const blueTeam = useDraftStore((s) => s.blueTeam)
  const redTeam = useDraftStore((s) => s.redTeam)
  const prediction = usePrediction()

  const ready = isDraftReady()
  const seriesComplete = isSeriesComplete()
  const isSeries = seriesFormat !== 'single'
  const hasPrediction = prediction.isSuccess

  function handlePredict() {
    const request = buildPredictRequest()
    prediction.mutate(request, {
      onSuccess: (data) => {
        onPrediction(data.blue_win_probability, data.red_win_probability)
      },
    })
  }

  function handleRecordResult(winner: 'blue' | 'red') {
    recordGameResult(winner)
    prediction.reset()
    onPrediction(0, 0) // Clear displayed prediction
  }

  function handleResetDraft() {
    resetDraft()
    prediction.reset()
  }

  function handleResetAll() {
    resetAll()
    prediction.reset()
    onPrediction(0, 0)
  }

  return (
    <div className="flex flex-col gap-2 items-center">
      {/* Predict + Reset row */}
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

      {/* Series result recording -- only in BO3/BO5 after prediction */}
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

      {/* Series complete -- offer new series */}
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
