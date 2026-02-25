import { useDraftStore, buildPredictRequest } from '../store/draftStore'
import { usePrediction } from '../api/predict'

export function DraftControls({ onPrediction }: { onPrediction: (blue: number, red: number) => void }) {
  const isDraftReady = useDraftStore((s) => s.isDraftReady)
  const resetDraft = useDraftStore((s) => s.resetDraft)
  const resetAll = useDraftStore((s) => s.resetAll)
  const prediction = usePrediction()

  const ready = isDraftReady()

  function handlePredict() {
    const request = buildPredictRequest()
    prediction.mutate(request, {
      onSuccess: (data) => {
        onPrediction(data.blue_win_probability, data.red_win_probability)
      },
    })
  }

  function handleResetDraft() {
    resetDraft()
    prediction.reset()
  }

  function handleResetAll() {
    resetAll()
    prediction.reset()
  }

  return (
    <div className="flex flex-col gap-2 items-center">
      <div className="flex gap-2 justify-center">
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
      {prediction.isError && (
        <p className="text-red-400 text-xs text-center max-w-xs">
          {prediction.error instanceof Error ? prediction.error.message : 'Prediction failed'}
        </p>
      )}
    </div>
  )
}
