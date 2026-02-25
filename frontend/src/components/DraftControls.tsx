import { useDraftStore } from '../store/draftStore'
import { usePrediction } from '../api/predict'
import type { PredictRequest, Role } from '../types'

const ROLES: Role[] = ['top', 'jungle', 'mid', 'bot', 'support']

function buildPredictRequest(state: ReturnType<typeof useDraftStore.getState>): PredictRequest {
  const bluePicksMap: Record<string, string> = {} as Record<string, string>
  const redPicksMap: Record<string, string> = {} as Record<string, string>

  for (const role of ROLES) {
    bluePicksMap[role] = state.blueRoles[role] ?? state.bluePicks[ROLES.indexOf(role)] ?? 'UNKNOWN'
    redPicksMap[role] = state.redRoles[role] ?? state.redPicks[ROLES.indexOf(role)] ?? 'UNKNOWN'
  }

  return {
    blue_team: state.blueTeam ?? '',
    red_team: state.redTeam ?? '',
    blue_picks: {
      top: bluePicksMap.top,
      jungle: bluePicksMap.jungle,
      mid: bluePicksMap.mid,
      bot: bluePicksMap.bot,
      support: bluePicksMap.support,
    },
    red_picks: {
      top: redPicksMap.top,
      jungle: redPicksMap.jungle,
      mid: redPicksMap.mid,
      bot: redPicksMap.bot,
      support: redPicksMap.support,
    },
    blue_bans: state.blueBans.filter((b): b is string => b !== null),
    red_bans: state.redBans.filter((b): b is string => b !== null),
  }
}

export function DraftControls({ onPrediction }: { onPrediction: (blue: number, red: number) => void }) {
  const isDraftReady = useDraftStore((s) => s.isDraftReady)
  const resetDraft = useDraftStore((s) => s.resetDraft)
  const resetAll = useDraftStore((s) => s.resetAll)
  const prediction = usePrediction()

  const ready = isDraftReady()

  function handlePredict() {
    const state = useDraftStore.getState()
    const request = buildPredictRequest(state)
    prediction.mutate(request, {
      onSuccess: (data) => {
        onPrediction(data.blue_win_probability, data.red_win_probability)
      },
    })
  }

  return (
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
        onClick={resetDraft}
        className="px-4 py-2 rounded text-sm bg-panel-light text-text-secondary hover:text-text-primary transition-colors"
      >
        Reset Draft
      </button>
      <button
        type="button"
        onClick={resetAll}
        className="px-4 py-2 rounded text-sm bg-panel-light text-text-secondary hover:text-text-primary transition-colors"
      >
        Reset All
      </button>
    </div>
  )
}
