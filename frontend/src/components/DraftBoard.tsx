import { useState } from 'react'
import { useDraftStore, countFilledSlots } from '../store/draftStore'
import { ModeToggle } from './ModeToggle'
import { SeriesTracker } from './SeriesTracker'
import { TeamPanel } from './TeamPanel'
import { RoleAssignment } from './RoleAssignment'
import { WinProbability } from './WinProbability'
import { DraftControls } from './DraftControls'

export function DraftBoard() {
  const [prediction, setPrediction] = useState<{ blue: number; red: number } | null>(null)
  const [isPredicting, setIsPredicting] = useState(false)

  const mode = useDraftStore((s) => s.mode)
  const currentDraftStep = useDraftStore((s) => s.currentDraftStep)
  const currentPhaseLabel = useDraftStore((s) => s.currentPhaseLabel)
  const undoLastStep = useDraftStore((s) => s.undoLastStep)
  const currentStep = useDraftStore((s) => s.currentStep)
  const blueBans = useDraftStore((s) => s.blueBans)
  const redBans = useDraftStore((s) => s.redBans)
  const bluePicks = useDraftStore((s) => s.bluePicks)
  const redPicks = useDraftStore((s) => s.redPicks)

  const step = currentDraftStep()
  const phaseLabel = currentPhaseLabel()
  const isDraftComplete = phaseLabel === 'Draft Complete'

  const filledCount = countFilledSlots({ blueBans, redBans, bluePicks, redPicks })

  function handlePrediction(blue: number, red: number) {
    setPrediction({ blue, red })
    setIsPredicting(false)
  }

  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center justify-center gap-4 flex-wrap">
        <ModeToggle />
        <SeriesTracker />
      </div>

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
              <span className="text-sm text-text-secondary ml-2">-- Assign Roles</span>
            )}
          </div>
          {currentStep > 0 && !isDraftComplete && (
            <button
              type="button"
              onClick={undoLastStep}
              className="text-xs px-2 py-1 rounded bg-panel-light text-text-secondary hover:text-gold transition-colors"
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

      <div className="grid grid-cols-[1fr_auto_1fr] gap-4 items-start">
        <div className="flex flex-col gap-3">
          <TeamPanel side="blue" />
          <RoleAssignment side="blue" />
        </div>

        <div className="flex flex-col gap-4 items-center min-w-[200px] pt-12">
          <WinProbability
            blue={prediction?.blue ?? null}
            red={prediction?.red ?? null}
            isPending={isPredicting}
          />
          <DraftControls onPrediction={handlePrediction} />
        </div>

        <div className="flex flex-col gap-3">
          <TeamPanel side="red" />
          <RoleAssignment side="red" />
        </div>
      </div>
    </div>
  )
}
