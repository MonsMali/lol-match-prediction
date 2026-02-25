import { useState } from 'react'
import { ModeToggle } from './ModeToggle'
import { TeamPanel } from './TeamPanel'
import { WinProbability } from './WinProbability'
import { DraftControls } from './DraftControls'

export function DraftBoard() {
  const [prediction, setPrediction] = useState<{ blue: number; red: number } | null>(null)
  const [isPredicting, setIsPredicting] = useState(false)

  function handlePrediction(blue: number, red: number) {
    setPrediction({ blue, red })
    setIsPredicting(false)
  }

  return (
    <div className="flex flex-col gap-4">
      <ModeToggle />

      <div className="grid grid-cols-[1fr_auto_1fr] gap-4 items-start">
        <TeamPanel side="blue" />

        <div className="flex flex-col gap-4 items-center min-w-[200px] pt-12">
          <WinProbability
            blue={prediction?.blue ?? null}
            red={prediction?.red ?? null}
            isPending={isPredicting}
          />
          <DraftControls onPrediction={handlePrediction} />
        </div>

        <TeamPanel side="red" />
      </div>
    </div>
  )
}
