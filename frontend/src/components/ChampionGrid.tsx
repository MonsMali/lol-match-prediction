import { useState } from 'react'
import { useChampions } from '../api/champions'
import { useDraftStore } from '../store/draftStore'
import { ChampionIcon } from './ChampionIcon'

interface ChampionGridProps {
  onSelect: (championName: string) => void
}

export function ChampionGrid({ onSelect }: ChampionGridProps) {
  const [searchQuery, setSearchQuery] = useState('')
  const { data: champions, isPending } = useChampions()
  const usedChampions = useDraftStore((s) => s.usedChampions)
  const mode = useDraftStore((s) => s.mode)
  const currentStep = useDraftStore((s) => s.currentStep)
  const activeSlot = useDraftStore((s) => s.activeSlot)
  const currentDraftStep = useDraftStore((s) => s.currentDraftStep)

  const used = usedChampions()

  // Determine if selection is allowed
  const canSelect = mode === 'live' ? currentStep < 20 : activeSlot !== null

  // Build context message
  let contextMessage: string | null = null
  if (mode === 'live') {
    if (currentStep >= 20) {
      contextMessage = 'Draft complete'
    } else {
      const step = currentDraftStep()
      if (step) {
        const teamLabel = step.team === 'blue' ? 'Blue Team' : 'Red Team'
        const actionLabel = step.action === 'ban' ? 'Banning' : 'Picking'
        contextMessage = `${actionLabel} for ${teamLabel}`
      }
    }
  } else {
    if (!activeSlot) {
      contextMessage = 'Select a slot first'
    }
  }

  const filtered = champions?.filter((c) =>
    c.name.toLowerCase().includes(searchQuery.toLowerCase())
  ) ?? []

  function handleSelect(name: string) {
    if (!canSelect) return
    if (used.has(name)) return
    onSelect(name)
  }

  return (
    <div className="flex flex-col gap-2">
      <input
        type="text"
        value={searchQuery}
        onChange={(e) => setSearchQuery(e.target.value)}
        placeholder="Search champions..."
        className="w-full px-3 py-2 rounded bg-panel text-text-primary placeholder-text-secondary
          border border-transparent focus:border-gold focus:outline-none transition-colors"
      />

      {contextMessage && (
        <p className={`text-xs text-center ${canSelect ? 'text-gold/70' : 'text-text-secondary'}`}>
          {contextMessage}
        </p>
      )}

      <div className="overflow-y-auto max-h-[320px] rounded bg-panel/50 p-2">
        {isPending ? (
          <div className="grid grid-cols-6 md:grid-cols-8 lg:grid-cols-10 gap-1">
            {Array.from({ length: 50 }).map((_, i) => (
              <div
                key={i}
                className="w-12 h-12 rounded bg-panel-light animate-pulse"
              />
            ))}
          </div>
        ) : (
          <div className="grid grid-cols-6 md:grid-cols-8 lg:grid-cols-10 gap-1">
            {filtered.map((champion) => (
              <ChampionIcon
                key={champion.name}
                champion={champion}
                disabled={used.has(champion.name) || !canSelect}
                onClick={() => handleSelect(champion.name)}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
