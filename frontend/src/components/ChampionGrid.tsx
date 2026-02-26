import { useState } from 'react'
import { useChampions } from '../api/champions'
import { useDraftStore, DRAFT_SEQUENCE } from '../store/draftStore'
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

  // Allow selection during live draft, in bulk mode, or when editing after draft complete
  const isDraftComplete = mode === 'live' && currentStep >= DRAFT_SEQUENCE.length
  const canSelect = mode === 'live'
    ? (currentStep < DRAFT_SEQUENCE.length || (isDraftComplete && activeSlot !== null))
    : activeSlot !== null

  // Build context message
  let contextMessage: string | null = null
  if (mode === 'live') {
    if (isDraftComplete && activeSlot) {
      contextMessage = 'Click a champion to replace the selected slot'
    } else if (isDraftComplete) {
      contextMessage = 'Click a slot to edit, or assign roles below'
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

  // When editing a slot, the champion currently in that slot should be selectable
  let editingChampion: string | null = null
  if (activeSlot && isDraftComplete) {
    const state = useDraftStore.getState()
    const { team, action, index } = activeSlot
    if (team === 'blue') {
      editingChampion = action === 'ban' ? state.blueBans[index] : state.bluePicks[index]
    } else {
      editingChampion = action === 'ban' ? state.redBans[index] : state.redPicks[index]
    }
  }

  function handleSelect(name: string) {
    if (!canSelect) return
    if (used.has(name) && name !== editingChampion) return
    onSelect(name)
  }

  return (
    <div className="flex flex-col gap-1.5">
      <input
        type="text"
        value={searchQuery}
        onChange={(e) => setSearchQuery(e.target.value)}
        placeholder="Search champions..."
        className="w-full px-3 py-1.5 rounded bg-panel text-text-primary text-sm placeholder-text-secondary
          border border-transparent focus:border-gold focus:outline-none transition-colors"
      />

      {contextMessage && (
        <p className={`text-xs text-center ${canSelect ? 'text-gold/70' : 'text-text-secondary'}`}>
          {contextMessage}
        </p>
      )}

      <div className="overflow-y-auto max-h-[360px] lg:max-h-[440px] rounded bg-panel/50 p-1.5">
        {isPending ? (
          <div className="grid grid-cols-6 sm:grid-cols-8 md:grid-cols-8 lg:grid-cols-10 xl:grid-cols-12 gap-1">
            {Array.from({ length: 50 }).map((_, i) => (
              <div
                key={i}
                className="w-12 h-12 rounded bg-panel-light animate-pulse"
              />
            ))}
          </div>
        ) : (
          <div className="grid grid-cols-6 sm:grid-cols-8 md:grid-cols-8 lg:grid-cols-10 xl:grid-cols-12 gap-1">
            {filtered.map((champion) => {
              const isUsed = used.has(champion.name)
              const isEditing = champion.name === editingChampion
              return (
                <ChampionIcon
                  key={champion.name}
                  champion={champion}
                  disabled={(isUsed && !isEditing) || !canSelect}
                  onClick={() => handleSelect(champion.name)}
                />
              )
            })}
          </div>
        )}
      </div>
    </div>
  )
}
