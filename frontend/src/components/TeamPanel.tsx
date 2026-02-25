import type { Side, Role } from '../types'
import { useDraftStore } from '../store/draftStore'
import { DRAFT_SEQUENCE } from '../store/draftStore'
import { BanRow } from './BanRow'
import { PickSlot } from './PickSlot'
import { TeamSelector } from './TeamSelector'

interface TeamPanelProps {
  side: Side
}

const ROLES: Role[] = ['top', 'jungle', 'mid', 'bot', 'support']

export function TeamPanel({ side }: TeamPanelProps) {
  const bans = useDraftStore((s) => side === 'blue' ? s.blueBans : s.redBans)
  const picks = useDraftStore((s) => side === 'blue' ? s.bluePicks : s.redPicks)
  const team = useDraftStore((s) => side === 'blue' ? s.blueTeam : s.redTeam)
  const roles = useDraftStore((s) => side === 'blue' ? s.blueRoles : s.redRoles)
  const mode = useDraftStore((s) => s.mode)
  const currentStep = useDraftStore((s) => s.currentStep)
  const activeSlot = useDraftStore((s) => s.activeSlot)
  const setTeam = useDraftStore((s) => s.setTeam)
  const setActiveSlot = useDraftStore((s) => s.setActiveSlot)

  // Determine active ban index for live mode
  let activeBanIndex: number | null = null
  if (mode === 'live' && currentStep < DRAFT_SEQUENCE.length) {
    const step = DRAFT_SEQUENCE[currentStep]
    if (step.team === side && step.action === 'ban') {
      activeBanIndex = step.slotIndex
    }
  } else if (mode === 'bulk' && activeSlot?.team === side && activeSlot.action === 'ban') {
    activeBanIndex = activeSlot.index
  }

  // Determine active pick index
  let activePickIndex: number | null = null
  if (mode === 'live' && currentStep < DRAFT_SEQUENCE.length) {
    const step = DRAFT_SEQUENCE[currentStep]
    if (step.team === side && step.action === 'pick') {
      activePickIndex = step.slotIndex
    }
  } else if (mode === 'bulk' && activeSlot?.team === side && activeSlot.action === 'pick') {
    activePickIndex = activeSlot.index
  }

  // Get role for a pick slot
  function getRoleForSlot(index: number): Role | null {
    const champion = picks[index]
    if (!champion) return null
    for (const role of ROLES) {
      if (roles[role] === champion) return role
    }
    return null
  }

  const borderSide = side === 'blue' ? 'border-l-2 border-l-blue-team' : 'border-r-2 border-r-red-team'

  return (
    <div className={`flex flex-col gap-3 p-4 bg-panel rounded ${borderSide}`}>
      <TeamSelector
        side={side}
        selectedTeam={team}
        onSelect={(name) => setTeam(side, name)}
      />

      <div className="flex flex-col items-center gap-1">
        <span className="text-text-secondary text-xs uppercase tracking-wide">Bans</span>
        <BanRow bans={bans} side={side} activeBanIndex={activeBanIndex} />
      </div>

      <div className="flex flex-col items-center gap-1.5">
        <span className="text-text-secondary text-xs uppercase tracking-wide">Picks</span>
        {picks.map((champion, i) => (
          <PickSlot
            key={i}
            champion={champion}
            role={getRoleForSlot(i)}
            side={side}
            slotIndex={i}
            isActive={activePickIndex === i}
            onClick={() => {
              if (mode === 'bulk' && !champion) {
                setActiveSlot(side, 'pick', i)
              }
            }}
          />
        ))}
      </div>
    </div>
  )
}
