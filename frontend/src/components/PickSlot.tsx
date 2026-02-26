import type { Side, Role } from '../types'
import { useChampionLookup } from '../hooks/useChampionLookup'
import { ChampionImage } from './ChampionImage'

interface PickSlotProps {
  champion: string | null
  role: Role | null
  side: Side
  slotIndex: number
  isActive: boolean
  onClick: () => void
}

const ROLE_LABELS: Record<string, string> = {
  top: 'TOP',
  jungle: 'JGL',
  mid: 'MID',
  bot: 'BOT',
  support: 'SUP',
}

export function PickSlot({ champion, role, side, slotIndex, isActive, onClick }: PickSlotProps) {
  const championLookup = useChampionLookup()
  const championInfo = champion ? championLookup.get(champion) : undefined
  const borderColor = side === 'blue' ? 'border-blue-team' : 'border-red-team'
  const glowColor = side === 'blue' ? 'shadow-blue-team/40' : 'shadow-red-team/40'

  return (
    <button
      type="button"
      onClick={onClick}
      className={`
        relative w-16 h-16 rounded bg-panel-light flex flex-col items-center justify-center
        border-2 transition-all duration-200
        ${isActive ? `${borderColor} ${glowColor} shadow-lg` : ''}
        ${!isActive && champion ? 'border-transparent cursor-default' : ''}
        ${!isActive && !champion ? 'border-white/5 cursor-pointer hover:border-gold/40' : ''}
      `}
    >
      {champion ? (
        <>
          <ChampionImage
            src={championInfo?.image_url}
            alt={champion}
            side={side}
            className="w-full h-full rounded"
          />
          {role && (
            <span className="absolute -bottom-1 left-1/2 -translate-x-1/2 text-[9px] font-bold bg-panel px-1 rounded text-text-secondary">
              {ROLE_LABELS[role] ?? role}
            </span>
          )}
        </>
      ) : (
        <span className={`text-xs font-mono ${isActive ? 'text-gold' : 'text-disabled'}`}>{slotIndex + 1}</span>
      )}
    </button>
  )
}
