import type { Side } from '../types'

interface BanRowProps {
  bans: (string | null)[]
  side: Side
  activeBanIndex: number | null
  onSlotClick?: (index: number) => void
}

const DDRAGON_BASE = 'https://ddragon.leagueoflegends.com/cdn/14.24.1/img/champion'

export function BanRow({ bans, side, activeBanIndex, onSlotClick }: BanRowProps) {
  const borderColor = side === 'blue' ? 'border-blue-team' : 'border-red-team'
  const shadowColor = side === 'blue' ? 'shadow-blue-team/30' : 'shadow-red-team/30'

  return (
    <div className="flex gap-1 justify-center">
      {bans.map((ban, i) => {
        const isActive = activeBanIndex === i
        const isEmpty = !ban
        const clickable = isEmpty && onSlotClick

        return (
          <button
            key={i}
            type="button"
            disabled={!clickable}
            onClick={() => clickable && onSlotClick(i)}
            className={`
              relative w-8 h-8 rounded bg-panel-light flex items-center justify-center
              border-2 transition-all duration-200
              ${isActive ? `${borderColor} ${shadowColor} shadow-md animate-pulse` : ''}
              ${!isActive && !isEmpty ? 'border-transparent' : ''}
              ${!isActive && isEmpty ? 'border-transparent opacity-40' : ''}
              ${clickable ? 'cursor-pointer hover:border-gold/40 hover:opacity-100' : ''}
              ${!clickable && !isEmpty ? 'cursor-default' : ''}
            `}
          >
            {ban ? (
              <>
                <img
                  src={`${DDRAGON_BASE}/${ban}.png`}
                  alt={ban}
                  className={`w-full h-full rounded object-cover ${isActive ? 'grayscale-0 opacity-80' : 'grayscale opacity-60'}`}
                />
                <span className="absolute inset-0 flex items-center justify-center text-red-team font-bold text-sm">
                  X
                </span>
              </>
            ) : (
              <span className={`text-xs ${isActive ? 'text-gold' : 'text-disabled'}`}>B</span>
            )}
          </button>
        )
      })}
    </div>
  )
}
