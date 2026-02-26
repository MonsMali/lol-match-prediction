import type { Side } from '../types'
import { useChampionLookup } from '../hooks/useChampionLookup'
import { ChampionImage } from './ChampionImage'

interface BanRowProps {
  bans: (string | null)[]
  side: Side
  activeBanIndex: number | null
  onSlotClick?: (index: number) => void
}

export function BanRow({ bans, side, activeBanIndex, onSlotClick }: BanRowProps) {
  const championLookup = useChampionLookup()
  const borderColor = side === 'blue' ? 'border-blue-team' : 'border-red-team'
  const shadowColor = side === 'blue' ? 'shadow-blue-team/30' : 'shadow-red-team/30'

  return (
    <div className="flex gap-1 justify-center">
      {bans.map((ban, i) => {
        const isActive = activeBanIndex === i
        const isEmpty = !ban
        const clickable = isEmpty && onSlotClick
        const banInfo = ban ? championLookup.get(ban) : undefined

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
                <div className={`w-full h-full ${isActive ? 'grayscale-0 opacity-80' : 'grayscale opacity-60'}`}>
                  <ChampionImage
                    src={banInfo?.image_url}
                    alt={ban}
                    side={side}
                    className="w-full h-full rounded"
                  />
                </div>
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
