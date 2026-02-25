import type { Side } from '../types'

interface BanRowProps {
  bans: (string | null)[]
  side: Side
  activeBanIndex: number | null
}

const DDRAGON_BASE = 'https://ddragon.leagueoflegends.com/cdn/14.24.1/img/champion'

export function BanRow({ bans, side, activeBanIndex }: BanRowProps) {
  const borderColor = side === 'blue' ? 'border-blue-team' : 'border-red-team'

  return (
    <div className="flex gap-1 justify-center">
      {bans.map((ban, i) => {
        const isActive = activeBanIndex === i
        return (
          <div
            key={i}
            className={`
              relative w-8 h-8 rounded bg-panel-light flex items-center justify-center
              border-2 ${isActive ? `${borderColor} animate-pulse` : 'border-transparent'}
            `}
          >
            {ban ? (
              <>
                <img
                  src={`${DDRAGON_BASE}/${ban}.png`}
                  alt={ban}
                  className="w-full h-full rounded object-cover grayscale opacity-60"
                />
                <span className="absolute inset-0 flex items-center justify-center text-red-team font-bold text-sm">
                  X
                </span>
              </>
            ) : (
              <span className="text-disabled text-xs">B</span>
            )}
          </div>
        )
      })}
    </div>
  )
}
