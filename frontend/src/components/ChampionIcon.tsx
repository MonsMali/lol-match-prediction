import React from 'react'
import type { ChampionInfo } from '../types'

interface ChampionIconProps {
  champion: ChampionInfo
  disabled: boolean
  onClick: () => void
}

export const ChampionIcon = React.memo(function ChampionIcon({
  champion,
  disabled,
  onClick,
}: ChampionIconProps) {
  const [hasError, setHasError] = React.useState(false)

  return (
    <button
      type="button"
      onClick={disabled ? undefined : onClick}
      className={`
        relative w-12 h-12 rounded bg-panel flex items-center justify-center
        transition-all duration-150
        ${disabled
          ? 'opacity-30 cursor-not-allowed'
          : 'cursor-pointer hover:brightness-125 hover:shadow-[0_0_6px_rgba(200,155,60,0.5)]'
        }
      `}
      title={champion.name}
      disabled={disabled}
    >
      {hasError ? (
        <span className="text-text-secondary text-xs font-bold">?</span>
      ) : (
        <img
          src={champion.image_url}
          alt={champion.name}
          width={48}
          height={48}
          loading="lazy"
          className="w-full h-full rounded object-cover"
          onError={() => setHasError(true)}
        />
      )}
    </button>
  )
})
