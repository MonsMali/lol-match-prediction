import { useState, useEffect } from 'react'

interface ChampionImageProps {
  src: string | undefined
  alt: string
  side: 'blue' | 'red'
  className?: string
}

export function ChampionImage({ src, alt, side, className }: ChampionImageProps) {
  const [status, setStatus] = useState<'loading' | 'loaded' | 'error'>('loading')

  // Reset status when src changes
  useEffect(() => {
    setStatus('loading')
  }, [src])

  const teamColor = side === 'blue' ? '#0088ff' : '#ff3333'

  return (
    <div className={`relative overflow-hidden ${className ?? ''}`}>
      {/* Shimmer layer */}
      {status === 'loading' && (
        <div className="absolute inset-0 bg-panel-light animate-pulse rounded" />
      )}

      {/* Error fallback */}
      {status === 'error' && (
        <svg
          viewBox="0 0 48 48"
          className="absolute inset-0 w-full h-full rounded"
          aria-hidden="true"
        >
          <rect width="48" height="48" fill="#252540" />
          <circle cx="24" cy="18" r="8" fill={teamColor} opacity="0.4" />
          <ellipse cx="24" cy="40" rx="14" ry="10" fill={teamColor} opacity="0.4" />
        </svg>
      )}

      {/* Image layer */}
      {src && (
        <img
          src={src}
          alt={alt}
          className={`absolute inset-0 w-full h-full object-cover rounded transition-opacity duration-150 ${
            status === 'loaded' ? 'opacity-100' : 'opacity-0'
          }`}
          onLoad={() => setStatus('loaded')}
          onError={() => setStatus('error')}
        />
      )}
    </div>
  )
}
