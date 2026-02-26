import type { ChampionSuggestion } from '../types'

interface SuggestionPanelProps {
  blueSuggestions: ChampionSuggestion[]
  redSuggestions: ChampionSuggestion[]
  blueTeam: string
  redTeam: string
}

const ROLE_LABELS: Record<string, string> = {
  top: 'Top',
  jungle: 'Jungle',
  mid: 'Mid',
  bot: 'Bot',
  support: 'Support',
}

function SuggestionList({ suggestions, side }: { suggestions: ChampionSuggestion[]; side: 'blue' | 'red' }) {
  if (suggestions.length === 0) {
    return <p className="text-xs text-text-secondary italic">No improvements found</p>
  }

  return (
    <div className="flex flex-col gap-1">
      {suggestions.map((s) => (
        <div key={`${s.role}-${s.champion}`} className="flex items-center gap-2 text-xs">
          <span className="text-text-secondary w-12 shrink-0">{ROLE_LABELS[s.role] ?? s.role}</span>
          <span className="text-text-secondary line-through">{s.current_champion}</span>
          <span className="text-text-secondary">-&gt;</span>
          <span className={`font-semibold ${side === 'blue' ? 'text-blue-team' : 'text-red-team'}`}>
            {s.champion}
          </span>
          <span className="text-green-400 font-mono ml-auto">+{s.delta_pct}%</span>
        </div>
      ))}
    </div>
  )
}

export function SuggestionPanel({ blueSuggestions, redSuggestions, blueTeam, redTeam }: SuggestionPanelProps) {
  if (blueSuggestions.length === 0 && redSuggestions.length === 0) return null

  return (
    <div className="w-full bg-panel rounded-lg p-3 mt-1">
      <h3 className="text-xs font-semibold text-text-secondary uppercase tracking-wider mb-3 text-center">
        Suggested Swaps
      </h3>
      <div className="grid grid-cols-2 gap-4">
        <div>
          <p className="text-xs font-semibold text-blue-team mb-2">{blueTeam}</p>
          <SuggestionList suggestions={blueSuggestions} side="blue" />
        </div>
        <div>
          <p className="text-xs font-semibold text-red-team mb-2">{redTeam}</p>
          <SuggestionList suggestions={redSuggestions} side="red" />
        </div>
      </div>
    </div>
  )
}
