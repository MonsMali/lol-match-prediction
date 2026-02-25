import type { Side } from '../types'
import { useTeams } from '../api/teams'

interface TeamSelectorProps {
  side: Side
  selectedTeam: string | null
  onSelect: (teamName: string) => void
}

const LEAGUE_ORDER = ['LCK', 'LEC', 'LCS', 'LPL']

export function TeamSelector({ side, selectedTeam, onSelect }: TeamSelectorProps) {
  const { data: teams, isPending } = useTeams()
  const borderColor = side === 'blue' ? 'border-blue-team' : 'border-red-team'

  if (isPending || !teams) {
    return (
      <select disabled className="w-full px-2 py-1.5 rounded bg-panel text-text-secondary text-sm border border-disabled">
        <option>Loading teams...</option>
      </select>
    )
  }

  const leagueGroups = LEAGUE_ORDER.filter((league) => league in teams)
  const otherLeagues = Object.keys(teams).filter((league) => !LEAGUE_ORDER.includes(league))

  return (
    <select
      value={selectedTeam ?? ''}
      onChange={(e) => onSelect(e.target.value)}
      className={`w-full px-2 py-1.5 rounded bg-panel text-text-primary text-sm border ${borderColor} focus:outline-none`}
    >
      <option value="">Select Team</option>
      {leagueGroups.map((league) => (
        <optgroup key={league} label={league}>
          {teams[league].map((team) => (
            <option key={team} value={team}>
              {team}
            </option>
          ))}
        </optgroup>
      ))}
      {otherLeagues.length > 0 && (
        <optgroup label="Other">
          {otherLeagues.flatMap((league) =>
            teams[league].map((team) => (
              <option key={team} value={team}>
                {team}
              </option>
            ))
          )}
        </optgroup>
      )}
    </select>
  )
}
