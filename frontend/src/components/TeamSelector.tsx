import type { Side } from '../types'
import { useTeams } from '../api/teams'
import { SearchableSelect } from './SearchableSelect'

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
      <div className="w-full px-2 py-1.5 rounded bg-panel text-text-secondary text-sm border border-disabled">
        Loading teams...
      </div>
    )
  }

  const leagueGroups = LEAGUE_ORDER.filter((league) => league in teams)
  const otherLeagues = Object.keys(teams).filter((league) => !LEAGUE_ORDER.includes(league))

  const options = [
    ...leagueGroups.flatMap((league) =>
      teams[league].map((team) => ({ value: team, label: team, group: league }))
    ),
    ...otherLeagues.flatMap((league) =>
      teams[league].map((team) => ({ value: team, label: team, group: 'Other' }))
    ),
  ]

  return (
    <SearchableSelect
      options={options}
      value={selectedTeam}
      onChange={onSelect}
      placeholder="Select Team"
      borderColor={borderColor}
    />
  )
}
