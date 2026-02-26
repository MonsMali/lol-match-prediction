import type { Side, Role } from '../types'
import { useDraftStore } from '../store/draftStore'
import { useChampionLookup } from '../hooks/useChampionLookup'
import { ChampionImage } from './ChampionImage'

const ROLES: Role[] = ['top', 'jungle', 'mid', 'bot', 'support']
const ROLE_LABELS: Record<Role, string> = {
  top: 'Top',
  jungle: 'Jungle',
  mid: 'Mid',
  bot: 'Bot',
  support: 'Support',
}

interface RoleAssignmentProps {
  side: Side
}

export function RoleAssignment({ side }: RoleAssignmentProps) {
  const championLookup = useChampionLookup()
  const picks = useDraftStore((s) => side === 'blue' ? s.bluePicks : s.redPicks)
  const roles = useDraftStore((s) => side === 'blue' ? s.blueRoles : s.redRoles)
  const setRole = useDraftStore((s) => s.setRole)

  // Only render when all 5 picks are filled
  if (!picks.every((p) => p !== null)) return null

  const assignedRoles = new Set(
    ROLES.filter((r) => roles[r] !== null)
  )

  const allAssigned = assignedRoles.size === 5

  const headerColor = side === 'blue' ? 'text-blue-team' : 'text-red-team'
  const borderColor = side === 'blue' ? 'border-blue-team/30' : 'border-red-team/30'

  return (
    <div className={`bg-panel-light rounded p-3 border ${borderColor}`}>
      <div className="flex items-center justify-between mb-2">
        <h3 className={`text-xs font-semibold uppercase tracking-wide ${headerColor}`}>
          {side === 'blue' ? 'Blue' : 'Red'} Team Roles
        </h3>
        {allAssigned && (
          <span className="text-green-400 text-xs font-semibold">All Assigned</span>
        )}
      </div>

      <div className="flex flex-col gap-1.5">
        {picks.map((champion, i) => {
          if (!champion) return null

          const championInfo = champion ? championLookup.get(champion) : undefined
          // Find which role this champion is currently assigned to
          const currentRole = ROLES.find((r) => roles[r] === champion) ?? ''

          return (
            <div key={i} className="flex items-center gap-2">
              <ChampionImage
                src={championInfo?.image_url}
                alt={champion}
                side={side}
                className="w-8 h-8 rounded"
              />
              <span className="text-text-primary text-xs flex-1 truncate">{champion}</span>
              <select
                value={currentRole}
                onChange={(e) => {
                  const newRole = e.target.value as Role | ''
                  // Clear this champion from any previous role assignment
                  for (const r of ROLES) {
                    if (roles[r] === champion) {
                      setRole(side, r, null)
                    }
                  }
                  // If a role was selected (not "-- Role --"), assign it
                  if (newRole) {
                    // If the target role is taken by another champion, swap them
                    const displaced = roles[newRole]
                    if (displaced && displaced !== champion) {
                      // Find the old role of the current champion to give to the displaced one
                      const oldRole = ROLES.find((r) => roles[r] === champion)
                      if (oldRole) {
                        setRole(side, oldRole, displaced)
                      }
                    }
                    setRole(side, newRole, champion)
                  }
                }}
                className="bg-panel text-text-primary text-xs rounded px-2 py-1 border border-panel-light focus:border-gold outline-none"
              >
                <option value="">-- Role --</option>
                {ROLES.map((role) => {
                  const taken = roles[role] !== null && roles[role] !== champion
                  return (
                    <option key={role} value={role}>
                      {ROLE_LABELS[role]}{taken ? ` (${roles[role]})` : ''}
                    </option>
                  )
                })}
              </select>
            </div>
          )
        })}
      </div>
    </div>
  )
}
