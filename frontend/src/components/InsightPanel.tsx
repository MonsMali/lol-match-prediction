import type { InsightFactor, PickImpact } from '../types'

interface InsightPanelProps {
  blueInsights: InsightFactor[]
  redInsights: InsightFactor[]
  bluePickImpacts: PickImpact[]
  redPickImpacts: PickImpact[]
  blueTeam: string
  redTeam: string
}

const ROLE_LABELS: Record<string, string> = {
  top: 'TOP',
  jungle: 'JNG',
  mid: 'MID',
  bot: 'BOT',
  support: 'SUP',
}

function PickImpactList({ impacts, side }: { impacts: PickImpact[]; side: 'blue' | 'red' }) {
  if (impacts.length === 0) return null

  // Find the worst pick (most negative impact -- first in the sorted list)
  const worstIdx = impacts.length > 0 && impacts[0].impact_pct < 0 ? 0 : -1

  return (
    <div className="flex flex-col gap-1">
      {impacts.map((pick, idx) => {
        const isPositive = pick.impact_pct > 0
        const isWorst = idx === worstIdx
        const absImpact = Math.abs(pick.impact_pct)
        const barWidth = Math.min(absImpact / 8 * 100, 100)
        const barColor = isPositive
          ? (side === 'blue' ? 'bg-blue-team' : 'bg-red-team')
          : 'bg-red-400/60'

        return (
          <div
            key={pick.role}
            className={`flex flex-col gap-0.5 rounded px-1.5 py-1 ${isWorst ? 'bg-red-500/10 ring-1 ring-red-500/30' : ''}`}
          >
            <div className="flex items-center justify-between text-xs">
              <div className="flex items-center gap-1.5">
                <span className="font-mono text-text-secondary w-6 text-[10px]">
                  {ROLE_LABELS[pick.role] || pick.role.toUpperCase()}
                </span>
                <span className={`font-medium truncate ${isWorst ? 'text-red-400' : 'text-text-primary'}`}>
                  {pick.champion}
                </span>
                {isWorst && (
                  <span className="text-[9px] font-semibold text-red-400 uppercase tracking-wider">
                    weakest
                  </span>
                )}
              </div>
              <span className={`font-mono font-semibold tabular-nums ${isPositive ? 'text-green-400' : 'text-red-400'}`}>
                {isPositive ? '+' : ''}{pick.impact_pct}%
              </span>
            </div>
            <div className="h-1 w-full rounded-full bg-panel-light overflow-hidden">
              <div
                className={`h-full rounded-full transition-all duration-500 ${barColor}`}
                style={{ width: `${barWidth}%` }}
              />
            </div>
          </div>
        )
      })}
    </div>
  )
}

function InsightList({ insights, side }: { insights: InsightFactor[]; side: 'blue' | 'red' }) {
  if (insights.length === 0) return null

  const barBg = side === 'blue' ? 'bg-blue-team' : 'bg-red-team'

  return (
    <div className="flex flex-col gap-1.5">
      {insights.map((ins) => {
        const isPositive = ins.impact_pct > 0
        const absImpact = Math.abs(ins.impact_pct)
        const barWidth = Math.min(absImpact / 10 * 100, 100)

        return (
          <div key={ins.label} className="flex flex-col gap-0.5">
            <div className="flex items-center justify-between text-xs">
              <span className="text-text-secondary truncate mr-2">{ins.label}</span>
              <span className={`font-mono font-semibold ${isPositive ? 'text-green-400' : 'text-red-400'}`}>
                {isPositive ? '+' : ''}{ins.impact_pct}%
              </span>
            </div>
            <div className="h-1 w-full rounded-full bg-panel-light overflow-hidden">
              <div
                className={`h-full rounded-full transition-all duration-500 ${isPositive ? barBg : 'bg-red-400/50'}`}
                style={{ width: `${barWidth}%` }}
              />
            </div>
          </div>
        )
      })}
    </div>
  )
}

export function InsightPanel({
  blueInsights, redInsights,
  bluePickImpacts, redPickImpacts,
  blueTeam, redTeam,
}: InsightPanelProps) {
  const hasImpacts = bluePickImpacts.length > 0 || redPickImpacts.length > 0
  const hasInsights = blueInsights.length > 0 || redInsights.length > 0

  if (!hasImpacts && !hasInsights) return null

  return (
    <div className="w-full bg-panel rounded-lg p-3 mt-2 flex flex-col gap-4">
      {hasImpacts && (
        <div>
          <h3 className="text-xs font-semibold text-text-secondary uppercase tracking-wider mb-3 text-center">
            Draft Breakdown
          </h3>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <p className="text-xs font-semibold text-blue-team mb-2">{blueTeam}</p>
              <PickImpactList impacts={bluePickImpacts} side="blue" />
            </div>
            <div>
              <p className="text-xs font-semibold text-red-team mb-2">{redTeam}</p>
              <PickImpactList impacts={redPickImpacts} side="red" />
            </div>
          </div>
        </div>
      )}

      {hasInsights && (
        <div>
          <h3 className="text-xs font-semibold text-text-secondary uppercase tracking-wider mb-3 text-center">
            Draft Analysis
          </h3>
          <div className="grid grid-cols-2 gap-4">
            <div>
              {!hasImpacts && <p className="text-xs font-semibold text-blue-team mb-2">{blueTeam}</p>}
              <InsightList insights={blueInsights} side="blue" />
            </div>
            <div>
              {!hasImpacts && <p className="text-xs font-semibold text-red-team mb-2">{redTeam}</p>}
              <InsightList insights={redInsights} side="red" />
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
