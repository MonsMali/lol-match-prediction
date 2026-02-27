import type { InsightFactor } from '../types'

interface InsightPanelProps {
  blueInsights: InsightFactor[]
  redInsights: InsightFactor[]
  blueTeam: string
  redTeam: string
}

function InsightList({ insights, side }: { insights: InsightFactor[]; side: 'blue' | 'red' }) {
  if (insights.length === 0) return null

  const barBg = side === 'blue' ? 'bg-blue-team' : 'bg-red-team'

  return (
    <div className="flex flex-col gap-1.5">
      {insights.map((ins) => {
        const isPositive = ins.impact_pct > 0
        const absImpact = Math.abs(ins.impact_pct)
        // Cap bar width at 10% impact = full bar
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

export function InsightPanel({ blueInsights, redInsights, blueTeam, redTeam }: InsightPanelProps) {
  if (blueInsights.length === 0 && redInsights.length === 0) return null

  return (
    <div className="w-full bg-panel rounded-lg p-3 mt-2">
      <h3 className="text-xs font-semibold text-text-secondary uppercase tracking-wider mb-3 text-center">
        Key Factors
      </h3>
      <div className="grid grid-cols-2 gap-4">
        <div>
          <p className="text-xs font-semibold text-blue-team mb-2">{blueTeam}</p>
          <InsightList insights={blueInsights} side="blue" />
        </div>
        <div>
          <p className="text-xs font-semibold text-red-team mb-2">{redTeam}</p>
          <InsightList insights={redInsights} side="red" />
        </div>
      </div>
    </div>
  )
}
