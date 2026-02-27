import type { TeamContext } from '../types'

interface TeamStrengthPanelProps {
  blueContext: TeamContext | null
  redContext: TeamContext | null
  blueTeam: string
  redTeam: string
}

function WinrateIndicator({ value }: { value: number }) {
  const pct = Math.round(value * 100)
  const color = pct > 55 ? 'text-green-400' : pct < 45 ? 'text-red-400' : 'text-yellow-400'
  return <span className={`font-mono font-bold text-lg ${color}`}>{pct}%</span>
}

function TrendArrow({ value }: { value: number }) {
  if (Math.abs(value) < 0.01) return <span className="text-text-secondary text-xs">--</span>
  const isUp = value > 0
  return (
    <span className={`text-xs font-semibold ${isUp ? 'text-green-400' : 'text-red-400'}`}>
      {isUp ? '+' : ''}{(value * 100).toFixed(1)}%
    </span>
  )
}

function MetaBar({ value }: { value: number }) {
  // value is meta_adaptation: positive = above average, negative = below
  const pct = Math.min(Math.abs(value) * 200, 100)
  const isPositive = value >= 0
  return (
    <div className="flex items-center gap-1.5">
      <div className="h-1.5 flex-1 rounded-full bg-panel-light overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-500 ${isPositive ? 'bg-green-400/70' : 'bg-red-400/60'}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className={`text-[10px] font-mono ${isPositive ? 'text-green-400' : 'text-red-400'}`}>
        {isPositive ? '+' : ''}{(value * 100).toFixed(0)}%
      </span>
    </div>
  )
}

function TeamCard({ context, team, side }: { context: TeamContext; team: string; side: 'blue' | 'red' }) {
  const teamColor = side === 'blue' ? 'text-blue-team' : 'text-red-team'

  return (
    <div className="flex flex-col gap-2">
      <p className={`text-xs font-semibold ${teamColor}`}>{team}</p>

      <div className="flex items-baseline justify-between">
        <span className="text-[10px] uppercase tracking-wider text-text-secondary">Win Rate</span>
        <WinrateIndicator value={context.historical_winrate} />
      </div>

      <div className="flex items-baseline justify-between">
        <span className="text-[10px] uppercase tracking-wider text-text-secondary">Recent Form</span>
        <div className="flex items-baseline gap-1.5">
          <span className="text-xs font-mono text-text-primary">
            {Math.round(context.recent_winrate * 100)}%
          </span>
          <TrendArrow value={context.form_trend} />
        </div>
      </div>

      <div>
        <span className="text-[10px] uppercase tracking-wider text-text-secondary block mb-0.5">
          Meta Adaptation
        </span>
        <MetaBar value={context.meta_adaptation} />
      </div>
    </div>
  )
}

export function TeamStrengthPanel({ blueContext, redContext, blueTeam, redTeam }: TeamStrengthPanelProps) {
  if (!blueContext && !redContext) return null

  return (
    <div className="w-full bg-panel rounded-lg p-3 mt-2">
      <h3 className="text-xs font-semibold text-text-secondary uppercase tracking-wider mb-3 text-center">
        Team Strength
      </h3>
      <div className="grid grid-cols-2 gap-4">
        <div>
          {blueContext ? (
            <TeamCard context={blueContext} team={blueTeam} side="blue" />
          ) : (
            <p className="text-xs text-text-secondary">No data</p>
          )}
        </div>
        <div>
          {redContext ? (
            <TeamCard context={redContext} team={redTeam} side="red" />
          ) : (
            <p className="text-xs text-text-secondary">No data</p>
          )}
        </div>
      </div>
    </div>
  )
}
