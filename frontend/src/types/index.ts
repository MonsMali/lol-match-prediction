export type Side = 'blue' | 'red'
export type DraftAction = 'ban' | 'pick'
export type DraftMode = 'live' | 'bulk'
export type SeriesFormat = 'single' | 'bo3' | 'bo5'
export type Role = 'top' | 'jungle' | 'mid' | 'bot' | 'support'

export interface ChampionInfo {
  name: string
  key: string
  image_url: string
}

export interface TeamDraft {
  top: string
  jungle: string
  mid: string
  bot: string
  support: string
  top_player?: string | null
  jungle_player?: string | null
  mid_player?: string | null
  bot_player?: string | null
  support_player?: string | null
}

export type Rosters = Record<string, Record<Role, string>>

export interface PredictRequest {
  blue_team: string
  red_team: string
  blue_picks: TeamDraft
  red_picks: TeamDraft
  blue_bans: string[]
  red_bans: string[]
  patch?: string | null
}

export interface InsightFactor {
  label: string
  impact_pct: number
  description: string
}

export interface ModelMeta {
  training_patch: string
  training_year: number
  domain: string
}

export interface ChampionSuggestion {
  role: string
  champion: string
  delta_pct: number
  current_champion: string
}

export interface PickImpact {
  role: string
  champion: string
  impact_pct: number
}

export interface TeamContext {
  historical_winrate: number
  recent_winrate: number
  form_trend: number
  meta_adaptation: number
}

export interface PredictResponse {
  blue_win_probability: number
  red_win_probability: number
  blue_insights: InsightFactor[]
  red_insights: InsightFactor[]
  blue_pick_impacts: PickImpact[]
  red_pick_impacts: PickImpact[]
  blue_team_context: TeamContext | null
  red_team_context: TeamContext | null
  model: ModelMeta
}

export interface SuggestionsResponse {
  blue_suggestions: ChampionSuggestion[]
  red_suggestions: ChampionSuggestion[]
}

export interface DraftStep {
  team: Side
  action: DraftAction
  slotIndex: number
}
