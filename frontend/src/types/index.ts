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
}

export interface PredictRequest {
  blue_team: string
  red_team: string
  blue_picks: TeamDraft
  red_picks: TeamDraft
  blue_bans: string[]
  red_bans: string[]
  patch?: string | null
}

export interface PredictResponse {
  blue_win_probability: number
  red_win_probability: number
}

export interface DraftStep {
  team: Side
  action: DraftAction
  slotIndex: number
}
