import { create } from 'zustand'
import type { Side, DraftAction, DraftMode, SeriesFormat, Role, DraftStep, PredictRequest, Rosters } from '../types'

// The 20-step professional draft sequence
export const DRAFT_SEQUENCE: DraftStep[] = [
  // Ban Phase 1: alternating, blue first
  { team: 'blue', action: 'ban', slotIndex: 0 },
  { team: 'red',  action: 'ban', slotIndex: 0 },
  { team: 'blue', action: 'ban', slotIndex: 1 },
  { team: 'red',  action: 'ban', slotIndex: 1 },
  { team: 'blue', action: 'ban', slotIndex: 2 },
  { team: 'red',  action: 'ban', slotIndex: 2 },
  // Pick Phase 1: B1, R1-R2, B2-B3, R3
  { team: 'blue', action: 'pick', slotIndex: 0 },
  { team: 'red',  action: 'pick', slotIndex: 0 },
  { team: 'red',  action: 'pick', slotIndex: 1 },
  { team: 'blue', action: 'pick', slotIndex: 1 },
  { team: 'blue', action: 'pick', slotIndex: 2 },
  { team: 'red',  action: 'pick', slotIndex: 2 },
  // Ban Phase 2: alternating, red first
  { team: 'red',  action: 'ban', slotIndex: 3 },
  { team: 'blue', action: 'ban', slotIndex: 3 },
  { team: 'red',  action: 'ban', slotIndex: 4 },
  { team: 'blue', action: 'ban', slotIndex: 4 },
  // Pick Phase 2: R4, B4-B5, R5
  { team: 'red',  action: 'pick', slotIndex: 3 },
  { team: 'blue', action: 'pick', slotIndex: 3 },
  { team: 'blue', action: 'pick', slotIndex: 4 },
  { team: 'red',  action: 'pick', slotIndex: 4 },
]

const EMPTY_ROLES: Record<Role, string | null> = {
  top: null,
  jungle: null,
  mid: null,
  bot: null,
  support: null,
}

function createEmptySlots(): (string | null)[] {
  return [null, null, null, null, null]
}

interface ActiveSlot {
  team: Side
  action: DraftAction
  index: number
}

interface DraftState {
  // Draft data
  blueBans: (string | null)[]
  redBans: (string | null)[]
  bluePicks: (string | null)[]
  redPicks: (string | null)[]
  blueTeam: string | null
  redTeam: string | null
  blueRoles: Record<Role, string | null>
  redRoles: Record<Role, string | null>
  bluePlayers: Record<Role, string | null>
  redPlayers: Record<Role, string | null>
  rosters: Rosters

  // Mode and state machine
  mode: DraftMode
  currentStep: number
  activeSlot: ActiveSlot | null

  // Series
  seriesFormat: SeriesFormat
  seriesScore: { blue: number; red: number }
  currentGame: number

  // Computed
  usedChampions: () => Set<string>
  isComplete: () => boolean
  isDraftReady: () => boolean
  isSeriesComplete: () => boolean
  currentDraftStep: () => DraftStep | null
  currentPhaseLabel: () => string

  // Actions
  selectChampion: (name: string) => void
  setActiveSlot: (team: Side, action: DraftAction, index: number) => void
  setTeam: (side: Side, name: string) => void
  setRole: (side: Side, role: Role, champion: string | null) => void
  setPlayer: (side: Side, role: Role, player: string | null) => void
  setRosters: (rosters: Rosters) => void
  setMode: (mode: DraftMode) => void
  setSeriesFormat: (format: SeriesFormat) => void
  recordGameResult: (winner: Side) => void
  undoLastStep: () => void
  resetDraft: () => void
  resetAll: () => void
}

function getSlotArray(
  state: DraftState,
  team: Side,
  action: DraftAction
): (string | null)[] {
  if (team === 'blue') {
    return action === 'ban' ? state.blueBans : state.bluePicks
  }
  return action === 'ban' ? state.redBans : state.redPicks
}

/** Scan for the next empty slot in bulk mode: blue bans, red bans, blue picks, red picks */
function findNextEmptySlot(state: {
  blueBans: (string | null)[]
  redBans: (string | null)[]
  bluePicks: (string | null)[]
  redPicks: (string | null)[]
}): ActiveSlot | null {
  const groups: { team: Side; action: DraftAction; slots: (string | null)[] }[] = [
    { team: 'blue', action: 'ban', slots: state.blueBans },
    { team: 'red', action: 'ban', slots: state.redBans },
    { team: 'blue', action: 'pick', slots: state.bluePicks },
    { team: 'red', action: 'pick', slots: state.redPicks },
  ]
  for (const g of groups) {
    for (let i = 0; i < g.slots.length; i++) {
      if (g.slots[i] === null) {
        return { team: g.team, action: g.action, index: i }
      }
    }
  }
  return null
}

/** Count how many of the 20 draft slots are filled */
export function countFilledSlots(state: {
  blueBans: (string | null)[]
  redBans: (string | null)[]
  bluePicks: (string | null)[]
  redPicks: (string | null)[]
}): number {
  return [
    ...state.blueBans,
    ...state.redBans,
    ...state.bluePicks,
    ...state.redPicks,
  ].filter((s) => s !== null).length
}

const ROLES: Role[] = ['top', 'jungle', 'mid', 'bot', 'support']

/** Build a PredictRequest from current store state.
 *  Convenience overload with no args reads from the store directly. */
export function buildPredictRequest(state?: Pick<DraftState, 'blueTeam' | 'redTeam' | 'bluePicks' | 'redPicks' | 'blueBans' | 'redBans' | 'blueRoles' | 'redRoles' | 'bluePlayers' | 'redPlayers'>): PredictRequest {
  const s = state ?? useDraftStore.getState()
  const bluePicksMap: Record<string, string> = {} as Record<string, string>
  const redPicksMap: Record<string, string> = {} as Record<string, string>

  for (const role of ROLES) {
    bluePicksMap[role] = s.blueRoles[role] ?? s.bluePicks[ROLES.indexOf(role)] ?? 'UNKNOWN'
    redPicksMap[role] = s.redRoles[role] ?? s.redPicks[ROLES.indexOf(role)] ?? 'UNKNOWN'
  }

  const bp = s.bluePlayers
  const rp = s.redPlayers

  return {
    blue_team: s.blueTeam ?? '',
    red_team: s.redTeam ?? '',
    blue_picks: {
      top: bluePicksMap.top,
      jungle: bluePicksMap.jungle,
      mid: bluePicksMap.mid,
      bot: bluePicksMap.bot,
      support: bluePicksMap.support,
      ...(bp?.top ? { top_player: bp.top } : {}),
      ...(bp?.jungle ? { jungle_player: bp.jungle } : {}),
      ...(bp?.mid ? { mid_player: bp.mid } : {}),
      ...(bp?.bot ? { bot_player: bp.bot } : {}),
      ...(bp?.support ? { support_player: bp.support } : {}),
    },
    red_picks: {
      top: redPicksMap.top,
      jungle: redPicksMap.jungle,
      mid: redPicksMap.mid,
      bot: redPicksMap.bot,
      support: redPicksMap.support,
      ...(rp?.top ? { top_player: rp.top } : {}),
      ...(rp?.jungle ? { jungle_player: rp.jungle } : {}),
      ...(rp?.mid ? { mid_player: rp.mid } : {}),
      ...(rp?.bot ? { bot_player: rp.bot } : {}),
      ...(rp?.support ? { support_player: rp.support } : {}),
    },
    blue_bans: s.blueBans.filter((b): b is string => b !== null),
    red_bans: s.redBans.filter((b): b is string => b !== null),
  }
}

/** Set a value into the correct slot array and return partial state update. */
function updateSlot(
  prev: DraftState,
  team: Side,
  action: DraftAction,
  index: number,
  value: string | null,
): Partial<DraftState> {
  const key = team === 'blue'
    ? (action === 'ban' ? 'blueBans' : 'bluePicks')
    : (action === 'ban' ? 'redBans' : 'redPicks')
  const arr = [...prev[key]]
  const oldValue = arr[index]
  arr[index] = value

  const update: Partial<DraftState> = { [key]: arr }

  // Clear role assignment when removing or replacing a pick
  if (action === 'pick' && oldValue && oldValue !== value) {
    const rolesKey = team === 'blue' ? 'blueRoles' : 'redRoles'
    const roles = { ...prev[rolesKey] }
    for (const r of ROLES) {
      if (roles[r] === oldValue) roles[r] = null
    }
    update[rolesKey] = roles
  }

  return update
}

export const useDraftStore = create<DraftState>()((set, get) => ({
  // Initial state
  blueBans: createEmptySlots(),
  redBans: createEmptySlots(),
  bluePicks: createEmptySlots(),
  redPicks: createEmptySlots(),
  blueTeam: null,
  redTeam: null,
  blueRoles: { ...EMPTY_ROLES },
  redRoles: { ...EMPTY_ROLES },
  bluePlayers: { ...EMPTY_ROLES },
  redPlayers: { ...EMPTY_ROLES },
  rosters: {},
  mode: 'live',
  currentStep: 0,
  activeSlot: null,
  seriesFormat: 'single',
  seriesScore: { blue: 0, red: 0 },
  currentGame: 1,

  // Computed
  usedChampions: () => {
    const state = get()
    const used = new Set<string>()
    const allSlots = [
      ...state.blueBans,
      ...state.redBans,
      ...state.bluePicks,
      ...state.redPicks,
    ]
    for (const name of allSlots) {
      if (name !== null) {
        used.add(name)
      }
    }
    return used
  },

  isComplete: () => {
    const state = get()
    const allSlots = [
      ...state.blueBans,
      ...state.redBans,
      ...state.bluePicks,
      ...state.redPicks,
    ]
    return allSlots.every((slot) => slot !== null)
  },

  isDraftReady: () => {
    const state = get()
    const allSlotsFilled = [
      ...state.blueBans,
      ...state.redBans,
      ...state.bluePicks,
      ...state.redPicks,
    ].every((slot) => slot !== null)

    const teamsSelected = state.blueTeam !== null && state.redTeam !== null

    const allRolesAssigned =
      Object.values(state.blueRoles).every((v) => v !== null) &&
      Object.values(state.redRoles).every((v) => v !== null)

    return allSlotsFilled && teamsSelected && allRolesAssigned
  },

  isSeriesComplete: () => {
    const state = get()
    if (state.seriesFormat === 'single') return false
    const threshold = state.seriesFormat === 'bo3' ? 2 : 3
    return state.seriesScore.blue >= threshold || state.seriesScore.red >= threshold
  },

  currentDraftStep: () => {
    const state = get()
    if (state.currentStep >= DRAFT_SEQUENCE.length) return null
    return DRAFT_SEQUENCE[state.currentStep]
  },

  currentPhaseLabel: () => {
    const step = get().currentStep
    if (step < 6) return 'Ban Phase 1'
    if (step < 12) return 'Pick Phase 1'
    if (step < 16) return 'Ban Phase 2'
    if (step < 20) return 'Pick Phase 2'
    return 'Draft Complete'
  },

  // Actions
  selectChampion: (name: string) => {
    const state = get()

    if (state.mode === 'live' && state.currentStep < DRAFT_SEQUENCE.length) {
      if (state.usedChampions().has(name)) return
      const { team, action, slotIndex } = DRAFT_SEQUENCE[state.currentStep]

      set((prev) => ({
        ...updateSlot(prev, team, action, slotIndex, name),
        currentStep: prev.currentStep + 1,
      }))
    } else {
      const slot = state.activeSlot
      if (!slot) return
      const currentInSlot = getSlotArray(state, slot.team, slot.action)[slot.index]
      if (state.usedChampions().has(name) && name !== currentInSlot) return

      set((prev) => {
        const slotUpdate = updateSlot(prev, slot.team, slot.action, slot.index, name)
        const nextState = {
          blueBans: (slotUpdate.blueBans as (string | null)[] | undefined) ?? prev.blueBans,
          redBans: (slotUpdate.redBans as (string | null)[] | undefined) ?? prev.redBans,
          bluePicks: (slotUpdate.bluePicks as (string | null)[] | undefined) ?? prev.bluePicks,
          redPicks: (slotUpdate.redPicks as (string | null)[] | undefined) ?? prev.redPicks,
        }
        return { ...slotUpdate, activeSlot: findNextEmptySlot(nextState) }
      })
    }
  },

  setActiveSlot: (team: Side, action: DraftAction, index: number) => {
    set({ activeSlot: { team, action, index } })
  },

  setTeam: (side: Side, name: string) => {
    const rosters = get().rosters
    const roster = rosters[name]
    const players: Record<Role, string | null> = { ...EMPTY_ROLES }
    if (roster) {
      for (const role of ROLES) {
        players[role] = roster[role] ?? null
      }
    }
    if (side === 'blue') {
      set({ blueTeam: name, bluePlayers: players })
    } else {
      set({ redTeam: name, redPlayers: players })
    }
  },

  setRole: (side: Side, role: Role, champion: string | null) => {
    set((prev) => {
      if (side === 'blue') {
        return { blueRoles: { ...prev.blueRoles, [role]: champion } }
      }
      return { redRoles: { ...prev.redRoles, [role]: champion } }
    })
  },

  setPlayer: (side: Side, role: Role, player: string | null) => {
    set((prev) => {
      if (side === 'blue') {
        return { bluePlayers: { ...prev.bluePlayers, [role]: player } }
      }
      return { redPlayers: { ...prev.redPlayers, [role]: player } }
    })
  },

  setRosters: (rosters: Rosters) => {
    set({ rosters })
  },

  setMode: (mode: DraftMode) => {
    if (mode === 'live') {
      // Recalculate currentStep: find first unfilled slot in DRAFT_SEQUENCE
      const state = get()
      let step = 0
      for (let i = 0; i < DRAFT_SEQUENCE.length; i++) {
        const { team, action, slotIndex } = DRAFT_SEQUENCE[i]
        const arr = getSlotArray(state, team, action)
        if (arr[slotIndex] === null) {
          step = i
          break
        }
        if (i === DRAFT_SEQUENCE.length - 1) {
          step = DRAFT_SEQUENCE.length
        }
      }
      set({ mode, currentStep: step, activeSlot: null })
    } else {
      set({ mode, activeSlot: null })
    }
  },

  setSeriesFormat: (format: SeriesFormat) => {
    set({ seriesFormat: format, seriesScore: { blue: 0, red: 0 }, currentGame: 1 })
  },

  recordGameResult: (winner: Side) => {
    const state = get()
    // Do not record if series is already complete
    if (state.isSeriesComplete()) return

    set((prev) => {
      const newScore = { ...prev.seriesScore }
      newScore[winner] += 1
      return {
        seriesScore: newScore,
        currentGame: prev.currentGame + 1,
        // Reset draft state for next game (preserves teams and series)
        blueBans: createEmptySlots(),
        redBans: createEmptySlots(),
        bluePicks: createEmptySlots(),
        redPicks: createEmptySlots(),
        blueRoles: { ...EMPTY_ROLES },
        redRoles: { ...EMPTY_ROLES },
        currentStep: 0,
        activeSlot: null,
      }
    })
  },

  undoLastStep: () => {
    const state = get()
    if (state.mode !== 'live' || state.currentStep <= 0) return
    const prevStep = state.currentStep - 1
    const { team, action, slotIndex } = DRAFT_SEQUENCE[prevStep]

    set((prev) => ({
      ...updateSlot(prev, team, action, slotIndex, null),
      currentStep: prevStep,
    }))
  },

  resetDraft: () => {
    set({
      blueBans: createEmptySlots(),
      redBans: createEmptySlots(),
      bluePicks: createEmptySlots(),
      redPicks: createEmptySlots(),
      blueRoles: { ...EMPTY_ROLES },
      redRoles: { ...EMPTY_ROLES },
      bluePlayers: { ...EMPTY_ROLES },
      redPlayers: { ...EMPTY_ROLES },
      currentStep: 0,
      activeSlot: null,
    })
  },

  resetAll: () => {
    set({
      blueBans: createEmptySlots(),
      redBans: createEmptySlots(),
      bluePicks: createEmptySlots(),
      redPicks: createEmptySlots(),
      blueTeam: null,
      redTeam: null,
      blueRoles: { ...EMPTY_ROLES },
      redRoles: { ...EMPTY_ROLES },
      bluePlayers: { ...EMPTY_ROLES },
      redPlayers: { ...EMPTY_ROLES },
      mode: 'live',
      currentStep: 0,
      activeSlot: null,
      seriesFormat: 'single',
      seriesScore: { blue: 0, red: 0 },
      currentGame: 1,
    })
  },
}))
