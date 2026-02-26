import { create } from 'zustand'
import type { Side, DraftAction, DraftMode, SeriesFormat, Role, DraftStep, PredictRequest } from '../types'

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

/** Build a PredictRequest from current store state */
export function buildPredictRequest(): PredictRequest {
  const state = useDraftStore.getState()
  const bluePicksMap: Record<string, string> = {} as Record<string, string>
  const redPicksMap: Record<string, string> = {} as Record<string, string>

  for (const role of ROLES) {
    bluePicksMap[role] = state.blueRoles[role] ?? state.bluePicks[ROLES.indexOf(role)] ?? 'UNKNOWN'
    redPicksMap[role] = state.redRoles[role] ?? state.redPicks[ROLES.indexOf(role)] ?? 'UNKNOWN'
  }

  return {
    blue_team: state.blueTeam ?? '',
    red_team: state.redTeam ?? '',
    blue_picks: {
      top: bluePicksMap.top,
      jungle: bluePicksMap.jungle,
      mid: bluePicksMap.mid,
      bot: bluePicksMap.bot,
      support: bluePicksMap.support,
    },
    red_picks: {
      top: redPicksMap.top,
      jungle: redPicksMap.jungle,
      mid: redPicksMap.mid,
      bot: redPicksMap.bot,
      support: redPicksMap.support,
    },
    blue_bans: state.blueBans.filter((b): b is string => b !== null),
    red_bans: state.redBans.filter((b): b is string => b !== null),
  }
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

    if (state.mode === 'live') {
      if (state.currentStep >= DRAFT_SEQUENCE.length) return
      // Reject duplicate champion selection
      if (state.usedChampions().has(name)) return
      const step = DRAFT_SEQUENCE[state.currentStep]
      const { team, action, slotIndex } = step

      set((prev) => {
        const update: Partial<DraftState> = { currentStep: prev.currentStep + 1 }

        if (team === 'blue' && action === 'ban') {
          const arr = [...prev.blueBans]
          arr[slotIndex] = name
          update.blueBans = arr
        } else if (team === 'red' && action === 'ban') {
          const arr = [...prev.redBans]
          arr[slotIndex] = name
          update.redBans = arr
        } else if (team === 'blue' && action === 'pick') {
          const arr = [...prev.bluePicks]
          arr[slotIndex] = name
          update.bluePicks = arr
        } else {
          const arr = [...prev.redPicks]
          arr[slotIndex] = name
          update.redPicks = arr
        }

        return update
      })
    } else {
      // Bulk mode: place into activeSlot, then auto-advance to next empty slot
      const slot = state.activeSlot
      if (!slot) return
      // Reject duplicate champion selection
      if (state.usedChampions().has(name)) return

      set((prev) => {
        const { team, action, index } = slot
        const update: Partial<DraftState> = {}

        if (team === 'blue' && action === 'ban') {
          const arr = [...prev.blueBans]
          arr[index] = name
          update.blueBans = arr
        } else if (team === 'red' && action === 'ban') {
          const arr = [...prev.redBans]
          arr[index] = name
          update.redBans = arr
        } else if (team === 'blue' && action === 'pick') {
          const arr = [...prev.bluePicks]
          arr[index] = name
          update.bluePicks = arr
        } else {
          const arr = [...prev.redPicks]
          arr[index] = name
          update.redPicks = arr
        }

        // Build a temporary view of the state after placement for auto-advance
        const nextState = {
          blueBans: update.blueBans ?? prev.blueBans,
          redBans: update.redBans ?? prev.redBans,
          bluePicks: update.bluePicks ?? prev.bluePicks,
          redPicks: update.redPicks ?? prev.redPicks,
        }
        update.activeSlot = findNextEmptySlot(nextState)

        return update
      })
    }
  },

  setActiveSlot: (team: Side, action: DraftAction, index: number) => {
    set({ activeSlot: { team, action, index } })
  },

  setTeam: (side: Side, name: string) => {
    if (side === 'blue') {
      set({ blueTeam: name })
    } else {
      set({ redTeam: name })
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

    set((prev) => {
      const update: Partial<DraftState> = { currentStep: prevStep }

      if (team === 'blue' && action === 'ban') {
        const arr = [...prev.blueBans]; arr[slotIndex] = null; update.blueBans = arr
      } else if (team === 'red' && action === 'ban') {
        const arr = [...prev.redBans]; arr[slotIndex] = null; update.redBans = arr
      } else if (team === 'blue' && action === 'pick') {
        const champion = prev.bluePicks[slotIndex]
        const arr = [...prev.bluePicks]; arr[slotIndex] = null; update.bluePicks = arr
        // Clear any role assignment for this champion
        if (champion) {
          const roles = { ...prev.blueRoles }
          for (const role of ROLES) {
            if (roles[role] === champion) roles[role] = null
          }
          update.blueRoles = roles
        }
      } else {
        const champion = prev.redPicks[slotIndex]
        const arr = [...prev.redPicks]; arr[slotIndex] = null; update.redPicks = arr
        // Clear any role assignment for this champion
        if (champion) {
          const roles = { ...prev.redRoles }
          for (const role of ROLES) {
            if (roles[role] === champion) roles[role] = null
          }
          update.redRoles = roles
        }
      }

      return update
    })
  },

  resetDraft: () => {
    set({
      blueBans: createEmptySlots(),
      redBans: createEmptySlots(),
      bluePicks: createEmptySlots(),
      redPicks: createEmptySlots(),
      blueRoles: { ...EMPTY_ROLES },
      redRoles: { ...EMPTY_ROLES },
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
      mode: 'live',
      currentStep: 0,
      activeSlot: null,
      seriesFormat: 'single',
      seriesScore: { blue: 0, red: 0 },
      currentGame: 1,
    })
  },
}))
