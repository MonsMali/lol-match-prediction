import { create } from 'zustand'
import type { Side, DraftAction, DraftMode, SeriesFormat, Role, DraftStep } from '../types'

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

  // Actions
  selectChampion: (name: string) => void
  setActiveSlot: (team: Side, action: DraftAction, index: number) => void
  setTeam: (side: Side, name: string) => void
  setRole: (side: Side, role: Role, champion: string) => void
  setMode: (mode: DraftMode) => void
  setSeriesFormat: (format: SeriesFormat) => void
  recordGameResult: (winner: Side) => void
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

  // Actions
  selectChampion: (name: string) => {
    const state = get()

    if (state.mode === 'live') {
      if (state.currentStep >= DRAFT_SEQUENCE.length) return
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
      // Bulk mode: place into activeSlot
      const slot = state.activeSlot
      if (!slot) return

      set((prev) => {
        const { team, action, index } = slot
        const update: Partial<DraftState> = { activeSlot: null }

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

  setRole: (side: Side, role: Role, champion: string) => {
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
    set((prev) => {
      const newScore = { ...prev.seriesScore }
      newScore[winner] += 1
      return {
        seriesScore: newScore,
        currentGame: prev.currentGame + 1,
      }
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
