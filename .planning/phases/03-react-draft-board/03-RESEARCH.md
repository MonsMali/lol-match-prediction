# Phase 3: React Draft Board - Research

**Researched:** 2026-02-24
**Domain:** React SPA with Vite, TypeScript, Tailwind CSS, Zustand, TanStack Query
**Confidence:** HIGH

## Summary

Phase 3 builds a React single-page application that serves as the complete frontend for the LoL Draft Predictor. The app consumes three Phase 2 API endpoints (`GET /api/champions`, `GET /api/teams`, `POST /api/predict`) and presents a two-sided draft board where users pick/ban champions, assign roles, select teams, and see win probabilities. Two draft modes are required: step-by-step (following the real 20-step professional draft order) and bulk entry (fill everything at once). A best-of-series tracker allows multi-game series scoring.

The frontend stack is React 19 + Vite 7 + TypeScript + Tailwind CSS v4, with Zustand for client-side draft state and TanStack Query for server data fetching. Champion portraits load directly from Riot's Data Dragon CDN (URLs are pre-constructed by the backend). The app will be built as a static SPA that Phase 4 serves via FastAPI's StaticFiles middleware -- there is no frontend server in production.

The core complexity lies in the draft state machine: tracking 20 sequential steps across two phases of bans and two phases of picks, managing which team acts at each step, preventing duplicate champion selections, and supporting mode switching (live vs bulk) without losing state. Zustand is the right tool for this -- it handles complex client state with minimal boilerplate and no provider wrapping.

**Primary recommendation:** Use Zustand for all draft state (picks, bans, active slot, mode, series score) and TanStack Query exclusively for API data (champions list, teams list, prediction results). Do not mix server state into the Zustand store.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Dark theme with dark background (#0a0a0f range), blue team accents (#0088ff), red team accents (#ff3333), white/light gray text
- Custom modern web app design -- LoL-themed but NOT a broadcast overlay replica
- Visual inspiration: the LoL game client (hextech-styled UI with dark panels, gold accents, clean typography)
- Champion portraits: standard DDragon square icons (not splash art crops)
- Left/right split layout: blue team on the left, red team on the right
- Bans above picks within each team's panel -- small ban icons in a row, larger pick icons below with role labels
- Win probability display centered between the two team panels (vertical split bar with percentages)
- Champion grid below the team panels with search bar
- LoL champion select style: search bar at top of champion grid, full grid of square icons below, typing filters in real-time, clicking assigns to active slot
- All ~160+ champions visible at once in a scrollable grid area (no pagination)
- Already-picked or already-banned champions shown as greyed out/dimmed in the grid (not hidden) -- grid layout stays stable
- Role assignment happens after all 5 picks are made per team, not during each pick
- Step-by-step (live draft) mode is the default on page load
- Toggle tabs at top of draft board: "Live Draft" / "Quick Entry" -- always visible, one click to switch
- Switching modes mid-draft preserves current draft state (picks/bans carry over)
- Best-of-series is a secondary toggle (dropdown or small control) -- not prominent unless activated, defaults to single game

### Claude's Discretion
- Viewport strategy (single-screen vs scrollable) -- pick what fits content best
- Loading states and skeleton screens
- Exact spacing, typography scale, and responsive breakpoints
- Error state handling (API failures, missing champions)
- Champion grid sort order (alphabetical, by role, etc.)
- Animations and transitions

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| DRAFT-01 | Champion portrait grid displays all champions using Riot Data Dragon CDN images | TanStack Query fetches `GET /api/champions` (returns name + image_url per champion). Images load directly from DDragon CDN. Grid renders ~167 48x48 icons in a CSS Grid layout. |
| DRAFT-02 | Champion search/filter allows finding champions by name with fuzzy matching | Client-side string filtering on the champions array from TanStack Query cache. `String.includes()` for simple substring matching is sufficient; the backend already validated names. |
| DRAFT-03 | Two-sided draft board shows blue team and red team with 5 pick slots and 5 ban slots each | Zustand store holds `bluePicks[5]`, `redPicks[5]`, `blueBans[5]`, `redBans[5]`. Layout uses CSS Grid or Flexbox for left/right split with centered probability bar. |
| DRAFT-04 | Team selection interface for major professional leagues (LCK, LEC, LCS, LPL) | TanStack Query fetches `GET /api/teams` (returns `{league: string[]}` dict). Dropdown or searchable select grouped by league. Zustand stores selected `blueTeam` and `redTeam`. |
| DRAFT-05 | Role assignment UI maps picked champions to positions (Top, Jungle, Mid, Bot, Support) | After 5 picks are placed per team, show a role assignment overlay/section. Drag-drop or dropdown per pick slot. Zustand stores role mapping as `{top, jungle, mid, bot, support}` matching the API schema. |
| DRAFT-06 | Win probability display shows percentage for each team after draft completes | TanStack Query `useMutation` posts to `POST /api/predict` with the full draft payload. Response is `{blue_win_probability, red_win_probability}`. Render as a horizontal split bar. |
| DRAFT-07 | LoL-themed visual design with champion portraits, team colors, and draft board | Tailwind CSS v4 with custom theme colors (#0a0a0f background, #0088ff blue, #ff3333 red, gold accents). Dark panels, clean borders, hextech-inspired styling. |
| LIVE-01 | Step-by-step draft mode follows the real professional draft order | Zustand state machine with `currentStep` index into the 20-step draft sequence array. Each step defines `{team, type, slotIndex}`. Advancing increments the step. |
| LIVE-02 | User enters one champion at a time as it happens on broadcast | Champion grid click assigns to `activeSlot` (determined by `currentStep`). Only one slot is active at a time in live mode. |
| LIVE-03 | Draft board updates visually with each pick/ban selection | Zustand state change triggers React re-render. Active slot highlighted, filled slots show champion portrait, upcoming slots dimmed. |
| BULK-01 | Bulk entry mode allows entering all 10 bans and 10 picks at once | In bulk mode, all 20 slots are clickable (no enforced order). User clicks a slot to make it active, then clicks a champion. Zustand tracks which slots are filled. |
| BULK-02 | Quick prediction without following step-by-step draft sequence | Submit button enabled when all 20 slots + team selections + role assignments are complete. Calls same `POST /api/predict` endpoint. |
| BOS-01 | Best-of-series toggle supports BO3 and BO5 series formats | Zustand stores `seriesFormat: 'single' | 'bo3' | 'bo5'` and `seriesScore: {blue: number, red: number}`. Toggle control in UI header area. |
| BOS-02 | Series score tracking across multiple games | After each game prediction, user can record the winner. Score displayed prominently. Series winner declared when threshold reached (2 for BO3, 3 for BO5). |
| BOS-03 | User can input a new draft for each game in the series | "Next Game" button resets draft state (picks, bans, roles) but preserves team selections and series score. Zustand `resetDraft()` action. |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| react | 19.2.x | UI framework | Latest stable; v19 is current major with Suspense, transitions |
| react-dom | 19.2.x | DOM renderer | Paired with React 19 |
| vite | 7.3.x | Build tool + dev server | De facto standard for React; HMR, TypeScript, fast builds |
| typescript | 5.x | Type safety | Vite react-ts template includes it; catches draft state bugs at compile time |
| tailwindcss | 4.x | Utility-first CSS | v4 has first-party Vite plugin, zero-config content detection, 5x faster builds |
| @tailwindcss/vite | 4.x | Tailwind Vite integration | Official plugin; replaces PostCSS setup from v3 |
| zustand | 5.0.x | Client draft state | Minimal boilerplate, no providers, TypeScript-first, perfect for complex form/wizard state |
| @tanstack/react-query | 5.90.x | Server state (API calls) | Caching, loading states, error handling, mutation support for predict endpoint |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| @tanstack/react-query-devtools | 5.x | Query debugging | Development only; inspect cache, refetch behavior |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Zustand | React Context + useReducer | Context causes full-subtree re-renders; Zustand allows selective subscriptions. Draft state is complex enough to benefit from Zustand. |
| TanStack Query | SWR | TanStack Query has better mutation support (needed for POST /api/predict) and devtools. SWR is lighter but less featureful. |
| Tailwind CSS | CSS Modules | Tailwind is faster for prototyping dark themed UIs with consistent design tokens. CSS Modules require more manual design system work. |

**Installation:**
```bash
npm create vite@latest frontend -- --template react-ts
cd frontend
npm install zustand @tanstack/react-query tailwindcss @tailwindcss/vite
npm install -D @tanstack/react-query-devtools
```

**Tailwind v4 Vite Config:**
```typescript
// vite.config.ts
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
})
```

**Tailwind v4 CSS Entry:**
```css
/* src/index.css */
@import "tailwindcss";
```

No `tailwind.config.js` needed in v4 -- configuration is done via CSS `@theme` directive.

## Architecture Patterns

### Recommended Project Structure
```
frontend/
├── index.html
├── package.json
├── tsconfig.json
├── vite.config.ts
├── src/
│   ├── main.tsx              # React root + QueryClientProvider
│   ├── index.css             # Tailwind import + custom theme
│   ├── App.tsx               # Main layout (single page)
│   ├── api/
│   │   ├── client.ts         # fetch wrapper with base URL
│   │   ├── champions.ts      # useChampions() query hook
│   │   ├── teams.ts          # useTeams() query hook
│   │   └── predict.ts        # usePrediction() mutation hook
│   ├── store/
│   │   └── draftStore.ts     # Zustand store: draft state, mode, series
│   ├── components/
│   │   ├── DraftBoard.tsx    # Main two-sided layout container
│   │   ├── TeamPanel.tsx     # One side (picks + bans + team selector)
│   │   ├── BanRow.tsx        # Row of 5 small ban slots
│   │   ├── PickSlot.tsx      # Single pick slot with role label
│   │   ├── ChampionGrid.tsx  # Searchable champion icon grid
│   │   ├── ChampionIcon.tsx  # Single champion portrait (48x48)
│   │   ├── TeamSelector.tsx  # League-grouped team dropdown
│   │   ├── RoleAssignment.tsx # Post-pick role assignment UI
│   │   ├── WinProbability.tsx # Centered split bar display
│   │   ├── ModeToggle.tsx    # Live Draft / Quick Entry tabs
│   │   ├── SeriesTracker.tsx # BO3/BO5 score and controls
│   │   └── DraftControls.tsx # Submit, Reset, Next Game buttons
│   └── types/
│       └── index.ts          # Shared TypeScript types
└── public/
    └── (empty -- all assets from CDN)
```

### Pattern 1: Zustand Draft Store with State Machine
**What:** Single Zustand store managing all draft state including a step counter for live mode.
**When to use:** Always -- this is the central state for the entire application.
**Example:**
```typescript
import { create } from 'zustand'

type Side = 'blue' | 'red'
type DraftAction = 'ban' | 'pick'
type DraftMode = 'live' | 'bulk'
type SeriesFormat = 'single' | 'bo3' | 'bo5'
type Role = 'top' | 'jungle' | 'mid' | 'bot' | 'support'

interface DraftStep {
  team: Side
  action: DraftAction
  slotIndex: number  // 0-4 for the team's ban/pick array
}

// The 20-step professional draft sequence
const DRAFT_SEQUENCE: DraftStep[] = [
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

interface DraftState {
  // Draft data
  blueBans: (string | null)[]   // length 5
  redBans: (string | null)[]    // length 5
  bluePicks: (string | null)[]  // length 5
  redPicks: (string | null)[]   // length 5
  blueTeam: string | null
  redTeam: string | null
  blueRoles: Record<Role, string | null>
  redRoles: Record<Role, string | null>

  // Mode and state machine
  mode: DraftMode
  currentStep: number  // index into DRAFT_SEQUENCE (live mode only)

  // Series
  seriesFormat: SeriesFormat
  seriesScore: { blue: number; red: number }
  currentGame: number

  // Computed/derived
  usedChampions: () => Set<string>

  // Actions
  selectChampion: (championName: string) => void
  setActiveSlot: (team: Side, action: DraftAction, index: number) => void
  setTeam: (side: Side, teamName: string) => void
  setRole: (side: Side, role: Role, championName: string) => void
  setMode: (mode: DraftMode) => void
  setSeriesFormat: (format: SeriesFormat) => void
  recordGameResult: (winner: Side) => void
  resetDraft: () => void
  resetAll: () => void
}
```

### Pattern 2: TanStack Query for API Data
**What:** Separate query hooks for each API endpoint. Champions and teams are fetched once and cached. Prediction is a mutation.
**When to use:** All API interactions.
**Example:**
```typescript
import { useQuery, useMutation, QueryClient } from '@tanstack/react-query'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

// Champions -- fetched once, cached indefinitely (data doesn't change at runtime)
export function useChampions() {
  return useQuery({
    queryKey: ['champions'],
    queryFn: async () => {
      const res = await fetch(`${API_BASE}/api/champions`)
      if (!res.ok) throw new Error('Failed to fetch champions')
      const data = await res.json()
      return data.champions as { name: string; key: string; image_url: string }[]
    },
    staleTime: Infinity,  // Never refetch -- champion list is static
  })
}

// Teams -- fetched once, cached indefinitely
export function useTeams() {
  return useQuery({
    queryKey: ['teams'],
    queryFn: async () => {
      const res = await fetch(`${API_BASE}/api/teams`)
      if (!res.ok) throw new Error('Failed to fetch teams')
      const data = await res.json()
      return data.teams as Record<string, string[]>
    },
    staleTime: Infinity,
  })
}

// Prediction -- mutation (POST), not a query
export function usePrediction() {
  return useMutation({
    mutationFn: async (draft: PredictRequest) => {
      const res = await fetch(`${API_BASE}/api/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(draft),
      })
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Prediction failed')
      }
      return res.json() as Promise<{ blue_win_probability: number; red_win_probability: number }>
    },
  })
}
```

### Pattern 3: Champion Grid with Virtual Stability
**What:** Render all ~167 champion icons in a CSS Grid. Greyed-out champions stay in place (never removed from DOM). Search filters via CSS `display: none` or conditional opacity.
**When to use:** Champion selection grid.
**Example:**
```typescript
function ChampionGrid({ searchQuery, usedChampions, onSelect }: Props) {
  const { data: champions, isPending } = useChampions()

  if (isPending) return <GridSkeleton />

  const filtered = champions.filter(c =>
    c.name.toLowerCase().includes(searchQuery.toLowerCase())
  )

  return (
    <div className="grid grid-cols-10 gap-1">
      {filtered.map(champ => (
        <ChampionIcon
          key={champ.key}
          champion={champ}
          disabled={usedChampions.has(champ.name)}
          onClick={() => onSelect(champ.name)}
        />
      ))}
    </div>
  )
}
```

### Pattern 4: API Request Schema Construction
**What:** Convert Zustand draft state into the PredictRequest shape expected by the API.
**When to use:** When submitting a completed draft for prediction.
**Example:**
```typescript
// The API expects this shape (from api/schemas.py):
interface PredictRequest {
  blue_team: string
  red_team: string
  blue_picks: { top: string; jungle: string; mid: string; bot: string; support: string }
  red_picks: { top: string; jungle: string; mid: string; bot: string; support: string }
  blue_bans: string[]  // exactly 5
  red_bans: string[]   // exactly 5
  patch?: string | null
}

function buildPredictRequest(state: DraftState): PredictRequest {
  return {
    blue_team: state.blueTeam!,
    red_team: state.redTeam!,
    blue_picks: state.blueRoles as Required<Record<Role, string>>,
    red_picks: state.redRoles as Required<Record<Role, string>>,
    blue_bans: state.blueBans.filter(Boolean) as string[],
    red_bans: state.redBans.filter(Boolean) as string[],
  }
}
```

### Anti-Patterns to Avoid
- **Storing API data in Zustand:** Champions and teams come from the server. Use TanStack Query for fetching and caching. Zustand is for client-only draft state.
- **Re-rendering entire grid on each selection:** The champion grid has ~167 items. Use `React.memo` on `ChampionIcon` and pass stable callbacks. The `usedChampions` set should be derived via Zustand selector, not recomputed in the parent.
- **Building custom dropdown from scratch:** For the team selector, a styled `<select>` with `<optgroup>` per league works well enough. If fancier UX is needed, use Headless UI's Listbox or Radix Select -- but do not hand-roll focus management and keyboard navigation.
- **Fetching DDragon images through the backend:** Images load directly from `ddragon.leagueoflegends.com` CDN. The backend provides the full URL string. No proxy needed.
- **Mixing draft modes in state logic:** Live mode and bulk mode share the same pick/ban arrays but differ only in how `selectChampion` determines the target slot. In live mode, the slot is auto-determined by `currentStep`. In bulk mode, the user explicitly clicks a slot first.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Client state management | Custom Context + Reducer state machine | Zustand store | Selective re-renders, no provider nesting, simpler TypeScript inference |
| API data caching | Manual fetch + useState + useEffect | TanStack Query `useQuery` | Handles loading, error, cache invalidation, deduplication, devtools |
| POST mutation handling | Manual fetch + loading state + error state | TanStack Query `useMutation` | Built-in `isPending`, `isError`, `isSuccess` states, retry logic |
| CSS design system | Custom CSS variables + utility classes | Tailwind CSS v4 `@theme` | Consistent spacing, colors, responsive utilities out of the box |
| Image lazy loading | Custom IntersectionObserver for champion grid | Native `loading="lazy"` attribute | Browser-native, zero JS, sufficient for a grid of ~167 small images |
| Keyboard navigation in dropdowns | Custom key event handlers | HTML `<select>` / `<optgroup>` or Headless UI Listbox | Accessibility compliance is complex; native elements handle it |

**Key insight:** The frontend is a single-page form/wizard with a specialized layout. There is no routing, no authentication, no complex data relationships. Keep the architecture flat: one Zustand store, three TanStack Query hooks, and presentational components.

## Common Pitfalls

### Pitfall 1: DDragon Image 404s for Mismatched Champion Names
**What goes wrong:** Some champion portraits fail to load, showing broken image icons.
**Why it happens:** The backend constructs DDragon URLs using a mapping dict that may miss edge cases (champions with apostrophes, spaces, or special characters).
**How to avoid:** The backend already handles this (Phase 2 built the mapping). The frontend should add an `onError` handler on `<img>` tags that falls back to a placeholder icon. This is defensive -- it should never trigger if the backend mapping is complete.
**Warning signs:** Broken images in the champion grid.

### Pitfall 2: Draft State Desync Between Live and Bulk Modes
**What goes wrong:** Switching from live to bulk mode (or back) causes picks to appear in wrong slots or the step counter to be inconsistent with filled slots.
**Why it happens:** Live mode auto-advances `currentStep` but bulk mode allows arbitrary slot filling. When switching back to live, `currentStep` may not reflect which slots are actually filled.
**How to avoid:** When switching from bulk to live, recalculate `currentStep` by scanning the DRAFT_SEQUENCE and finding the first unfilled slot. When switching from live to bulk, no recalculation needed (bulk mode ignores `currentStep`).
**Warning signs:** Clicking a champion in live mode fills the wrong slot after mode switch.

### Pitfall 3: Stale Closure in Champion Selection Handler
**What goes wrong:** Clicking a champion always fills the first slot, or the `usedChampions` set is stale.
**Why it happens:** If the `onSelect` callback captures a stale Zustand state snapshot (common when passing callbacks as props without memoization), selections use outdated state.
**How to avoid:** Use Zustand's `getState()` inside the action (not a React state snapshot) or ensure the callback reads from the store directly. Zustand actions defined inside `create()` always have access to current state via `set` and `get`.
**Warning signs:** Duplicate champions appearing in slots, or champion grid not greying out after selection.

### Pitfall 4: Role Assignment Data Shape Mismatch
**What goes wrong:** API returns 422 even though all picks are filled.
**Why it happens:** The API expects `blue_picks` as `{top: "Ahri", jungle: "LeeSin", ...}` (role-keyed object) but the frontend sends an array of champion names or wrong field names.
**How to avoid:** Verify the PredictRequest shape against `api/schemas.py` (already documented above). The Zustand store should maintain role assignments as a `Record<Role, string | null>` that maps directly to the API schema's `TeamDraft` type.
**Warning signs:** 422 errors from the predict endpoint with validation details about missing fields.

### Pitfall 5: CORS Errors During Development
**What goes wrong:** Fetch requests from Vite dev server (port 5173) to FastAPI (port 8000) are blocked.
**Why it happens:** CORS middleware not configured, or the frontend uses the wrong base URL.
**How to avoid:** Phase 2 already configured CORS for `http://localhost:5173`. Use `VITE_API_URL` environment variable (defaults to `http://localhost:8000`). In production, CORS is irrelevant since the SPA is served from the same origin.
**Warning signs:** Browser console shows "Access-Control-Allow-Origin" errors.

### Pitfall 6: Champion Grid Performance with 167 Items
**What goes wrong:** The grid feels sluggish when typing in the search box or selecting champions.
**Why it happens:** Each keystroke re-renders 167 `ChampionIcon` components if not memoized.
**How to avoid:** Use `React.memo` on `ChampionIcon`. Derive `usedChampions` as a stable Set from Zustand (it only changes when a champion is added/removed, not on every render). Use `useDeferredValue` for the search input if needed.
**Warning signs:** Typing lag in the search input, especially on lower-end machines.

## Code Examples

### Tailwind v4 Custom Theme (CSS-based)
```css
/* src/index.css */
@import "tailwindcss";

@theme {
  --color-background: #0a0a0f;
  --color-panel: #1a1a2e;
  --color-panel-light: #252540;
  --color-blue-team: #0088ff;
  --color-red-team: #ff3333;
  --color-gold: #c89b3c;
  --color-gold-light: #f0e6d2;
  --color-text-primary: #f0f0f0;
  --color-text-secondary: #8888aa;
  --color-disabled: #444466;
}
```

### QueryClient Setup in main.tsx
```typescript
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import App from './App'
import './index.css'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 2,
      staleTime: 5 * 60 * 1000,  // 5 minutes default
    },
  },
})

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>
  </StrictMode>
)
```

### Win Probability Split Bar
```typescript
function WinProbability({ blue, red }: { blue: number; red: number }) {
  const bluePercent = Math.round(blue * 100)
  const redPercent = Math.round(red * 100)

  return (
    <div className="flex flex-col items-center gap-2">
      <div className="flex w-full h-8 rounded overflow-hidden">
        <div
          className="bg-blue-team flex items-center justify-center text-sm font-bold"
          style={{ width: `${bluePercent}%` }}
        >
          {bluePercent}%
        </div>
        <div
          className="bg-red-team flex items-center justify-center text-sm font-bold"
          style={{ width: `${redPercent}%` }}
        >
          {redPercent}%
        </div>
      </div>
    </div>
  )
}
```

### Professional Draft Sequence (Complete 20 Steps)
```typescript
// Verified from multiple sources including Riot's official tournament rules.
// 2026 "First Selection" system changes side choice logic but NOT the
// ban/pick sequence itself. The 20-step order remains unchanged.

const DRAFT_SEQUENCE = [
  // === Ban Phase 1 (6 steps) ===
  { step: 1,  team: 'blue', action: 'ban',  slotIndex: 0 },  // BB1
  { step: 2,  team: 'red',  action: 'ban',  slotIndex: 0 },  // RB1
  { step: 3,  team: 'blue', action: 'ban',  slotIndex: 1 },  // BB2
  { step: 4,  team: 'red',  action: 'ban',  slotIndex: 1 },  // RB2
  { step: 5,  team: 'blue', action: 'ban',  slotIndex: 2 },  // BB3
  { step: 6,  team: 'red',  action: 'ban',  slotIndex: 2 },  // RB3
  // === Pick Phase 1 (6 steps) ===
  { step: 7,  team: 'blue', action: 'pick', slotIndex: 0 },  // BP1
  { step: 8,  team: 'red',  action: 'pick', slotIndex: 0 },  // RP1
  { step: 9,  team: 'red',  action: 'pick', slotIndex: 1 },  // RP2
  { step: 10, team: 'blue', action: 'pick', slotIndex: 1 },  // BP2
  { step: 11, team: 'blue', action: 'pick', slotIndex: 2 },  // BP3
  { step: 12, team: 'red',  action: 'pick', slotIndex: 2 },  // RP3
  // === Ban Phase 2 (4 steps) ===
  { step: 13, team: 'red',  action: 'ban',  slotIndex: 3 },  // RB4
  { step: 14, team: 'blue', action: 'ban',  slotIndex: 3 },  // BB4
  { step: 15, team: 'red',  action: 'ban',  slotIndex: 4 },  // RB5
  { step: 16, team: 'blue', action: 'ban',  slotIndex: 4 },  // BB5
  // === Pick Phase 2 (4 steps) ===
  { step: 17, team: 'red',  action: 'pick', slotIndex: 3 },  // RP4
  { step: 18, team: 'blue', action: 'pick', slotIndex: 3 },  // BP4
  { step: 19, team: 'blue', action: 'pick', slotIndex: 4 },  // BP5
  { step: 20, team: 'red',  action: 'pick', slotIndex: 4 },  // RP5
] as const
```

### API Contract Reference (from Phase 2)
```typescript
// GET /api/champions response:
{ champions: Array<{ name: string; key: string; image_url: string }> }
// ~167 champions, image_url is full DDragon CDN URL

// GET /api/teams response:
{ teams: Record<string, string[]> }
// Keys: "LCK", "LEC", "LCS", "LPL", "Other"
// Values: canonical team names matching training data

// POST /api/predict request:
{
  blue_team: string,
  red_team: string,
  blue_picks: { top: string, jungle: string, mid: string, bot: string, support: string },
  red_picks: { top: string, jungle: string, mid: string, bot: string, support: string },
  blue_bans: string[],  // exactly 5
  red_bans: string[],   // exactly 5
  patch?: string | null
}

// POST /api/predict response:
{ blue_win_probability: number, red_win_probability: number }
// Both are floats in [0, 1], summing to 1.0

// GET /health response:
{ status: "ready" | "loading", model_name?: string, ... }
// 503 status code when model is loading (for predict/champions/teams endpoints)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Create React App | Vite 7 with `react-ts` template | 2023+ (CRA deprecated) | Vite is 40x faster, actively maintained, ecosystem default |
| Tailwind v3 + PostCSS + `tailwind.config.js` | Tailwind v4 + `@tailwindcss/vite` + CSS `@theme` | January 2025 (v4.0 release) | Zero-config content detection, CSS-native configuration, 5x faster builds |
| Redux / Redux Toolkit | Zustand v5 | 2023+ (Zustand matured) | 90% less boilerplate for client state; Redux is overkill for single-page forms |
| React Query v3/v4 | TanStack Query v5 | October 2023 (v5.0 release) | Simplified single-object API, 20% smaller bundle, Suspense support |
| Pydantic v1 schemas | Pydantic v2 schemas in API | 2023 | Frontend types should match v2 field naming (snake_case) |
| LoL draft: side = first pick | LoL 2026 "First Selection" system | January 2026 | Side and draft order are now decoupled. The 20-step ban/pick sequence is unchanged. Our tool lets users assign teams freely, so this has no impact. |

**Deprecated/outdated:**
- Create React App: Officially deprecated, do not use.
- Tailwind v3 `tailwind.config.js`: Still works but v4 uses CSS `@theme` directive instead. No config file needed.
- `@tanstack/react-query` v4 API (separate `queryKey` and `queryFn` arguments): v5 uses single object parameter always.

## Open Questions

1. **Vite Proxy for Development**
   - What we know: The Vite dev server runs on port 5173, FastAPI on port 8000. CORS is configured in Phase 2 for `localhost:5173`.
   - What's unclear: Whether to use Vite's built-in proxy (`server.proxy` in vite.config.ts) to avoid CORS entirely during development, or rely on the CORS middleware.
   - Recommendation: Use Vite proxy in development (avoids CORS entirely, simpler). Configure `server.proxy: { '/api': 'http://localhost:8000', '/health': 'http://localhost:8000' }` in vite.config.ts. This means the frontend fetches from its own origin and Vite forwards to the backend. In production, no proxy is needed since both are served from the same origin.

2. **Role Assignment UX for Bulk Mode**
   - What we know: Role assignment happens after all 5 picks are made per team (CONTEXT.md decision). In live mode, this is natural -- after step 20, show the role assignment. In bulk mode, the user fills slots in any order.
   - What's unclear: Whether role assignment should appear per-team (show after that team's 5 picks are filled) or after all 10 picks are filled.
   - Recommendation: Show role assignment for a team as soon as that team's 5 pick slots are filled, regardless of mode. This gives earlier feedback in bulk mode.

3. **Responsive Design Breakpoints**
   - What we know: The draft board is a complex two-sided layout with a champion grid. On small screens, a left/right split becomes very cramped.
   - What's unclear: Whether mobile support is important for this thesis project.
   - Recommendation: Design for desktop-first (1280px+). Add a minimum viewport warning for screens below 768px rather than investing in full responsive redesign. The draft board's information density is fundamentally desktop-oriented.

## Sources

### Primary (HIGH confidence)
- [Vite Official Docs - Getting Started](https://vite.dev/guide/) - Verified `npm create vite@latest` command, react-ts template, version 7.3.x
- [Tailwind CSS v4 Announcement](https://tailwindcss.com/blog/tailwindcss-v4) - Verified Vite plugin, CSS @theme, zero-config content detection
- [Tailwind CSS v4 Installation](https://tailwindcss.com/docs) - Verified `@import "tailwindcss"` setup
- [Zustand Official Docs](https://zustand.docs.pmnd.rs/) - Verified TypeScript `create<T>()(...)` pattern, `useShallow`, v5 API
- [TanStack Query v5 Docs](https://tanstack.com/query/v5/docs/framework/react/overview) - Verified `useQuery`, `useMutation`, single-object API, TypeScript inference
- [React 19 Release](https://react.dev/blog/2024/12/05/react-19) - Verified React 19.2.x is current stable
- [Phase 2 API Implementation](api/schemas.py, api/champion_mapping.py, api/team_data.py) - Exact API contract shapes

### Secondary (MEDIUM confidence)
- [League of Legends Wiki - Team Drafting](https://leagueoflegends.fandom.com/wiki/Team_drafting) - Professional draft order verified (20-step sequence)
- [LoL Esports 2026 First Selection](https://esportsinsider.com/2026/01/league-of-legends-side-selection-draft-order-revamp) - Confirmed draft sequence unchanged; only side/order decoupled
- [npm zustand](https://www.npmjs.com/package/zustand) - Version 5.0.11 confirmed as latest
- [npm @tanstack/react-query](https://www.npmjs.com/package/@tanstack/react-query) - Version 5.90.x confirmed as latest

### Tertiary (LOW confidence)
- Vite proxy configuration for development: Based on training knowledge of Vite's `server.proxy` feature. Should be verified against Vite 7 docs during implementation.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All library versions verified via npm/official docs, all are current stable releases
- Architecture: HIGH - Zustand + TanStack Query pattern is well-documented and widely adopted; draft state machine is a straightforward finite state machine
- Pitfalls: HIGH - DDragon mapping inherited from Phase 2 (tested); CORS configured in Phase 2; draft sequence verified against multiple sources
- API contract: HIGH - Exact schemas read from Phase 2 source code (api/schemas.py)

**Research date:** 2026-02-24
**Valid until:** 2026-03-24 (stable stack, pinned DDragon version, no breaking changes expected)
