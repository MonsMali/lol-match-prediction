# Phase 3: React Draft Board - Context

**Gathered:** 2026-02-24
**Status:** Ready for planning

<domain>
## Phase Boundary

Build the complete frontend draft simulator: a React SPA where users pick/ban champions for two professional teams, assign roles, and see win probability. Includes step-by-step live draft mode, bulk entry mode, and best-of-series tracking. All against the Phase 2 API contract.

</domain>

<decisions>
## Implementation Decisions

### Visual theme and atmosphere
- Dark theme with dark background (#0a0a0f range), blue team accents (#0088ff), red team accents (#ff3333), white/light gray text
- Custom modern web app design — LoL-themed but NOT a broadcast overlay replica
- Visual inspiration: the LoL game client (hextech-styled UI with dark panels, gold accents, clean typography)
- Champion portraits: standard DDragon square icons (not splash art crops)

### Draft board arrangement
- Left/right split layout: blue team on the left, red team on the right
- Bans above picks within each team's panel — small ban icons in a row, larger pick icons below with role labels
- Win probability display centered between the two team panels (vertical split bar with percentages)
- Champion grid below the team panels with search bar

### Champion selection flow
- LoL champion select style: search bar at top of champion grid, full grid of square icons below, typing filters in real-time, clicking assigns to active slot
- All ~160+ champions visible at once in a scrollable grid area (no pagination)
- Already-picked or already-banned champions shown as greyed out/dimmed in the grid (not hidden) — grid layout stays stable
- Role assignment happens after all 5 picks are made per team, not during each pick

### Mode switching and defaults
- Step-by-step (live draft) mode is the default on page load
- Toggle tabs at top of draft board: "Live Draft" / "Quick Entry" — always visible, one click to switch
- Switching modes mid-draft preserves current draft state (picks/bans carry over)
- Best-of-series is a secondary toggle (dropdown or small control) — not prominent unless activated, defaults to single game

### Claude's Discretion
- Viewport strategy (single-screen vs scrollable) — pick what fits content best
- Loading states and skeleton screens
- Exact spacing, typography scale, and responsive breakpoints
- Error state handling (API failures, missing champions)
- Champion grid sort order (alphabetical, by role, etc.)
- Animations and transitions

</decisions>

<specifics>
## Specific Ideas

- "I think the best reference would be the LoL client" — hextech-styled dark UI with clean panels, gold accents
- Champion select should feel like the in-game champion select experience: search + browse grid, click to assign
- The win probability split bar in the center should be the visual payoff after completing a draft

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 03-react-draft-board*
*Context gathered: 2026-02-24*
