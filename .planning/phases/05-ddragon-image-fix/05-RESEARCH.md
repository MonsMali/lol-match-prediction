# Phase 5: DDragon Image URL Fix - Research

**Researched:** 2026-02-26
**Domain:** React component image URL handling, DDragon CDN integration
**Confidence:** HIGH

## Summary

Three React components (PickSlot, BanRow, RoleAssignment) construct DDragon image URLs by interpolating the champion's display name directly into the URL path. This breaks for 12 champions whose DDragon asset ID differs from their display name (e.g., `Wukong` needs `MonkeyKing.png`, `Kai'Sa` needs `Kaisa.png`). The API already returns a correct `image_url` for every champion via `get_ddragon_url()` in `api/champion_mapping.py`, and the `ChampionIcon` component already uses it correctly.

The core problem is a data flow gap: the draft store stores only champion **names** (strings), but the broken components need **image URLs**. The fix requires either (a) making the image URL available where these components render, or (b) storing image URLs alongside names in the store. The user also wants shimmer loading states, team-colored SVG fallback silhouettes, and a fade-in transition on load.

**Primary recommendation:** Create a champion lookup map (name -> ChampionInfo) from the `useChampions` query data, pass `image_url` to PickSlot/BanRow/RoleAssignment via props or a shared hook, and add a reusable `ChampionImage` wrapper component that handles loading shimmer, error fallback, and fade-in.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Show a generic champion silhouette when an image fails to load
- Silhouette rendered as an inline SVG (no external dependency, ships with the app)
- Silhouette tinted in team color (blue/red) for visual context
- Same fallback treatment across all slot types (picks, bans, role assignment) -- bans already have their own visual treatment so the fallback does not need to differ
- On image error, show fallback immediately with no retry -- DDragon is a static CDN, so transient failures are unlikely to resolve on retry
- Show a shimmer/skeleton animation while champion portraits load
- Champion name appears immediately on selection (instant feedback); only the portrait area shows shimmer
- Portrait fades in with a quick opacity transition (~150-200ms) when loaded
- Shimmer + fade-in applies consistently to all slot types: PickSlot, BanRow, and RoleAssignment

### Claude's Discretion
- Exact shimmer animation CSS implementation
- SVG silhouette design details
- Exact fade-in timing within the ~150-200ms range

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| DRAFT-07 | LoL-themed visual design with champion portraits, team colors, and draft board resembling pro broadcast | Shimmer loading, team-colored fallback silhouettes, and fade-in transitions enhance the LoL broadcast aesthetic. Fixing broken portraits for 12 champions directly supports this requirement. |
| API-08 | Data Dragon champion name mapping handles mismatches (Wukong=MonkeyKing, Nunu, etc.) | The backend mapping already works correctly via `api/champion_mapping.py`. This phase ensures the frontend actually *uses* those correct URLs instead of constructing broken ones from display names. |
</phase_requirements>

## Standard Stack

### Core (already in project)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| React | 19.x | UI framework | Already in use |
| Zustand | 5.x | State management (draftStore) | Already in use |
| TanStack Query | 5.x | API data fetching (useChampions) | Already in use |
| Tailwind CSS | 4.x | Styling (via @import "tailwindcss") | Already in use |

### No New Dependencies Needed
This phase requires zero new libraries. All work is CSS animations (Tailwind), inline SVG, and React component refactoring.

## Architecture Patterns

### Current Data Flow (broken)
```
API /api/champions -> useChampions() -> ChampionGrid -> ChampionIcon (uses image_url -- WORKS)
                                                     -> selectChampion(name) -> draftStore (stores name only)

draftStore.bluePicks[i] = "Wukong"  ->  PickSlot constructs URL:
                                        `${DDRAGON_BASE}/Wukong.png`  -- BROKEN (should be MonkeyKing.png)
```

### Required Data Flow (fixed)
```
API /api/champions -> useChampions() -> champion lookup map (name -> ChampionInfo)
                                     -> ChampionGrid -> ChampionIcon (unchanged)
                                     -> selectChampion(name) -> draftStore (name stays as string)

draftStore.bluePicks[i] = "Wukong"  ->  PickSlot receives image_url via lookup
                                        uses `champion.image_url` -- CORRECT
```

### Pattern 1: Champion Lookup Hook
**What:** A custom hook or derived map that converts champion names to full ChampionInfo objects including image_url.
**When to use:** Wherever a component has a champion name from the store but needs the image URL.
**Example:**
```typescript
// useChampionLookup.ts
import { useMemo } from 'react'
import { useChampions } from '../api/champions'
import type { ChampionInfo } from '../types'

export function useChampionLookup(): Map<string, ChampionInfo> {
  const { data: champions } = useChampions()
  return useMemo(() => {
    const map = new Map<string, ChampionInfo>()
    if (champions) {
      for (const c of champions) {
        map.set(c.name, c)
      }
    }
    return map
  }, [champions])
}
```

### Pattern 2: Reusable ChampionImage Component
**What:** A shared image component that handles shimmer, error fallback (team-colored SVG silhouette), and fade-in transition.
**When to use:** In PickSlot, BanRow, and RoleAssignment wherever `<img src={...}>` currently appears.
**Example:**
```typescript
// ChampionImage.tsx
interface ChampionImageProps {
  src: string
  alt: string
  side: 'blue' | 'red'
  className?: string
}

export function ChampionImage({ src, alt, side, className }: ChampionImageProps) {
  const [status, setStatus] = React.useState<'loading' | 'loaded' | 'error'>('loading')

  return (
    <div className={`relative ${className}`}>
      {/* Shimmer while loading */}
      {status === 'loading' && (
        <div className="absolute inset-0 rounded bg-panel-light animate-pulse" />
      )}

      {/* Error fallback: inline SVG silhouette tinted in team color */}
      {status === 'error' && (
        <ChampionSilhouette side={side} />
      )}

      {/* Actual image with fade-in */}
      <img
        src={src}
        alt={alt}
        className={`w-full h-full rounded object-cover transition-opacity duration-150
          ${status === 'loaded' ? 'opacity-100' : 'opacity-0'}`}
        onLoad={() => setStatus('loaded')}
        onError={() => setStatus('error')}
      />
    </div>
  )
}
```

### Pattern 3: Inline SVG Silhouette
**What:** A simple champion silhouette SVG rendered inline, tinted with team color.
**When to use:** Error fallback for all slot types.
**Example:**
```typescript
function ChampionSilhouette({ side }: { side: 'blue' | 'red' }) {
  const fillColor = side === 'blue' ? '#0088ff' : '#ff3333'  // matches --color-blue-team / --color-red-team
  return (
    <svg viewBox="0 0 48 48" className="w-full h-full rounded">
      <rect width="48" height="48" fill="#252540" />
      {/* Simple head-and-shoulders silhouette */}
      <circle cx="24" cy="18" r="8" fill={fillColor} opacity="0.4" />
      <ellipse cx="24" cy="40" rx="14" ry="10" fill={fillColor} opacity="0.4" />
    </svg>
  )
}
```

### Anti-Patterns to Avoid
- **Storing image_url in the draft store:** The store stores champion names as strings for API request construction. Adding image URLs there would bloat the store and complicate serialization. Use a lookup map instead.
- **Re-implementing the DDragon mapping in the frontend:** The backend already handles this correctly. Never duplicate `DDRAGON_ID_MAP` on the frontend.
- **Using onError to retry loading:** The user explicitly decided no retry -- DDragon is a static CDN.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| DDragon URL construction | Frontend URL builder with name-to-ID mapping | API-provided `image_url` from `useChampions()` | Backend already handles all 12 edge cases in `champion_mapping.py` |
| Shimmer animation | Custom CSS keyframes | Tailwind's `animate-pulse` class | Already used in ChampionGrid's loading skeleton |
| Champion name-to-info resolution | Manual fetching per champion | `useMemo` lookup map over `useChampions()` data | Data is already cached with `staleTime: Infinity` |

## Common Pitfalls

### Pitfall 1: Image onLoad fires before React state update is painted
**What goes wrong:** The fade-in transition is skipped because the browser already painted the image before React applies the opacity class change.
**Why it happens:** If the image is in the browser cache, `onLoad` fires synchronously during render.
**How to avoid:** Start with `opacity-0` on the `<img>` and only transition to `opacity-100` after the state updates. The CSS `transition-opacity` will handle the animation even for cached images because the class change still triggers a CSS transition.
**Warning signs:** Champions appear instantly without fade on subsequent visits.

### Pitfall 2: Shimmer flicker for cached images
**What goes wrong:** A brief shimmer flash appears even though the image loads instantly from cache.
**Why it happens:** The initial state is 'loading', and the cached image's `onLoad` triggers on the same frame or next frame.
**How to avoid:** This is acceptable UX for a sub-frame shimmer. The 150ms fade-in masks any flicker. Do not add complexity to detect cache hits.
**Warning signs:** Noticeable flickering on repeat loads.

### Pitfall 3: Props change but stale error/loading state persists
**What goes wrong:** A `ChampionImage` component shows the error fallback for a new champion because the `status` state was not reset when `src` changed.
**Why it happens:** React reuses component instances when keys don't change.
**How to avoid:** Reset the `status` state when the `src` prop changes using a `useEffect` or by keying the component with the champion name.
**Warning signs:** Wrong fallback silhouette showing for a champion that should load fine.

### Pitfall 4: BanRow passes champion names as strings, not objects
**What goes wrong:** BanRow currently receives `bans: (string | null)[]` from the store. It cannot access `image_url` without either (a) changing its props interface to accept image URLs, or (b) using the champion lookup hook internally.
**Why it happens:** The store intentionally stores names only.
**How to avoid:** Either pass the lookup map as a prop from TeamPanel, or have BanRow call `useChampionLookup()` directly. The latter is simpler and avoids prop drilling.

## Code Examples

### Current broken URL construction (all three components)
```typescript
// PickSlot.tsx, BanRow.tsx, RoleAssignment.tsx all have this:
const DDRAGON_BASE = 'https://ddragon.leagueoflegends.com/cdn/14.24.1/img/champion'
// ...
<img src={`${DDRAGON_BASE}/${champion}.png`} />
// Breaks for: Wukong, Kai'Sa, Nunu & Willump, Kha'Zix, Cho'Gath,
//             Vel'Koz, Rek'Sai, Kog'Maw, Bel'Veth, K'Sante, LeBlanc, Renata Glasc
```

### ChampionIcon reference pattern (already working)
```typescript
// ChampionIcon.tsx -- uses image_url from API response
<img
  src={champion.image_url}    // Correct URL from backend
  alt={champion.name}
  onError={() => setHasError(true)}
/>
```

### Fix pattern: lookup in consuming component
```typescript
// In PickSlot (after refactor)
const championLookup = useChampionLookup()
const championInfo = champion ? championLookup.get(champion) : null
const imageUrl = championInfo?.image_url ?? null

// Then use ChampionImage with the resolved URL
{imageUrl && <ChampionImage src={imageUrl} alt={champion} side={side} />}
```

## Affected Files Summary

| File | Current Behavior | Required Change |
|------|-----------------|----------------|
| `frontend/src/components/PickSlot.tsx` | Constructs `DDRAGON_BASE/${champion}.png` | Use `image_url` from lookup; add ChampionImage with shimmer/fallback |
| `frontend/src/components/BanRow.tsx` | Constructs `DDRAGON_BASE/${ban}.png` | Use `image_url` from lookup; add ChampionImage with shimmer/fallback |
| `frontend/src/components/RoleAssignment.tsx` | Constructs `DDRAGON_BASE/${champion}.png` | Use `image_url` from lookup; add ChampionImage with shimmer/fallback |
| `frontend/src/components/ChampionIcon.tsx` | Uses `champion.image_url` correctly | No change needed (reference pattern) |
| `frontend/src/components/ChampionImage.tsx` | Does not exist | **New file**: shared image component with shimmer, fade-in, SVG fallback |
| `frontend/src/api/champions.ts` (or new hook file) | Returns raw array | Add `useChampionLookup()` hook returning `Map<string, ChampionInfo>` |

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Construct DDragon URLs from display names | Use API-provided `image_url` | Phase 2 (backend) | Backend already fixed; frontend still uses old approach in 3 components |

## Open Questions

None. The fix is well-scoped and all technical details are clear from the existing codebase.

## Sources

### Primary (HIGH confidence)
- Direct code inspection: `frontend/src/components/PickSlot.tsx` (lines 12, 40)
- Direct code inspection: `frontend/src/components/BanRow.tsx` (lines 10, 42)
- Direct code inspection: `frontend/src/components/RoleAssignment.tsx` (lines 4, 57)
- Direct code inspection: `frontend/src/components/ChampionIcon.tsx` (line 36) -- reference pattern
- Direct code inspection: `api/champion_mapping.py` -- complete DDRAGON_ID_MAP with all 12 mismatched champions
- Direct code inspection: `api/routers/champions.py` -- confirms `image_url` is returned per champion
- Direct code inspection: `frontend/src/types/index.ts` -- ChampionInfo type includes `image_url`
- Direct code inspection: `frontend/src/store/draftStore.ts` -- stores champion names as `string | null`
- Direct code inspection: `frontend/src/index.css` -- team color definitions (`--color-blue-team: #0088ff`, `--color-red-team: #ff3333`)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - all code inspected directly, no external dependencies needed
- Architecture: HIGH - data flow gap clearly identified, fix pattern proven by ChampionIcon
- Pitfalls: HIGH - standard React image loading patterns, no novel complexity

**Research date:** 2026-02-26
**Valid until:** Indefinite (static codebase analysis, no external API version concerns)
