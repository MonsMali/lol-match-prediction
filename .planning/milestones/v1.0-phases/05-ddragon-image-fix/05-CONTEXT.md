# Phase 5: DDragon Image URL Fix - Context

**Gathered:** 2026-02-26
**Status:** Ready for planning

<domain>
## Phase Boundary

Fix broken champion portrait URLs in PickSlot, BanRow, and RoleAssignment components. These components currently construct DDragon URLs from display names, which breaks for 12 champions whose DDragon ID differs from their display name (e.g., Wukong -> MonkeyKing). The fix is to use the API-provided `image_url` field instead. ChampionIcon.tsx already uses `image_url` correctly and serves as the reference pattern.

</domain>

<decisions>
## Implementation Decisions

### Fallback behavior
- Show a generic champion silhouette when an image fails to load
- Silhouette rendered as an inline SVG (no external dependency, ships with the app)
- Silhouette tinted in team color (blue/red) for visual context
- Same fallback treatment across all slot types (picks, bans, role assignment) -- bans already have their own visual treatment so the fallback does not need to differ
- On image error, show fallback immediately with no retry -- DDragon is a static CDN, so transient failures are unlikely to resolve on retry

### Image loading states
- Show a shimmer/skeleton animation while champion portraits load
- Champion name appears immediately on selection (instant feedback); only the portrait area shows shimmer
- Portrait fades in with a quick opacity transition (~150-200ms) when loaded
- Shimmer + fade-in applies consistently to all slot types: PickSlot, BanRow, and RoleAssignment

### Claude's Discretion
- Exact shimmer animation CSS implementation
- SVG silhouette design details
- Exact fade-in timing within the ~150-200ms range

</decisions>

<specifics>
## Specific Ideas

- ChampionIcon.tsx already uses `champion.image_url` from the API -- use this as the reference pattern for the fix
- The 12 affected champions: Wukong, Kai'Sa, Nunu & Willump, Kha'Zix, Cho'Gath, Vel'Koz, Rek'Sai, Kog'Maw, Bel'Veth, K'Sante, LeBlanc, Renata Glasc

</specifics>

<deferred>
## Deferred Ideas

None -- discussion stayed within phase scope

</deferred>

---

*Phase: 05-ddragon-image-fix*
*Context gathered: 2026-02-26*
