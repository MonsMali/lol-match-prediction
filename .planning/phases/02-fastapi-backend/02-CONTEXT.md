# Phase 2: FastAPI Backend - Context

**Gathered:** 2026-02-24
**Status:** Ready for planning

<domain>
## Phase Boundary

Build the REST API layer that exposes the Phase 1 ML adapter as HTTP endpoints. Includes prediction, champion list, team list, health check, and model upload endpoints. CORS middleware for frontend consumption. The frontend itself is Phase 3; deployment is Phase 4.

</domain>

<decisions>
## Implementation Decisions

### Prediction response shape
- Response contains only `blue_win_probability` and `red_win_probability` -- no metadata, no confidence scores, no feature breakdowns (those are v2: EPRED-02, EPRED-03)
- Do not echo the submitted draft back in the response
- Teams are required in the predict payload (not optional)
- Unknown team names accepted with fallback to average/neutral team stats (no error, silent fallback matching adapter behavior)

### Champion data & DDragon mapping
- Pin a specific DDragon version (hardcoded, e.g., 14.24.1) -- no runtime fetch of latest version
- Champion name mismatch mapping (Wukong=MonkeyKing, Nunu, etc.) lives as a hardcoded Python dictionary in the backend
- /api/champions returns only champions the model knows about (from training data), not the full DDragon roster
- Each champion entry includes the full DDragon CDN image URL (backend constructs it, frontend just uses it)

### Model upload & admin security
- Model upload endpoint secured with a bearer token read from ADMIN_TOKEN environment variable
- Hot-swap model in memory on upload -- requests during reload get the old model, zero downtime
- Upload accepts a full artifact bundle (model + scaler + encoders) to ensure consistency
- Validate the new model before swapping by running a test prediction with a dummy draft; only swap if it succeeds

### Error responses & validation
- Strict draft validation: require exactly 10 picks (5 per team), 10 bans (5 per team), 2 teams, all roles assigned
- Use FastAPI's standard error format: `{"detail": "message"}` with appropriate HTTP status codes
- Validate champion names at the API layer against the known list before calling the adapter (fail fast with 422)
- Return 503 Service Unavailable if model is still loading when a predict request arrives; frontend retries based on /health status

### Claude's Discretion
- API route prefix structure and versioning
- Pydantic schema field naming conventions
- CORS allowed origins configuration
- Internal code organization (routers, dependencies, utils)
- asyncio.to_thread usage for blocking ML inference

</decisions>

<specifics>
## Specific Ideas

No specific requirements -- open to standard approaches. User consistently chose the recommended (simplest, most conventional) option for each decision.

</specifics>

<deferred>
## Deferred Ideas

None -- discussion stayed within phase scope

</deferred>

---

*Phase: 02-fastapi-backend*
*Context gathered: 2026-02-24*
