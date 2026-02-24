# Phase 1: ML Adapter - Context

**Gathered:** 2026-02-24
**Status:** Ready for planning

<domain>
## Phase Boundary

Refactor the inference path to eliminate the full 37K-row CSV load and expose a clean `predict_from_draft` Python interface. The adapter loads only pre-serialized `.joblib` artifacts and returns win probabilities as a typed result object. This is an internal building block -- the REST API (Phase 2) and frontend (Phase 3) build on top of it.

</domain>

<decisions>
## Implementation Decisions

### Draft input contract
- Champions identified by **display name** (e.g., "Jinx", "Lee Sin"); adapter handles normalization internally (Wukong/MonkeyKing, Nunu mismatches)
- **Teams are required** -- both blue and red team names must be provided (team historical performance is a key model feature)
- **Role assignments are explicit** in the input -- caller provides Top/Jungle/Mid/Bot/Support for each pick (no auto-detection)
- **Patch defaults to latest** in training data; no patch field required in input (can be overridden optionally)

### Prediction result shape
- Return **blue_win_prob and red_win_prob only** -- no confidence bands or uncertainty scores (the probability itself conveys confidence)
- **No feature contribution/explainability** data in the result -- keep it simple
- Include **basic model metadata**: model_name and model_version alongside probabilities
- Result is a **typed object** (dataclass or similar) with named fields, not a plain dict -- enables autocomplete and easy Pydantic conversion in Phase 2

### Error and edge cases
- **Unknown champion names fail with a clear validation error** listing the unrecognized name and suggesting the closest match (fuzzy suggestion in error message, not auto-correction)
- **Unknown team names fail with error** and list valid teams -- since teams are required, unknown means bad input
- **Complete drafts required** -- all 10 picks and 10 bans must be provided; no partial draft support
- **Adapter validates for duplicate champions** across all picks and bans -- defense in depth even if frontend prevents it

### Artifact loading
- **Eager loading at init** -- all joblib artifacts loaded when the adapter is instantiated; no lazy loading
- **Fail fast on missing/corrupt artifacts** -- if any required artifact can't be loaded, raise immediately with a clear error listing what's missing
- **Expose health status** via a `.get_status()` method returning loaded artifacts, model version, and memory usage (feeds Phase 2's /health endpoint)
- **Singleton pattern** -- one adapter instance shared across the app; prevents duplicate artifact loading; natural fit for FastAPI's app.state

### Claude's Discretion
- Internal normalization approach for champion name mismatches
- Exact dataclass field naming and structure
- Memory optimization techniques within the 512MB constraint
- Artifact file path resolution strategy

</decisions>

<specifics>
## Specific Ideas

No specific requirements -- open to standard approaches. The existing `InteractiveLoLPredictor` is the reference implementation to wrap/refactor.

</specifics>

<deferred>
## Deferred Ideas

None -- discussion stayed within phase scope

</deferred>

---

*Phase: 01-ml-adapter*
*Context gathered: 2026-02-24*
