# Feature Landscape

**Domain:** LoL professional draft prediction web app
**Researched:** 2026-02-24
**Context:** Public-facing web UI wrapping an existing Python ML prediction system (Logistic Regression, 82.97% AUC-ROC, trained on 37,502 pro matches 2014-2024)

---

## Table Stakes

Features users expect. Missing = product feels incomplete or untrustworthy.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Champion grid with portraits | Every draft tool uses champion portraits; text-only feels unfinished | Low | Use Riot Data Dragon CDN: `https://ddragon.leagueoflegends.com/cdn/{version}/img/champion/{Name}.png` — no API key required |
| Champion search / filter by name | Champion pool is 160+; scrolling without search is unusable | Low | Client-side filter on keystroke. Fuzzy match already exists in `InteractiveLoLPredictor`. |
| Correct professional draft order | Blue side picks 1, red side picks 2-3, etc. — violating this breaks trust immediately | Low | Order logic already exists in the CLI predictor. Web layer enforces it. |
| 5 bans per team, 5 picks per team | Pro format; any deviation breaks the expected structure | Low | Hardcoded constraint — 10 bans, 10 picks total |
| Visual draft board (two-sided layout) | Users expect the split-screen layout matching actual pro champ select | Medium | Left panel (blue team) / Right panel (red team) with slots for bans and picks |
| Win probability output | The core product promise — if it does not display a number, the tool has no value | Low | Single API call to the prediction backend after draft completes |
| Blue side / Red side labeling | Pro matches are always framed as blue vs red side — essential context | Low | Must be visible throughout and in the result |
| Professional team selection | The thesis system is built around team identities (LCK, LEC, LCS, LPL) — selecting teams is required for the prediction features to fire correctly | Low | Existing `teams_db` in the predictor: T1, GenG, G2, FNC, C9, TL, JDG, etc. |
| Prevent duplicate champion selection | Selecting the same champion twice must be blocked — a universally enforced constraint in every draft tool | Low | Client-side validation, echoed server-side |
| Mobile-readable layout | Users on phones watching broadcasts will try to use the tool; a broken mobile layout causes immediate exit | Medium | Responsive CSS — not pixel-perfect mobile app, but must not be broken |
| Loading state indicator | Backend cold-starts on free tier can take ~30 seconds; users need feedback that something is happening | Low | Spinner or progress indicator while awaiting prediction |
| Error messaging | If the backend is down or the draft is invalid, show a human-readable message — not a blank screen | Low | Catch API errors, show "Prediction unavailable — please try again" |

---

## Differentiators

Features that set this product apart. Not universally expected, but meaningfully increase value.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Thesis-backed 82.97% AUC model | Most public draft tools use simple win-rate lookups or unreported accuracy; a peer-validated ML model is a genuine credibility marker | Low to surface | Display the accuracy claim and methodology link prominently. The model already exists — this is a copy + attribution. |
| Pre-match only (zero data leakage) | Unlike in-game win probability tools (Riot/AWS), this predicts purely from the draft — a narrower, more honest scope that is defensible and novel | Low to surface | Messaging: "Based on draft only — no in-game stats." Distinction matters to a technical LoL audience. |
| Step-by-step mode (live follow-along) | Users can follow a live broadcast pick by pick and see probability shift after each selection | Medium | Requires intermediate prediction after each pick, not just at draft end. Needs backend support for partial-draft inference. The existing `InteractiveLoLPredictor` handles UNKNOWN tokens — this is already supported. |
| Bulk entry mode (full draft at once) | Users who want a quick historical lookup enter all 20 picks/bans at once and get instant output | Low | Simpler than step-by-step; should be the default flow given backend cold-start latency |
| Role assignment UI | Mapping picks to positions (top/jungle/mid/bot/support) allows richer prediction context and reflects how the model was trained | Medium | Drag-and-drop or dropdown per slot. Needs clear visual distinction between roles. |
| Confidence indicator alongside probability | Showing "Low confidence" for unusual draft compositions sets user expectations correctly and demonstrates model self-awareness | Medium | Requires backend to return a confidence score or flag. The existing `confidence.py` module may provide this. |
| Shareable draft URL | Users watching the same match can share a pre-filled draft for others to see the prediction | Medium | Encode draft state as URL query params (no server storage needed). Common in tools like draftlol and drafting.gg. |
| Best-of-series framing | Pro matches are BO1, BO3, or BO5 — the existing system already supports this; surfacing it in the UI adds authenticity | Low | Simple toggle: BO1 / BO3 / BO5. Show game number (Game 1, Game 2, etc.). |
| Model swap / upload capability | Allows the researcher to retrain on Colab and hot-swap the model file without redeployment | Medium | POST endpoint accepting a `.joblib` file. Basic auth or secret key needed to prevent abuse. Scoped to researcher use, not public. |
| Probability bar visualization | A visual split bar (blue 62% / red 38%) communicates at a glance better than raw numbers | Low | Simple CSS progress bar. Well established in win probability UIs per Riot/AWS broadcast design. |
| Patch / year context display | Showing which training data era the model learned from helps users calibrate trust for current-patch predictions | Low | Static label from model metadata. |

---

## Anti-Features

Features to explicitly NOT build. Common failure modes in this domain.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Per-champion pick recommendations ("best pick here") | This tool is a predictor, not a drafter — adding suggestions changes the product scope entirely, introduces complexity, and the model was not trained for counterfactual ranking at scale | Show the probability for the chosen draft; let users compare by entering different drafts |
| In-game live stats feed | Out of scope per PROJECT.md; model uses pre-match data only — adding in-game data would require a separate model and a fundamentally different architecture | Be explicit in the UI: "Pre-match draft prediction only" |
| User accounts and history | Adds backend complexity (auth, database, session management) for minimal value in v1; the tool is a calculator, not a social platform | Use shareable URLs for persistence instead |
| Historical prediction accuracy dashboard | Requires storing past predictions and ground-truth outcomes; significant backend complexity for a v1 thesis demo | Link to the thesis PDF for documented accuracy; do not build a live tracker |
| Solo queue champion data | The model is trained on professional matches only; applying it to solo queue would be misleading and inaccurate | Clearly label: "Professional play only — LCK, LEC, LCS, LPL" |
| Fearless Draft mode | A tournament format variant (champions cannot be re-picked across games); adds state tracking complexity across games that is out of scope | BO-series tracking is sufficient — no cross-game champion memory |
| Real-time patch syncing | Auto-detecting and adjusting to the current patch requires continuous data ingestion, model retraining, and deployment automation — none of which exist in this system | Let the researcher manually retrain and upload new model files; display the model's training patch in the UI |
| Champion tier lists or meta ratings | Contextual ratings invite comparison with tools like Lolalytics or METAsrc that have far more data; this tool's differentiator is the model accuracy, not meta curation | Stick to win probability — one clear output |
| Chat, comments, or social features | Moderation burden for zero user value in a research tool | Not applicable to the use case |
| Animated draft reveal sequence | Reproducing the broadcast pick reveal animation is high CSS/JS effort for low functional value | Static reveal with portrait shown immediately on selection |

---

## Feature Dependencies

```
Team selection
  └── Required before draft begins (team identities feed model features)

Champion selection (pick/ban slots)
  └── Requires: champion grid + portrait assets loaded
  └── Requires: duplicate prevention logic
  └── Requires: draft order enforcement

Role assignment
  └── Depends on: champion picks completed (5 per team)
  └── Required before: prediction API call (roles feed model)

Prediction output
  └── Depends on: all 10 picks assigned + roles assigned
  └── Depends on: backend API live and model loaded

Step-by-step mode
  └── Depends on: partial-draft inference support in backend
  └── Depends on: draft order state machine

Shareable URL
  └── Depends on: draft state serialization (can be independent of backend)

Model upload
  └── Depends on: backend upload endpoint + protected route
  └── Independent of: UI draft flow
```

---

## MVP Recommendation

The minimal version that delivers the core value proposition with confidence.

**Build for MVP:**

1. Bulk entry draft mode — enter all 10 picks and 10 bans at once, get win probability. Lowest latency risk, simplest state management.
2. Champion grid with portrait search and selection — Data Dragon CDN, client-side filter, duplicate prevention.
3. Professional team selection — four leagues (LCK, LEC, LCS, LPL) with known team names from existing `teams_db`.
4. Role assignment per slot — dropdown or drag-to-position after picks are entered.
5. Win probability output with split bar visualization — single number + visual bar.
6. Cold-start loading indicator — spinner with "Waking up prediction engine..." message to set expectations.
7. Blue/red side labeling throughout.

**Defer to post-MVP:**

| Feature | Reason to Defer |
|---------|-----------------|
| Step-by-step live mode | Requires partial-draft inference API and more complex UI state machine; the bulk mode already delivers the core value |
| Shareable URL | Useful but not essential for v1; add once the draft flow is stable |
| Confidence indicator | Requires backend work to expose; add in a second pass |
| Model upload endpoint | Researcher-only feature; can be done via direct file replacement in deployment dashboard initially |
| Best-of-series mode | Low-effort enhancement but MVP needs one game to validate the UX first |

---

## Competitive Landscape Summary

Tools surveyed for feature benchmarking:

| Tool | Scope | Key Feature | Differentiator vs This App |
|------|-------|-------------|---------------------------|
| LoLDraftAI | Solo queue + pro | AI pick recommendations, synergy scoring | Recommendation-first; this app is prediction-first |
| DraftGap | Solo queue | Client sync, matchup stats | Solo queue focused; this app is pro-only |
| Diff15 | Pro (LEC/LCK/LPL) | Dual-algorithm prediction, no signup | Closest competitor; this app has thesis-validated accuracy |
| drafting.gg (LS) | Pro | Draft simulation, shareable links, tier lists | No prediction model; this app has ML prediction |
| draftlol | General | Multiplayer draft practice | Simulation-only, no prediction |
| DraftVision | Team practice | Map drawing, Fearless draft | Coaching tool, not prediction |
| Riot/AWS broadcast | Pro (in-game) | Real-time in-game win probability | In-game stats required; this app is pre-match only |

**This app's niche:** Pre-match draft prediction using a peer-validated ML model, scoped to professional play, requiring no account — the only tool in this list that combines professional-only scope with a thesis-documented accuracy claim.

---

## Confidence Assessment

| Finding | Confidence | Source |
|---------|------------|--------|
| Champion portrait URLs via Data Dragon CDN | HIGH | Riot official documentation (`ddragon.leagueoflegends.com`) |
| Portrait grid + search as table stakes | HIGH | Observed across all 8+ tools surveyed (ProComps, DraftGap, LoLDraftAI, draftlol, etc.) |
| Step-by-step vs bulk modes | HIGH | Both modes exist in the current CLI predictor; web tools universally offer sequential draft flow |
| Shareable URL as differentiator pattern | MEDIUM | Confirmed in drafting.gg and draftlol; implementation detail varies |
| Win probability as sole output (not recommendations) | HIGH | Consistent with project scope in PROJECT.md and thesis methodology |
| Model accuracy claim (82.97% AUC) | HIGH | Documented in CLAUDE.md and thesis |
| Partial-draft inference support (UNKNOWN tokens) | MEDIUM | Referenced in LoLDraftAI blog; the existing predictor handles partial drafts per code inspection |
| Anti-recommendation stance | HIGH | Derived from model design (win probability classifier, not ranking model) |

---

## Sources

- [LoLDraftAI — Draft Analysis Tool](https://loldraftai.com/)
- [LoLDraftAI Explained — Blog](https://loldraftai.com/blog/loldraftai-explained)
- [DraftGap — Drafting Companion](https://draftgap.com/)
- [DraftGap GitHub](https://github.com/vigovlugt/draftgap)
- [Diff15 — Pro Draft Analyzer](https://diff15.com/)
- [Drafting.gg (LS)](https://drafting.gg/)
- [Draftlol](https://draftlol.dawe.gg/)
- [DraftVision](https://loldraftvision.fun/)
- [ProComps.gg](https://procomps.gg/)
- [Riot Data Dragon CDN Documentation](https://riot-api-libraries.readthedocs.io/en/latest/ddragon.html)
- [Riot/AWS Win Probability at Worlds 2023 — AWS Blog](https://aws.amazon.com/blogs/gametech/riot-games-and-aws-bring-esports-win-probability-stat-to-2023-league-of-legends-world-championships-broadcasts/)
- [Role-Based Win Probability Models in LoL Esports](https://bravewords.com/partners/role-based-win-probability-models-in-lol-esports/)
- [RCVolus lol-pick-ban-ui — GitHub](https://github.com/RCVolus/lol-pick-ban-ui)
- [TechLabs Aachen — Win percentage from draft phase (Medium)](https://techlabs-aachen.medium.com/determining-win-percentage-from-draft-phase-in-a-professional-league-of-legends-game-59ea4e4d5c55)
