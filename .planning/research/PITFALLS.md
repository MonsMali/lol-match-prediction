# Domain Pitfalls

**Domain:** ML model web app — Python prediction pipeline on free cloud hosting
**Researched:** 2026-02-24
**Confidence:** HIGH (free tier constraints verified via Render community forums; joblib security via official joblib docs and OWASP; Data Dragon quirks via Riot API libraries docs)

---

## Critical Pitfalls

Mistakes that cause the app to be unusable, crash on first user, or require architectural rewrites.

---

### Pitfall 1: Render Free Tier OOM Kill on Model Startup

**What goes wrong:**
The Python process is killed by the OS before the first request completes because importing pandas + numpy + scikit-learn + loading joblib files consumes more than 512 MB RAM. Render's free tier hard-limits at 512 MB. The existing `AdvancedFeatureEngineering` class loads the full 37,502-match CSV dataset into memory during `__init__` via `load_and_analyze_data()`. This alone, combined with the ML stack imports, is highly likely to exceed 512 MB.

**Why it happens:**
Teams assume model file size (small .joblib files) equals runtime RAM usage. It does not. pandas DataFrames, numpy arrays, and scikit-learn internals all allocate memory far beyond the serialized file size. The feature engineering class was designed for local training runs, not for a constrained server environment.

**Consequences:**
- Server health check fails immediately after deploy
- Render marks the service unhealthy and loops restart attempts
- App is never reachable; the project appears broken

**Warning signs:**
- Render deployment logs show "Ran out of memory (used over 512MB)"
- Health check endpoint times out during startup
- Service restarts repeatedly in Render dashboard

**Prevention:**
1. Profile memory usage locally before deploying: `memory_profiler` or `tracemalloc`
2. Strip the dataset load from the web server's feature engineering path. The full CSV is needed for training, NOT for inference. Precompute all lookup tables (team win rates, champion meta strength) into lightweight `.joblib` artifacts at training time. The server should load only these small artifacts, not the raw dataset.
3. Replace `load_and_analyze_data()` in the web server's `InteractiveLoLPredictor` with a pre-serialized lookup approach.
4. If 512 MB is genuinely insufficient even with stripping, target Railway's hobby plan ($5/month, 512 MB guaranteed with no cold starts) or Render's $7/month starter tier.

**Phase to address:** Foundation / Backend setup phase. This is a prerequisite — the server cannot launch until memory fits the constraint.

---

### Pitfall 2: Joblib Model Upload as a Remote Code Execution Vector

**What goes wrong:**
The project spec includes a "model file upload/swap capability." Joblib uses Python pickle internally. Loading a user-uploaded `.joblib` file executes arbitrary Python code embedded in the file. Any visitor to the public URL can upload a malicious model and gain full shell access to the Render/Railway container.

**Why it happens:**
Developers conflate "the model file is ours" with "any uploaded model file is safe." On a public web app with a model upload endpoint, the attacker IS a user. The joblib documentation explicitly warns: "joblib.load() should never be used to load objects from an untrusted source."

**Consequences:**
- Full remote code execution on the server
- Attacker can exfiltrate environment variables, API keys, other files
- Platform account suspension for abuse

**Warning signs:**
- An unauthenticated `/upload-model` or similar endpoint exists and accepts `.joblib` files directly
- No file type validation beyond extension check

**Prevention:**
1. The safest option: remove the file upload feature entirely from v1. Use Git-based deployment for model swaps instead (push new `.joblib` to the repo, redeploy).
2. If upload is kept: restrict it with a secret token (passed in request header), validate file size limits, never load uploaded files in the same process as the web server (use a sandboxed subprocess or separate worker).
3. Document the risk explicitly in the roadmap phase that adds this feature.

**Phase to address:** Whichever phase implements "model swap" functionality. This pitfall must be flagged before that phase begins.

---

### Pitfall 3: Cold Start Timeout Kills First Real Request

**What goes wrong:**
Render free tier spins the service down after 15 minutes of inactivity. The next inbound request triggers a cold start. With the Python ML stack (pandas, numpy, scikit-learn, joblib), cold start can take 30-90 seconds. Render applies a 30-second request timeout by default. The first user's prediction request times out with a 504 error even though the server eventually recovers.

**Why it happens:**
The cold start is unavoidable on free tier, but the first request bearing the full startup cost is a design oversight. Most simple web apps have sub-second cold starts; an ML stack is 10-30x heavier.

**Consequences:**
- First user after any idle period sees a timeout error, not a slow response
- Users perceive the app as broken, not just slow

**Warning signs:**
- First request after >15 min idle returns 504/502
- Server logs show startup still running when request arrived

**Prevention:**
1. Add a lightweight `/health` or `/ping` endpoint that returns immediately (before model load completes) so the platform's health check passes.
2. Implement a "wake-up" frontend pattern: on page load, immediately fire a `/ping` call before the user interacts. Show a "warming up, this takes ~20 seconds on first load" banner.
3. Use a scheduled keep-alive ping (external cron service like cron-job.org, free) to hit the service every 10 minutes and prevent spin-down entirely.
4. Structure startup so model loading happens lazily on first prediction request, not on import, giving the HTTP server time to start and pass health checks before the heavy work begins.

**Phase to address:** Initial deployment phase. The keep-alive strategy and loading UX must be designed before first public deployment.

---

## Moderate Pitfalls

Mistakes that degrade user experience or create technical debt requiring significant rework.

---

### Pitfall 4: Multiple Gunicorn Workers Multiply RAM Usage Linearly

**What goes wrong:**
A common deployment pattern is `gunicorn -w 4 app:app`. Each worker is a separate OS process. If one worker holding the loaded model uses 300 MB, four workers use 1.2 GB — instantly exceeding free tier limits and crashing all workers.

**Why it happens:**
The standard gunicorn worker recommendation (2 * CPU cores + 1) is designed for I/O-bound apps, not memory-bound ML inference. ML model deployments need a single worker or use `--preload` to share memory via copy-on-write forking.

**Prevention:**
- Run a single worker on free tier: `gunicorn -w 1 -k uvicorn.workers.UvicornWorker app:main`
- If multiple workers are needed, add `--preload` to gunicorn config so the model is loaded before forking (copy-on-write reduces per-worker overhead)
- Keep concurrency via async (uvicorn/FastAPI async endpoints) within a single worker, not via multiple processes

**Phase to address:** Backend deployment configuration phase.

---

### Pitfall 5: scikit-learn Version Mismatch Breaks Model Loading

**What goes wrong:**
Models are trained on Google Colab (one scikit-learn version) and deployed to Render (a different version pinned in requirements.txt). The `joblib.load()` call raises an `InconsistentVersionWarning` or fails outright. In scikit-learn 1.3.0, backward compatibility for loading older models was broken, causing silent incorrect predictions or crashes.

**Why it happens:**
`requirements.txt` is not kept in sync with the Colab training environment. Colab auto-installs latest scikit-learn; the server runs whatever version was pinned at deployment time.

**Warning signs:**
- `InconsistentVersionWarning` in server logs on startup
- Prediction endpoint returns unexpected results or 500 errors after a model swap

**Prevention:**
1. Pin the exact scikit-learn version in `requirements.txt` that matches Colab training. After every Colab training run, check `sklearn.__version__` and update the pin.
2. Store the scikit-learn version as metadata alongside the model file (a simple text file committed alongside the `.joblib`).
3. Add a startup check that logs the loaded scikit-learn version and warns if it differs from a `TRAINED_WITH_SKLEARN_VERSION` environment variable.

**Phase to address:** Initial backend setup and model integration phase.

---

### Pitfall 6: Dataset-Dependent Feature Engineering Running on Web Server

**What goes wrong:**
`AdvancedFeatureEngineering.load_and_analyze_data()` reads the full 37,502-row CSV to build champion meta strength tables, team performance lookups, and synergy maps. This is appropriate for training. On the web server, it runs every cold start, adds 30+ seconds to startup, and consumes hundreds of MB just to reconstruct lookup tables that could have been pre-serialized.

**Why it happens:**
The feature engineering class was not designed with a train/serve split. There is no separation between "build the lookup tables" (training time) and "use the lookup tables" (inference time).

**Consequences:**
- The CSV file must be bundled into the Docker image or mounted on the server (adds ~10-50 MB to image, or requires a file storage mount)
- Cold starts are dramatically slower
- RAM consumption is doubled unnecessarily

**Prevention:**
1. At training time, serialize the finished lookup tables (champion meta strength, team historical performance, champion synergies) as separate `.joblib` files. These already exist in `models/champion_meta_strength.joblib`, `models/team_historical_performance.joblib`, `models/champion_synergies.joblib`.
2. Create a lean `InferenceFeatureEngineering` class (or add an inference-mode flag) that loads only the pre-serialized artifacts, not the raw CSV.
3. Do NOT bundle the full dataset CSV in the web server deployment. It is not needed for inference.

**Phase to address:** Backend API design phase — this is the single highest-value optimization before deployment.

---

### Pitfall 7: Data Dragon Champion Name Key Mismatches Break Portrait Loading

**What goes wrong:**
Data Dragon's `champion.json` uses internal champion keys that do not always match the display name. `Wukong` is keyed as `MonkeyKing`. `Fiddlesticks` has a known misspelling in some JSON responses. `Nunu & Willump` is keyed as `Nunu`. Constructing portrait URLs by directly using the display name or the model's internal champion representation will produce 404s for specific champions.

**Why it happens:**
The Data Dragon API was designed for programmatic use, not display-name lookup. The mismatch between display names, internal keys, and image filenames is a well-known quirk documented in the Riot API libraries community.

**Warning signs:**
- Champion portraits for certain champions show broken image icons
- URL pattern `ddragon.leagueoflegends.com/cdn/{version}/img/champion/{name}.png` 404s for specific champions

**Prevention:**
1. On app startup (or at build time), fetch `champion.json` from Data Dragon and build a local lookup map: `{display_name: internal_key}`. Use the `key` field from the JSON, not the top-level object key.
2. Cache this map in memory; refresh it at most once per day or on version change.
3. Use the `id` field (not the `name` field) from `champion.json` to construct portrait URLs, as `id` matches the image filename exactly.
4. Include a fallback image for portrait load failures so broken portraits do not break the draft UI.

**Phase to address:** Frontend draft board implementation phase.

---

### Pitfall 8: Data Dragon Patch Version Lag Breaks New Champion Portraits

**What goes wrong:**
Riot releases a new champion (e.g., Ambessa in patch 14.21). The existing `AdvancedFeatureEngineering` fuzzy matching may not recognize the new champion name if it was not in the training data. Separately, Data Dragon may not publish the new version immediately — sometimes 1-2 days after patch release. Users who enter a recently released champion get validation errors and broken portraits.

**Why it happens:**
The model's champion validation is trained on historical data. New champions are unknown to the feature engineering pipeline. Data Dragon's version endpoint may still point to the previous patch during the lag window.

**Consequences:**
- New champion causes 422 validation error from the API
- Portrait fetch returns 404 during the patch lag window
- User experience degrades precisely when user interest in the new champion is highest

**Prevention:**
1. Implement a "known champion" allowlist that can be updated independently of model retraining. Store it in a simple JSON config file committed to the repo.
2. For unknown champions, return a fallback response (neutral 50% probability, or a clear "champion not in training data" message) rather than a hard error.
3. Fetch the Data Dragon version list at startup and select the latest stable version. Cache it for 24 hours to avoid unnecessary requests.
4. Add an image fallback for any portrait that returns a non-200 response from Data Dragon.

**Phase to address:** Backend API validation phase.

---

## Minor Pitfalls

Mistakes that cause annoyance but are fixable without architectural change.

---

### Pitfall 9: CORS Misconfiguration Blocks Frontend from Backend

**What goes wrong:**
If the frontend (e.g., served from a different origin or during local development) calls the FastAPI backend, the browser blocks requests with "CORS policy" errors because the backend does not include the correct `Access-Control-Allow-Origin` headers.

**Prevention:**
Add `fastapi.middleware.cors.CORSMiddleware` to the FastAPI app from day one. Configure it to allow the specific frontend origin in production and `*` during development. Do not leave this to "fix later" — it blocks all frontend testing.

**Phase to address:** Backend setup phase.

---

### Pitfall 10: Render/Railway Free Tier Build Timeout on Heavy Dependencies

**What goes wrong:**
Installing the full ML stack (scikit-learn, XGBoost, LightGBM, CatBoost, pandas, numpy, optuna) during the platform build step takes 5-10 minutes. Free tier build timeouts (typically 15-20 minutes) are sometimes hit if the base image is cold or the network is slow.

**Prevention:**
1. Trim `requirements.txt` for production to only what the inference server needs. XGBoost, LightGBM, CatBoost, and optuna are NOT needed if only the Logistic Regression model is deployed. Remove training-only dependencies from the production requirements file.
2. Use `requirements-prod.txt` vs `requirements-dev.txt` separation. This also reduces the installed package footprint and RAM at runtime.

**Phase to address:** Initial deployment configuration phase.

---

### Pitfall 11: Champion Fuzzy Matching Produces Wrong Champion on Ambiguous Input

**What goes wrong:**
The existing `InteractiveLoLPredictor` uses fuzzy matching for champion name input. "Lee" could match "Lee Sin" or "Leona." "Malz" could match "Malzahar" or "Maokai." In a web UI where users type into a search box, a wrong fuzzy match silently produces an incorrect prediction without the user realizing which champion was actually used.

**Prevention:**
In the web UI, always show the matched champion's name and portrait as a confirmation before adding it to the draft. Never silently accept a fuzzy match. The UI pattern "Did you mean: [Champion Name + Portrait]?" eliminates silent mismatch errors.

**Phase to address:** Frontend draft entry UX phase.

---

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|---------------|------------|
| Backend API setup | OOM kill on startup (Pitfall 1) | Strip dataset load from server; use pre-serialized artifacts |
| Backend API setup | Multi-worker RAM multiplication (Pitfall 4) | Single worker + async; no multi-process on free tier |
| Backend API setup | scikit-learn version mismatch (Pitfall 5) | Pin exact version; add startup version check |
| Backend API setup | CORS blocking frontend (Pitfall 9) | Add CORSMiddleware before first frontend integration test |
| Initial deployment | Cold start timeout (Pitfall 3) | Health endpoint + frontend warm-up UX + keep-alive cron |
| Initial deployment | Build timeout (Pitfall 10) | Separate prod requirements file; remove training-only deps |
| Model swap feature | Joblib upload RCE (Pitfall 2) | Authentication token or remove feature; never load untrusted pickles |
| Draft board UI | Data Dragon name key mismatches (Pitfall 7) | Build ID-to-key map from champion.json at startup |
| Draft board UI | New champion portrait 404 (Pitfall 8) | Image fallback; version lag handling |
| Draft entry UX | Fuzzy match ambiguity (Pitfall 11) | Always confirm match with portrait before committing to draft |

---

## Sources

- Render free tier memory limits: [Render Community — Deployment Error: Ran out of memory (used over 512MB)](https://community.render.com/t/deployment-error-ran-out-of-memory-used-over-512mb/14215) — HIGH confidence (official Render community, real user reports)
- Render cold start and timeout behavior: [How to Keep Your FastAPI Server Active on Render's Free Tier](https://medium.com/@saveriomazza/how-to-keep-your-fastapi-server-active-on-renders-free-tier-93767b70365c) — MEDIUM confidence (community article, consistent with Render docs)
- Gunicorn multiple worker memory multiplication: [FastAPI GitHub Discussion #2425 — Serving ML models with multiple workers](https://github.com/fastapi/fastapi/issues/2425) — HIGH confidence (official FastAPI repo, widely reproduced)
- joblib pickle security risk: [joblib Persistence documentation](https://joblib.readthedocs.io/en/latest/persistence.html) — HIGH confidence (official joblib docs explicitly warn against loading untrusted sources)
- joblib RCE in practice: [AWS Security Blog — The Little Pickle Story](https://aws.amazon.com/blogs/security/enhancing-cloud-security-in-ai-ml-the-little-pickle-story/) — HIGH confidence (official AWS security blog)
- scikit-learn version mismatch: [scikit-learn Model Persistence documentation](https://scikit-learn.org/stable/model_persistence.html) — HIGH confidence (official scikit-learn docs)
- scikit-learn 1.3.0 backward compatibility break: [GitHub Issue #26798](https://github.com/scikit-learn/scikit-learn/issues/26798) — HIGH confidence (official scikit-learn issue tracker)
- Data Dragon champion key mismatches: [Riot API Libraries — Data Dragon documentation](https://riot-api-libraries.readthedocs.io/en/latest/ddragon.html) — HIGH confidence (official community documentation for Riot API)
- Data Dragon patch version lag: [freepublicapis.com — Data Dragon API](https://www.freepublicapis.com/data-dragon-api) — MEDIUM confidence (third-party monitoring; consistent with community reports)
- FastAPI model loading patterns: [Loading Models into FastAPI Applications](https://apxml.com/courses/fastapi-ml-deployment/chapter-3-integrating-ml-models/loading-models-fastapi) — MEDIUM confidence (course material, consistent with FastAPI docs)
