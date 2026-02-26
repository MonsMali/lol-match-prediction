# Phase 1: ML Adapter - Research

**Researched:** 2026-02-24
**Domain:** Python ML inference refactoring (scikit-learn, joblib, dataclass design)
**Confidence:** HIGH

## Summary

The current `InteractiveLoLPredictor` loads the full 37K-row CSV at startup via `AdvancedFeatureEngineering.load_and_analyze_data()`, which iterates through every row to compute champion characteristics, meta indicators, team performance, and ban priority dictionaries. This is the bottleneck that Phase 1 must eliminate. The good news: the training pipeline already serializes the entire `AdvancedFeatureEngineering` instance to `ultimate_feature_engineering.joblib` (9.5 MB), which contains all pre-computed lookup dictionaries needed for inference.

The adapter must extract and use these pre-computed lookup dicts (champion_characteristics, champion_meta_strength, champion_popularity, team_historical_performance, ban_priority) from the serialized feature engineering object. Combined with the model (~232 KB), scaler (~4 KB), and encoders (~12 KB), total artifact size is under 11 MB -- well within the 512 MB RAM constraint.

**Primary recommendation:** Build a new `LoLDraftAdapter` class that loads pre-serialized artifacts (model, scaler, encoders, and the feature engineering object's lookup dicts), re-implements the feature computation for a single match without pandas DataFrame manipulation of the training dataset, and exposes a clean `predict_from_draft(draft_dict) -> PredictionResult` interface.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Champions identified by **display name** (e.g., "Jinx", "Lee Sin"); adapter handles normalization internally (Wukong/MonkeyKing, Nunu mismatches)
- **Teams are required** -- both blue and red team names must be provided (team historical performance is a key model feature)
- **Role assignments are explicit** in the input -- caller provides Top/Jungle/Mid/Bot/Support for each pick (no auto-detection)
- **Patch defaults to latest** in training data; no patch field required in input (can be overridden optionally)
- Return **blue_win_prob and red_win_prob only** -- no confidence bands or uncertainty scores
- **No feature contribution/explainability** data in the result
- Include **basic model metadata**: model_name and model_version alongside probabilities
- Result is a **typed object** (dataclass or similar) with named fields, not a plain dict
- **Unknown champion names fail with a clear validation error** listing the unrecognized name and suggesting the closest match (fuzzy suggestion in error message, not auto-correction)
- **Unknown team names fail with error** and list valid teams
- **Complete drafts required** -- all 10 picks and 10 bans must be provided; no partial draft support
- **Adapter validates for duplicate champions** across all picks and bans
- **Eager loading at init** -- all joblib artifacts loaded when adapter is instantiated
- **Fail fast on missing/corrupt artifacts** -- raise immediately with clear error listing what is missing
- **Expose health status** via a `.get_status()` method returning loaded artifacts, model version, and memory usage
- **Singleton pattern** -- one adapter instance shared across the app

### Claude's Discretion
- Internal normalization approach for champion name mismatches
- Exact dataclass field naming and structure
- Memory optimization techniques within the 512MB constraint
- Artifact file path resolution strategy

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| scikit-learn | >=1.0.0 (match training env) | Model inference via `predict_proba` | Already used for training; model is serialized sklearn LogisticRegression |
| joblib | >=1.1.0 | Artifact loading | Standard for sklearn model serialization; already used throughout project |
| pandas | existing | Single-row feature computation | Required by `create_advanced_features_vectorized` pattern |
| numpy | existing | Numeric operations | Required by sklearn and feature math |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| dataclasses | stdlib | Typed result objects (PredictionResult, AdapterStatus) | Always -- locked decision for typed result |
| pathlib | stdlib | Artifact path resolution | Always -- already used in `src/config.py` |
| difflib | stdlib | Fuzzy champion name matching (`get_close_matches`) | For unknown champion error messages |
| resource / psutil | stdlib / pip | Memory usage reporting in `.get_status()` | For health status method |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| dataclass | Pydantic BaseModel | Pydantic adds dep but converts easier in Phase 2; dataclass is simpler for Phase 1, Pydantic adapter trivial later |
| pandas for single-row | Pure dict/numpy math | Would eliminate pandas dep for inference, but feature computation logic is tightly coupled to pandas in existing code; refactoring risk outweighs benefit |
| difflib.get_close_matches | rapidfuzz | rapidfuzz is faster but adds dependency; difflib is stdlib and sufficient for ~200 champion names |

## Architecture Patterns

### Recommended Project Structure
```
src/
├── adapter/
│   ├── __init__.py           # Export LoLDraftAdapter, predict_from_draft
│   ├── adapter.py            # Main LoLDraftAdapter class (singleton)
│   ├── schemas.py            # DraftInput, PredictionResult, AdapterStatus dataclasses
│   ├── features.py           # Feature computation from lookup dicts (extracted from AdvancedFeatureEngineering)
│   ├── validation.py         # Champion/team/draft validation with fuzzy matching
│   └── normalization.py      # Champion name normalization (Wukong/MonkeyKing mapping)
```

### Pattern 1: Artifact-Only Inference
**What:** Load pre-computed lookup dictionaries from serialized feature engineering object instead of computing from CSV.
**When to use:** Always -- this is the core requirement (INFER-01, INFER-02).
**Critical detail:** The training pipeline serializes the entire `AdvancedFeatureEngineering` instance to `ultimate_feature_engineering.joblib`. This object contains:
- `champion_characteristics` -- dict mapping champion name to {win_rate, early_game_strength, late_game_strength, scaling_factor, flexibility}
- `champion_meta_strength` -- dict mapping (patch, champion) tuple to float
- `champion_popularity` -- dict mapping (patch, champion) tuple to {popularity: float}
- `team_historical_performance` -- dict mapping team name to {overall_winrate, recent_winrate, form_trend, games_played}
- `ban_priority` -- dict mapping champion name to {early_bans, total_bans}

**Example:**
```python
# Load feature engineering object and extract lookup dicts
fe_obj = joblib.load("models/ultimate_feature_engineering.joblib")
champion_chars = fe_obj.champion_characteristics
meta_strength = fe_obj.champion_meta_strength
team_perf = fe_obj.team_historical_performance
ban_priority = fe_obj.ban_priority
champion_pop = fe_obj.champion_popularity
```

### Pattern 2: Feature Computation for Single Match
**What:** Re-implement `create_advanced_features_vectorized` + `apply_prediction_time_encoding` for a single match using the extracted lookup dicts.
**When to use:** At prediction time.
**Critical detail:** The existing code creates a 1-row DataFrame, swaps it into `self.feature_engineering.df`, calls `create_advanced_features_vectorized()`, then aligns to exactly 37 features. The adapter should replicate this exact computation but without mutating shared state.

**The 37 expected features (from apply_prediction_time_encoding):**
```python
expected_features = [
    'team_avg_win_rate', 'team_avg_early_game_strength', 'team_avg_late_game_strength',
    'team_scaling', 'composition_balance', 'team_avg_flexibility', 'team_avg_winrate',
    'team_early_strength', 'team_late_strength', 'team_flexibility', 'team_meta_strength',
    'team_meta_consistency', 'team_popularity', 'meta_advantage', 'ban_count',
    'ban_diversity', 'high_priority_bans', 'team_overall_winrate', 'team_recent_winrate',
    'team_form_trend', 'team_experience', 'playoffs', 'side_blue', 'year', 'champion_count',
    'meta_form_interaction', 'scaling_experience_interaction', 'composition_historical_winrate',
    'league_target_encoded', 'team_target_encoded', 'patch_target_encoded', 'split_target_encoded',
    'top_champion_target_encoded', 'jng_champion_target_encoded', 'mid_champion_target_encoded',
    'bot_champion_target_encoded', 'sup_champion_target_encoded'
]
```

### Pattern 3: Dual-Perspective Prediction
**What:** The existing prediction logic computes features from BOTH blue and red perspective, then averages.
**When to use:** Always -- this is how the model was designed.
**Critical detail from `predict_match`:**
```python
# Blue perspective: team=blue_team, side='Blue', champions=blue picks
# Red perspective: team=red_team, side='Red', champions=red picks
# Then: avg_blue_win = (blue_pred[1] + (1 - red_pred[1])) / 2
```

### Pattern 4: Singleton Adapter with Eager Loading
**What:** Single adapter instance, artifacts loaded at `__init__`, fail-fast on missing files.
**When to use:** Always -- locked decision.
**Example:**
```python
class LoLDraftAdapter:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, artifacts_dir: Path = None):
        if hasattr(self, '_initialized'):
            return
        self._load_artifacts(artifacts_dir)
        self._initialized = True
```

### Anti-Patterns to Avoid
- **Mutating shared state during prediction:** The existing code does `self.feature_engineering.df = blue_df` which is not thread-safe. The adapter must compute features without mutating any instance state.
- **Loading CSV at any point:** The entire point of Phase 1 is eliminating this. No `pd.read_csv` in the inference path.
- **Silently falling back on missing artifacts:** Locked decision says fail fast. No try/except with fallback paths.
- **Lazy loading:** Locked decision says eager loading at init.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Fuzzy string matching | Custom Levenshtein implementation | `difflib.get_close_matches` | stdlib, handles the ~200 champion name use case; existing code already has a hand-rolled Levenshtein that should be replaced |
| Champion name normalization | Regex-based parsing | Static alias dict (Wukong->MonkeyKing, etc.) | Finite and well-known set of mismatches; existing code has `common_misspellings` dict as reference |
| Singleton pattern | Custom metaclass | `__new__` override or module-level instance | Standard Python pattern; no framework needed |
| Memory profiling | Manual `sys.getsizeof` | `resource.getrusage` (stdlib) or `psutil.Process().memory_info()` | Accurate RSS measurement for health endpoint |

## Common Pitfalls

### Pitfall 1: scikit-learn Version Mismatch
**What goes wrong:** Model trained with sklearn X.Y won't load (or silently misbehaves) under sklearn A.B.
**Why it happens:** `requirements.txt` says `>=1.0.0` which is very broad. The model files were serialized with a specific version.
**How to avoid:** Before building the adapter, inspect the serialized model to determine the sklearn version it was trained with. Pin that exact version (or at least the same minor version) in production requirements.
**Warning signs:** `UserWarning: Trying to unpickle estimator LogisticRegression from version X.Y when using version A.B`
**Detection:** `joblib.load(model_path).__class__.__module__` can reveal version info; also check `model.__sklearn_tags__` or `model._sklearn_version` attribute.

### Pitfall 2: Feature Order Mismatch
**What goes wrong:** The 37 features must be in the EXACT order the model was trained with. Wrong order = wrong predictions with no error.
**Why it happens:** Dict ordering, DataFrame column ordering, or adding features in different order than training.
**How to avoid:** Hard-code the exact feature list (shown above in Pattern 2) and always construct the feature vector in that specific order.
**Warning signs:** Predictions are always near 50% or always the same regardless of input.

### Pitfall 3: The 27 MB Model File
**What goes wrong:** The `production/best_model.joblib` is 27 MB, which is suspiciously large for a Logistic Regression model. It may be a deployment package dict containing the model + extra data.
**Why it happens:** The training code wraps models in dicts: `{'model': model_obj, 'scaler': scaler_obj, 'performance': {...}, ...}`.
**How to avoid:** Inspect the file first. If it's a dict, extract just the model. Consider whether the `ultimate_best_model.joblib` (232 KB) is the correct/better artifact to use.
**Warning signs:** Excessive memory usage on load.

### Pitfall 4: Missing Lookup Dict Keys at Inference Time
**What goes wrong:** A champion or team not in the training data causes a KeyError when looking up characteristics.
**Why it happens:** New champions, renamed teams, or champions that had fewer than 5 games in training data.
**How to avoid:** All lookup dict accesses MUST use `.get()` with sensible defaults. The existing code already does this: `self.champion_characteristics.get(x, default_char)`.
**Warning signs:** KeyError during prediction for valid-looking input.

### Pitfall 5: Patch-Champion Tuple Lookup
**What goes wrong:** `champion_meta_strength` is keyed by `(patch, champion)` tuples. If the adapter defaults to a different patch string format than what's in the dict, all lookups return the default 0.5.
**Why it happens:** Patch formatting inconsistency (e.g., "14.1" vs "14.01" vs "14.1.1").
**How to avoid:** Determine the "latest patch" from the keys of `champion_meta_strength` dict itself, not from hardcoded values. Use `max(set(patch for patch, _ in meta_strength.keys()))` or similar.
**Warning signs:** All meta_strength features are 0.5 (the default).

### Pitfall 6: Dual-Perspective Averaging Logic
**What goes wrong:** The model predicts P(team_wins | team_is_the_subject). Blue perspective gives P(blue wins), red perspective gives P(red wins). These must be averaged correctly.
**Why it happens:** Misunderstanding which index of `predict_proba` corresponds to which class.
**How to avoid:** Keep the exact averaging formula from `predict_match`: `avg_blue_win = (blue_pred[1] + (1 - red_pred[1])) / 2`.
**Warning signs:** Probabilities that don't sum to 1, or predictions that don't change when you swap blue/red.

## Code Examples

### Draft Input Schema
```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class DraftInput:
    blue_team: str
    red_team: str
    blue_picks: dict  # {"top": "Gnar", "jungle": "Viego", "mid": "Azir", "bot": "Jinx", "support": "Thresh"}
    red_picks: dict   # same structure
    blue_bans: list[str]  # ["Yone", "LeBlanc", "Kai'Sa", "Graves", "Nautilus"]
    red_bans: list[str]   # 5 bans
    patch: Optional[str] = None  # defaults to latest in training data
```

### Prediction Result Schema
```python
@dataclass
class PredictionResult:
    blue_win_prob: float
    red_win_prob: float
    model_name: str
    model_version: str
```

### Adapter Status Schema
```python
@dataclass
class AdapterStatus:
    loaded_artifacts: list[str]
    model_name: str
    model_version: str
    memory_usage_mb: float
    champion_count: int
    team_count: int
```

### Feature Computation (Single Match)
```python
# Source: derived from create_advanced_features_vectorized + apply_prediction_time_encoding
def compute_features_for_side(
    picks: dict,  # {role: champion_name}
    bans: list[str],
    team: str,
    side: str,  # "Blue" or "Red"
    patch: str,
    league: str,
    # Lookup dicts loaded from artifacts:
    champion_chars: dict,
    meta_strength: dict,
    champion_pop: dict,
    team_perf: dict,
    ban_priority: dict,
    encoders: dict,
) -> list[float]:
    """Compute the 37-feature vector for one side of the match."""
    default_char = {'win_rate': 0.5, 'early_game_strength': 0.5, 'late_game_strength': 0.5,
                    'scaling_factor': 0, 'flexibility': 1}
    default_perf = {'overall_winrate': 0.5, 'recent_winrate': 0.5, 'form_trend': 0, 'games_played': 0}

    # 1. Champion characteristics (averaged across 5 picks)
    chars = [champion_chars.get(c, default_char) for c in picks.values()]
    team_avg_win_rate = np.mean([c['win_rate'] for c in chars])
    # ... etc for all 37 features
```

### Validation Example
```python
from difflib import get_close_matches

def validate_champion(name: str, valid_champions: set[str], aliases: dict[str, str]) -> str:
    """Validate and normalize a champion name. Raises ValueError if unknown."""
    # Check alias map first
    normalized = aliases.get(name.lower(), name)
    # Case-insensitive match
    for valid in valid_champions:
        if valid.lower() == normalized.lower():
            return valid
    # Fuzzy match for error message
    close = get_close_matches(name, list(valid_champions), n=3, cutoff=0.6)
    msg = f"Unknown champion: '{name}'"
    if close:
        msg += f". Did you mean: {', '.join(close)}?"
    raise ValueError(msg)
```

## State of the Art

| Old Approach (current) | New Approach (adapter) | Impact |
|------------------------|------------------------|--------|
| Load 37K-row CSV + iterate to build lookup dicts | Load pre-serialized lookup dicts from joblib | Eliminates ~90% of startup time and memory |
| Mutate `feature_engineering.df` for each prediction | Stateless feature computation function | Thread-safe, no shared state mutation |
| Interactive CLI with `input()` calls | Programmatic API with typed input/output | API-ready for Phase 2 |
| Multiple fallback paths for model files | Single canonical path with fail-fast | Predictable behavior |
| Hand-rolled Levenshtein distance | stdlib `difflib.get_close_matches` | Simpler, tested, sufficient |

## Open Questions

1. **Which model file is the "correct" production artifact?**
   - `models/production/best_model.joblib` (27 MB) -- may be a deployment package dict
   - `models/ultimate_best_model.joblib` (232 KB) -- more reasonable size for LogisticRegression
   - **What we know:** The code checks multiple paths and handles both dict-wrapped and raw model formats
   - **Recommendation:** Inspect both files at plan execution time. Determine which has the best AUC and use that. Document the choice.

2. **Exact scikit-learn version used during training**
   - **What we know:** `requirements.txt` says `>=1.0.0`. STATE.md flags this as a blocker.
   - **Recommendation:** At implementation time, load the model and check `model._sklearn_version` attribute. Pin that version.

3. **`ultimate_feature_engineering.joblib` contents verification**
   - **What we know:** The trainer serializes the entire `AdvancedFeatureEngineering` object. It should contain all lookup dicts.
   - **What's unclear:** Whether the 2302 variant or the original is the correct one to use. Both are 9.5 MB.
   - **Recommendation:** Load and inspect both at implementation time. Check which has more champion/team coverage.

4. **Champion name format in lookup dicts**
   - **What we know:** The CSV contains names like "Kai'Sa", "Kha'Zix", "Lee Sin". The existing code uses `.title()` normalization.
   - **What's unclear:** Exact name format in the serialized lookup dicts (e.g., is it "Wukong" or "MonkeyKing" in champion_characteristics?).
   - **Recommendation:** Inspect the loaded dicts to determine canonical names, then build the alias mapping from that ground truth.

## Sources

### Primary (HIGH confidence)
- **Source code analysis** (`src/prediction/interactive_match_predictor.py`) -- full prediction flow, feature list, dual-perspective logic
- **Source code analysis** (`src/feature_engineering/advanced_feature_engineering.py`) -- lookup dict structures, feature computation, `create_advanced_features_vectorized` implementation
- **Source code analysis** (`src/config.py`) -- artifact paths, project structure
- **Source code analysis** (`src/models/trainer.py` line 990) -- confirms `ultimate_feature_engineering.joblib` serialization
- **File system inspection** -- artifact sizes (model 27MB/232KB, feature eng 9.5MB, meta 308KB, synergies 212KB, team perf 8KB)

### Secondary (MEDIUM confidence)
- **requirements.txt** -- version constraints (scikit-learn>=1.0.0, joblib>=1.1.0)
- **STATE.md blockers** -- sklearn version mismatch risk, RAM profiling needed

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries are already in the project; no new dependencies needed
- Architecture: HIGH -- derived directly from reading existing source code and understanding the exact data flow
- Pitfalls: HIGH -- identified from actual code inspection (feature ordering, state mutation, version mismatch)
- Open questions: MEDIUM -- require runtime inspection of serialized artifacts to resolve fully

**Research date:** 2026-02-24
**Valid until:** 2026-03-24 (stable domain; artifacts won't change unless model is retrained)
