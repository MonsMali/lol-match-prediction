"""Admin endpoints for model management.

Provides the model upload endpoint that allows hot-swapping ML
artifacts without redeploying. Secured with Bearer token
authentication via the ADMIN_TOKEN environment variable.
"""

import logging
import shutil
import tempfile
from pathlib import Path

import joblib
from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile

from api.dependencies import require_admin_token
from src.adapter import LoLDraftAdapter
from src.config import PRODUCTION_MODELS_DIR

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin", tags=["admin"])


@router.post("/upload-model")
async def upload_model(
    request: Request,
    model_file: UploadFile,
    scaler_file: UploadFile,
    encoders_file: UploadFile,
    _: None = Depends(require_admin_token),
) -> dict:
    """Upload new model artifacts and hot-swap the live model.

    Accepts three multipart file uploads: model, scaler, and encoders.
    The new artifacts are validated by loading them and running a test
    prediction before the live model is replaced. Requests during
    validation continue using the old model (zero downtime).

    Returns:
        Success message on successful swap.

    Raises:
        HTTPException: 422 if artifacts are invalid or test prediction
            fails. 401 if admin token is missing or incorrect.
    """
    tmp_dir = None
    try:
        # 1. Save uploaded files to a temporary directory
        tmp_dir = tempfile.mkdtemp(prefix="lol_model_upload_")
        tmp_path = Path(tmp_dir)

        file_mapping = {
            "best_model.joblib": model_file,
            "scaler.joblib": scaler_file,
            "encoders.joblib": encoders_file,
        }

        for filename, upload in file_mapping.items():
            content = await upload.read()
            (tmp_path / filename).write_bytes(content)

        # 2. Validate artifacts by loading them
        loaded = {}
        for filename in file_mapping:
            try:
                loaded[filename] = joblib.load(tmp_path / filename)
            except Exception as exc:
                raise HTTPException(
                    status_code=422,
                    detail=f"Invalid artifact '{filename}': {exc}",
                )

        # 3. Run a test prediction with the new artifacts
        #    Copy lookup dicts from the existing production directory so
        #    the test adapter can initialize fully.
        models_root = PRODUCTION_MODELS_DIR.parent
        for lookup_file in [
            "champion_meta_strength.joblib",
            "champion_synergies.joblib",
            "team_historical_performance.joblib",
        ]:
            src = models_root / lookup_file
            if src.exists():
                shutil.copy2(str(src), str(tmp_path / lookup_file))

        # Create a non-singleton adapter instance for testing
        test_adapter = object.__new__(LoLDraftAdapter)
        try:
            # Manually run __init__ logic on the fresh instance (bypassing
            # singleton __new__) by pointing to the temp directory.  The
            # adapter __init__ checks _initialized, so we ensure it is
            # absent.
            test_adapter.__init__(artifacts_dir=tmp_path)
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail=f"New artifacts failed to initialize adapter: {exc}",
            )

        # Build a minimal dummy draft for the test prediction
        roles = ["top", "jungle", "mid", "bot", "support"]
        champions = sorted(test_adapter.valid_champions)
        teams = sorted(test_adapter.valid_teams)

        dummy_draft = {
            "blue_team": teams[0] if teams else "Team A",
            "red_team": teams[1] if len(teams) > 1 else "Team B",
            "blue_picks": dict(zip(roles, champions[:5])) if len(champions) >= 5 else dict(zip(roles, ["Aatrox"] * 5)),
            "blue_bans": champions[5:10] if len(champions) >= 10 else ["Zed"] * 5,
            "red_picks": dict(zip(roles, champions[10:15])) if len(champions) >= 15 else dict(zip(roles, ["Ahri"] * 5)),
            "red_bans": champions[15:20] if len(champions) >= 20 else ["Lux"] * 5,
        }

        try:
            test_adapter.predict_from_draft(dummy_draft)
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Test prediction failed with new artifacts: {exc}",
            )

        # 4. Swap artifacts: copy validated files to production directory
        for filename in file_mapping:
            shutil.copy2(
                str(tmp_path / filename),
                str(PRODUCTION_MODELS_DIR / filename),
            )

        # 5. Reset the singleton so the next access loads fresh artifacts
        LoLDraftAdapter._instance = None
        if hasattr(LoLDraftAdapter, "_initialized"):
            del LoLDraftAdapter._initialized

        new_adapter = LoLDraftAdapter()
        request.app.state.adapter = new_adapter
        request.app.state.model_ready = True

        logger.info("Model swapped successfully: %s", new_adapter.model_name)

        return {"status": "success", "message": "Model swapped successfully"}

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as exc:
        logger.error("Model upload failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Model upload failed: {exc}",
        )
    finally:
        # 6. Clean up temporary directory
        if tmp_dir and Path(tmp_dir).exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
