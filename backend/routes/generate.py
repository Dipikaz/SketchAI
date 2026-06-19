from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException

from backend.models.request_models import (
    GenerateRequest, GenerateResponse, JobStatus,
)
from backend.services.prompt_builder   import build_prompt
from backend.services.sketch_generator import generate_sketch
from backend.services.layer_splitter   import split_layers
from backend.services.procreate_exporter import export_procreate

router = APIRouter()

# ---------------------------------------------------------------------------
# In-memory job store — shared with status.py and download.py via import
# ---------------------------------------------------------------------------

jobs: dict[str, dict] = {}

OUTPUTS_DIR = Path.home() / "sketchai" / "outputs"


def _update(job_id: str, **kwargs) -> None:
    jobs[job_id].update(kwargs)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

async def _run_pipeline(job_id: str, request: GenerateRequest) -> None:
    job_dir = OUTPUTS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 1 — Build prompt
        _update(job_id, status=JobStatus.generating, progress=5)
        positive, negative = build_prompt(
            request.scene, request.style.value, request.vanishing_points
        )

        # 2 — Generate via ComfyUI (~20-30 s)
        sketch_path = await generate_sketch(positive, negative, job_dir)
        _update(job_id, progress=60)

        # 3 — Split into 6 layers
        _update(job_id, status=JobStatus.splitting, progress=65)
        layers_dir = job_dir / "layers"
        split_layers(sketch_path, layers_dir)
        _update(job_id, progress=80)

        # 4 — Package .procreate
        _update(job_id, status=JobStatus.packaging, progress=85)
        out_path = job_dir / "output.procreate"
        export_procreate(layers_dir, out_path)

        _update(job_id, status=JobStatus.done, progress=100,
                file_path=str(out_path))

    except Exception as exc:
        _update(job_id, status=JobStatus.failed, progress=0,
                error=str(exc))


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@router.post("/generate", response_model=GenerateResponse, status_code=202)
async def generate(request: GenerateRequest, background_tasks: BackgroundTasks):
    """
    Start a sketch generation job.
    Returns immediately with a job_id; poll GET /status/{job_id} for progress.
    """
    job_id = uuid.uuid4().hex[:10]
    jobs[job_id] = {
        "status":    JobStatus.queued,
        "progress":  0,
        "file_path": None,
        "error":     None,
    }
    background_tasks.add_task(_run_pipeline, job_id, request)
    return GenerateResponse(job_id=job_id, status=JobStatus.queued)
