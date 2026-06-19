from fastapi import APIRouter, HTTPException

from backend.models.request_models import StatusResponse
from backend.routes.generate import jobs

router = APIRouter()


@router.get("/status/{job_id}", response_model=StatusResponse)
async def status(job_id: str):
    """Return current status and progress percentage for a job."""
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
    return StatusResponse(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        error=job.get("error"),
    )
