from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from backend.models.request_models import JobStatus
from backend.routes.generate import jobs

router = APIRouter()


@router.get("/download/{job_id}")
async def download(job_id: str):
    """
    Download the finished .procreate file for a completed job.
    Returns 404 if the job doesn't exist, 409 if it isn't done yet.
    """
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")

    if job["status"] == JobStatus.failed:
        raise HTTPException(status_code=500,
                            detail=f"Job failed: {job.get('error')}")

    if job["status"] != JobStatus.done:
        raise HTTPException(
            status_code=409,
            detail=f"Job not ready — current status: {job['status']} ({job['progress']}%)",
        )

    file_path = Path(job["file_path"])
    if not file_path.exists():
        raise HTTPException(status_code=500, detail="Output file missing on disk")

    return FileResponse(
        path=file_path,
        media_type="application/octet-stream",
        filename=f"sketchai-{job_id}.procreate",
    )
