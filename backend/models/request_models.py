from __future__ import annotations
from enum import Enum
from pydantic import BaseModel, Field


class SketchStyle(str, Enum):
    construction = "construction lines"
    ink = "ink"
    pencil = "pencil"


class SketchFormat(str, Enum):
    procreate = "procreate"
    png = "png"


class GenerateRequest(BaseModel):
    scene: str = Field(..., min_length=3, max_length=300,
                       example="forest path, evening light")
    style: SketchStyle = SketchStyle.construction
    vanishing_points: int = Field(1, ge=1, le=3)
    format: SketchFormat = SketchFormat.procreate


class JobStatus(str, Enum):
    queued     = "queued"
    generating = "generating"
    splitting  = "splitting"
    packaging  = "packaging"
    done       = "done"
    failed     = "failed"


class GenerateResponse(BaseModel):
    job_id: str
    status: JobStatus


class StatusResponse(BaseModel):
    job_id:   str
    status:   JobStatus
    progress: int = Field(..., ge=0, le=100, description="0-100")
    error:    str | None = None
