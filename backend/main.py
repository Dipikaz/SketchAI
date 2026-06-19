from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes import generate, status, download

app = FastAPI(
    title="SketchAI",
    description="AI sketch generator — generates layered .procreate files from text prompts.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(generate.router, tags=["generation"])
app.include_router(status.router,   tags=["generation"])
app.include_router(download.router, tags=["generation"])


@app.get("/health")
async def health():
    return {"status": "ok"}
