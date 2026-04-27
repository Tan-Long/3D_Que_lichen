"""FastAPI application entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from app.routers.upload import router as upload_router
from app.routers.jobs import router as jobs_router, ws_router
from app.routers.viewer import router as viewer_router
from app.routers.refine import router as refine_router
from app.routers.samples import router as samples_router
from app.routers.import_router import router as import_router
from app.services.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize database
    await init_db()
    yield
    # Shutdown: nothing to clean up


app = FastAPI(title="3D Scan Processor", version="2.0.0", lifespan=lifespan)

# API routes
app.include_router(upload_router)
app.include_router(jobs_router)
app.include_router(viewer_router)
app.include_router(refine_router)
app.include_router(samples_router)
app.include_router(import_router)
app.include_router(ws_router)

# Serve frontend static files (index.html, JS, CSS)
static_dir = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
