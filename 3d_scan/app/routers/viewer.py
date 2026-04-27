"""Serve generated PLY and other output files."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.config import UPLOADS_DIR

router = APIRouter(prefix="/api", tags=["files"])


@router.get("/files/{job_id}/{file_path:path}")
async def serve_file(job_id: str, file_path: str):
    """Serve an output file (mesh.ply, pointcloud.ply, etc.)."""
    full_path = UPLOADS_DIR / job_id / file_path

    # Security: ensure the resolved path stays within uploads
    try:
        full_path.resolve().relative_to(UPLOADS_DIR.resolve())
    except ValueError:
        raise HTTPException(403, "Access denied.")

    if not full_path.is_file():
        raise HTTPException(404, "File not found.")

    media_type = "application/octet-stream"
    if full_path.suffix == ".ply":
        media_type = "application/x-ply"
    elif full_path.suffix == ".json":
        media_type = "application/json"

    return FileResponse(
        full_path,
        media_type=media_type,
        filename=full_path.name,
    )
