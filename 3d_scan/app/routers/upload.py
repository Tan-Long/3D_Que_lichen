"""Upload endpoint: accepts zip, kicks off processing pipeline."""

from __future__ import annotations

import asyncio
import uuid

from fastapi import APIRouter, HTTPException, UploadFile

from app.config import MAX_UPLOAD_SIZE, UPLOADS_DIR
from app.services import job_manager
from app.services.pipeline import run_pipeline

router = APIRouter(prefix="/api", tags=["upload"])


@router.post("/upload")
async def upload_dataset(file: UploadFile):
    """Upload a Stray Scanner .zip dataset for processing."""
    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(400, "Please upload a .zip file.")

    # Read file (with size check)
    contents = await file.read()
    if len(contents) > MAX_UPLOAD_SIZE:
        raise HTTPException(413, f"File too large. Max {MAX_UPLOAD_SIZE // (1024*1024)} MB.")

    # Create job
    job_id = uuid.uuid4().hex[:12]
    job_dir = UPLOADS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    zip_path = job_dir / file.filename
    zip_path.write_bytes(contents)

    await job_manager.create_job(job_id)

    # Start pipeline in background
    asyncio.create_task(run_pipeline(job_id, str(zip_path)))

    return {"job_id": job_id, "status": "queued"}
