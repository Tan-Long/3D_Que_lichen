"""Refine endpoint: accepts cleaned PLY and re-processes at high quality with RGB."""

from __future__ import annotations

import asyncio
import uuid

from fastapi import APIRouter, Form, HTTPException, UploadFile

from app.config import UPLOADS_DIR
from app.services import job_manager
from app.services.pipeline import _find_dataset_dir, run_refine_pipeline

router = APIRouter(prefix="/api", tags=["refine"])


@router.post("/refine")
async def refine_mesh(file: UploadFile, job_id: str = Form(...)):
    """Accept a cleaned PLY and re-process at higher quality with RGB."""
    job_dir = UPLOADS_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(404, "Original job not found.")

    dataset_dir = _find_dataset_dir(job_dir)
    if dataset_dir is None:
        raise HTTPException(404, "Dataset directory not found in job.")

    # Save cleaned PLY
    contents = await file.read()
    output_dir = dataset_dir / "output"
    output_dir.mkdir(exist_ok=True)
    cleaned_path = output_dir / "cleaned.ply"
    cleaned_path.write_bytes(contents)

    # Create refine job with unique ID
    refine_job_id = job_id + "_ref_" + uuid.uuid4().hex[:6]
    await job_manager.create_job(refine_job_id)

    # Start refine pipeline in background
    asyncio.create_task(
        run_refine_pipeline(refine_job_id, job_id, str(dataset_dir))
    )

    return {"job_id": refine_job_id, "status": "queued"}
