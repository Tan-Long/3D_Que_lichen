"""Samples management endpoints – backed by SQLite via database module."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.config import UPLOADS_DIR
from app.services import database as db

router = APIRouter(prefix="/api/samples", tags=["samples"])

SAMPLES_DIR = UPLOADS_DIR / "samples"
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)


# --------------- Models ---------------

class CreateSampleRequest(BaseModel):
    name: str
    job_id: str


class RenameSampleRequest(BaseModel):
    name: str


class RenameFileRequest(BaseModel):
    new_filename: str


# --------------- Helpers ---------------

def _file_type(suffix: str) -> str:
    suffix = suffix.lower()
    if suffix in (".png", ".jpg", ".jpeg", ".bmp", ".gif"):
        return "image"
    if suffix in (".ply", ".obj", ".stl", ".glb", ".gltf"):
        return "mesh"
    return "other"


def _get_sample_dir(sample_id: str) -> Path:
    return SAMPLES_DIR / sample_id


# --------------- Endpoints ---------------

@router.get("")
async def list_samples(status: str | None = None, assigned_to: str | None = None,
                       search: str | None = None):
    return await db.list_samples(status=status, assigned_to=assigned_to, search=search)


@router.post("")
async def create_sample(req: CreateSampleRequest):
    """Save the outputs of a processed job as a named sample."""
    job_dir = UPLOADS_DIR / req.job_id
    if not job_dir.is_dir():
        raise HTTPException(404, f"Job directory not found: {req.job_id}")

    # Generate sample ID and create directory
    sample_id = uuid.uuid4().hex[:12]
    sample_dir = SAMPLES_DIR / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Copy output files
    files = []
    for pattern in ("*.ply", "*.png", "*.jpg", "*.jpeg", "*.obj", "*.stl"):
        for f in job_dir.rglob(pattern):
            dest = sample_dir / f.name
            if dest.exists():
                dest = sample_dir / f"{f.stem}_{uuid.uuid4().hex[:4]}{f.suffix}"
            shutil.copy2(str(f), str(dest))
            files.append({
                "filename": dest.name,
                "original_name": f.name,
                "size": dest.stat().st_size,
                "type": _file_type(dest.suffix),
            })

    # Create in database (use the pre-generated sample_id)
    sample = await db.create_sample(
        name=req.name,
        job_id=req.job_id,
        files=files,
    )
    return sample


@router.get("/{sample_id}")
async def get_sample(sample_id: str):
    sample = await db.get_sample(sample_id)
    if not sample:
        raise HTTPException(404, "Sample not found")
    return sample


@router.put("/{sample_id}/rename")
async def rename_sample(sample_id: str, req: RenameSampleRequest):
    sample = await db.update_sample(sample_id, name=req.name)
    if not sample:
        raise HTTPException(404, "Sample not found")
    return sample


@router.delete("/{sample_id}")
async def delete_sample(sample_id: str):
    # Get sample to check for files on disk
    sample = await db.get_sample(sample_id)
    if not sample:
        raise HTTPException(404, "Sample not found")

    # Remove files on disk (only from samples dir, not source_path)
    sample_dir = _get_sample_dir(sample_id)
    if sample_dir.exists():
        shutil.rmtree(str(sample_dir))

    await db.delete_sample(sample_id)
    return {"ok": True}


@router.get("/{sample_id}/files/{filename}")
async def serve_sample_file(sample_id: str, filename: str):
    """Serve a file from a sample directory or its source_path."""
    sample = await db.get_sample(sample_id)
    if not sample:
        raise HTTPException(404, "Sample not found")

    # Try sample directory first
    sample_dir = _get_sample_dir(sample_id)
    file_path = sample_dir / filename
    if file_path.is_file():
        return _file_response(file_path)

    # Try source_path/output directory (for imported samples)
    if sample.get("source_path"):
        source_file = Path(sample["source_path"]) / "output" / filename
        if source_file.is_file():
            return _file_response(source_file)

    raise HTTPException(404, "File not found")


def _file_response(file_path: Path) -> FileResponse:
    media_type = "application/octet-stream"
    suffix = file_path.suffix.lower()
    if suffix in (".png",):
        media_type = "image/png"
    elif suffix in (".jpg", ".jpeg"):
        media_type = "image/jpeg"
    elif suffix == ".ply":
        media_type = "application/x-ply"
    return FileResponse(file_path, media_type=media_type, filename=file_path.name)


@router.put("/{sample_id}/files/{filename}/rename")
async def rename_file(sample_id: str, filename: str, req: RenameFileRequest):
    # Rename on disk
    sample_dir = _get_sample_dir(sample_id)
    old_path = sample_dir / filename
    if old_path.exists():
        old_path.rename(sample_dir / req.new_filename)

    sample = await db.rename_sample_file(sample_id, filename, req.new_filename)
    if not sample:
        raise HTTPException(404, "Sample or file not found")
    return sample


@router.delete("/{sample_id}/files/{filename}")
async def delete_file(sample_id: str, filename: str):
    # Delete on disk
    sample_dir = _get_sample_dir(sample_id)
    file_path = sample_dir / filename
    if file_path.exists():
        file_path.unlink()

    sample = await db.delete_sample_file(sample_id, filename)
    if not sample:
        raise HTTPException(404, "Sample or file not found")
    return sample


@router.post("/{sample_id}/files")
async def upload_file_to_sample(sample_id: str, file: UploadFile):
    """Upload/overwrite a file in a sample (e.g. after editing)."""
    sample = await db.get_sample(sample_id)
    if not sample:
        raise HTTPException(404, "Sample not found")

    filename = file.filename or "untitled.ply"
    data = await file.read()

    sample_dir = _get_sample_dir(sample_id)
    sample_dir.mkdir(parents=True, exist_ok=True)
    file_path = sample_dir / filename
    file_path.write_bytes(data)

    result = await db.add_sample_file(
        sample_id, filename,
        file_type=_file_type(Path(filename).suffix),
        size=len(data),
    )
    return result


@router.post("/{sample_id}/reset")
async def reset_sample_to_original(sample_id: str):
    """Reset sample files to source_path/output (discard edited overrides)."""
    sample = await db.get_sample(sample_id)
    if not sample:
        raise HTTPException(404, "Sample not found")

    source_path = sample.get("source_path")
    if not source_path:
        raise HTTPException(400, "This sample has no source_path to reset from")

    output_dir = Path(source_path) / "output"
    if not output_dir.is_dir():
        raise HTTPException(400, "Original output directory not found")

    # Remove local override files so file serving falls back to source_path/output
    sample_dir = _get_sample_dir(sample_id)
    if sample_dir.exists():
        shutil.rmtree(str(sample_dir), ignore_errors=True)

    files = []
    for f in output_dir.iterdir():
        if f.is_file() and f.suffix.lower() in (".ply", ".png", ".jpg", ".jpeg", ".obj", ".stl"):
            files.append({
                "filename": f.name,
                "original_name": f.name,
                "type": _file_type(f.suffix),
                "size": f.stat().st_size,
            })

    updated = await db.replace_sample_files(sample_id, files)
    return {"ok": True, "sample": updated}