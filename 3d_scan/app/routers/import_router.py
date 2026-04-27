"""Bulk import endpoints for registering existing dataset directories."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.config import EXTERNAL_DATA_DIR
from app.services import database as db

router = APIRouter(prefix="/api/import", tags=["import"])


class ImportDirectoryRequest(BaseModel):
    path: str
    name: str | None = None


class BulkImportRequest(BaseModel):
    base_path: str | None = None  # defaults to EXTERNAL_DATA_DIR


@router.get("/preview")
async def preview_import(base_path: str | None = None):
    """Dry run: show what directories would be imported."""
    path = base_path or EXTERNAL_DATA_DIR
    results = await db.preview_import(path)
    return {"base_path": path, "directories": results}


@router.post("/directory")
async def import_single_directory(req: ImportDirectoryRequest):
    """Import a single dataset directory."""
    result = await db.import_directory(req.path, req.name)
    if not result:
        raise HTTPException(400, "Invalid directory: camera_matrix.csv not found")
    return result


@router.post("/bulk")
async def bulk_import(req: BulkImportRequest):
    """Scan a base directory and import all valid dataset subdirectories."""
    path = req.base_path or EXTERNAL_DATA_DIR
    imported = await db.bulk_import(path)
    return {"imported_count": len(imported), "samples": imported}
