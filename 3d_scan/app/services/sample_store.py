"""Persistent sample store backed by a JSON file.

Each sample represents a named 3D scan with associated output files (images,
meshes, point clouds).  The store lives at ``<UPLOADS_DIR>/samples.json``.
"""

from __future__ import annotations

import json
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

from app.config import UPLOADS_DIR

SAMPLES_DIR = UPLOADS_DIR / "samples"
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

_DB_PATH = SAMPLES_DIR / "samples.json"


def _load_db() -> dict[str, Any]:
    if _DB_PATH.exists():
        return json.loads(_DB_PATH.read_text())
    return {}


def _save_db(db: dict[str, Any]) -> None:
    _DB_PATH.write_text(json.dumps(db, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def list_samples() -> list[dict]:
    """Return all samples sorted by creation time (newest first)."""
    db = _load_db()
    samples = list(db.values())
    samples.sort(key=lambda s: s.get("created_at", 0), reverse=True)
    return samples


def get_sample(sample_id: str) -> dict | None:
    return _load_db().get(sample_id)


def create_sample(name: str, job_id: str, source_dir: str) -> dict:
    """Create a new sample by copying output files from a processed job."""
    sample_id = uuid.uuid4().hex[:12]
    sample_dir = SAMPLES_DIR / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    src = Path(source_dir)
    files: list[dict] = []

    # Copy all relevant output files
    for pattern in ("*.ply", "*.png", "*.jpg", "*.jpeg", "*.obj", "*.stl"):
        for f in src.rglob(pattern):
            dest = sample_dir / f.name
            # Avoid name collisions
            if dest.exists():
                dest = sample_dir / f"{f.stem}_{uuid.uuid4().hex[:4]}{f.suffix}"
            shutil.copy2(str(f), str(dest))
            files.append({
                "filename": dest.name,
                "original_name": f.name,
                "size": dest.stat().st_size,
                "type": _file_type(dest.suffix),
            })

    sample = {
        "id": sample_id,
        "name": name,
        "job_id": job_id,
        "created_at": time.time(),
        "updated_at": time.time(),
        "files": files,
    }

    db = _load_db()
    db[sample_id] = sample
    _save_db(db)
    return sample


def rename_sample(sample_id: str, new_name: str) -> dict | None:
    db = _load_db()
    if sample_id not in db:
        return None
    db[sample_id]["name"] = new_name
    db[sample_id]["updated_at"] = time.time()
    _save_db(db)
    return db[sample_id]


def delete_sample(sample_id: str) -> bool:
    db = _load_db()
    if sample_id not in db:
        return False
    # Remove files on disk
    sample_dir = SAMPLES_DIR / sample_id
    if sample_dir.exists():
        shutil.rmtree(str(sample_dir))
    del db[sample_id]
    _save_db(db)
    return True


def rename_file(sample_id: str, old_filename: str, new_filename: str) -> dict | None:
    db = _load_db()
    sample = db.get(sample_id)
    if not sample:
        return None

    sample_dir = SAMPLES_DIR / sample_id
    old_path = sample_dir / old_filename
    if not old_path.exists():
        return None

    # Keep the same extension
    new_path = sample_dir / new_filename
    old_path.rename(new_path)

    for f in sample["files"]:
        if f["filename"] == old_filename:
            f["filename"] = new_filename
            break

    sample["updated_at"] = time.time()
    _save_db(db)
    return sample


def delete_file(sample_id: str, filename: str) -> dict | None:
    db = _load_db()
    sample = db.get(sample_id)
    if not sample:
        return None

    sample_dir = SAMPLES_DIR / sample_id
    file_path = sample_dir / filename
    if file_path.exists():
        file_path.unlink()

    sample["files"] = [f for f in sample["files"] if f["filename"] != filename]
    sample["updated_at"] = time.time()
    _save_db(db)
    return sample


def save_edited_file(sample_id: str, filename: str, data: bytes) -> dict | None:
    """Save/overwrite a file in a sample (e.g. after editing in the viewer)."""
    db = _load_db()
    sample = db.get(sample_id)
    if not sample:
        return None

    sample_dir = SAMPLES_DIR / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    file_path = sample_dir / filename
    file_path.write_bytes(data)

    # Update or add file entry
    found = False
    for f in sample["files"]:
        if f["filename"] == filename:
            f["size"] = len(data)
            found = True
            break
    if not found:
        sample["files"].append({
            "filename": filename,
            "original_name": filename,
            "size": len(data),
            "type": _file_type(file_path.suffix),
        })

    sample["updated_at"] = time.time()
    _save_db(db)
    return sample


def get_sample_dir(sample_id: str) -> Path | None:
    d = SAMPLES_DIR / sample_id
    return d if d.is_dir() else None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _file_type(suffix: str) -> str:
    suffix = suffix.lower()
    if suffix in (".png", ".jpg", ".jpeg", ".bmp", ".gif"):
        return "image"
    if suffix in (".ply", ".obj", ".stl", ".glb", ".gltf"):
        return "mesh"
    return "other"
