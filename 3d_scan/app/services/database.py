"""SQLite persistence layer with async support via aiosqlite.

Replaces the JSON-based sample_store for concurrent multi-user access.
Uses WAL mode for safe concurrent reads/writes.
"""

from __future__ import annotations

import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

import aiosqlite

from app.config import DATABASE_PATH

_db_path: str = str(DATABASE_PATH)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS samples (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    source_path TEXT,
    job_id TEXT,
    status TEXT NOT NULL DEFAULT 'unprocessed',
    assigned_to TEXT,
    assigned_at REAL,
    notes TEXT DEFAULT '',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS sample_files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sample_id TEXT NOT NULL REFERENCES samples(id) ON DELETE CASCADE,
    filename TEXT NOT NULL,
    original_name TEXT,
    file_type TEXT,
    size INTEGER DEFAULT 0,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS labels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    color TEXT NOT NULL,
    sort_order INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS annotations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sample_id TEXT NOT NULL REFERENCES samples(id) ON DELETE CASCADE,
    mesh_filename TEXT NOT NULL,
    label_data TEXT NOT NULL DEFAULT '{}',
    created_by TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
"""

_DEFAULT_LABELS = [
    ("wall", "#4A90D9", 0),
    ("floor", "#7EC850", 1),
    ("ceiling", "#D94A4A", 2),
    ("door", "#D9A74A", 3),
    ("window", "#9B59B6", 4),
    ("furniture", "#E67E22", 5),
    ("object", "#1ABC9C", 6),
    ("other", "#95A5A6", 7),
]

VALID_STATUSES = [
    "unprocessed",
    "rgb_extracted",
    "meshed",
    "filtered",
    "labeled",
    "complete",
]


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------

async def init_db() -> None:
    """Create tables and seed default labels."""
    async with aiosqlite.connect(_db_path) as db:
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("PRAGMA foreign_keys=ON")
        await db.executescript(_SCHEMA)

        # Seed labels if empty
        cursor = await db.execute("SELECT COUNT(*) FROM labels")
        count = (await cursor.fetchone())[0]
        if count == 0:
            await db.executemany(
                "INSERT INTO labels (name, color, sort_order) VALUES (?, ?, ?)",
                _DEFAULT_LABELS,
            )
        await db.commit()


def _row_to_dict(row: aiosqlite.Row, description: list) -> dict:
    return {desc[0]: row[i] for i, desc in enumerate(description)}


# ---------------------------------------------------------------------------
# Samples CRUD
# ---------------------------------------------------------------------------

async def list_samples(
    status: str | None = None,
    assigned_to: str | None = None,
    search: str | None = None,
) -> list[dict]:
    """List samples with optional filters."""
    query = "SELECT * FROM samples WHERE 1=1"
    params: list[Any] = []

    if status:
        query += " AND status = ?"
        params.append(status)
    if assigned_to:
        query += " AND assigned_to = ?"
        params.append(assigned_to)
    if search:
        query += " AND name LIKE ?"
        params.append(f"%{search}%")

    query += " ORDER BY created_at DESC"

    async with aiosqlite.connect(_db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(query, params)
        rows = await cursor.fetchall()
        samples = [dict(row) for row in rows]

        # Attach files to each sample
        for sample in samples:
            cursor = await db.execute(
                "SELECT * FROM sample_files WHERE sample_id = ? ORDER BY created_at",
                (sample["id"],),
            )
            files = await cursor.fetchall()
            sample["files"] = [dict(f) for f in files]

        return samples


async def get_sample(sample_id: str) -> dict | None:
    async with aiosqlite.connect(_db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM samples WHERE id = ?", (sample_id,))
        row = await cursor.fetchone()
        if not row:
            return None
        sample = dict(row)

        cursor = await db.execute(
            "SELECT * FROM sample_files WHERE sample_id = ? ORDER BY created_at",
            (sample_id,),
        )
        files = await cursor.fetchall()
        sample["files"] = [dict(f) for f in files]
        return sample


async def create_sample(
    name: str,
    source_path: str | None = None,
    job_id: str | None = None,
    status: str = "unprocessed",
    files: list[dict] | None = None,
) -> dict:
    """Create a new sample record."""
    sample_id = uuid.uuid4().hex[:12]
    now = time.time()

    async with aiosqlite.connect(_db_path) as db:
        await db.execute(
            """INSERT INTO samples (id, name, source_path, job_id, status, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (sample_id, name, source_path, job_id, status, now, now),
        )

        if files:
            for f in files:
                await db.execute(
                    """INSERT INTO sample_files (sample_id, filename, original_name, file_type, size, created_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (sample_id, f["filename"], f.get("original_name", f["filename"]),
                     f.get("type", "other"), f.get("size", 0), now),
                )

        await db.commit()

    return await get_sample(sample_id)


async def update_sample(sample_id: str, **kwargs) -> dict | None:
    """Update sample fields (name, status, assigned_to, notes, etc.)."""
    allowed = {"name", "status", "assigned_to", "assigned_at", "notes", "job_id"}
    updates = {k: v for k, v in kwargs.items() if k in allowed}
    if not updates:
        return await get_sample(sample_id)

    updates["updated_at"] = time.time()
    set_clause = ", ".join(f"{k} = ?" for k in updates)
    values = list(updates.values()) + [sample_id]

    async with aiosqlite.connect(_db_path) as db:
        cursor = await db.execute(
            f"UPDATE samples SET {set_clause} WHERE id = ?", values
        )
        if cursor.rowcount == 0:
            return None
        await db.commit()

    return await get_sample(sample_id)


async def delete_sample(sample_id: str) -> bool:
    async with aiosqlite.connect(_db_path) as db:
        await db.execute("PRAGMA foreign_keys=ON")
        cursor = await db.execute("DELETE FROM samples WHERE id = ?", (sample_id,))
        await db.commit()
        return cursor.rowcount > 0


# ---------------------------------------------------------------------------
# Sample Files
# ---------------------------------------------------------------------------

async def add_sample_file(sample_id: str, filename: str, original_name: str | None = None,
                          file_type: str = "other", size: int = 0) -> dict | None:
    now = time.time()
    async with aiosqlite.connect(_db_path) as db:
        # Check if file already exists, update if so
        cursor = await db.execute(
            "SELECT id FROM sample_files WHERE sample_id = ? AND filename = ?",
            (sample_id, filename),
        )
        existing = await cursor.fetchone()
        if existing:
            await db.execute(
                "UPDATE sample_files SET size = ?, created_at = ? WHERE id = ?",
                (size, now, existing[0]),
            )
        else:
            await db.execute(
                """INSERT INTO sample_files (sample_id, filename, original_name, file_type, size, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (sample_id, filename, original_name or filename, file_type, size, now),
            )
        await db.execute(
            "UPDATE samples SET updated_at = ? WHERE id = ?", (now, sample_id)
        )
        await db.commit()
    return await get_sample(sample_id)


async def rename_sample_file(sample_id: str, old_filename: str, new_filename: str) -> dict | None:
    now = time.time()
    async with aiosqlite.connect(_db_path) as db:
        cursor = await db.execute(
            "UPDATE sample_files SET filename = ? WHERE sample_id = ? AND filename = ?",
            (new_filename, sample_id, old_filename),
        )
        if cursor.rowcount == 0:
            return None
        await db.execute(
            "UPDATE samples SET updated_at = ? WHERE id = ?", (now, sample_id)
        )
        await db.commit()
    return await get_sample(sample_id)


async def delete_sample_file(sample_id: str, filename: str) -> dict | None:
    now = time.time()
    async with aiosqlite.connect(_db_path) as db:
        cursor = await db.execute(
            "DELETE FROM sample_files WHERE sample_id = ? AND filename = ?",
            (sample_id, filename),
        )
        if cursor.rowcount == 0:
            return None
        await db.execute(
            "UPDATE samples SET updated_at = ? WHERE id = ?", (now, sample_id)
        )
        await db.commit()
    return await get_sample(sample_id)


async def replace_sample_files(sample_id: str, files: list[dict]) -> dict | None:
    """Replace all file rows for a sample (used by reset-to-original)."""
    now = time.time()
    async with aiosqlite.connect(_db_path) as db:
        await db.execute("DELETE FROM sample_files WHERE sample_id = ?", (sample_id,))
        for f in files:
            await db.execute(
                """INSERT INTO sample_files (sample_id, filename, original_name, file_type, size, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    sample_id,
                    f["filename"],
                    f.get("original_name", f["filename"]),
                    f.get("type", "other"),
                    f.get("size", 0),
                    now,
                ),
            )
        await db.execute(
            "UPDATE samples SET updated_at = ? WHERE id = ?", (now, sample_id)
        )
        await db.commit()
    return await get_sample(sample_id)


# ---------------------------------------------------------------------------
# Bulk Import
# ---------------------------------------------------------------------------

async def import_directory(dir_path: str, name: str | None = None) -> dict | None:
    """Register an existing dataset directory as a sample.

    The directory must contain camera_matrix.csv to be valid.
    """
    p = Path(dir_path)
    if not (p / "camera_matrix.csv").exists():
        return None

    sample_name = name or p.name

    # Check if already imported
    async with aiosqlite.connect(_db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT id FROM samples WHERE source_path = ?", (str(p),)
        )
        existing = await cursor.fetchone()

    # Detect status based on existing outputs
    status = "unprocessed"
    output_dir = p / "output"
    if (output_dir / "mesh.ply").exists():
        status = "meshed"
    elif (p / "rgb").is_dir() and any((p / "rgb").iterdir()):
        status = "rgb_extracted"

    # Collect existing output files
    files = []
    if output_dir.is_dir():
        for f in output_dir.iterdir():
            if f.is_file() and f.suffix.lower() in (".ply", ".png", ".jpg", ".jpeg", ".obj"):
                files.append({
                    "filename": f.name,
                    "original_name": f.name,
                    "type": _file_type(f.suffix),
                    "size": f.stat().st_size,
                })

    # If already imported: refresh status & file list from disk.
    if existing:
        sample_id = existing["id"]
        await update_sample(sample_id, status=status)
        await replace_sample_files(sample_id, files)
        return await get_sample(sample_id)

    return await create_sample(
        name=sample_name,
        source_path=str(p),
        status=status,
        files=files,
    )


async def bulk_import(base_path: str) -> list[dict]:
    """Scan base_path for subdirectories with camera_matrix.csv and import them."""
    base = Path(base_path)
    imported = []

    if not base.is_dir():
        return imported

    for child in sorted(base.iterdir()):
        if child.is_dir() and (child / "camera_matrix.csv").exists():
            result = await import_directory(str(child))
            if result:
                imported.append(result)

    return imported


async def preview_import(base_path: str) -> list[dict]:
    """Dry run: show what would be imported from base_path."""
    base = Path(base_path)
    results = []

    if not base.is_dir():
        return results

    for child in sorted(base.iterdir()):
        if child.is_dir() and (child / "camera_matrix.csv").exists():
            # Check if already imported
            async with aiosqlite.connect(_db_path) as db:
                cursor = await db.execute(
                    "SELECT id FROM samples WHERE source_path = ?", (str(child),)
                )
                existing = await cursor.fetchone()

            has_output = (child / "output" / "mesh.ply").exists()
            results.append({
                "path": str(child),
                "name": child.name,
                "already_imported": existing is not None,
                "has_output": has_output,
            })

    return results


# ---------------------------------------------------------------------------
# Workflow helpers
# ---------------------------------------------------------------------------

async def get_workflow_summary() -> dict:
    """Get counts by status and by assigned_to."""
    async with aiosqlite.connect(_db_path) as db:
        # By status
        cursor = await db.execute(
            "SELECT status, COUNT(*) as count FROM samples GROUP BY status"
        )
        by_status = {row[0]: row[1] for row in await cursor.fetchall()}

        # By assignee
        cursor = await db.execute(
            "SELECT assigned_to, COUNT(*) as count FROM samples WHERE assigned_to IS NOT NULL GROUP BY assigned_to"
        )
        by_assignee = {row[0]: row[1] for row in await cursor.fetchall()}

        # Total
        cursor = await db.execute("SELECT COUNT(*) FROM samples")
        total = (await cursor.fetchone())[0]

        return {
            "total": total,
            "by_status": by_status,
            "by_assignee": by_assignee,
        }


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

async def list_labels() -> list[dict]:
    async with aiosqlite.connect(_db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM labels ORDER BY sort_order")
        return [dict(row) for row in await cursor.fetchall()]


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
