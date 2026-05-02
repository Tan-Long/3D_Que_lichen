"""Orchestrates the 3-step processing pipeline via subprocesses."""

from __future__ import annotations

import asyncio
import os
import re
import subprocess
import zipfile
from pathlib import Path

from app.config import SCRIPTS_DIR, UPLOADS_DIR, VENV_PYTHON
from app.services.job_manager import JobStatus, update_job


def _find_dataset_dir(job_dir: Path) -> Path | None:
    """After extracting a zip, find the dataset folder containing camera_matrix.csv."""
    # Direct files in job_dir
    if (job_dir / "camera_matrix.csv").exists():
        return job_dir
    # One level deep (typical: zip contains a named folder)
    for child in job_dir.iterdir():
        if child.is_dir() and (child / "camera_matrix.csv").exists():
            return child
    return None


def _parse_progress(line: str, total_frames: int) -> float | None:
    """Try to extract progress from script output lines like '(51/370)'."""
    m = re.search(r"\((\d+)/(\d+)\)", line)
    if m:
        current = int(m.group(1))
        total = int(m.group(2))
        if total > 0:
            return current / total
    return None


async def _run_script(
    job_id: str,
    script_name: str,
    args: list[str],
    step_name: str,
    base_progress: float,
    step_weight: float,
) -> bool:
    """Run a processing script as subprocess with progress tracking.

    Args:
        job_id: Job identifier.
        script_name: Script filename (e.g., '01_extract_rgb_frames.py').
        args: CLI arguments for the script.
        step_name: Human-readable step name.
        base_progress: Starting progress value for this step (0.0 - 1.0).
        step_weight: How much of the total progress this step represents.

    Returns:
        True if successful, False otherwise.
    """
    await update_job(
        job_id,
        step=step_name,
        progress=base_progress,
        message=f"Starting {step_name}...",
    )

    script_path = str(SCRIPTS_DIR / script_name)
    cmd = [VENV_PYTHON, script_path] + args

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(SCRIPTS_DIR),
    )

    total_frames = 0
    output_lines: list[str] = []
    async for raw_line in proc.stdout:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line:
            continue

        output_lines.append(line)

        # Try to get total frames count
        m = re.search(r"Found (\d+) depth frames", line)
        if m:
            total_frames = int(m.group(1))

        # Parse step-internal progress
        pct = _parse_progress(line, total_frames)
        step_progress = base_progress
        if pct is not None:
            step_progress = base_progress + pct * step_weight

        await update_job(job_id, progress=step_progress, message=line)

    await proc.wait()

    if proc.returncode != 0:
        tail = "\n".join(output_lines[-15:]) if output_lines else "(no output captured)"
        await update_job(
            job_id,
            status=JobStatus.ERROR,
            error=f"{step_name} failed (exit code {proc.returncode}):\n{tail}",
        )
        return False

    await update_job(job_id, progress=base_progress + step_weight)
    return True


async def run_pipeline(job_id: str, zip_path: str) -> None:
    """Full processing pipeline: extract zip -> extract frames -> generate mesh."""
    job_dir = UPLOADS_DIR / job_id

    try:
        # --- Step 0: Extract ZIP ---
        await update_job(
            job_id,
            status=JobStatus.PROCESSING,
            step="Extracting ZIP",
            progress=0.0,
            message="Extracting uploaded archive...",
        )

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(job_dir)

        # Remove __MACOSX junk if present
        macosx_dir = job_dir / "__MACOSX"
        if macosx_dir.exists():
            import shutil
            shutil.rmtree(macosx_dir)

        dataset_dir = _find_dataset_dir(job_dir)
        if dataset_dir is None:
            await update_job(
                job_id,
                status=JobStatus.ERROR,
                error="Invalid dataset: camera_matrix.csv not found in zip.",
            )
            return

        dataset_str = str(dataset_dir)
        await update_job(job_id, progress=0.05, message=f"Dataset: {dataset_dir.name}")

        # --- Step 1: Extract RGB frames (5% -> 20%) ---
        ok = await _run_script(
            job_id,
            "01_extract_rgb_frames.py",
            [dataset_str],
            step_name="Extracting RGB frames",
            base_progress=0.05,
            step_weight=0.15,
        )
        if not ok:
            return

        # --- Step 2: Generate point cloud & mesh (20% -> 95%) ---
        ok = await _run_script(
            job_id,
            "02_pointcloud_and_mesh.py",
            [dataset_str, "--no-visualize", "--skip-every", "5"],
            step_name="Generating 3D mesh",
            base_progress=0.20,
            step_weight=0.75,
        )
        if not ok:
            return

        # --- Done ---
        output_dir = dataset_dir / "output"
        mesh_path = output_dir / "mesh.ply"
        pcd_path = output_dir / "pointcloud.ply"

        result = {}
        if mesh_path.exists():
            result["mesh"] = f"/api/files/{job_id}/{dataset_dir.name}/output/mesh.ply"
        if pcd_path.exists():
            result["pointcloud"] = f"/api/files/{job_id}/{dataset_dir.name}/output/pointcloud.ply"

        await update_job(
            job_id,
            status=JobStatus.COMPLETE,
            step="Complete",
            progress=1.0,
            message="Processing complete!",
            result=result,
        )

    except Exception as e:
        await update_job(
            job_id,
            status=JobStatus.ERROR,
            error=str(e),
        )


async def run_refine_pipeline(
    refine_job_id: str,
    original_job_id: str,
    dataset_dir_str: str,
) -> None:
    """Re-process at high quality using the cleaned mesh as spatial mask."""
    try:
        await update_job(
            refine_job_id,
            status=JobStatus.PROCESSING,
            step="Refining mesh",
            progress=0.0,
            message="Starting high-quality refinement...",
        )

        dataset_dir = Path(dataset_dir_str)

        # Run refine — use skip_every=2 and voxel_size=0.004 for
        # a good balance of quality vs speed
        ok = await _run_script(
            refine_job_id,
            "02_pointcloud_and_mesh.py",
            [
                dataset_dir_str,
                "--no-visualize",
                "--refine-from",
                str(dataset_dir / "output" / "cleaned.ply"),
                "--voxel-size", "0.004",
                "--skip-every", "3",
            ],
            step_name="Refining mesh (high-res RGB)",
            base_progress=0.0,
            step_weight=0.95,
        )
        if not ok:
            return

        refined_path = dataset_dir / "output" / "refined_mesh.ply"
        if refined_path.exists():
            result = {
                "mesh": f"/api/files/{original_job_id}/{dataset_dir.name}/output/refined_mesh.ply"
            }
            await update_job(
                refine_job_id,
                status=JobStatus.COMPLETE,
                step="Complete",
                progress=1.0,
                message="Refinement complete!",
                result=result,
            )
        else:
            await update_job(
                refine_job_id,
                status=JobStatus.ERROR,
                error="Refined mesh was not generated.",
            )

    except Exception as e:
        await update_job(
            refine_job_id,
            status=JobStatus.ERROR,
            error=str(e),
        )
