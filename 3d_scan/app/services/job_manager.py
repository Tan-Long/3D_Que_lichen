"""In-memory job state tracker with WebSocket broadcast."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from fastapi import WebSocket


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class Job:
    job_id: str
    status: JobStatus = JobStatus.QUEUED
    step: str = ""
    progress: float = 0.0
    message: str = ""
    result: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


# Global state
_jobs: dict[str, Job] = {}
_subscribers: dict[str, list[WebSocket]] = {}
_lock = asyncio.Lock()


async def create_job(job_id: str) -> Job:
    job = Job(job_id=job_id)
    _jobs[job_id] = job
    return job


async def get_job(job_id: str) -> Job | None:
    return _jobs.get(job_id)


async def update_job(job_id: str, **kwargs: Any) -> None:
    """Update job fields and notify all WebSocket subscribers."""
    job = _jobs.get(job_id)
    if not job:
        return
    for key, value in kwargs.items():
        if hasattr(job, key):
            setattr(job, key, value)
    await _broadcast(job_id)


async def subscribe(job_id: str, ws: WebSocket) -> None:
    if job_id not in _subscribers:
        _subscribers[job_id] = []
    _subscribers[job_id].append(ws)


async def unsubscribe(job_id: str, ws: WebSocket) -> None:
    if job_id in _subscribers:
        _subscribers[job_id] = [w for w in _subscribers[job_id] if w is not ws]


async def _broadcast(job_id: str) -> None:
    job = _jobs.get(job_id)
    if not job:
        return
    data = {
        "job_id": job.job_id,
        "status": job.status.value,
        "step": job.step,
        "progress": round(job.progress, 3),
        "message": job.message,
        "result": job.result,
        "error": job.error,
    }
    dead: list[WebSocket] = []
    for ws in _subscribers.get(job_id, []):
        try:
            await ws.send_json(data)
        except Exception:
            dead.append(ws)
    for ws in dead:
        await unsubscribe(job_id, ws)


def list_jobs() -> list[dict]:
    """Return summary of all jobs."""
    return [
        {
            "job_id": j.job_id,
            "status": j.status.value,
            "step": j.step,
            "progress": round(j.progress, 3),
        }
        for j in _jobs.values()
    ]
