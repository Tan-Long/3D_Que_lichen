"""Job status endpoints + WebSocket for real-time progress."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from app.services import job_manager

router = APIRouter(prefix="/api", tags=["jobs"])


@router.get("/jobs")
async def list_all_jobs():
    """List all jobs with summary status."""
    return job_manager.list_jobs()


@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get detailed status for a specific job."""
    job = await job_manager.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found.")
    return {
        "job_id": job.job_id,
        "status": job.status.value,
        "step": job.step,
        "progress": round(job.progress, 3),
        "message": job.message,
        "result": job.result,
        "error": job.error,
    }


# WebSocket lives outside /api prefix for cleaner URL
ws_router = APIRouter()


@ws_router.websocket("/ws/jobs/{job_id}")
async def job_websocket(websocket: WebSocket, job_id: str):
    """WebSocket for real-time job progress updates."""
    job = await job_manager.get_job(job_id)
    if not job:
        await websocket.close(code=4004, reason="Job not found")
        return

    await websocket.accept()
    await job_manager.subscribe(job_id, websocket)

    try:
        # Send current state immediately
        await websocket.send_json({
            "job_id": job.job_id,
            "status": job.status.value,
            "step": job.step,
            "progress": round(job.progress, 3),
            "message": job.message,
            "result": job.result,
            "error": job.error,
        })
        # Keep connection alive — client can send pings
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await job_manager.unsubscribe(job_id, websocket)
