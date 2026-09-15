"""
Disk-backed background jobs.

Training runs in a separate process pool so the API stays responsive and a
long job is not cut off by proxy timeouts. Job state lives in
``<DATA_ROOT>/jobs/<job_id>/`` (status.json, result.json), so it survives
API restarts and can be read by any worker.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import traceback
import uuid
from concurrent.futures import Future, ProcessPoolExecutor
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

TERMINAL_STATES = {"succeeded", "failed"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, path)


def _update_status(job_dir: Path, **fields: Any) -> None:
    status_path = job_dir / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    status.update(fields)
    _write_json(status_path, status)


WORKER_THREADS = int(os.environ.get("JOB_THREADS", "16"))


def _limit_threads() -> None:
    """Cap BLAS/OpenMP threads before numpy & friends are imported in the worker.

    Each model already parallelises with n_jobs; without this cap every
    library also spins up one thread per core, and two concurrent jobs on a
    64-core host push the load average past 100.
    """
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, str(WORKER_THREADS))
    try:
        import torch
        torch.set_num_threads(WORKER_THREADS)
    except Exception:
        pass


def _execute(job_dir_str: str, kind: str, params: Dict[str, Any]) -> None:
    """Entry point inside the worker process."""
    _limit_threads()
    job_dir = Path(job_dir_str)
    _update_status(job_dir, status="running", started_at=_now(), message="Training models")

    def progress(message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        _update_status(job_dir, message=message, progress=extra or {}, updated_at=_now())

    try:
        from . import pipeline

        if kind == "synthetic":
            result = pipeline.run_synthetic(**params, artifact_dir=job_dir, progress=progress)
        elif kind == "session":
            session_dir = params.pop("session_dir")
            result = pipeline.run_session(**params, artifact_dir=job_dir, progress=progress)
        else:
            raise ValueError(f"Unknown job kind: {kind}")
        _write_json(job_dir / "result.json", result)
        _update_status(job_dir, status="succeeded", finished_at=_now(), message="Done")
        # Uploaded data is deleted as soon as the results are produced; only
        # aggregate metrics (result.json) remain.
        if kind == "session":
            shutil.rmtree(session_dir, ignore_errors=True)
    except Exception as exc:
        # TargetError messages are user-facing; everything else gets the type too.
        from .pipeline import TargetError

        message = str(exc) if isinstance(exc, TargetError) else f"{type(exc).__name__}: {exc}"
        _update_status(
            job_dir,
            status="failed",
            finished_at=_now(),
            message=message,
            traceback=traceback.format_exc(limit=20),
        )


class JobManager:
    def __init__(self, root: Path, max_workers: int) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._max_workers = max_workers
        self._pool: Optional[ProcessPoolExecutor] = None
        self._fail_orphans()

    def _get_pool(self) -> ProcessPoolExecutor:
        if self._pool is None:
            # spawn: xgboost/lightgbm OpenMP state does not survive fork cleanly.
            self._pool = ProcessPoolExecutor(
                max_workers=self._max_workers, mp_context=get_context("spawn")
            )
        return self._pool

    def _fail_orphans(self) -> None:
        """Jobs left queued/running by a previous process will never finish."""
        for status_path in self.root.glob("*/status.json"):
            try:
                status = json.loads(status_path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if status.get("status") not in TERMINAL_STATES:
                _update_status(
                    status_path.parent,
                    status="failed",
                    finished_at=_now(),
                    message="Interrupted by a server restart. Please run it again.",
                )

    def submit(self, kind: str, params: Dict[str, Any]) -> str:
        job_id = uuid.uuid4().hex
        job_dir = self.root / job_id
        job_dir.mkdir()
        _write_json(job_dir / "status.json", {
            "job_id": job_id,
            "kind": kind,
            "status": "queued",
            "message": "Waiting for a free worker",
            "created_at": _now(),
        })
        future = self._get_pool().submit(_execute, str(job_dir), kind, params)
        future.add_done_callback(lambda f: self._on_done(f, job_dir))
        return job_id

    def _on_done(self, future: Future, job_dir: Path) -> None:
        exc = future.exception()
        if exc is None:
            return
        # The worker process itself died (e.g. out of memory): the child never
        # got to record the failure, and the pool is unusable afterwards.
        logger.error("Job worker crashed for %s: %r", job_dir.name, exc)
        _update_status(
            job_dir,
            status="failed",
            finished_at=_now(),
            message=f"Worker crashed: {type(exc).__name__}: {exc}",
        )
        self._pool = None

    def status(self, job_id: str) -> Optional[Dict[str, Any]]:
        path = self._job_dir(job_id) / "status.json"
        if not path.exists():
            return None
        status = json.loads(path.read_text())
        status.pop("traceback", None)
        return status

    def result_path(self, job_id: str) -> Path:
        return self._job_dir(job_id) / "result.json"

    # Scored output files live next to the model that produced them; they are
    # the customer's own rows plus scores, kept only until downloaded/expired.
    def submit_file(self, job_id: str, content: bytes, filename: str) -> str:
        out_dir = self._job_dir(job_id) / "scored"
        out_dir.mkdir(parents=True, exist_ok=True)
        file_id = uuid.uuid4().hex
        (out_dir / f"{file_id}.csv").write_bytes(content)
        return file_id

    def file_path(self, job_id: str, file_id: str) -> Optional[Path]:
        if not file_id.isalnum():
            return None
        path = self._job_dir(job_id) / "scored" / f"{file_id}.csv"
        return path if path.exists() else None

    def _job_dir(self, job_id: str) -> Path:
        # job ids are uuid hex; reject anything that could escape the jobs root.
        if not job_id.isalnum():
            return self.root / "__invalid__"
        return self.root / job_id
