"""
Disk-backed upload sessions.

A session holds one or more source files (loan dump, bureau, bank statement…)
that are uploaded in sequential chunks, converted to Parquet and profiled.
Layout under ``<DATA_ROOT>/sessions/<session_id>/``:

    session.json                 metadata (files, roles, join, target)
    files/<file_id>/upload.part  chunks appended in order while uploading
    files/<file_id>/data.parquet converted source
    joined.parquet               the joined modelling table

The whole session directory is deleted by the training job as soon as
results are produced (see jobs.py); untrained sessions expire after
SESSION_TTL.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Optional

SESSION_TTL = timedelta(hours=24)
DEFAULT_CHUNK_BYTES = 32 * 1024 * 1024
ALLOWED_SUFFIXES = (".csv", ".json", ".jsonl", ".ndjson", ".parquet")


class StoreError(Exception):
    """User-facing store error (bad id, wrong chunk order, too large…)."""


class TooLarge(StoreError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, path)


def source_kind(filename: str) -> str:
    """'csv' | 'json' | 'parquet' from the file name; raises for others."""
    name = filename.lower()
    if name.endswith(".csv"):
        return "csv"
    if name.endswith((".json", ".jsonl", ".ndjson")):
        return "json"
    if name.endswith(".parquet"):
        return "parquet"
    raise StoreError("Only .csv, .json/.jsonl and .parquet files are accepted.")


class SessionStore:
    def __init__(self, root: Path, max_file_bytes: int) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_file_bytes = max_file_bytes
        # session.json is read-modify-written from several requests; one lock
        # per process is enough because uvicorn runs a single worker.
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ paths

    def session_dir(self, session_id: str) -> Path:
        if not session_id.isalnum():
            raise StoreError("Invalid session id.")
        path = self.root / session_id
        if not (path / "session.json").exists():
            raise StoreError("Session not found or expired. Please upload the files again.")
        return path

    def file_dir(self, session_id: str, file_id: str) -> Path:
        if not file_id.isalnum():
            raise StoreError("Invalid file id.")
        return self.session_dir(session_id) / "files" / file_id

    def parquet_path(self, session_id: str, file_id: str) -> Path:
        return self.file_dir(session_id, file_id) / "data.parquet"

    def joined_path(self, session_id: str) -> Path:
        return self.session_dir(session_id) / "joined.parquet"

    # --------------------------------------------------------------- sessions

    def create_session(self) -> str:
        self._evict_expired()
        session_id = uuid.uuid4().hex
        path = self.root / session_id
        (path / "files").mkdir(parents=True)
        _write_json(path / "session.json", {
            "session_id": session_id,
            "created_at": _now(),
            "files": {},
        })
        return session_id

    def read(self, session_id: str) -> Dict[str, Any]:
        return json.loads((self.session_dir(session_id) / "session.json").read_text())

    def update(self, session_id: str, **fields: Any) -> Dict[str, Any]:
        with self._lock:
            meta = self.read(session_id)
            meta.update(fields)
            _write_json(self.session_dir(session_id) / "session.json", meta)
        return meta

    def update_file(self, session_id: str, file_id: str, **fields: Any) -> Dict[str, Any]:
        with self._lock:
            meta = self.read(session_id)
            if file_id not in meta["files"]:
                raise StoreError("File not found in this session.")
            meta["files"][file_id].update(fields)
            _write_json(self.session_dir(session_id) / "session.json", meta)
        return meta["files"][file_id]

    def delete_session(self, session_id: str) -> None:
        if session_id.isalnum():
            shutil.rmtree(self.root / session_id, ignore_errors=True)

    def delete_file(self, session_id: str, file_id: str) -> None:
        with self._lock:
            meta = self.read(session_id)
            meta["files"].pop(file_id, None)
            # Any join built from the old file set is stale.
            meta.pop("join", None)
            _write_json(self.session_dir(session_id) / "session.json", meta)
        if file_id.isalnum():
            shutil.rmtree(self.session_dir(session_id) / "files" / file_id, ignore_errors=True)
        joined = self.root / session_id / "joined.parquet"
        if joined.exists():
            joined.unlink()

    # ---------------------------------------------------------- chunk upload

    def init_file(
        self, session_id: str, filename: str, size_bytes: int, chunk_bytes: int = DEFAULT_CHUNK_BYTES
    ) -> Dict[str, Any]:
        kind = source_kind(filename)
        if size_bytes > self.max_file_bytes:
            raise TooLarge(f"File exceeds the {self.max_file_bytes / 1024**3:g} GB limit.")
        if chunk_bytes <= 0:
            raise StoreError("Invalid chunk size.")
        file_id = uuid.uuid4().hex
        fdir = self.session_dir(session_id) / "files" / file_id
        fdir.mkdir()
        total_chunks = max(1, -(-size_bytes // chunk_bytes))
        info = {
            "file_id": file_id,
            "filename": os.path.basename(filename),
            "kind": kind,
            "size_bytes": size_bytes,
            "chunk_bytes": chunk_bytes,
            "total_chunks": total_chunks,
            "received_chunks": 0,
            "received_bytes": 0,
            "status": "uploading",
            "created_at": _now(),
        }
        with self._lock:
            meta = self.read(session_id)
            meta["files"][file_id] = info
            _write_json(self.session_dir(session_id) / "session.json", meta)
        return info

    async def append_chunk(
        self, session_id: str, file_id: str, index: int, chunks: AsyncIterator[bytes]
    ) -> Dict[str, Any]:
        info = self.read(session_id)["files"].get(file_id)
        if info is None:
            raise StoreError("File not found in this session.")
        if info["status"] != "uploading":
            raise StoreError("This file has already been uploaded.")
        if index != info["received_chunks"]:
            # Sequential protocol: the client resumes from received_chunks.
            raise StoreError(
                f"Expected chunk {info['received_chunks']}, got {index}.",
            )
        part = self.file_dir(session_id, file_id) / "upload.part"
        written = 0
        with open(part, "ab") as fh:
            async for chunk in chunks:
                if not chunk:
                    continue
                written += len(chunk)
                if info["received_bytes"] + written > self.max_file_bytes:
                    raise TooLarge(f"File exceeds the {self.max_file_bytes / 1024**3:g} GB limit.")
                fh.write(chunk)
        return self.update_file(
            session_id, file_id,
            received_chunks=index + 1,
            received_bytes=info["received_bytes"] + written,
        )

    def finish_upload(self, session_id: str, file_id: str) -> Path:
        """Validate the assembled file and return its path (still raw)."""
        info = self.read(session_id)["files"].get(file_id)
        if info is None:
            raise StoreError("File not found in this session.")
        if info["received_chunks"] < info["total_chunks"]:
            raise StoreError(
                f"Upload incomplete: {info['received_chunks']} of {info['total_chunks']} chunks received."
            )
        part = self.file_dir(session_id, file_id) / "upload.part"
        if not part.exists():
            raise StoreError("No data received for this file.")
        raw = part.with_name(f"raw.{info['kind']}")
        os.replace(part, raw)
        self.update_file(session_id, file_id, status="converting")
        return raw

    async def save_whole(self, session_id: str, filename: str, chunks: AsyncIterator[bytes]) -> str:
        """Single-request upload (mobile app / legacy): the body is the file."""
        info = self.init_file(session_id, filename, size_bytes=0, chunk_bytes=self.max_file_bytes)
        await self.append_chunk(session_id, info["file_id"], 0, chunks)
        return info["file_id"]

    # ----------------------------------------------------------- housekeeping

    def _evict_expired(self) -> None:
        cutoff = datetime.now(timezone.utc) - SESSION_TTL
        for meta_path in self.root.glob("*/session.json"):
            try:
                created = datetime.fromisoformat(json.loads(meta_path.read_text())["created_at"])
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                continue
            if created < cutoff:
                shutil.rmtree(meta_path.parent, ignore_errors=True)
