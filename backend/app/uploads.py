"""
Disk-backed store for uploaded datasets.

Uploads are streamed straight to ``<DATA_ROOT>/uploads/<upload_id>/data.csv``
so multi-GB files never sit in memory or in the small system /tmp.

An upload is deleted by the training job as soon as results are produced
(see jobs.py). The TTL below only catches uploads that were never trained on
or whose training failed.
"""

from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Optional

import pandas as pd

# Rows read for schema analysis; enough for stable stats without loading GBs.
ANALYSIS_SAMPLE_ROWS = 200_000
UPLOAD_TTL = timedelta(hours=24)


class UploadTooLarge(Exception):
    pass


class UploadStore:
    def __init__(self, root: Path, max_bytes: int) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes

    def path(self, upload_id: str) -> Optional[Path]:
        if not upload_id.isalnum():
            return None
        csv_path = self.root / upload_id / "data.csv"
        return csv_path if csv_path.exists() else None

    async def save_stream(self, filename: str, chunks: AsyncIterator[bytes]) -> Dict[str, Any]:
        self._evict_expired()
        upload_id = uuid.uuid4().hex
        upload_dir = self.root / upload_id
        upload_dir.mkdir()
        csv_path = upload_dir / "data.csv"
        size = 0
        newlines = 0
        last_byte = b""
        try:
            with open(csv_path, "wb") as fh:
                async for chunk in chunks:
                    if not chunk:
                        continue
                    size += len(chunk)
                    if size > self.max_bytes:
                        raise UploadTooLarge()
                    newlines += chunk.count(b"\n")
                    last_byte = chunk[-1:]
                    fh.write(chunk)
        except BaseException:
            shutil.rmtree(upload_dir, ignore_errors=True)
            raise

        # Line count minus header; a quoted multi-line cell makes this approximate.
        lines = newlines + (1 if last_byte not in (b"", b"\n") else 0)
        meta = {
            "upload_id": upload_id,
            "filename": filename,
            "size_bytes": size,
            "row_count": max(lines - 1, 0),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        (upload_dir / "meta.json").write_text(json.dumps(meta))
        return meta

    def read_sample(self, upload_id: str) -> pd.DataFrame:
        csv_path = self.path(upload_id)
        if csv_path is None:
            raise FileNotFoundError(upload_id)
        return pd.read_csv(csv_path, nrows=ANALYSIS_SAMPLE_ROWS, low_memory=False)

    def delete(self, upload_id: str) -> None:
        if upload_id.isalnum():
            shutil.rmtree(self.root / upload_id, ignore_errors=True)

    def _evict_expired(self) -> None:
        cutoff = datetime.now(timezone.utc) - UPLOAD_TTL
        for meta_path in self.root.glob("*/meta.json"):
            try:
                created = datetime.fromisoformat(json.loads(meta_path.read_text())["created_at"])
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                continue
            if created < cutoff:
                shutil.rmtree(meta_path.parent, ignore_errors=True)
