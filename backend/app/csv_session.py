"""
In-memory session store for uploaded CSV DataFrames.
Sessions expire after 60 minutes and are evicted on access.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Optional

import pandas as pd

_TTL = timedelta(minutes=60)


@dataclass
class CSVSession:
    session_id: str
    df: pd.DataFrame
    created_at: datetime = field(default_factory=datetime.utcnow)


class CSVSessionStore:
    def __init__(self) -> None:
        self._store: Dict[str, CSVSession] = {}

    def _evict_expired(self) -> None:
        now = datetime.utcnow()
        expired = [sid for sid, s in self._store.items() if now - s.created_at > _TTL]
        for sid in expired:
            del self._store[sid]

    def create(self, df: pd.DataFrame) -> str:
        self._evict_expired()
        session_id = str(uuid.uuid4())
        self._store[session_id] = CSVSession(session_id=session_id, df=df)
        return session_id

    def get(self, session_id: str) -> Optional[CSVSession]:
        self._evict_expired()
        return self._store.get(session_id)


csv_sessions = CSVSessionStore()
