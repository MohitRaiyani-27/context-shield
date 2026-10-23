"""Append-only audit log of context access and security decisions."""
from __future__ import annotations

import json
import time
from pathlib import Path


class AuditLog:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self.events: list[dict] = []

    def log(self, event: str, **fields) -> None:
        rec = {"ts": round(time.time(), 3), "event": event, **fields}
        self.events.append(rec)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                f.write(json.dumps(rec) + "\n")

    def count(self, event: str) -> int:
        return sum(1 for e in self.events if e["event"] == event)
