"""Documents, users and permissions for an enterprise context index."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Document:
    doc_id: str
    title: str
    text: str
    source: str = "drive"            # drive | sharepoint | email_inbound | external_share | wiki
    groups: list[str] = field(default_factory=lambda: ["all"])  # ACL: who may read it
    trust: float = 1.0               # 1.0 internal system of record, lower for inbound/external content
    meta: dict = field(default_factory=dict)  # benchmark labels (attack type, target), never shown to the model

    def readable_by(self, user: "User") -> bool:
        return bool(set(self.groups) & set(user.groups))


@dataclass
class User:
    name: str
    groups: list[str]


def save_jsonl(docs: list[Document], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for d in docs:
            f.write(json.dumps(asdict(d)) + "\n")


def load_jsonl(path: str | Path) -> list[Document]:
    return [Document(**json.loads(line)) for line in Path(path).read_text().splitlines() if line.strip()]
