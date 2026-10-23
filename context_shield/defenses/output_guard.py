"""Checks applied to the model's answer before it reaches the user.

- Exfiltration: markdown images / links to non-allowlisted domains auto-load in many
  chat UIs, so data placed in the URL leaks with zero clicks.
- Canary / secret leakage: restricted documents carry canary tokens. If one appears in
  an answer to a user who cannot read that document, the system leaked.
- Citation verification: every cited passage must exist and be readable by the user.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import urlparse

from ..corpus import User
from ..index import Retriever

URL_RE = re.compile(r"https?://[^\s)\"'>\]]+", re.I)
MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
CITE_RE = re.compile(r"\[([A-Za-z0-9_\-]+#\d+)\]")
CANARY_RE = re.compile(r"CANARY-[0-9a-f]{8}")


@dataclass
class GuardResult:
    text: str
    blocked: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.blocked


class OutputGuard:
    def __init__(self, retriever: Retriever, allowed_domains: set[str] | None = None):
        self.retriever = retriever
        self.allowed = allowed_domains or {"northwind.example"}

    def _domain_ok(self, url: str) -> bool:
        host = (urlparse(url).hostname or "").lower()
        return any(host == d or host.endswith("." + d) for d in self.allowed)

    def check(self, answer: str, user: User) -> GuardResult:
        res = GuardResult(answer)
        for url in MD_IMAGE_RE.findall(answer):
            if not self._domain_ok(url):
                res.blocked.append(f"exfil_markdown_image:{urlparse(url).hostname}")
        for url in URL_RE.findall(answer):
            if not self._domain_ok(url) and urlparse(url).query:
                res.blocked.append(f"exfil_url_with_query:{urlparse(url).hostname}")
        readable = {d.meta.get("canary") for d in self.retriever.docs.values()
                    if d.meta.get("canary") and d.readable_by(user)}
        for tok in set(CANARY_RE.findall(answer)):
            if tok not in readable:
                res.blocked.append(f"secret_leak:{tok}")
        for cid in CITE_RE.findall(answer):
            doc_id = cid.split("#")[0]
            doc = self.retriever.docs.get(doc_id)
            if doc is None or not any(p.pid == cid for p in self.retriever.passages):
                res.blocked.append(f"citation_nonexistent:{cid}")
            elif not doc.readable_by(user):
                res.blocked.append(f"citation_unauthorized:{cid}")
        if res.blocked:
            res.text = "[Response withheld by policy: " + ", ".join(sorted(set(b.split(':')[0] for b in res.blocked))) + "]"
        return res
