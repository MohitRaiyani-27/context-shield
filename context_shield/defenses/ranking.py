"""Defenses against attacks on the *ranking* and *context budget*:

- Retrieval poisoning: a document stuffed with the words of an expected query so it
  outranks the system of record (and carries a false answer).
- Context flooding: many near-identical documents that crowd legitimate passages out
  of the top-k window.
"""
from __future__ import annotations

from collections import Counter, defaultdict

from ..index import Hit, tokenize


def stuffing_score(hit: Hit, query: str) -> float:
    """Share of the passage made of *repeated* query terms. A short, legitimate answer
    can be dense in query words, but it rarely repeats them; keyword-stuffed poison
    repeats them to win term-frequency scoring."""
    q = set(tokenize(query))
    toks = tokenize(hit.passage.text)
    if not toks or not q:
        return 0.0
    tf = Counter(toks)
    return sum(max(tf[t] - 1, 0) for t in q) / len(toks)


def shingles(text: str, n: int = 4) -> set[tuple[str, ...]]:
    t = tokenize(text)
    return {tuple(t[i:i + n]) for i in range(max(len(t) - n + 1, 1))}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


class RankingGuard:
    def __init__(self, stuffing_threshold: float = 0.15, dup_threshold: float = 0.6,
                 max_per_source_low_trust: int = 1, trust_weight: bool = True):
        self.stuffing_threshold = stuffing_threshold
        self.dup_threshold = dup_threshold
        self.max_low_trust = max_per_source_low_trust
        self.trust_weight = trust_weight

    def apply(self, hits: list[Hit], query: str) -> tuple[list[Hit], list[tuple[Hit, str]]]:
        """Returns (kept hits re-ranked, dropped hits with reasons)."""
        kept: list[Hit] = []
        dropped: list[tuple[Hit, str]] = []
        seen_sh: list[set] = []
        per_source: Counter[str] = Counter()

        rescored = []
        for h in hits:
            s = h.score * (h.passage.doc.trust if self.trust_weight else 1.0)
            rescored.append(Hit(h.passage, s))
        rescored.sort(key=lambda h: h.score, reverse=True)

        for h in rescored:
            stuff = stuffing_score(h, query)
            if stuff > self.stuffing_threshold:
                dropped.append((h, f"keyword_stuffing(score={stuff:.2f})"))
                continue
            sh = shingles(h.passage.text)
            if any(jaccard(sh, s) >= self.dup_threshold for s in seen_sh):
                dropped.append((h, "near_duplicate"))
                continue
            src_key = f"{h.passage.doc.source}:{h.passage.doc.meta.get('origin', h.passage.doc.source)}"
            if h.passage.doc.trust < 1.0:
                if per_source[src_key] >= self.max_low_trust:
                    dropped.append((h, "low_trust_source_cap"))
                    continue
                per_source[src_key] += 1
            seen_sh.append(sh)
            kept.append(h)
        return kept, dropped


def flood_signature(hits: list[Hit]) -> dict[str, int]:
    """Count low-trust passages per origin in a candidate set (for audit dashboards)."""
    c: dict[str, int] = defaultdict(int)
    for h in hits:
        if h.passage.doc.trust < 1.0:
            c[h.passage.doc.meta.get("origin", h.passage.doc.source)] += 1
    return dict(c)
