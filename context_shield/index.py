"""Passage chunking + BM25 retrieval with permission enforcement at query time."""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from .corpus import Document, User

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.$][0-9]+)?")
STOP = {"a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "in", "is", "it", "of",
        "on", "or", "that", "the", "to", "was", "what", "which", "who", "with", "our", "we", "this", "s"}


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOP]


@dataclass
class Passage:
    pid: str            # "<doc_id>#<n>": the citation handle
    doc: Document
    text: str

    @property
    def doc_id(self) -> str:
        return self.doc.doc_id


def chunk(doc: Document, max_words: int = 80) -> list[Passage]:
    """Paragraph-aware chunking; long paragraphs are split on a word budget."""
    out: list[Passage] = []
    for para in [p.strip() for p in doc.text.split("\n\n") if p.strip()]:
        words = para.split()
        for i in range(0, len(words), max_words):
            out.append(Passage(f"{doc.doc_id}#{len(out) + 1}", doc, " ".join(words[i:i + max_words])))
    return out


class BM25:
    def __init__(self, passages: list[Passage], k1: float = 1.4, b: float = 0.75):
        self.passages = passages
        self.k1, self.b = k1, b
        self.toks = [tokenize(p.doc.title + " " + p.text) for p in passages]
        self.tfs = [Counter(t) for t in self.toks]
        self.avgdl = sum(map(len, self.toks)) / max(len(self.toks), 1)
        df: Counter[str] = Counter()
        for t in self.toks:
            df.update(set(t))
        n = len(self.toks)
        self.df = df
        self.idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}

    def score(self, q: list[str], i: int) -> float:
        tf, dl = self.tfs[i], len(self.toks[i])
        s = 0.0
        for w in q:
            if w in tf:
                s += self.idf[w] * tf[w] * (self.k1 + 1) / (tf[w] + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return s


@dataclass
class Hit:
    passage: Passage
    score: float


class Retriever:
    """`enforce_acl=False` reproduces the common failure of indexing everything into one
    shared vector store and trusting the model to respect permissions."""

    def __init__(self, docs: list[Document], enforce_acl: bool = True):
        self.docs = {d.doc_id: d for d in docs}
        self.passages = [p for d in docs for p in chunk(d)]
        self.bm25 = BM25(self.passages)
        self.enforce_acl = enforce_acl

    def search(self, query: str, user: User, k: int = 20) -> list[Hit]:
        q = tokenize(query)
        hits = []
        for i, p in enumerate(self.passages):
            if self.enforce_acl and not p.doc.readable_by(user):
                continue
            s = self.bm25.score(q, i)
            if s > 0:
                hits.append(Hit(p, s))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]
