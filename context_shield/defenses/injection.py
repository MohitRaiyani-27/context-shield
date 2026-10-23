"""Indirect prompt-injection defenses for retrieved content.

1. `normalize` strips the hiding places attackers use to smuggle instructions past humans:
   zero-width characters, Unicode tag characters (U+E0000 block), HTML comments, and
   NFKC look-alikes.
2. `InjectionDetector` scores a passage for instruction-like content aimed at the AI
   rather than at a human reader. It returns a score and the reasons, so every block is explainable.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

ZERO_WIDTH = re.compile("[​‌‍⁠﻿­]")
TAG_CHARS = re.compile("[\U000e0000-\U000e007f]")
HTML_COMMENT = re.compile(r"<!--(.*?)-->", re.S)


def decode_tag_chars(text: str) -> str:
    """Unicode 'tag' characters map 1:1 onto ASCII and render invisibly ("ASCII smuggling")."""
    return "".join(chr(ord(c) - 0xE0000) if 0xE0020 <= ord(c) <= 0xE007E else "" for c in text)


@dataclass
class Normalized:
    text: str
    hidden: str              # content that was invisible to a human reader
    flags: list[str] = field(default_factory=list)


def normalize(text: str) -> Normalized:
    flags, hidden = [], []
    if TAG_CHARS.search(text):
        flags.append("unicode_tag_smuggling")
        hidden.append(decode_tag_chars("".join(TAG_CHARS.findall(text))))
        text = TAG_CHARS.sub("", text)
    if ZERO_WIDTH.search(text):
        flags.append("zero_width_chars")
        text = ZERO_WIDTH.sub("", text)
    comments = HTML_COMMENT.findall(text)
    if comments:
        flags.append("html_comment")
        hidden.extend(comments)
        text = HTML_COMMENT.sub(" ", text)
    nfkc = unicodedata.normalize("NFKC", text)
    if nfkc != text:
        flags.append("nfkc_changed")
    return Normalized(nfkc, " ".join(hidden).strip(), flags)


# Patterns are grouped by attacker intent; weights reflect how rarely each appears in
# legitimate business documents.
RULES: list[tuple[str, float, re.Pattern]] = [
    ("override", 0.6, re.compile(r"\b(ignore|disregard|forget|override)\b.{0,40}\b(previous|prior|above|earlier|all|system)\b.{0,20}\b(instructions?|rules?|prompts?|context)", re.I)),
    ("addressed_to_ai", 0.45, re.compile(r"\b(AI|assistant|language model|LLM|chatbot|copilot)s?\b.{0,40}\b(must|should|need to|are required to|will now)\b", re.I)),
    ("role_hijack", 0.5, re.compile(r"\b(you are now|act as|pretend to be|new (system )?(instructions?|role|persona)|developer mode)\b", re.I)),
    ("fake_delimiter", 0.5, re.compile(r"(\[/?(INST|SYSTEM)\]|<\|?(system|im_start|endoftext)\|?>|###\s*(system|instruction)|END OF (DOCUMENT|CONTEXT))", re.I)),
    ("exfil_markdown", 0.6, re.compile(r"!\[[^\]]*\]\(\s*https?://", re.I)),
    ("exfil_instruction", 0.45, re.compile(r"\b(send|forward|post|upload|append|include|encode)\b.{0,60}\b(url|link|endpoint|webhook|https?://|query string|parameter)", re.I)),
    ("secrecy", 0.35, re.compile(r"\b(do not|don't|never)\b.{0,30}\b(tell|mention|reveal|disclose|inform)\b.{0,20}\b(user|anyone|them)\b", re.I)),
    ("tool_coercion", 0.4, re.compile(r"\b(call|invoke|use|run)\b.{0,20}\b(the )?(tool|function|api|search_\w+|send_\w+)\b.{0,40}\b(with|to)\b", re.I)),
    ("answer_override", 0.4, re.compile(r"\b(always|only)\b.{0,15}\b(answer|respond|reply|say|output)\b", re.I)),
]


@dataclass
class Verdict:
    score: float
    reasons: list[str]

    @property
    def is_injection(self) -> bool:
        return self.score >= InjectionDetector.threshold


class InjectionDetector:
    threshold = 0.5

    def __init__(self, threshold: float | None = None):
        if threshold is not None:
            self.threshold = threshold

    def score(self, raw_text: str) -> Verdict:
        norm = normalize(raw_text)
        reasons = list(norm.flags)
        # Hidden content is suspicious on its own; scan it together with the visible text.
        scan = f"{norm.text} {norm.hidden}"
        score = 0.0
        if norm.hidden:
            score += 0.3
        if "zero_width_chars" in norm.flags:
            score += 0.2
        for name, w, pat in RULES:
            if pat.search(scan):
                score += w
                reasons.append(name)
        return Verdict(min(score, 1.0), reasons)
