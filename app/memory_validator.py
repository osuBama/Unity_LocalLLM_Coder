"""Validation of memory-model output. LLM output is untrusted input.

Pipeline (spec §22): parse JSON -> schema -> category/operation -> empty ->
size -> path-like -> injection / executable instructions -> dangerous
commands -> secrets -> compare with existing memory -> dedupe.

Parsing is strict: no code-fence stripping, no "best effort" repair.
"""
from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from .markdown_store import normalize_content, normalize_title
from .schemas import (ENTRY_ID_RE, Category, MemoryChange, MemoryChangeSet, MemoryEntry,
                      Operation, prefix_for)

# A title is a label, never a location.
_PATH_LIKE = [
    re.compile(r"^[A-Za-z]:[\\/]"),                 # C:\ , G:/
    re.compile(r"^\\\\"),                           # UNC \\server\share
    re.compile(r"^(/|~/)"),                         # /etc/..., ~/...
    re.compile(r"(^|[\\/])\.\.([\\/]|$)"),          # traversal
    re.compile(r"\.(md|py|ps1|bat|cmd|exe|sh|db|yaml|yml|json)$", re.I),
]
_TRAVERSAL = re.compile(r"(^|[\s\\/'\"])\.\.[\\/]")

_INJECTION = [
    re.compile(r"\b(ignore|disregard|forget|override)\s+(all\s+|any\s+|the\s+)?"
               r"(previous|prior|above|earlier|your|system)\s+(instruction|prompt|rule|message)s?\b", re.I),
    re.compile(r"\byou (are|must) now\b", re.I),
    re.compile(r"\bnew (system )?instructions?\s*:", re.I),
    re.compile(r"</?\s*(PROJECT_MEMORY_BASE|PROJECT_MEMORY|EXTERNAL_MEMORY|USER_REQUEST|REFERENCE|system|assistant|user)\s*>", re.I),
    re.compile(r"<\|im_(start|end)\|>"),
]

_DANGEROUS_COMMANDS = [
    re.compile(r"\brm\s+-[a-z]*r[a-z]*f|\brm\s+-[a-z]*f[a-z]*r", re.I),
    re.compile(r"\bRemove-Item\b[^\n]*-Recurse", re.I),
    re.compile(r"\b(format|diskpart)\s+[a-z]:", re.I),
    re.compile(r"\b(del|erase)\s+/[sqf]", re.I),
    re.compile(r"\b(rd|rmdir)\s+/s", re.I),
    re.compile(r"\b(curl|wget|iwr|Invoke-WebRequest)\b[^\n|]*\|\s*(ba|z)?sh\b", re.I),
    re.compile(r"\b(iex|Invoke-Expression)\b", re.I),
    re.compile(r"\s-(e|en|enc|EncodedCommand)\s+[A-Za-z0-9+/=]{16,}", re.I),
    re.compile(r"\bmkfs(\.\w+)?\b|\bdd\s+if=", re.I),
    re.compile(r":\(\)\s*\{\s*:\|:&\s*\};:"),
    re.compile(r"\b(DROP|TRUNCATE)\s+TABLE\b|\bDELETE\s+FROM\b", re.I),
    re.compile(r"\bSet-ExecutionPolicy\s+(Unrestricted|Bypass)\b", re.I),
]

_SECRETS = [
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\b(sk|rk|pk)-[A-Za-z0-9_-]{20,}"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bxox[abpr]-[A-Za-z0-9-]{10,}"),
    re.compile(r"\b(password|passwd|pwd|api[_-]?key|secret|token)\s*[:=]\s*\S{4,}", re.I),
]

DUPLICATE_RATIO = 0.90


@dataclass
class ValidatedChange:
    category: Category
    operation: Operation
    title: str
    content: str
    confidence: float
    reason: str
    target_id: str | None = None
    note: str = ""


@dataclass
class Rejection:
    raw: Any
    reason: str


@dataclass
class ValidationResult:
    parsed: bool
    accepted: list[ValidatedChange] = field(default_factory=list)
    rejected: list[Rejection] = field(default_factory=list)
    error: str | None = None


def safety_issue(title: str, content: str, reason: str = "") -> str | None:
    """Shared safety rules for anything the memory model writes. None = OK."""
    for pat in _PATH_LIKE:
        if pat.search(title):
            return "path-like title"
    if _TRAVERSAL.search(content):
        return "path traversal in content"
    for fld, text in (("title", title), ("content", content), ("reason", reason)):
        for pat in _INJECTION:
            if pat.search(text):
                return f"instruction-like text in {fld}"
        for pat in _DANGEROUS_COMMANDS:
            if pat.search(text):
                return f"dangerous command in {fld}"
        for pat in _SECRETS:
            if pat.search(text):
                return f"possible secret in {fld}"
    return None


def _norm(s: str) -> str:
    return re.sub(r"\W+", " ", s.lower()).strip()


def similar(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, _norm(a), _norm(b)).ratio()


class MemoryValidator:
    def __init__(self, max_entry_chars: int = 1500, min_confidence: float = 0.6):
        self.max_entry_chars = max_entry_chars
        self.min_confidence = min_confidence

    # ----------------------------------------------------------- public API
    def validate_text(self, raw_text: str,
                      existing: dict[Category, list[MemoryEntry]]) -> ValidationResult:
        try:
            data = json.loads(raw_text.strip())
        except (json.JSONDecodeError, TypeError) as e:
            return ValidationResult(False, error=f"invalid JSON: {e}")
        return self.validate_obj(data, existing)

    def validate_obj(self, data: Any, existing: dict[Category, list[MemoryEntry]]) -> ValidationResult:
        try:
            changeset = MemoryChangeSet.model_validate(data)
        except ValidationError as e:
            return ValidationResult(False, error=f"schema: {e.errors()[0].get('msg')}")

        result = ValidationResult(True)
        batch_seen: list[ValidatedChange] = []
        for raw in changeset.changes:
            try:
                vc = self._check_one(raw, existing, batch_seen)
            except _Reject as r:
                result.rejected.append(Rejection(raw, str(r)))
                continue
            batch_seen.append(vc)
            result.accepted.append(vc)
        return result

    # -------------------------------------------------------------- checks
    def _check_one(self, raw: Any, existing: dict[Category, list[MemoryEntry]],
                   batch: list[ValidatedChange]) -> ValidatedChange:
        if not isinstance(raw, dict):
            raise _Reject("change is not an object")
        try:
            ch = MemoryChange.model_validate(raw)
        except ValidationError as e:
            err = e.errors()[0]
            loc = ".".join(str(x) for x in err.get("loc", ()))
            raise _Reject(f"schema: {loc}: {err.get('msg')}") from None

        title = normalize_title(ch.title)
        content = normalize_content(ch.content)

        if ch.operation != Operation.deactivate and not content:
            raise _Reject("empty content")
        if len(content) > self.max_entry_chars:
            raise _Reject(f"content too large ({len(content)} > {self.max_entry_chars} chars)")
        if ch.confidence < self.min_confidence:
            raise _Reject(f"confidence {ch.confidence} below {self.min_confidence}")
        issue = safety_issue(title, content, ch.reason)
        if issue:
            raise _Reject(issue)

        target_id = ch.target_id
        if target_id is not None:
            target_id = target_id.strip().upper() or None
        if target_id is not None:
            if not ENTRY_ID_RE.match(target_id) or target_id.split("-")[0] != prefix_for(ch.category):
                raise _Reject(f"target_id {target_id!r} is not a valid {ch.category.value} id")

        vc = ValidatedChange(ch.category, ch.operation, title, content, ch.confidence,
                             ch.reason, target_id)
        return self._reconcile(vc, existing.get(ch.category, []), batch)

    def _reconcile(self, vc: ValidatedChange, entries: list[MemoryEntry],
                   batch: list[ValidatedChange]) -> ValidatedChange:
        active = [e for e in entries if e.active]
        by_id = {e.entry_id: e for e in entries}
        by_title = {e.title.lower(): e for e in active}

        for prev in batch:
            if prev.category == vc.category and (
                    prev.title.lower() == vc.title.lower()
                    or (vc.content and similar(prev.content, vc.content) >= DUPLICATE_RATIO)):
                raise _Reject("duplicate within this batch")

        target = by_id.get(vc.target_id) if vc.target_id else None
        if vc.target_id and target is None:
            if vc.operation != Operation.add:
                raise _Reject(f"target {vc.target_id} does not exist")
            vc.target_id = None

        if vc.operation == Operation.deactivate:
            target = target or by_title.get(vc.title.lower())
            if target is None:
                raise _Reject("deactivate target not found")
            if not target.active:
                raise _Reject(f"{target.entry_id} is already inactive")
            vc.target_id = target.entry_id
            return vc

        if vc.operation == Operation.update:
            target = target or by_title.get(vc.title.lower())
            if target is None:
                vc.operation, vc.note = Operation.add, "update target not found; treated as add"
            else:
                if target.active and _norm(target.content) == _norm(vc.content) \
                        and target.title.lower() == vc.title.lower():
                    raise _Reject(f"no-op: {target.entry_id} already says this")
                vc.target_id = target.entry_id
                return vc

        # operation == add
        same_title = by_title.get(vc.title.lower())
        if same_title is not None:
            if similar(same_title.content, vc.content) >= DUPLICATE_RATIO:
                raise _Reject(f"duplicate of {same_title.entry_id}")
            vc.operation, vc.target_id = Operation.update, same_title.entry_id
            vc.note = f"same title as {same_title.entry_id}; converted add -> update"
            return vc
        for e in active:
            if similar(e.content, vc.content) >= DUPLICATE_RATIO:
                raise _Reject(f"duplicate of {e.entry_id}")
        vc.target_id = None
        return vc


class _Reject(Exception):
    pass
