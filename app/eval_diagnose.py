"""Why did a golden question pass or fail? Trace the expected answer through the system.

The "evidence" is what the golden case checks for (expect_all / expect_any). For each
answer we look at the exact prompt the primary received and at what the system had
available at that moment, and classify:

  answered_from_prompt       pass, and the evidence was in the prompt
  answered_without_evidence  pass, but the evidence was not in the prompt (model knew or guessed)
  model_missed               fail, although the evidence WAS in the prompt
  not_retrieved              fail; evidence existed in memory or past conversations, not in the prompt
  lost_from_session          fail; said earlier in this session, but trimmed/compressed away
  never_available            fail; nothing the system had contained it
  no_expectation             forbid-only case: nothing to trace
"""
from __future__ import annotations

import re

CATEGORIES = ["answered_from_prompt", "answered_without_evidence", "model_missed", "not_retrieved",
              "lost_from_session", "never_available", "no_expectation"]

LABELS = {
    "answered_from_prompt": "Answered from the prompt",
    "answered_without_evidence": "Answered without evidence",
    "model_missed": "Model missed it",
    "not_retrieved": "Not retrieved",
    "lost_from_session": "Lost from the session",
    "never_available": "Never available",
    "no_expectation": "Nothing to trace",
}

_BASE_RE = re.compile(r"<PROJECT_MEMORY_BASE>(.*?)</PROJECT_MEMORY_BASE>", re.S)
_BLOCK_RE = re.compile(r"<PROJECT_MEMORY>(.*?)</PROJECT_MEMORY>", re.S)
_REF_RE = re.compile(r"<REFERENCE>(.*?)</REFERENCE>", re.S)
_SESSION = "SESSION SO FAR"
_HISTORY = "RELATED PAST EXCHANGES"
_SECTION_RE = re.compile(r"^([A-Z][A-Z /()]+?)(?: \(.*?\))?:\s*$", re.M)


def _match(pattern: str, text: str) -> bool:
    if len(pattern) > 2 and pattern.startswith("/") and pattern.endswith("/"):
        return re.search(pattern[1:-1], text, re.I | re.S) is not None
    return pattern.lower() in text.lower()


def evidence_in(case, text: str) -> bool:
    """True if `text` contains what the case expects (all of expect_all, one of expect_any)."""
    if not (case.expect_all or case.expect_any):
        return False
    if case.expect_all and not all(_match(p, text) for p in case.expect_all):
        return False
    if case.expect_any and not any(_match(p, text) for p in case.expect_any):
        return False
    return True


def split_prompt(messages: list[dict], question: str) -> dict[str, str]:
    """Split the prompt actually sent into the places evidence can come from."""
    parts = {"conversation": "", "memory_base": "", "memory_turn": "", "session_summary": "",
             "past_exchanges": "", "reference": ""}
    convo: list[str] = []
    for i, m in enumerate(messages):
        content = m.get("content")
        if not isinstance(content, str):
            continue
        if m.get("role") == "system":
            for b in _BASE_RE.findall(content):
                parts["memory_base"] += b
            continue
        blocks = _BLOCK_RE.findall(content)
        for r in _REF_RE.findall(content):
            parts["reference"] += r
        visible = _REF_RE.sub("", _BLOCK_RE.sub("", content))
        for b in blocks:
            # Split the per-turn block into its sections.
            current = "memory_turn"
            for line in b.splitlines():
                head = _SECTION_RE.match(line.strip())
                if head:
                    title = head.group(1).strip()
                    current = ("session_summary" if title.startswith(_SESSION)
                               else "past_exchanges" if title.startswith(_HISTORY) else "memory_turn")
                    continue
                parts[current] += line + "\n"
        if i == len(messages) - 1 and m.get("role") == "user":
            visible = visible.replace(question, "", 1)   # the question itself is not evidence
        convo.append(visible)
    parts["conversation"] = "\n".join(convo)
    return parts


def diagnose(case, passed: bool, prompt_parts: dict[str, str], *, memory_text: str = "",
             history_text: str = "", session_text: str = "") -> tuple[str, list[str]]:
    """(category, locations where the evidence was in the prompt)."""
    if not (case.expect_all or case.expect_any):
        return "no_expectation", []
    where = [k for k, v in prompt_parts.items() if v and evidence_in(case, v)]
    if passed:
        return ("answered_from_prompt" if where else "answered_without_evidence"), where
    if where:
        return "model_missed", where
    if evidence_in(case, memory_text) or evidence_in(case, history_text):
        return "not_retrieved", []
    if evidence_in(case, session_text):
        return "lost_from_session", []
    return "never_available", []


def summarize_diagnoses(results: list[dict], variants: list[str]) -> dict:
    """Per variant: counts per category, and how often available evidence reached the prompt."""
    out = {}
    for v in variants:
        rs = [r for r in results if r.get("variant") == v and r.get("diagnosis")]
        counts = {c: sum(1 for r in rs if r["diagnosis"] == c) for c in CATEGORIES}
        # Of the answers whose evidence existed in memory/past conversations, how many got it into the prompt?
        from_store = [r for r in rs if r.get("evidence_in_store")]
        reached = [r for r in from_store if any(w in ("memory_base", "memory_turn", "past_exchanges")
                                                for w in r.get("evidence_in") or [])]
        out[v] = {"counts": counts,
                  "memory_retrieval": {"available": len(from_store), "reached_prompt": len(reached),
                                       "rate": round(len(reached) / len(from_store), 3) if from_store else None}}
    return out
