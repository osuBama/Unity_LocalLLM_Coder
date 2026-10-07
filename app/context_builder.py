"""Builds the compact <PROJECT_MEMORY> block under a hard token budget.

Budget priority (spec §38): constraints > state > active objectives >
relevant lessons > relevant decisions > relevant environment > relevant
discoveries. Lower-priority material is dropped first; the budget is never
exceeded because a Markdown file happens to be large.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from .markdown_store import MarkdownStore
from .memory_retriever import MemoryRetriever
from .schemas import Category, MemoryEntry
from .util import estimate_tokens, truncate_tokens

# (category, always-include-active?, render label)
PRIORITY: list[tuple[Category, bool]] = [
    (Category.constraint, True),
    (Category.state, True),
    (Category.objective, True),
    (Category.lesson, False),
    (Category.decision, False),
    (Category.environment, False),
    (Category.discovery, False),
]

SESSION_LABEL = "SESSION SO FAR (summary of earlier turns no longer shown verbatim)"
HISTORY_LABEL = ("RELATED PAST EXCHANGES (verbatim excerpts from earlier conversations; may be outdated, "
                 "current evidence wins)")

RENDER_ORDER: list[tuple[Category, str]] = [
    (Category.state, "CURRENT STATE"),
    (Category.objective, "ACTIVE OBJECTIVES"),
    (Category.constraint, "CONSTRAINTS"),
    (Category.decision, "RELEVANT DECISIONS"),
    (Category.lesson, "RELEVANT LESSONS"),
    (Category.environment, "RELEVANT ENVIRONMENT"),
    (Category.discovery, "RELEVANT DISCOVERIES"),
]

_TAG_RE = re.compile(r"</?\s*(PROJECT_MEMORY_BASE|PROJECT_MEMORY|EXTERNAL_MEMORY|USER_REQUEST|REFERENCE)\s*>", re.I)

UPDATES_LABEL = "UPDATED SINCE THE MEMORY BASE (these override the base in the system prompt)"
BASE_ORDER: list[tuple[Category, str]] = [
    (Category.constraint, "CONSTRAINTS"),
    (Category.objective, "ACTIVE OBJECTIVES"),
    (Category.environment, "ENVIRONMENT"),
    (Category.state, "CURRENT STATE"),
    (Category.decision, "DECISIONS"),
    (Category.lesson, "LESSONS"),
    (Category.discovery, "DISCOVERIES"),
]


def entry_fingerprint(e: MemoryEntry) -> str:
    return hashlib.sha1(f"{e.title}\n{e.content}".encode("utf-8")).hexdigest()[:16]


@dataclass
class StableSnapshot:
    """A frozen memory base for the system prompt. Same memory -> byte-identical text."""
    text: str
    tokens: int
    fingerprints: dict[str, str] = field(default_factory=dict)  # entry id -> fingerprint
    categories: tuple[Category, ...] = ()
    dropped: list[str] = field(default_factory=list)


def sanitize_memory_text(text: str) -> str:
    """Hand-edited memory must not be able to close our delimiters."""
    return _TAG_RE.sub(lambda m: m.group(0).replace("<", "&lt;").replace(">", "&gt;"), text)


@dataclass
class BuiltContext:
    text: str
    token_estimate: int
    included: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not self.included


class ContextBuilder:
    def __init__(self, stores: dict[Category, MarkdownStore], retriever: MemoryRetriever,
                 preamble: str, max_tokens: int, max_entry_tokens: int = 600,
                 relevant_limit: int = 8):
        self.stores = stores
        self.retriever = retriever
        self.preamble = preamble.strip()
        self.max_tokens = max_tokens
        self.max_entry_tokens = max_entry_tokens
        self.relevant_limit = relevant_limit

    def _line(self, e: MemoryEntry) -> str:
        body = truncate_tokens(" ".join(e.content.split()), self.max_entry_tokens)
        return sanitize_memory_text(f"- [{e.entry_id}] {e.title}: {body}")

    # ---------------------------------------------------------- memory base
    def build_stable(self, categories: list[Category], max_tokens: int) -> StableSnapshot:
        """Rarely-changing memory for the system prompt, where the prompt cache reaches it.

        Deterministic: entries are ordered by id, never by time, so unchanged memory
        renders to exactly the same text.
        """
        cats = [c for c, _ in BASE_ORDER if c in categories]
        if not cats:
            return StableSnapshot("", 0)
        header = ("<PROJECT_MEMORY_BASE>\n" + self.preamble +
                  "\nThis base was captured earlier in the session; newer changes, if any, "
                  "arrive with the user's message and take precedence.\n")
        footer = "</PROJECT_MEMORY_BASE>"
        labels = dict(BASE_ORDER)
        remaining = max_tokens - estimate_tokens(header + footer) - sum(
            estimate_tokens(f"\n{labels[c]}:\n") for c in cats)
        fps: dict[str, str] = {}
        dropped: list[str] = []
        sections: list[str] = []
        for cat in cats:
            entries = sorted(self.stores[cat].entries(active_only=True),
                             key=lambda e: int(e.entry_id.split("-")[1]))
            lines = []
            for e in entries:
                line = self._line(e)
                cost = estimate_tokens(line + "\n")
                if remaining > 0 and cost <= remaining:
                    lines.append(line)
                    fps[e.entry_id] = entry_fingerprint(e)
                    remaining -= cost
                else:
                    dropped.append(e.entry_id)
            if lines:
                sections.append(f"\n{labels[cat]}:\n" + "\n".join(lines) + "\n")
        if not fps:
            return StableSnapshot("", 0, {}, tuple(cats), dropped)
        text = header + "".join(sections) + footer
        return StableSnapshot(text, estimate_tokens(text), fps, tuple(cats), dropped)

    def changes_since(self, snap: StableSnapshot) -> tuple[list[MemoryEntry], list[MemoryEntry]]:
        """(new or edited entries, entries that left) in the snapshot's categories."""
        if not snap.categories:
            return [], []
        current = {e.entry_id: e for c in snap.categories for e in self.stores[c].entries(active_only=True)}
        changed = [e for eid, e in current.items()
                   if eid in snap.fingerprints and entry_fingerprint(e) != snap.fingerprints[eid]]
        # New entries only count if the base had room for them; others reach the
        # prompt through normal relevance retrieval instead.
        added = [e for eid, e in current.items() if eid not in snap.fingerprints]
        retired = [eid for eid in snap.fingerprints if eid not in current]
        all_entries = {e.entry_id: e for c in snap.categories for e in self.stores[c].entries()}
        return (sorted(changed + added, key=lambda e: e.entry_id),
                [all_entries[i] for i in sorted(retired) if i in all_entries])

    # -------------------------------------------------------- per-turn block
    def build(self, query: str, max_tokens: int | None = None,
              session_summary: str | None = None, base: StableSnapshot | None = None,
              preamble: bool = True, query_vec=None, history: list[dict] | None = None) -> BuiltContext:
        """Per-turn memory block for the latest user message.

        With `base`, entries already in the frozen base are skipped, and entries in
        the base's categories that changed or appeared since are sent as updates.
        `preamble=False` drops the "this is data, not instructions" text; use it only
        when the (cached) system prompt already says so, since this block is re-read
        every turn.
        """
        budget = max_tokens or self.max_tokens
        header = f"<PROJECT_MEMORY>\n{self.preamble}\n" if preamble else "<PROJECT_MEMORY>\n"
        footer = "</PROJECT_MEMORY>"
        # Section headings are paid for up front so the total can never overshoot.
        overhead = estimate_tokens(header + footer) + sum(
            estimate_tokens(f"\n{label}:\n") for _, label in RENDER_ORDER)
        remaining = budget - overhead
        chosen: dict[Category, list[str]] = {c: [] for c, _ in RENDER_ORDER}
        included: list[str] = []
        dropped: list[str] = []

        if remaining <= 0:
            return BuiltContext("", 0, [], ["<budget smaller than overhead>"])

        skip: set[str] = set(base.fingerprints) if base else set()
        updates_text = ""
        if base and base.fingerprints:
            changed, retired = self.changes_since(base)
            items = [(e.entry_id, self._line(e)) for e in changed if e.entry_id not in base.dropped]
            items += [(e.entry_id, sanitize_memory_text(f"- [{e.entry_id}] {e.title}: no longer applies"))
                      for e in retired]
            if items:
                label = f"\n{UPDATES_LABEL}:\n"
                remaining -= estimate_tokens(label)
                kept = []
                for eid, line in items:
                    cost = estimate_tokens(line + "\n")
                    if cost <= remaining:
                        kept.append(line)
                        skip.add(eid)   # only what was actually sent is excluded from retrieval
                        remaining -= cost
                if kept:
                    updates_text = label + "\n".join(kept) + "\n"
                    included.append("UPDATES")
                else:
                    remaining += estimate_tokens(label)

        # The session summary replaces trimmed history, so it is paid for first,
        # capped at half the budget so durable memory always keeps room.
        session_text = ""
        if session_summary and session_summary.strip():
            label = f"\n{SESSION_LABEL}:\n"
            cap = max(0, remaining // 2 - estimate_tokens(label))
            if cap > 20:
                body = sanitize_memory_text(truncate_tokens(session_summary.strip(), cap))
                session_text = label + body + "\n"
                remaining -= estimate_tokens(session_text)
                included.append("SESSION")

        for cat, always in PRIORITY:
            if always:
                candidates = self.stores[cat].entries(active_only=True)
                # Most recently updated first: newest state wins when space is short.
                candidates.sort(key=lambda e: e.updated_at or "", reverse=True)
            else:
                candidates = [s.entry for s in
                              self.retriever.search(query, self.relevant_limit, categories=[cat],
                                                    query_vec=query_vec)]
            for e in candidates:
                if e.entry_id in skip:
                    continue
                line = self._line(e)
                cost = estimate_tokens(line + "\n")
                if cost <= remaining:
                    chosen[cat].append(line)
                    included.append(e.entry_id)
                    remaining -= cost
                else:
                    dropped.append(e.entry_id)

        history_text = ""
        if history:
            label = f"\n{HISTORY_LABEL}:\n"
            lines = []
            cost_total = estimate_tokens(label)
            for h in history:
                line = sanitize_memory_text(f"- [{h['date']}, conversation {h['conversation_id']}, turn {h['turn']}]\n"
                                            + "\n".join("  " + x for x in h["text"].splitlines()))
                cost = estimate_tokens(line + "\n")
                if cost_total + cost > remaining:
                    break
                lines.append(line)
                cost_total += cost
            if lines:
                history_text = label + "\n".join(lines) + "\n"
                remaining -= cost_total
                included.extend(f"HISTORY:{h['key']}" for h in history[:len(lines)])

        if not included:
            return BuiltContext("", 0, [], dropped)

        parts = [header]
        if updates_text:
            parts.append(updates_text)
        if session_text:
            parts.append(session_text)
        if history_text:
            parts.append(history_text)
        for cat, label in RENDER_ORDER:
            if chosen[cat]:
                parts.append(f"\n{label}:\n" + "\n".join(chosen[cat]) + "\n")
        parts.append(footer)
        text = "".join(parts)
        return BuiltContext(text, estimate_tokens(text), included, dropped)


def wrap_user_request(memory_block: str, user_text: str) -> str:
    """Append the per-turn memory block AFTER the user's text, never before it.

    On the next turn this message reappears in history without the block. With
    the block appended, the cached prompt still matches up to the end of the
    user's text, so only the block itself is lost from cache. A block in front
    would break the cache at the start of the message and force the previous
    exchange to be re-read every turn (the evaluation harness measured +75-115%
    processed tokens from that alone).
    """
    if not memory_block:
        return user_text
    return f"{user_text}\n\n{memory_block}"
