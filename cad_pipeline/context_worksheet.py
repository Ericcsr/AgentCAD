#!/usr/bin/env python3
"""Persistent context worksheet for CAD agent crash recovery."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


WORKSHEET_NAME = "context_worksheet.md"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def truncate(text: str, max_chars: int) -> str:
    text = text or ""
    if len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head - 40
    return text[:head] + "\n\n…(truncated)…\n\n" + text[-tail:]


def truncate_code(code: str, max_chars: int = 8000) -> str:
    code = (code or "").strip()
    if len(code) <= max_chars:
        return code
    lines = code.splitlines()
    if len(lines) <= 120:
        return truncate(code, max_chars)
    keep_head = "\n".join(lines[:80])
    keep_tail = "\n".join(lines[-25:])
    return f"{keep_head}\n\n# …({len(lines) - 105} lines omitted)…\n\n{keep_tail}"


def compress_history(history: list[dict[str, str]], *, max_items: int = 10, max_chars: int = 2000) -> str:
    if not history:
        return "(empty)"
    items = history[-max_items:]
    lines: list[str] = []
    for entry in items:
        role = entry.get("role", "?")
        content = (entry.get("content") or "").strip().replace("\n", " ")
        if len(content) > 220:
            content = content[:200] + "…"
        lines.append(f"- [{role}] {content}")
    return truncate("\n".join(lines), max_chars)


def looks_like_context_error(exc: BaseException | str) -> bool:
    text = str(exc).lower()
    needles = (
        "context length",
        "context window",
        "maximum context",
        "max context",
        "too many tokens",
        "token limit",
        "prompt is too long",
        "prompt too long",
        "context_overflow",
        "context overflow",
        "exceeds the context",
        "reducing the length",
    )
    return any(n in text for n in needles)


def looks_like_recoverable_cursor_error(exc: BaseException) -> bool:
    if getattr(exc, "is_retryable", None) is True:
        return True
    name = type(exc).__name__
    if name in {
        "RateLimitError",
        "APITimeoutError",
        "NetworkError",
        "AgentBusyError",
        "InternalServerError",
    }:
        return True
    text = str(exc).lower()
    if looks_like_context_error(text):
        return True
    if "agent_busy" in text or "agent busy" in text:
        return True
    if "timeout" in text or "temporar" in text or "rate limit" in text:
        return True
    if "connection" in text or "unavailable" in text:
        return True
    return False


@dataclass
class ContextWorksheet:
    """All durable context needed to relaunch a crashed design agent."""

    requirements: str = ""
    features_text: str = ""
    phase: str = "idle"
    task_instruction: str = ""
    pending_prompt: str = ""
    code_snapshot: str = ""
    geometry: str = ""
    last_review: str = ""
    conversation_digest: str = ""
    task_summary: str = ""
    recovery_count: int = 0
    last_error: str = ""
    model: str = ""
    updated_at: str = field(default_factory=_now)
    notes: str = ""
    references_text: str = ""

    def render(self) -> str:
        sections = [
            "# CAD Agent Context Worksheet",
            "",
            "This file is the durable memory for the design agent. After a crash, a new",
            "agent must READ this worksheet (and `generated/current_design.py` /",
            "`generated/feature_list.json` when present) before continuing work.",
            "",
            "## Session",
            f"- updated: {self.updated_at or _now()}",
            f"- model: {self.model or '(unset)'}",
            f"- recovery_count: {self.recovery_count}",
            f"- last_error: {truncate(self.last_error, 500) or '(none)'}",
            "",
            "## Active Task",
            f"- phase: {self.phase or 'idle'}",
            f"- instruction: {self.task_instruction or '(none)'}",
            "",
            "## Task Summary",
            self.task_summary.strip() or "(not summarized yet)",
            "",
            "## Requirements",
            self.requirements.strip() or "(none)",
            "",
            "## Key Features",
            self.features_text.strip() or "(none)",
            "",
            "## Imported STEP References",
            self.references_text.strip() or "(none)",
            "",
            "## Geometry",
            self.geometry.strip() or "(none)",
            "",
            "## Last Review",
            self.last_review.strip() or "(none)",
            "",
            "## Conversation Digest",
            self.conversation_digest.strip() or "(empty)",
            "",
            "## Design Code Snapshot",
            "Source of truth is `generated/current_design.py` when present; snapshot below",
            "is a recovery fallback.",
            "",
            "```python",
            truncate_code(self.code_snapshot, 12000) or "# (empty)",
            "```",
            "",
            "## Pending Prompt (resume payload)",
            "```",
            truncate(self.pending_prompt, 12000) or "(none)",
            "```",
            "",
            "## Notes",
            self.notes.strip() or "(none)",
            "",
        ]
        return "\n".join(sections)

    def save(self, path: Path) -> Path:
        path = path.resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.updated_at = _now()
        path.write_text(self.render() + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "ContextWorksheet":
        if not path.exists():
            return cls()
        text = path.read_text(encoding="utf-8")
        return cls.from_markdown(text)

    @classmethod
    def from_markdown(cls, text: str) -> "ContextWorksheet":
        ws = cls()
        session = _section(text, "Session")
        active = _section(text, "Active Task")
        ws.task_summary = _section(text, "Task Summary").strip()
        ws.requirements = _section(text, "Requirements").strip()
        ws.features_text = _section(text, "Key Features").strip()
        ws.references_text = _section(text, "Imported STEP References").strip()
        ws.geometry = _section(text, "Geometry").strip()
        ws.last_review = _section(text, "Last Review").strip()
        ws.conversation_digest = _section(text, "Conversation Digest").strip()
        ws.notes = _section(text, "Notes").strip()
        ws.pending_prompt = _fenced_or_body(_section(text, "Pending Prompt (resume payload)"))
        code_sec = _section(text, "Design Code Snapshot")
        ws.code_snapshot = _fenced_or_body(code_sec)

        ws.model = _bullet_value(session, "model") or ""
        rec = _bullet_value(session, "recovery_count")
        if rec.isdigit():
            ws.recovery_count = int(rec)
        ws.last_error = _bullet_value(session, "last_error") or ""
        ws.updated_at = _bullet_value(session, "updated") or ""
        ws.phase = _bullet_value(active, "phase") or "idle"
        ws.task_instruction = _bullet_value(active, "instruction") or ""
        return ws

    def apply_context_overflow_summary(self, history: list[dict[str, str]] | None = None) -> None:
        """Compress worksheet fields after a context-window failure."""
        parts = [
            f"Phase: {self.phase or 'unknown'}.",
            f"Current instruction: {truncate(self.task_instruction, 400)}.",
            f"Requirements: {truncate(self.requirements, 700)}.",
            f"Features: {truncate(self.features_text, 700)}.",
        ]
        if self.references_text:
            parts.append(f"STEP references: {truncate(self.references_text, 500)}.")
        if self.last_review:
            parts.append(f"Last review: {truncate(self.last_review, 400)}.")
        parts.append(
            "Prior agent hit a context-window / oversized-prompt failure; "
            "continue from this compressed worksheet and current_design.py."
        )
        self.task_summary = " ".join(parts)
        if history:
            self.conversation_digest = compress_history(history, max_items=8, max_chars=1500)
        self.code_snapshot = truncate_code(self.code_snapshot, max_chars=5000)
        # Prefer short resume instruction over a huge pending prompt
        self.pending_prompt = truncate(
            self.task_instruction or self.pending_prompt,
            2000,
        )
        note = (
            "Context overflow recovery: worksheet summarized; new agent should rely on "
            "Task Summary + Features + Design Code Snapshot rather than full chat history."
        )
        self.notes = f"{self.notes}\n{note}".strip() if self.notes else note


def _section(text: str, title: str) -> str:
    pattern = rf"^## {re.escape(title)}\s*\n(.*?)(?=^## |\Z)"
    m = re.search(pattern, text, flags=re.M | re.S)
    return m.group(1).strip() if m else ""


def _bullet_value(section: str, key: str) -> str:
    for line in section.splitlines():
        s = line.strip()
        if s.lower().startswith(f"- {key.lower()}:"):
            return s.split(":", 1)[-1].strip()
    return ""


def _fenced_or_body(section: str) -> str:
    if "```" not in section:
        return section.strip()
    parts = section.split("```")
    # content after opening fence
    if len(parts) >= 2:
        body = parts[1]
        if body.startswith("python") or body.startswith("text"):
            body = body.split("\n", 1)[-1]
        return body.strip()
    return section.strip()
