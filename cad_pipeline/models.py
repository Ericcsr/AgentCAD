#!/usr/bin/env python3
"""Cursor SDK model ids and friendly aliases for AgentCAD."""

from __future__ import annotations

DEFAULT_MODEL = "grok-4.5"

# Friendly names → Cursor SDK model ids. Unknown values pass through unchanged.
MODEL_ALIASES: dict[str, str] = {
    "grok": "grok-4.5",
    "grok-4.5": "grok-4.5",
    "grok4.5": "grok-4.5",
    "grok-4.6": "grok-4.6",
    "grok4.6": "grok-4.6",
    "claude": "claude-sonnet-5",
    "sonnet": "claude-sonnet-5",
    "claude-sonnet": "claude-sonnet-5",
    "claude-sonnet-5": "claude-sonnet-5",
    "claude-4.6-sonnet": "claude-4.6-sonnet",
    "claude-4.5-sonnet": "claude-4.5-sonnet",
    "opus": "claude-opus-5",
    "claude-opus": "claude-opus-5",
    "claude-opus-5": "claude-opus-5",
    "openai": "gpt-5.4",
    "gpt": "gpt-5.4",
    "gpt-5": "gpt-5",
    "gpt-5.2": "gpt-5.2",
    "gpt-5.4": "gpt-5.4",
    "gpt-5.6": "gpt-5.6-sol",
    "gpt-5.6-sol": "gpt-5.6-sol",
    "composer": "composer-2.5",
    "composer-2.5": "composer-2.5",
    "auto": "auto",
}

# Shown by --list-models / argparse help (alias, resolved id, note).
MODEL_PRESETS: tuple[tuple[str, str, str], ...] = (
    ("grok-4.5", "grok-4.5", "Cursor Grok 4.5 (default)"),
    ("grok-4.6", "grok-4.6", "Cursor Grok 4.6"),
    ("claude", "claude-sonnet-5", "Claude Sonnet 5"),
    ("claude-opus", "claude-opus-5", "Claude Opus 5"),
    ("openai", "gpt-5.4", "OpenAI GPT-5.4"),
    ("composer", "composer-2.5", "Cursor Composer 2.5"),
    ("auto", "auto", "Cursor server-selected model"),
)


def normalize_model_key(value: str) -> str:
    return value.strip().lower().replace("_", "-").replace(" ", "-")


def resolve_model(value: str | None, *, default: str = DEFAULT_MODEL) -> str:
    """Map a CLI/env alias to a Cursor SDK model id."""
    if value is None or not str(value).strip():
        return default
    key = normalize_model_key(str(value))
    return MODEL_ALIASES.get(key, key)


def model_help_text() -> str:
    lines = [
        "Model aliases (Cursor SDK). Pass a raw id for anything else:",
        "",
        f"  {'alias':<16} {'id':<22} note",
        f"  {'-' * 16} {'-' * 22} {'-' * 28}",
    ]
    for alias, model_id, note in MODEL_PRESETS:
        lines.append(f"  {alias:<16} {model_id:<22} {note}")
    lines.append("")
    lines.append("Also: DESIGN_MODEL=grok-4.6  (CLI --model overrides env)")
    return "\n".join(lines)
