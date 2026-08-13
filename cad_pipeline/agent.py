#!/usr/bin/env python3
"""Design agent: language → CadQuery via Cursor SDK or mock."""

from __future__ import annotations

import base64
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from dotenv import load_dotenv

from cad_pipeline.context_worksheet import (
    ContextWorksheet,
    WORKSHEET_NAME,
    compress_history,
    looks_like_context_error,
    looks_like_recoverable_cursor_error,
    truncate,
)
from cad_pipeline.mesh_utils import geometry_summary
from cad_pipeline.models import DEFAULT_MODEL, resolve_model
from cad_pipeline.render import RENDER_DIR, MeshRenderer, list_views
from cad_pipeline.runtime import DesignResult, run_design_code
from cad_pipeline.step_import import (
    StepReference,
    format_references_block,
    import_step_file,
)
from cad_pipeline.versioning import DesignVersionStore, VersionMeta, VersionSnapshot

ROOT = Path(__file__).resolve().parent.parent
GENERATED_DIR = ROOT / "generated"
DESIGN_FILE = GENERATED_DIR / "current_design.py"
FEATURES_FILE = GENERATED_DIR / "feature_list.json"
WORKSHEET_FILE = GENERATED_DIR / WORKSHEET_NAME

DEFAULT_VIEWS = ("isometric", "front", "side")
REVIEW_VIEWS = ("isometric", "front", "side", "top", "back")

# Pipeline quality modes
MODE_DRAFT = "draft"  # compile + tessellate only — fast iteration
MODE_REFINE = "refine"  # full feature / physics review loop


@dataclass
class DesignFeature:
    """One checklist item derived from the user brief / revisions."""

    id: str
    text: str
    status: str = "pending"  # pending | met | unmet
    evidence: str = ""

    def line(self) -> str:
        extra = f" — {self.evidence}" if self.evidence else ""
        return f"{self.id}: [{self.status.upper()}] {self.text}{extra}"


@dataclass
class ReviewOutcome:
    passed: bool
    raw: str
    issues: list[str]
    actions: list[str]
    feature_results: list[DesignFeature] = field(default_factory=list)


def format_feature_list(features: Sequence[DesignFeature]) -> str:
    if not features:
        return "(no features yet)"
    return "\n".join(f"- {f.id}: {f.text}" for f in features)


def format_feature_status(features: Sequence[DesignFeature]) -> str:
    if not features:
        return "(no features yet)"
    return "\n".join(f"- {f.line()}" for f in features)


def parse_features_block(text: str) -> list[DesignFeature]:
    """Parse a FEATURES: bullet list into DesignFeature rows."""
    raw = text or ""
    features: list[DesignFeature] = []
    in_block = False
    for line in raw.splitlines():
        s = line.strip()
        su = s.upper()
        if su.startswith("FEATURES:") or su.startswith("FEATURE LIST:"):
            in_block = True
            rest = s.split(":", 1)[-1].strip()
            if rest:
                features.append(_parse_feature_item(rest, len(features) + 1))
            continue
        if su.startswith("FEATURES_CHECK:") or su.startswith("VERDICT:"):
            in_block = False
            continue
        if not in_block:
            if re.match(r"^[-*•]?\s*F\d+\s*:", s, re.I):
                in_block = True
            else:
                continue
        if not s or s.startswith("#"):
            continue
        if s.startswith(("-", "*", "•")) or re.match(r"^F\d+\s*:", s, re.I):
            item = s.lstrip("-*• ").strip()
            if item:
                features.append(_parse_feature_item(item, len(features) + 1))
    out: list[DesignFeature] = []
    seen: set[str] = set()
    for f in features:
        text_key = f.text.strip().lower()
        if not text_key or text_key in seen:
            continue
        seen.add(text_key)
        out.append(DesignFeature(id=f"F{len(out) + 1}", text=f.text.strip(), status="pending"))
    return out


def _parse_feature_item(item: str, fallback_index: int) -> DesignFeature:
    m = re.match(r"^(F\d+)\s*[:.\-–—]\s*(.+)$", item.strip(), re.I)
    if m:
        return DesignFeature(id=m.group(1).upper(), text=m.group(2).strip())
    return DesignFeature(id=f"F{fallback_index}", text=item.strip())


def parse_features_check(
    text: str,
    features: Sequence[DesignFeature],
) -> list[DesignFeature]:
    """Parse FEATURES_CHECK: lines and merge MET/UNMET into a copy of features."""
    by_id = {f.id.upper(): DesignFeature(f.id, f.text, f.status, f.evidence) for f in features}
    order = [f.id.upper() for f in features]
    in_block = False
    for line in (text or "").splitlines():
        s = line.strip()
        su = s.upper()
        if su.startswith("FEATURES_CHECK:"):
            in_block = True
            continue
        if su.startswith("VERDICT:") or su.startswith("ISSUES:") or su.startswith("ACTIONS:"):
            in_block = False
            continue
        if not in_block:
            continue
        item = s.lstrip("-*• ").strip()
        if not item:
            continue
        m = re.match(
            r"^(F\d+)\s*[:.\-–—]\s*(MET|UNMET|PARTIAL)?\s*[:.\-–—]?\s*(.*)$",
            item,
            re.I,
        )
        if not m:
            continue
        fid = m.group(1).upper()
        status_raw = (m.group(2) or "").upper()
        evidence = (m.group(3) or "").strip(" :.-–—")
        if status_raw == "MET":
            status = "met"
        elif status_raw in {"UNMET", "PARTIAL"}:
            status = "unmet"
        else:
            # "F1: something" without explicit MET — treat evidence for lookup
            low = item.lower()
            if re.search(r"\bunmet\b", low) or re.search(r"\bmissing\b", low):
                status = "unmet"
            elif re.search(r"\bmet\b", low) or re.search(r"\bpass\b", low):
                status = "met"
            else:
                status = "unmet"
            if not evidence:
                evidence = item
        if fid not in by_id:
            by_id[fid] = DesignFeature(id=fid, text=evidence or fid, status=status, evidence=evidence)
            order.append(fid)
        else:
            by_id[fid].status = status
            by_id[fid].evidence = evidence
    return [by_id[i] for i in order if i in by_id]


def parse_review(text: str, features: Sequence[DesignFeature] | None = None) -> ReviewOutcome:
    """Parse VERDICT / ISSUES / ACTIONS / FEATURES_CHECK from a review reply."""
    raw = (text or "").strip()
    upper = raw.upper()
    passed = bool(re.search(r"\bVERDICT:\s*PASS\b", upper))
    if re.search(r"\bVERDICT:\s*FAIL\b", upper):
        passed = False
    elif not passed and "FAIL" in upper.split("\n", 1)[0]:
        passed = False

    issues: list[str] = []
    actions: list[str] = []
    section = None
    for line in raw.splitlines():
        s = line.strip()
        su = s.upper()
        if su.startswith("VERDICT:") or su.startswith("FEATURES_CHECK:") or su.startswith("FEATURES:"):
            section = None
            continue
        if su.startswith("ISSUES:"):
            section = "issues"
            rest = s.split(":", 1)[-1].strip()
            if rest:
                issues.append(rest)
            continue
        if su.startswith("ACTIONS:"):
            section = "actions"
            rest = s.split(":", 1)[-1].strip()
            if rest:
                actions.append(rest)
            continue
        if s.startswith(("-", "*", "•")):
            item = s.lstrip("-*• ").strip()
            if not item:
                continue
            if section == "actions":
                actions.append(item)
            elif section == "issues":
                issues.append(item)
    feature_results: list[DesignFeature] = []
    if features:
        feature_results = parse_features_check(raw, features)
        unmet = [f for f in feature_results if f.status == "unmet"]
        if unmet:
            passed = False
            for f in unmet:
                note = f"{f.id} unmet: {f.text}"
                if f.evidence:
                    note += f" ({f.evidence})"
                if note not in issues:
                    issues.append(note)
                action = f"Implement feature {f.id}: {f.text}"
                if action not in actions:
                    actions.append(action)
    return ReviewOutcome(
        passed=passed,
        raw=raw,
        issues=issues,
        actions=actions,
        feature_results=feature_results,
    )


def heuristic_features_from_prompt(prompt: str) -> list[DesignFeature]:
    """Offline / fallback feature extraction from a language brief."""
    text = (prompt or "").strip()
    if not text:
        return []
    features: list[DesignFeature] = []

    def add(desc: str) -> None:
        features.append(DesignFeature(id=f"F{len(features) + 1}", text=desc))

    low = text.lower()
    add(f"Satisfy the design brief: {text[:160]}{'…' if len(text) > 160 else ''}")
    if any(w in low for w in ("chair", "seat")):
        add("Include a seating surface at a usable sitting height")
        add("Provide stable support to the floor (legs or base)")
        add("Include a backrest unless the brief says otherwise")
    if "table" in low:
        add("Include a flat tabletop at a usable height")
        add("Provide stable legs or a base to the floor")
    if "arm" in low:
        add("Include armrests")
    if any(w in low for w in ("drawer", "shelf", "storage")):
        add("Include the requested storage feature (drawer/shelf)")
    if any(w in low for w in ("round", "circular", "cylinder")):
        add("Use rounded / circular geometry where requested")
    if any(w in low for w in ("mm", "cm", "inch", "tall", "wide", "height", "width", "depth")):
        add("Match any explicit dimensions stated in the brief")
    # De-dupe by text
    seen: set[str] = set()
    out: list[DesignFeature] = []
    for f in features:
        key = f.text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(DesignFeature(id=f"F{len(out) + 1}", text=f.text))
    return out


def merge_feature_lists(
    existing: Sequence[DesignFeature],
    incoming: Sequence[DesignFeature],
) -> list[DesignFeature]:
    """Merge new/updated features; keep prior text when ids collide unless incoming is richer."""
    by_text = {f.text.strip().lower(): f for f in existing}
    out = [DesignFeature(f.id, f.text, "pending", "") for f in existing]
    for neu in incoming:
        key = neu.text.strip().lower()
        if not key:
            continue
        # Fuzzy: if incoming is a refinement of an existing line, replace it
        replaced = False
        for i, old in enumerate(out):
            old_key = old.text.strip().lower()
            if key == old_key or key in old_key or old_key in key:
                out[i] = DesignFeature(id=old.id, text=neu.text.strip(), status="pending")
                replaced = True
                break
        if not replaced and key not in by_text:
            out.append(DesignFeature(id=f"F{len(out) + 1}", text=neu.text.strip(), status="pending"))
    # Re-number
    return [DesignFeature(id=f"F{i}", text=f.text, status="pending") for i, f in enumerate(out, start=1)]


def heuristic_physics_notes(result: DesignResult) -> list[str]:
    """Cheap geometric sanity notes included in the review context."""
    notes: list[str] = []
    mins = result.vertices.min(axis=0)
    maxs = result.vertices.max(axis=0)
    size = maxs - mins
    notes.append(
        f"Measured bbox mm: X={size[0]:.1f}, Y={size[1]:.1f}, Z={size[2]:.1f}; "
        f"z_min={mins[2]:.1f}, z_max={maxs[2]:.1f}"
    )
    if mins[2] > 2.0:
        notes.append(f"WARNING: solid appears to float above the floor (z_min={mins[2]:.1f} mm).")
    if mins[2] < -2.0:
        notes.append(f"WARNING: solid extends below the floor (z_min={mins[2]:.1f} mm).")
    if size[2] < 5.0 or size[0] < 5.0 or size[1] < 5.0:
        notes.append("WARNING: bbox has a very small dimension — possible collapsed/missing geometry.")
    if size[2] > 5000 or size[0] > 5000 or size[1] > 5000:
        notes.append("WARNING: bbox is extremely large (>5 m) — check units/scale.")
    # Thin aspect: one axis tiny relative to others
    axes = sorted(size)
    if axes[0] > 0 and axes[2] / axes[0] > 200:
        notes.append("WARNING: extremely thin aspect ratio — possible paper-thin feature.")
    return notes

SYSTEM_BRIEF = """You are an expert mechanical CAD designer writing CadQuery (Python) code
inside the ai_design project.

Hard requirements:
- Write the FULL design to the single file: generated/current_design.py
- Define parts() -> dict[str, cq.Workplane] with clear part names (seat, leg_fl, backrest, …)
- Define build() -> cq.Workplane as the assembled union of parts() (or return the dict from
  build() alone if you prefer — runtime accepts either)
- Units are millimeters. Z is up. Object sits on the XY floor (z=0).
- Prefer robust boolean unions of simple boxes/cylinders/extrusions.
- Do not use `__import__` / `import` in that file. Injected at runtime: `cq` (cadquery),
  `math`, and `import_reference(name)` for user-attached STEP files.
- Do not print or network inside build()/parts().
- Do NOT modify any other project files.
- Do NOT run CadQuery yourself; just write the source file.
- After writing the file, reply with a one-line confirmation.

CadQuery API tips (avoid common failures):
- Rotate solids with Workplane.rotate((x1,y1,z1), (x2,y2,z2), angle_deg) — NOT Vector.rotate.
- Translate with .translate((x,y,z)). Fillet with .edges(...).fillet(r).
- Boolean: .union(), .cut(), .intersect(). Keep fillet radii smaller than feature size.
- Prefer boxes/cylinders/extrude; avoid fragile edge selections when possible.

Parts:
- Keep each logical component as its own named entry in parts().
- build() should union those parts into the full product.
- When a revision is scoped to ONE part, change only that part's geometry; keep other
  parts' names and interfaces stable unless the user asks otherwise.

Imported STEP references:
- The user may attach existing STEP models as constraints (mating, envelope, holes).
- Measured facts are in the prompt and in generated/references/<name>.md.
- Do NOT open, read, cat, or glob any .step/.stp files — they are large binaries
  and will stall this session. Facts are already extracted.
- To use the exact B-rep in CadQuery, call import_reference("name") (returns Workplane).
- Prefer copying measured dimensions into parametric code; import the STEP only when
  you need the exact shape (cut/union/mate/envelope).

Vision / camera:
- RGB preview images of the current design may be attached (labeled by view name).
- Use them to judge proportions, gaps, and mistakes before editing.
- You can call the custom tool `render_cad_view` to render another camera angle
  (view: isometric|front|back|side|left|top|bottom|three_quarter, or elev+azim degrees).
- Z is up. front = looking from -Y, side = from +X, top = from +Z.
"""

DEBUG_BRIEF = """You are fixing a broken CadQuery design. READ-ONLY except for the design file.

The previous build() failed. Fix the Python so build() succeeds.

Hard requirements:
- Overwrite generated/current_design.py with a FULL corrected build()/parts() implementation.
- Do not use `__import__`; `cq`, `math`, and `import_reference(name)` are injected.
- Keep the user's design intent; only fix the error (and closely related issues).
- Prefer simple CadQuery patterns: box/cylinder/extrude + union/cut + Workplane.rotate/translate.
- Keep a parts() dict of named components when the design has multiple pieces.
- Do NOT use Vector.rotate — use Workplane.rotate(axisStart, axisEnd, angleDegrees).
- After writing the fixed file, reply with a one-line confirmation.
"""

ASK_BRIEF = """You are in ASK mode for a CAD design review — read-only.

Hard requirements:
- Answer the user's question about the current design clearly and concisely.
- Use the CadQuery source, measured geometry, and imported STEP *facts* (text) as
  ground truth. RGB renders are optional; do not wait on extra views.
- Units are millimeters unless the user asks otherwise.
- Do NOT modify any files. Do NOT write code. Do NOT edit the workspace.
- Do NOT open or read any .step/.stp files.
- Do NOT suggest you changed the model; Ask mode never changes geometry.
- If a dimension is not explicit in the source, say so and give the best estimate from context.
- Do not call tools unless the question is specifically about another camera angle.
"""

REVIEW_BRIEF = """You are a senior CAD design reviewer. READ-ONLY — do not edit files.

Inspect the CadQuery source AND the attached RGB renders (multiple camera angles).
Decide whether the design is acceptable.

Check all of the following:
1) Key features: for EACH item in the feature list, decide MET or UNMET using code + images
2) Visual integrity: missing parts, wrong proportions, disconnected pieces, obvious artifacts
3) Physics / manufacturing sanity (units mm, Z-up, floor at z=0):
   - Object should rest on or near the floor (z_min ≈ 0), not float or dig deep below z=0
   - Stable support (legs/base) for furniture-like objects
   - Feature sizes plausible; no paper-thin walls or absurd scales unless requested
   - Parts that should touch should not have large unintended gaps

Output format — EXACTLY like this (no markdown fences):
FEATURES_CHECK:
- F1: MET — short evidence from code or renders
- F2: UNMET — what is missing or wrong
VERDICT: PASS
or
VERDICT: FAIL
ISSUES:
- concrete issue 1
ACTIONS:
- specific CadQuery change 1

VERDICT: PASS only if EVERY feature is MET and physics/visual checks are acceptable.
If any feature is UNMET, VERDICT must be FAIL and ACTIONS must cover those features.
You may call render_cad_view for an extra angle before deciding.
"""

FEATURES_EXTRACT_BRIEF = """You extract a concise key-feature checklist from a CAD design brief.
READ-ONLY — do not edit files.

Rules:
- List concrete, verifiable product features (geometry, parts, proportions, function).
- Prefer 4–10 items. Merge vague wishes into testable features.
- Do NOT invent features the user did not imply.
- Units mm when dimensions are mentioned.

Output format — EXACTLY (no markdown fences):
FEATURES:
- F1: ...
- F2: ...
"""

FEATURES_UPDATE_BRIEF = """You maintain a CAD key-feature checklist.
READ-ONLY — do not edit files.

Given the EXISTING feature list and a NEW user requirement / clarification, produce an
UPDATED full list:
- Keep features that still apply
- Refine wording when the user adds detail to an existing feature
- Add new features for genuinely new requirements
- Remove or replace features the user explicitly cancelled
- Prefer 4–12 items total

Output format — EXACTLY (no markdown fences):
FEATURES:
- F1: ...
- F2: ...
"""

REFINE_BRIEF = """You are refining a CadQuery design after a failed design review.

Hard requirements:
- Overwrite generated/current_design.py with a FULL updated build() implementation.
- Do not use `__import__`; `cq`, `math`, and `import_reference(name)` are injected.
- Address EVERY listed ACTION / ISSUE from the review while keeping the user's brief.
- Satisfy every UNMET key feature listed in the checklist.
- Prefer robust CadQuery patterns (box/cylinder/extrude + union/cut + Workplane.rotate/translate).
- Do NOT use Vector.rotate.
- After writing the file, reply with a one-line confirmation.
"""

CONTEXT_RESUME_BRIEF = """You are RESUMING a CAD design session after the previous agent crashed
or hit a context/API failure.

Hard requirements:
1) FIRST read generated/context_worksheet.md end-to-end.
2) Also read generated/current_design.py, generated/feature_list.json, and
   generated/references/*.md if they exist. Never read .step/.stp files.
3) Treat the worksheet Task Summary / Requirements / Key Features as ground truth.
4) Continue the Active Task — do not restart the whole product from scratch unless the
   design file is missing or empty.
5) Then perform the pending work described below.
"""

MOCK_CHAIR = '''def parts():
    """Named parts for the mock chair."""
    SEAT_W, SEAT_D, SEAT_T, SEAT_H = 450.0, 420.0, 28.0, 450.0
    LEG_W, LEG_INSET = 38.0, 25.0
    BACK_H, BACK_T, BACK_RECLINE = 420.0, 22.0, 8.0
    STRETCHER_T, STRETCHER_H = 22.0, 120.0
    SLATS, GAP = 4, 18.0

    half_w = SEAT_W / 2 - LEG_INSET - LEG_W / 2
    half_d = SEAT_D / 2 - LEG_INSET - LEG_W / 2
    span_w = 2 * half_w - LEG_W
    span_d = 2 * half_d - LEG_W

    def union(items):
        s = items[0]
        for p in items[1:]:
            s = s.union(p)
        return s

    out = {}
    out["seat"] = (
        cq.Workplane("XY")
        .workplane(offset=SEAT_H - SEAT_T)
        .box(SEAT_W, SEAT_D, SEAT_T, centered=(True, True, False))
        .edges("|Z").fillet(6)
    )
    for label, (x, y) in {
        "leg_fl": (-half_w, -half_d),
        "leg_fr": (half_w, -half_d),
        "leg_bl": (-half_w, half_d),
        "leg_br": (half_w, half_d),
    }.items():
        out[label] = (
            cq.Workplane("XY")
            .box(LEG_W, LEG_W, SEAT_H, centered=(True, True, False))
            .translate((x, y, 0))
        )
    out["stretcher_l"] = (
        cq.Workplane("XY")
        .workplane(offset=STRETCHER_H - STRETCHER_T / 2)
        .box(STRETCHER_T, span_d, STRETCHER_T, centered=(True, True, False))
        .translate((-half_w, 0, 0))
    )
    out["stretcher_r"] = (
        cq.Workplane("XY")
        .workplane(offset=STRETCHER_H - STRETCHER_T / 2)
        .box(STRETCHER_T, span_d, STRETCHER_T, centered=(True, True, False))
        .translate((half_w, 0, 0))
    )
    out["stretcher_f"] = (
        cq.Workplane("XY")
        .workplane(offset=STRETCHER_H - STRETCHER_T / 2)
        .box(span_w, STRETCHER_T, STRETCHER_T, centered=(True, True, False))
        .translate((0, -half_d, 0))
    )

    back = []
    for x in (-half_w, half_w):
        back.append(
            cq.Workplane("XY")
            .workplane(offset=SEAT_H)
            .box(LEG_W, LEG_W, BACK_H, centered=(True, True, False))
            .translate((x, half_d, 0))
        )
    back.append(
        cq.Workplane("XY")
        .workplane(offset=SEAT_H + BACK_H - BACK_T)
        .box(span_w + LEG_W, BACK_T, BACK_T, centered=(True, True, False))
        .translate((0, half_d, 0))
    )
    slat_w = (span_w - (SLATS - 1) * GAP) / SLATS
    start_x = -span_w / 2 + slat_w / 2
    for i in range(SLATS):
        x = start_x + i * (slat_w + GAP)
        back.append(
            cq.Workplane("XY")
            .workplane(offset=SEAT_H + 5)
            .box(slat_w, BACK_T * 0.7, BACK_H - BACK_T - 10, centered=(True, True, False))
            .translate((x, half_d, 0))
        )
    out["backrest"] = union(back).rotate((0, half_d, SEAT_H), (1, half_d, SEAT_H), BACK_RECLINE)
    return out

def build():
    p = parts()
    solid = None
    for item in p.values():
        solid = item if solid is None else solid.union(item)
    return solid
'''

MOCK_TABLE = '''def parts():
    TOP_W, TOP_D, TOP_T, TOP_H = 1200.0, 700.0, 30.0, 750.0
    LEG_W, INSET = 50.0, 40.0
    half_w = TOP_W / 2 - INSET - LEG_W / 2
    half_d = TOP_D / 2 - INSET - LEG_W / 2
    out = {
        "top": (
            cq.Workplane("XY")
            .workplane(offset=TOP_H - TOP_T)
            .box(TOP_W, TOP_D, TOP_T, centered=(True, True, False))
            .edges("|Z").fillet(4)
        )
    }
    for label, (x, y) in {
        "leg_fl": (-half_w, -half_d),
        "leg_fr": (half_w, -half_d),
        "leg_bl": (-half_w, half_d),
        "leg_br": (half_w, half_d),
    }.items():
        out[label] = (
            cq.Workplane("XY")
            .box(LEG_W, LEG_W, TOP_H - TOP_T, centered=(True, True, False))
            .translate((x, y, 0))
        )
    return out

def build():
    p = parts()
    solid = None
    for item in p.values():
        solid = item if solid is None else solid.union(item)
    return solid
'''

MOCK_BOX = '''def parts():
    return {
        "body": cq.Workplane("XY").box(200, 120, 80, centered=(True, True, False)).edges("|Z").fillet(6)
    }

def build():
    return parts()["body"]
'''


def _scale_assignments(code: str, names: tuple[str, ...], factor: float) -> str:
    pattern = r"(" + "|".join(names) + r")\s*=\s*([0-9.]+)"

    def repl(m: re.Match[str]) -> str:
        return f"{m.group(1)} = {float(m.group(2)) * factor:.1f}"

    return re.sub(pattern, repl, code)


def _extract_code(text: str) -> str:
    text = text.strip()
    if "```" in text:
        parts = text.split("```")
        for i, part in enumerate(parts):
            if i % 2 == 1:
                body = part
                if body.startswith("python"):
                    body = body[len("python") :]
                elif body.startswith("py"):
                    body = body[len("py") :]
                body = body.strip()
                if "def build" in body:
                    return body
    if "def build" in text:
        idx = text.find("def build")
        return text[idx:].strip()
    return text


# Prefix for status-bar-only updates (studio will not append these to the chat log).
_HEARTBEAT_PREFIX = "~ "


def _tool_arg_hint(name: str, args: object) -> str:
    """Short path/command/view snippet for a Cursor tool call."""
    data = args
    if not isinstance(data, dict):
        return ""
    path = data.get("path") or data.get("file") or data.get("target_file") or data.get("file_path")
    if path:
        return f" · {path}"
    cmd = data.get("command") or data.get("cmd")
    if cmd:
        text = str(cmd).replace("\n", " ").strip()
        if len(text) > 90:
            text = text[:87] + "…"
        return f" · {text}"
    view = data.get("view")
    if view:
        return f" · view={view}"
    query = data.get("pattern") or data.get("query") or data.get("glob")
    if query:
        text = str(query).replace("\n", " ").strip()
        if len(text) > 70:
            text = text[:67] + "…"
        return f" · {text}"
    return ""


def _assistant_text_from_message(message: object) -> str:
    content = getattr(getattr(message, "message", None), "content", ()) or ()
    chunks: list[str] = []
    for block in content:
        text = getattr(block, "text", None)
        if text:
            chunks.append(str(text))
        elif isinstance(block, dict) and block.get("text"):
            chunks.append(str(block["text"]))
    return "".join(chunks)


def _update_type(update: object) -> str:
    kind = getattr(update, "type", None)
    if kind:
        return str(kind)
    if isinstance(update, dict):
        return str(update.get("type") or "")
    return ""


def _tool_from_update(update: object) -> tuple[str, object]:
    tc = getattr(update, "tool_call", None)
    if tc is None and isinstance(update, dict):
        tc = update.get("toolCall") or update.get("tool_call")
    if not isinstance(tc, dict):
        return "tool", None
    nested = tc.get("tool")
    name = (
        tc.get("name")
        or tc.get("toolName")
        or tc.get("tool_name")
        or (nested.get("name") if isinstance(nested, dict) else None)
        or "tool"
    )
    args = tc.get("args") or tc.get("arguments") or tc.get("input") or tc.get("params")
    return str(name), args


def _consume_cursor_run(run: object, status: Callable[[str], None]) -> tuple[str, object]:
    """
    Wait for the Cursor run while logging thinking/tool deltas.

    Uses run.events() (not messages()) so interaction_update deltas are visible.
    Same terminal result as run.text() + run.wait().
    """
    started = time.monotonic()
    last_hb = [0.0]
    texts: list[str] = []
    saw_thinking = False
    saw_assistant = False
    tool_counts: dict[str, int] = {}

    def elapsed() -> float:
        return time.monotonic() - started

    def heartbeat(msg: str) -> None:
        now = time.monotonic()
        if now - last_hb[0] < 0.45:
            return
        last_hb[0] = now
        status(f"{_HEARTBEAT_PREFIX}{msg}  ({elapsed():.0f}s)")

    def on_delta(update: object) -> None:
        nonlocal saw_thinking, saw_assistant
        kind = _update_type(update)
        if kind == "thinking-delta":
            snippet = (getattr(update, "text", None) or "").strip().replace("\n", " ")
            if not saw_thinking:
                saw_thinking = True
                status(f"Cursor thinking…  ({elapsed():.0f}s)")
            if snippet:
                if len(snippet) > 100:
                    snippet = snippet[:97] + "…"
                heartbeat(f"Cursor thinking · {snippet}")
            else:
                heartbeat("Cursor thinking…")
        elif kind == "thinking-completed":
            ms = getattr(update, "thinking_duration_ms", None)
            extra = f" ({ms}ms)" if ms else ""
            status(f"Cursor thinking done{extra}  ({elapsed():.0f}s)")
        elif kind == "tool-call-started":
            name, args = _tool_from_update(update)
            tool_counts[name] = tool_counts.get(name, 0) + 1
            hint = _tool_arg_hint(name, args)
            status(f"Cursor tool `{name}` started{hint}  ({elapsed():.0f}s)")
        elif kind == "tool-call-completed":
            name, args = _tool_from_update(update)
            hint = _tool_arg_hint(name, args)
            status(f"Cursor tool `{name}` completed{hint}  ({elapsed():.0f}s)")
        elif kind == "partial-tool-call":
            name, args = _tool_from_update(update)
            heartbeat(f"Cursor tool `{name}` streaming{_tool_arg_hint(name, args)}")
        elif kind == "text-delta":
            chunk = getattr(update, "text", None) or ""
            if chunk:
                texts.append(str(chunk))
                if not saw_assistant:
                    saw_assistant = True
                    status(f"Cursor assistant reply started  ({elapsed():.0f}s)")
                else:
                    heartbeat(f"Cursor writing reply · {sum(len(t) for t in texts):,} chars")
        elif kind == "shell-output-delta":
            heartbeat("Cursor shell output…")
        elif kind in {"step-started", "step-completed"}:
            step_id = getattr(update, "step_id", "")
            status(f"Cursor {kind} step={step_id}  ({elapsed():.0f}s)")
        elif kind == "turn-ended":
            status(f"Cursor turn ended  ({elapsed():.0f}s)")
        elif kind:
            heartbeat(f"Cursor delta · {kind}")

    # Attach delta listener if the run was created with on_delta already;
    # events() still surfaces sdk_message + interaction_update.
    run_id = getattr(run, "id", "") or ""
    agent_id = getattr(run, "agent_id", "") or ""
    short_run = (str(run_id)[:14] + "…") if len(str(run_id)) > 14 else str(run_id)
    status(
        f"Cursor run started"
        + (f"  id={short_run}" if short_run else "")
        + (f"  agent={str(agent_id)[:16]}" if agent_id else "")
        + " — waiting for thinking/tools (deltas on)…"
    )

    stop_wait_tick = threading.Event()

    def wait_tick() -> None:
        while not stop_wait_tick.wait(2.0):
            heartbeat("Cursor still waiting for first thinking/tool event…")

    waiter = threading.Thread(target=wait_tick, daemon=True, name="cursor-wait-tick")
    waiter.start()

    events = getattr(run, "events", None)
    try:
        if callable(events):
            for event in events():
                stop_wait_tick.set()
                msg = getattr(event, "sdk_message", None)
                update = getattr(event, "interaction_update", None)
                if update is not None:
                    on_delta(update)
                if msg is None:
                    continue
                kind = getattr(msg, "type", "") or ""
                if kind == "thinking":
                    snippet = (getattr(msg, "text", "") or "").strip().replace("\n", " ")
                    if not saw_thinking:
                        saw_thinking = True
                        status(f"Cursor thinking…  ({elapsed():.0f}s)")
                    if snippet:
                        heartbeat(f"Cursor thinking · {snippet[:100]}")
                elif kind == "tool_call":
                    name = str(getattr(msg, "name", "") or "tool")
                    st = str(getattr(msg, "status", "") or "running")
                    hint = _tool_arg_hint(name, getattr(msg, "args", None))
                    if st in {"", "running"}:
                        tool_counts[name] = tool_counts.get(name, 0) + 1
                        status(f"Cursor tool `{name}` running{hint}  ({elapsed():.0f}s)")
                    else:
                        status(f"Cursor tool `{name}` {st}{hint}  ({elapsed():.0f}s)")
                elif kind == "assistant":
                    chunk = _assistant_text_from_message(msg)
                    if chunk:
                        texts.append(chunk)
                        if not saw_assistant:
                            saw_assistant = True
                            status(f"Cursor assistant reply started  ({elapsed():.0f}s)")
                        else:
                            heartbeat(f"Cursor writing reply · {sum(len(t) for t in texts):,} chars")
                elif kind == "status":
                    st = str(getattr(msg, "status", "") or "")
                    extra = str(getattr(msg, "message", "") or "").strip()
                    line = " ".join(p for p in (st, extra) if p)
                    if line:
                        heartbeat(f"Cursor status: {line}")
                elif kind == "task":
                    extra = str(getattr(msg, "text", "") or getattr(msg, "status", "") or "").strip()
                    if extra:
                        status(f"Cursor task: {extra[:120]}  ({elapsed():.0f}s)")
    finally:
        stop_wait_tick.set()

    wait = getattr(run, "wait", None)
    result = wait() if callable(wait) else None
    took = elapsed()
    run_status = getattr(result, "status", None) or getattr(run, "status", "") or "unknown"
    tools_txt = ", ".join(f"{k}×{v}" for k, v in tool_counts.items() if v) or "none"
    status(f"Cursor run finished in {took:.0f}s  status={run_status}  tools={tools_txt}")
    reply = "".join(texts).strip()
    if not reply:
        reply = (getattr(run, "result", None) or getattr(result, "result", None) or "") or ""
        reply = str(reply).strip()
    return reply, result


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "y"}


@dataclass
class DesignAgent:
    """Conversational CAD agent powered by Cursor SDK."""

    model: str = field(default_factory=lambda: os.getenv("DESIGN_MODEL", DEFAULT_MODEL))
    api_key: str | None = field(default_factory=lambda: os.getenv("CURSOR_API_KEY"))
    history: list[dict[str, str]] = field(default_factory=list)
    current_code: str | None = None
    last_requirements: str = ""
    features: list[DesignFeature] = field(default_factory=list)
    backend: str = field(default_factory=lambda: os.getenv("DESIGN_LLM", "auto"))
    # None → read DESIGN_FAST from env after dotenv loads
    fast: bool | None = None
    # draft = build/render only; refine = full constraint review
    design_mode: str | None = None
    worksheet: ContextWorksheet = field(default_factory=ContextWorksheet)
    references: list[StepReference] = field(default_factory=list)
    _cursor_agent: object | None = field(default=None, init=False, repr=False)
    _renderer: MeshRenderer = field(default_factory=MeshRenderer, init=False, repr=False)
    _vertices: np.ndarray | None = field(default=None, init=False, repr=False)
    _faces: np.ndarray | None = field(default=None, init=False, repr=False)
    # Optional UI-hosted renderer (single OpenGL context) + main-thread marshal
    _host_render: object | None = field(default=None, init=False, repr=False)
    _ui_marshal: object | None = field(default=None, init=False, repr=False)
    _status_hook: Callable[[str], None] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        load_dotenv(ROOT / ".env")
        self.api_key = os.getenv("CURSOR_API_KEY") or self.api_key
        self.model = resolve_model(os.getenv("DESIGN_MODEL", self.model))
        self.backend = os.getenv("DESIGN_LLM", self.backend)
        if self.fast is None:
            self.fast = _env_flag("DESIGN_FAST", False)
        if self.design_mode is None:
            raw = (os.getenv("DESIGN_MODE") or MODE_DRAFT).strip().lower()
            self.design_mode = MODE_REFINE if raw in {"refine", "refinement", "review"} else MODE_DRAFT
        elif self.design_mode in {"refinement", "review"}:
            self.design_mode = MODE_REFINE
        elif self.design_mode in {"drafting", "draft"}:
            self.design_mode = MODE_DRAFT
        if self.backend == "auto":
            self.backend = "cursor" if self.api_key else "mock"
        if self.backend == "openai":
            self.backend = "cursor"
        # Restore durable worksheet if present from a prior crash
        if WORKSHEET_FILE.exists():
            try:
                self.worksheet = ContextWorksheet.load(WORKSHEET_FILE)
                if self.worksheet.requirements and not self.last_requirements:
                    self.last_requirements = self.worksheet.requirements
            except Exception:
                self.worksheet = ContextWorksheet()
        if FEATURES_FILE.exists() and not self.features:
            try:
                import json

                raw = json.loads(FEATURES_FILE.read_text(encoding="utf-8"))
                self.features = [
                    DesignFeature(
                        id=str(item.get("id", f"F{i}")),
                        text=str(item.get("text", "")),
                        status=str(item.get("status", "pending")),
                        evidence=str(item.get("evidence", "")),
                    )
                    for i, item in enumerate(raw, start=1)
                    if item.get("text")
                ]
            except Exception:
                pass

    def _model_selection(self):
        """Build Cursor ModelSelection, optionally with fast mode enabled."""
        from cursor_sdk import ModelParameterValue, ModelSelection

        if self.fast:
            return ModelSelection(
                id=self.model,
                params=[ModelParameterValue(id="fast", value="true")],
            )
        return self.model

    def set_status_hook(self, hook: Callable[[str], None] | None) -> None:
        """UI callback for live process traces (chat log + status bar)."""
        self._status_hook = hook

    def _trace(self, msg: str) -> None:
        hook = self._status_hook
        if hook:
            hook(msg)

    def _heartbeat(self, msg: str) -> None:
        hook = self._status_hook
        if hook:
            hook(_HEARTBEAT_PREFIX + msg)

    def references_block(self, *, compact: bool = False) -> str:
        return format_references_block(self.references, compact=compact)

    def _prompt_references(self, *, compact: bool = False) -> str:
        block = self.references_block(compact=compact)
        return f"{block}\n" if block else ""

    def add_step_reference(self, path: Path) -> StepReference:
        """Import a STEP file, extract facts, and attach it as design context."""
        existing = {ref.name for ref in self.references}
        # Reuse the same name if this exact source was already loaded
        for ref in self.references:
            if ref.source_path.resolve() == Path(path).expanduser().resolve():
                existing.discard(ref.name)
                break
        new_ref = import_step_file(Path(path), existing_names=existing)
        self.references = [r for r in self.references if r.name != new_ref.name]
        self.references.append(new_ref)
        # Drop the live Cursor agent so the next turn reloads .cursorignore
        # and does not keep a previously indexed STEP binary in context.
        if self._cursor_agent is not None:
            self._close_cursor_agent()
        try:
            self.update_worksheet(
                notes=f"Imported STEP reference `{new_ref.name}` from {new_ref.source_path.name}."
            )
        except Exception:
            pass
        return new_ref

    # --- context worksheet --------------------------------------------------

    def update_worksheet(
        self,
        *,
        phase: str | None = None,
        task_instruction: str | None = None,
        pending_prompt: str | None = None,
        last_review: str | None = None,
        geometry: str | None = None,
        notes: str | None = None,
        last_error: str | None = None,
    ) -> ContextWorksheet:
        """Refresh and persist the durable context worksheet."""
        ws = self.worksheet
        ws.model = self.model
        ws.requirements = self.last_requirements or ws.requirements
        ws.features_text = format_feature_status(self.features) if self.features else ws.features_text
        ws.references_text = self.references_block()
        if phase is not None:
            ws.phase = phase
        if task_instruction is not None:
            ws.task_instruction = task_instruction
        if pending_prompt is not None:
            ws.pending_prompt = truncate(pending_prompt, 14000)
        if last_review is not None:
            ws.last_review = truncate(last_review, 4000)
        if geometry is not None:
            ws.geometry = geometry
        if notes is not None:
            ws.notes = notes
        if last_error is not None:
            ws.last_error = truncate(last_error, 1000)
        code = self.current_code
        if not code and DESIGN_FILE.exists():
            code = DESIGN_FILE.read_text(encoding="utf-8")
        if code:
            ws.code_snapshot = code
        ws.conversation_digest = compress_history(self.history)
        if not ws.task_summary and ws.task_instruction:
            ws.task_summary = (
                f"Phase {ws.phase}: {truncate(ws.task_instruction, 300)}. "
                f"Requirements: {truncate(ws.requirements, 400)}."
            )
        ws.save(WORKSHEET_FILE)
        return ws

    def is_draft_mode(self) -> bool:
        return (self.design_mode or MODE_DRAFT) == MODE_DRAFT

    def is_refine_mode(self) -> bool:
        return not self.is_draft_mode()

    def set_design_mode(self, mode: str) -> str:
        """Set draft|refine and persist a worksheet note."""
        raw = (mode or "").strip().lower()
        if raw in {"refine", "refinement", "review"}:
            self.design_mode = MODE_REFINE
        else:
            self.design_mode = MODE_DRAFT
        try:
            self.update_worksheet(
                notes=f"Design mode set to {self.design_mode}.",
            )
        except Exception:
            pass
        return self.design_mode

    def mode_label(self) -> str:
        return "Drafting" if self.is_draft_mode() else "Refinement"

    def worksheet_summary(self) -> str:
        ws = self.worksheet
        return (
            f"Worksheet phase={ws.phase} recoveries={ws.recovery_count} "
            f"file={WORKSHEET_FILE.relative_to(ROOT)}"
        )

    # --- version history ----------------------------------------------------

    def version_store(self) -> DesignVersionStore:
        return DesignVersionStore(ROOT)

    def commit_version(
        self,
        result: DesignResult,
        *,
        label: str = "",
        note: str = "",
        chat_log: str = "",
    ) -> VersionMeta:
        """Snapshot design + agent conversation state for later rollback."""
        store = self.version_store()
        worksheet_md = ""
        if WORKSHEET_FILE.exists():
            worksheet_md = WORKSHEET_FILE.read_text(encoding="utf-8")
        # Keep on-disk design file aligned
        DESIGN_FILE.parent.mkdir(parents=True, exist_ok=True)
        DESIGN_FILE.write_text(result.code.rstrip() + "\n", encoding="utf-8")
        self.current_code = result.code
        meta = store.commit(
            code=result.code,
            history=list(self.history),
            features=list(self.features),
            requirements=self.last_requirements,
            design_mode=self.design_mode or MODE_DRAFT,
            vertices=result.vertices,
            faces=result.faces,
            chat_log=chat_log,
            worksheet_md=worksheet_md,
            label=label,
            note=note,
            parent_id=store.current_id(),
        )
        self.history.append(
            {"role": "assistant", "content": f"[version] committed {meta.id}: {meta.label}"}
        )
        return meta

    def list_versions(self) -> list[VersionMeta]:
        return self.version_store().list_versions()

    def restore_version(
        self,
        version_id: str,
        *,
        truncate_newer: bool = True,
    ) -> tuple[DesignResult, VersionSnapshot]:
        """
        Roll back design, features, requirements, and agent history.

        Also closes the Cursor agent so the LLM conversation does not keep
        the discarded revisions.
        """
        store = self.version_store()
        snap = store.get(version_id)
        if snap is None:
            raise RuntimeError(f"Unknown version: {version_id}")

        # Always rebuild solid so STEP export still works after rollback
        rebuilt = run_design_code(snap.code)
        code = rebuilt.code
        vertices, faces = rebuilt.vertices, rebuilt.faces

        self.current_code = code
        DESIGN_FILE.parent.mkdir(parents=True, exist_ok=True)
        DESIGN_FILE.write_text(code.rstrip() + "\n", encoding="utf-8")
        self.history = list(snap.history)
        self.last_requirements = snap.meta.requirements or self.last_requirements
        self.features = [
            DesignFeature(
                id=str(f.get("id", f"F{i}")),
                text=str(f.get("text", "")),
                status=str(f.get("status", "pending")),
                evidence=str(f.get("evidence", "")),
            )
            for i, f in enumerate(snap.features, start=1)
            if f.get("text")
        ]
        self._save_features()
        if snap.meta.design_mode:
            self.set_design_mode(snap.meta.design_mode)
        if snap.worksheet_md:
            WORKSHEET_FILE.write_text(snap.worksheet_md.rstrip() + "\n", encoding="utf-8")
            try:
                self.worksheet = ContextWorksheet.load(WORKSHEET_FILE)
            except Exception:
                pass
        self.set_mesh(vertices, faces)
        # Drop Cursor conversation so recovery starts clean from worksheet/design
        self._close_cursor_agent()
        if truncate_newer:
            store.delete_after(version_id)
        else:
            store.set_current(version_id)
        self.history.append(
            {
                "role": "assistant",
                "content": f"[version] restored {snap.meta.id}: {snap.meta.label}",
            }
        )
        self.update_worksheet(
            phase="restored",
            task_instruction=f"Rolled back to {snap.meta.id}",
            notes=f"Restored version {snap.meta.id}",
        )
        return rebuilt, snap

    # --- mesh / renders -----------------------------------------------------

    def set_mesh(self, vertices: np.ndarray, faces: np.ndarray) -> None:
        """Update the mesh used for RGB previews and render_cad_view."""
        self._vertices = np.asarray(vertices, dtype=np.float64)
        self._faces = np.asarray(faces, dtype=np.int32)

    def features_summary(self) -> str:
        """Human-readable feature checklist for UI / status."""
        if not self.features:
            return "Key features: (none yet)"
        return "Key features:\n" + format_feature_status(self.features)

    def sync_features(self, requirement: str, *, replace: bool) -> list[DesignFeature]:
        """
        Build or update the key-feature list from a user requirement.

        replace=True  → initial brief (replace list)
        replace=False → revision / added detail (merge into existing)

        Drafting mode always uses local heuristics (no extra Cursor round-trip).
        """
        text = (requirement or "").strip()
        if not text:
            return list(self.features)

        use_heuristic = self.backend == "mock" or self.is_draft_mode()
        if use_heuristic:
            incoming = heuristic_features_from_prompt(text)
            if replace or not self.features:
                self.features = incoming
            else:
                self.features = merge_feature_lists(self.features, incoming)
            self._save_features()
            self.update_worksheet(
                phase="features",
                task_instruction=("Replace" if replace else "Update") + " key feature list",
            )
            return list(self.features)

        self.update_worksheet(
            phase="features_extract" if (replace or not self.features) else "features_update",
            task_instruction=text[:500],
        )

        if replace or not self.features:
            prompt = (
                f"{FEATURES_EXTRACT_BRIEF}\n\n"
                f"Design brief:\n{text}\n"
            )
            raw = self._send_cursor(prompt)
            parsed = parse_features_block(raw)
            self.features = parsed or heuristic_features_from_prompt(text)
        else:
            prompt = (
                f"{FEATURES_UPDATE_BRIEF}\n\n"
                f"Existing features:\n{format_feature_list(self.features)}\n\n"
                f"New user requirement / clarification:\n{text}\n"
            )
            # Snapshot — feature extraction is read-only
            snapshot = DESIGN_FILE.read_text(encoding="utf-8") if DESIGN_FILE.exists() else None
            try:
                raw = self._send_cursor(prompt)
            finally:
                if snapshot is None:
                    if DESIGN_FILE.exists():
                        DESIGN_FILE.unlink()
                elif DESIGN_FILE.read_text(encoding="utf-8") != snapshot:
                    DESIGN_FILE.write_text(snapshot, encoding="utf-8")
            parsed = parse_features_block(raw)
            if parsed:
                self.features = parsed
            else:
                self.features = merge_feature_lists(
                    self.features, heuristic_features_from_prompt(text)
                )
        # Reset status after any requirement change
        for f in self.features:
            f.status = "pending"
            f.evidence = ""
        self._save_features()
        return list(self.features)

    def _save_features(self) -> None:
        import json

        GENERATED_DIR.mkdir(parents=True, exist_ok=True)
        payload = [
            {"id": f.id, "text": f.text, "status": f.status, "evidence": f.evidence}
            for f in self.features
        ]
        FEATURES_FILE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def set_host_renderer(self, render_fn) -> None:
        """Use the studio preview's OpenGL context for all agent renders."""
        self._host_render = render_fn
        # Drop the secondary VTK window to avoid VAO/context clashes
        try:
            self._renderer.close()
        except Exception:
            pass

    def set_ui_marshal(self, marshal_fn) -> None:
        """Schedule callables on the Tk UI thread (required for OpenGL)."""
        self._ui_marshal = marshal_fn

    def _call_on_ui(self, fn):
        """Run fn on the UI thread if a marshal is configured; else run inline."""
        import threading

        if self._ui_marshal is None or threading.current_thread() is threading.main_thread():
            return fn()
        holder: dict = {}
        done = threading.Event()

        def job() -> None:
            try:
                holder["value"] = fn()
            except Exception as exc:  # noqa: BLE001
                holder["error"] = exc
            finally:
                done.set()

        try:
            self._ui_marshal(job)
        except Exception as exc:  # noqa: BLE001 — e.g. destroyed Tk window
            raise RuntimeError(f"Failed to schedule UI-thread VTK work: {exc}") from exc
        # Multi-view review can take a while on HiDPI; allow headroom
        if not done.wait(timeout=120):
            raise TimeoutError("Timed out waiting for UI-thread VTK render")
        if "error" in holder:
            raise holder["error"]
        return holder.get("value")

    def render_views(
        self,
        views: Sequence[str] | None = None,
        *,
        vertices: np.ndarray | None = None,
        faces: np.ndarray | None = None,
    ) -> list[Path]:
        selected = list(views) if views is not None else list(DEFAULT_VIEWS)

        def _do() -> list[Path]:
            if self._host_render is not None:
                return [self._host_render(view=v).path for v in selected]
            verts = vertices if vertices is not None else self._vertices
            tris = faces if faces is not None else self._faces
            if verts is None or tris is None:
                return []
            return [r.path for r in self._renderer.render_views(verts, tris, selected)]

        return self._call_on_ui(_do)

    def _render_tool(self, args: dict, _context) -> dict:
        view = args.get("view") or "custom"
        self._trace(f"Tool render_cad_view · {view} (waiting on UI-thread VTK)…")
        if self._vertices is None or self._faces is None:
            return {
                "content": [{"type": "text", "text": "No mesh loaded yet; cannot render."}],
                "isError": True,
            }
        try:

            def _do():
                if self._host_render is not None:
                    return self._host_render(
                        view=args.get("view"),
                        elev=args.get("elev"),
                        azim=args.get("azim"),
                    )
                return self._renderer.render(
                    self._vertices,
                    self._faces,
                    view=args.get("view"),
                    elev=args.get("elev"),
                    azim=args.get("azim"),
                )

            result = self._call_on_ui(_do)
        except Exception as exc:  # noqa: BLE001
            self._trace(f"Tool render_cad_view failed: {exc}")
            return {
                "content": [{"type": "text", "text": f"Render failed: {exc}"}],
                "isError": True,
            }

        rel = result.path.relative_to(ROOT) if result.path.is_relative_to(ROOT) else result.path
        self._trace(f"Tool render_cad_view done · {rel.name}")
        data_b64 = base64.b64encode(result.path.read_bytes()).decode("ascii")
        return {
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"Rendered view={result.view} elev={result.elev:g} azim={result.azim:g} "
                        f"→ {rel}"
                    ),
                },
                {
                    "type": "image",
                    "mimeType": "image/png",
                    "data": data_b64,
                },
            ]
        }

    # --- Cursor SDK ---------------------------------------------------------

    def _ensure_cursor_agent(self) -> None:
        if self._cursor_agent is not None:
            return
        if not self.api_key:
            raise RuntimeError(
                "CURSOR_API_KEY is required for Cursor SDK designs.\n"
                "Create one at https://cursor.com/dashboard/integrations\n"
                "Then: export CURSOR_API_KEY=...   or put it in .env\n"
                "Or run offline: DESIGN_LLM=mock python start_design.py"
            )
        from cursor_sdk import Agent, AgentOptions, CustomTool, LocalAgentOptions

        GENERATED_DIR.mkdir(parents=True, exist_ok=True)
        RENDER_DIR.mkdir(parents=True, exist_ok=True)
        self._trace(f"Cursor · creating local agent  model={self.model}  fast={'on' if self.fast else 'off'}")

        self._cursor_agent = Agent.create(
            AgentOptions(
                # No shell: follow-ups used to `cat` imported STEP binaries and hang.
                tools=["edit", "read"],
                disallowed_tools=["shell"],
            ),
            model=self._model_selection(),
            api_key=self.api_key,
            local=LocalAgentOptions(
                cwd=str(ROOT),
                custom_tools={
                    "render_cad_view": CustomTool(
                        description=(
                            "Render an RGB preview of the current CAD solid from a camera angle. "
                            f"Named views: {', '.join(list_views())}. "
                            "Or pass elev and azim in degrees (Z-up; front≈elev 8 azim -90)."
                        ),
                        input_schema={
                            "type": "object",
                            "properties": {
                                "view": {
                                    "type": "string",
                                    "description": "Named camera preset",
                                    "enum": list_views(),
                                },
                                "elev": {
                                    "type": "number",
                                    "description": "Elevation degrees (optional with azim)",
                                },
                                "azim": {
                                    "type": "number",
                                    "description": "Azimuth degrees (optional with elev)",
                                },
                            },
                        },
                        execute=self._render_tool,
                    ),
                },
            ),
        )

    def _close_cursor_agent(self) -> None:
        """Dispose the Cursor agent only (keep mesh / worksheet / renderer)."""
        agent = self._cursor_agent
        self._cursor_agent = None
        if agent is None:
            return
        close = getattr(agent, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
        exit_fn = getattr(agent, "__exit__", None)
        if callable(exit_fn):
            try:
                exit_fn(None, None, None)
            except Exception:
                pass

    def _recover_cursor_agent(
        self,
        error: BaseException | str,
        *,
        context_overflow: bool,
        on_status: Callable[[str], None] | None = None,
    ) -> None:
        """Persist worksheet, summarize if needed, and launch a fresh Cursor agent."""
        status = on_status or (lambda _s: None)
        ws = self.worksheet
        ws.recovery_count = int(ws.recovery_count or 0) + 1
        ws.last_error = truncate(str(error), 1000)
        if context_overflow:
            status("Context window issue — summarizing worksheet…")
            ws.apply_context_overflow_summary(self.history)
        else:
            status("Agent crashed — refreshing context worksheet…")
            ws.conversation_digest = compress_history(self.history)
            if self.current_code:
                ws.code_snapshot = self.current_code
        ws.notes = (
            (ws.notes + "\n" if ws.notes else "")
            + f"Recovery #{ws.recovery_count}: relaunching Cursor agent after: "
            + truncate(str(error), 240)
        ).strip()
        ws.save(WORKSHEET_FILE)
        self._close_cursor_agent()
        if self.backend == "mock":
            status(f"Worksheet updated for recovery #{ws.recovery_count} (mock backend).")
            return
        status(f"Launching new agent (recovery #{ws.recovery_count})…")
        self._ensure_cursor_agent()

    def _build_resume_prompt(self, original_prompt: str, *, context_overflow: bool) -> str:
        ws = self.worksheet
        body = original_prompt
        if context_overflow:
            # Prefer compressed resume payload over the huge original prompt
            body = (
                f"Active phase: {ws.phase}\n"
                f"Instruction: {ws.task_instruction or '(see worksheet)'}\n\n"
                f"Task summary:\n{ws.task_summary or '(see worksheet)'}\n\n"
                f"Key features:\n{ws.features_text or '(see worksheet)'}\n\n"
                f"Continue the pending work. Prefer generated/current_design.py as the design "
                f"source of truth. Pending detail:\n{ws.pending_prompt or truncate(original_prompt, 3000)}"
            )
        return (
            f"{CONTEXT_RESUME_BRIEF}\n\n"
            f"Worksheet file: {WORKSHEET_FILE.relative_to(ROOT)}\n\n"
            f"{body}"
        )

    def _send_cursor(
        self,
        prompt: str,
        images: Sequence[Path] | None = None,
        *,
        on_status: Callable[[str], None] | None = None,
        force: bool = False,
        allow_recovery: bool = True,
    ) -> str:
        from cursor_sdk import (
            CursorAgentError,
            LocalSendOptions,
            SDKImage,
            SendOptions,
            UserMessage,
        )

        max_recoveries = max(1, int(os.getenv("DESIGN_AGENT_RECOVERIES", "3")))
        recoveries = 0
        use_images = list(images or [])
        use_force = force
        active_prompt = prompt
        resumed = False
        status = on_status or self._status_hook or (lambda _s: None)

        while True:
            self._ensure_cursor_agent()
            assert self._cursor_agent is not None

            # Keep worksheet current before each attempt
            self.update_worksheet(pending_prompt=active_prompt)

            image_paths = [p for p in use_images if p and Path(p).exists()]
            captions = []
            for p in image_paths:
                captions.append(f"- {p.name}: RGB preview ({p})")
            text = active_prompt
            if captions:
                text += "\n\nAttached RGB renders (inspect carefully):\n" + "\n".join(captions)

            message: str | UserMessage
            if image_paths:
                message = UserMessage(
                    text=text,
                    images=[SDKImage.from_file(str(p)) for p in image_paths],
                )
            else:
                message = text

            send_opts = SendOptions(
                local=LocalSendOptions(force=True) if use_force else None,
                # enableDeltas=true — without this, follow-ups sit on RUNNING
                # with no thinking/tool events until the whole turn finishes.
                on_delta=lambda _update: None,
            )

            try:
                self._trace(
                    f"Cursor send · {len(image_paths)} image(s), "
                    f"{len(text):,} prompt chars"
                    + (" (force)" if use_force else "")
                )
                run = self._cursor_agent.send(message, send_opts)
                reply, result = _consume_cursor_run(run, status)
            except CursorAgentError as err:
                context_hit = looks_like_context_error(err)
                recoverable = allow_recovery and looks_like_recoverable_cursor_error(err)
                if recoverable and recoveries < max_recoveries:
                    recoveries += 1
                    status(
                        f"Cursor API error ({type(err).__name__}); "
                        f"recovering via worksheet ({recoveries}/{max_recoveries})…"
                    )
                    self._recover_cursor_agent(err, context_overflow=context_hit, on_status=status)
                    active_prompt = self._build_resume_prompt(prompt, context_overflow=context_hit)
                    resumed = True
                    use_force = True
                    if context_hit:
                        use_images = []  # drop heavy RGB payload after overflow
                    continue
                raise RuntimeError(
                    f"Cursor agent failed: {err}\n"
                    f"retryable={getattr(err, 'is_retryable', None)} "
                    f"type={type(err).__name__}\n"
                    f"Worksheet: {WORKSHEET_FILE}"
                ) from err

            if getattr(result, "status", None) == "error":
                err_obj = getattr(result, "error", None)
                err_text = (
                    str(err_obj)
                    if err_obj is not None
                    else f"run status=error id={getattr(result, 'id', '?')}"
                )
                context_hit = looks_like_context_error(err_text)
                if allow_recovery and recoveries < max_recoveries:
                    recoveries += 1
                    status(
                        f"Cursor run failed; recovering via worksheet "
                        f"({recoveries}/{max_recoveries})…"
                    )
                    self._recover_cursor_agent(err_text, context_overflow=context_hit, on_status=status)
                    active_prompt = self._build_resume_prompt(prompt, context_overflow=context_hit)
                    resumed = True
                    use_force = True
                    if context_hit:
                        use_images = []
                    continue
                raise RuntimeError(
                    f"Cursor agent run failed: {err_text}\nWorksheet: {WORKSHEET_FILE}"
                )

            if resumed:
                status("Recovered agent completed the task from the context worksheet.")
            return (reply or "").strip()

    def _run_cursor(self, prompt: str, images: Sequence[Path] | None = None) -> str:
        # Snapshot code into worksheet before unlink so recovery still has source
        if self.current_code:
            self.worksheet.code_snapshot = self.current_code
            self.update_worksheet(pending_prompt=prompt)
        elif DESIGN_FILE.exists():
            self.worksheet.code_snapshot = DESIGN_FILE.read_text(encoding="utf-8")
            self.update_worksheet(pending_prompt=prompt)
        if DESIGN_FILE.exists():
            DESIGN_FILE.unlink()
        self._trace("Cursor · waiting for agent to write generated/current_design.py")
        text = self._send_cursor(prompt, images=images)
        self._trace("Cursor · loading design source")
        return self._load_design_code(assistant_text=text)

    def _load_design_code(self, assistant_text: str) -> str:
        if DESIGN_FILE.exists():
            code = DESIGN_FILE.read_text(encoding="utf-8").strip()
            if "def build" in code:
                return code
        # After crash recovery the agent may have restored from worksheet snapshot
        extracted = _extract_code(assistant_text)
        if "def build" in extracted:
            DESIGN_FILE.parent.mkdir(parents=True, exist_ok=True)
            DESIGN_FILE.write_text(extracted.rstrip() + "\n", encoding="utf-8")
            return extracted
        snap = (self.worksheet.code_snapshot or "").strip()
        if "def build" in snap:
            DESIGN_FILE.parent.mkdir(parents=True, exist_ok=True)
            DESIGN_FILE.write_text(snap.rstrip() + "\n", encoding="utf-8")
            return snap
        raise RuntimeError(
            "Cursor agent did not produce generated/current_design.py with build().\n"
            f"Assistant reply was:\n{assistant_text[:1200]}\n"
            f"See also {WORKSHEET_FILE}"
        )

    def close(self) -> None:
        try:
            self.update_worksheet(phase="closed", notes="Agent session closed.")
        except Exception:
            pass
        try:
            self._renderer.close()
        except Exception:
            pass
        self._close_cursor_agent()

    # --- mock backend -------------------------------------------------------

    def _mock_code(self, prompt: str, revision: bool) -> str:
        text = prompt.lower()
        if not revision:
            if "table" in text:
                return MOCK_TABLE
            if "box" in text or "block" in text or "cube" in text:
                return MOCK_BOX
            return MOCK_CHAIR

        code = self.current_code or MOCK_CHAIR
        if "taller" in text:
            return _scale_assignments(code, ("SEAT_H", "TOP_H", "BACK_H"), 1.15)
        if "wider" in text:
            return _scale_assignments(code, ("SEAT_W", "TOP_W"), 1.12)
        if "thicker" in text:
            return _scale_assignments(code, ("SEAT_T", "TOP_T", "BACK_T"), 1.25)
        if "shorter" in text:
            return _scale_assignments(code, ("SEAT_H", "TOP_H", "BACK_H"), 0.9)
        if "narrower" in text:
            return _scale_assignments(code, ("SEAT_W", "TOP_W"), 0.9)
        return code

    # --- public API ---------------------------------------------------------

    def generate(self, prompt: str) -> str:
        """Create an initial design from a natural-language brief."""
        self._trace("Generate · starting initial design")
        self.history = [{"role": "user", "content": prompt}]
        req = prompt.strip()
        if self.references:
            names = ", ".join(f"`{r.name}`" for r in self.references)
            req = f"{req}\n\nImported STEP constraints: {names}"
        self.last_requirements = req
        self.worksheet = ContextWorksheet(
            model=self.model,
            requirements=req,
            references_text=self.references_block(),
        )
        self.update_worksheet(phase="generate", task_instruction=prompt.strip())
        self.sync_features(req, replace=True)
        feature_block = format_feature_list(self.features)
        self.update_worksheet(phase="generate", task_instruction=prompt.strip())
        if self.backend == "mock":
            self._trace("Generate · mock backend")
            code = self._mock_code(prompt, revision=False)
            GENERATED_DIR.mkdir(parents=True, exist_ok=True)
            DESIGN_FILE.write_text(code.rstrip() + "\n", encoding="utf-8")
        else:
            self._trace("Generate · sending brief to Cursor")
            code = self._run_cursor(
                f"{SYSTEM_BRIEF}\n\n"
                f"{self._prompt_references()}"
                f"Key features to satisfy:\n{feature_block}\n\n"
                f"Create a NEW CadQuery design for this brief:\n{prompt}\n\n"
                f"Write it to {DESIGN_FILE.relative_to(ROOT)} now."
            )
        self.current_code = code
        self.history.append({"role": "assistant", "content": code})
        self.history.append(
            {"role": "assistant", "content": f"[features]\n{format_feature_list(self.features)}"}
        )
        self.update_worksheet(phase="generate_done", task_instruction=prompt.strip())
        return code

    def revise(
        self,
        instruction: str,
        *,
        images: Sequence[Path] | None = None,
        views: Sequence[str] | None = None,
        scope: str | None = None,
    ) -> str:
        """Revise the current design; optional RGB previews / part scope."""
        from cad_pipeline.runtime import WHOLE_DESIGN

        if not self.current_code:
            return self.generate(instruction)

        self._trace("Revise · follow-up instruction")

        scope_name = (scope or WHOLE_DESIGN).strip() or WHOLE_DESIGN
        scoped = scope_name not in {WHOLE_DESIGN, "", "whole", "assembly", "all"}

        # Accumulate requirements so later review still knows the original brief
        scope_note = f" [part: {scope_name}]" if scoped else ""
        if self.last_requirements:
            self.last_requirements = (
                f"{self.last_requirements.rstrip()}\n\n"
                f"Later revision{scope_note}: {instruction.strip()}"
            )
        else:
            self.last_requirements = instruction.strip()

        self.sync_features(instruction, replace=False)
        self._trace("Revise · feature checklist updated")
        feature_block = format_feature_list(self.features)
        self.update_worksheet(
            phase="revise",
            task_instruction=f"{instruction.strip()}{scope_note}",
        )

        render_paths = list(images or [])
        if not render_paths and self._vertices is not None:
            render_paths = self.render_views(views or DEFAULT_VIEWS)

        self.history.append(
            {
                "role": "user",
                "content": f"{instruction}" + (f"\n(scope: part `{scope_name}`)" if scoped else ""),
            }
        )
        self.history.append(
            {"role": "assistant", "content": f"[features]\n{feature_block}"}
        )
        if self.backend == "mock":
            self._trace("Revise · mock backend")
            code = self._mock_code(instruction, revision=True)
            DESIGN_FILE.write_text(code.rstrip() + "\n", encoding="utf-8")
            if self._vertices is not None:
                self.render_views(views or DEFAULT_VIEWS)
        else:
            if scoped:
                scope_block = (
                    f"EDIT SCOPE: only modify the part named `{scope_name}` inside parts().\n"
                    f"Keep every other part's name and geometry unless a tiny interface tweak "
                    f"is required for the change. Preserve parts() keys.\n"
                )
            else:
                scope_block = (
                    "EDIT SCOPE: whole design — you may update any parts, but keep a coherent "
                    "parts() dict and build() assembly.\n"
                )
            self._trace("Revise · sending instruction to Cursor (follow-up on same agent)")
            code = self._run_cursor(
                f"{SYSTEM_BRIEF}\n\n"
                f"Revise the existing design in {DESIGN_FILE.relative_to(ROOT)}.\n"
                f"{scope_block}\n"
                f"{self._prompt_references(compact=True)}"
                f"Key features that must remain satisfied (updated):\n{feature_block}\n\n"
                f"Current source for reference:\n```python\n{self.current_code}\n```\n\n"
                f"If RGB renders are attached, use them for proportions. Do not open "
                f".step files. Then apply this revision and overwrite the file with the "
                f"full updated parts()/build() source:\n{instruction}",
                images=render_paths,
            )
        self.current_code = code
        self.history.append({"role": "assistant", "content": code})
        self.update_worksheet(phase="revise_done", task_instruction=instruction.strip())
        return code

    def build_with_repair(
        self,
        code: str,
        *,
        images: Sequence[Path] | None = None,
        max_attempts: int | None = None,
        on_status: Callable[[str], None] | None = None,
    ) -> DesignResult:
        """
        Execute CadQuery code; on failure, feed the error back to the agent and retry.

        Continues until build succeeds or max_attempts is exhausted.
        """
        attempts = max_attempts
        if attempts is None:
            attempts = int(os.getenv("DESIGN_DEBUG_RETRIES", "5"))
        attempts = max(1, attempts)
        status = on_status or (lambda _s: None)

        last_error = ""
        current = code
        for attempt in range(1, attempts + 1):
            status(f"Building geometry (attempt {attempt}/{attempts})…")
            try:
                result = run_design_code(current)
                self.current_code = result.code
                DESIGN_FILE.parent.mkdir(parents=True, exist_ok=True)
                DESIGN_FILE.write_text(result.code.rstrip() + "\n", encoding="utf-8")
                if attempt > 1:
                    status(f"Fixed after {attempt} attempt(s).")
                return result
            except Exception as exc:  # noqa: BLE001 — fed back to the agent
                last_error = str(exc)
                status(f"Build failed (attempt {attempt}/{attempts}): {last_error.splitlines()[0]}")
                if attempt >= attempts:
                    break

                if self.backend == "mock":
                    # Offline: fall back to a known-good chair and continue
                    current = self._mock_code("chair", revision=False)
                    DESIGN_FILE.write_text(current.rstrip() + "\n", encoding="utf-8")
                    self.current_code = current
                    continue

                status(f"Asking agent to debug (attempt {attempt}/{attempts})…")
                current = self._repair_code(current, last_error, images=images)
                self.current_code = current
                self.history.append(
                    {
                        "role": "user",
                        "content": f"[debug attempt {attempt}] {last_error[:500]}",
                    }
                )
                self.history.append({"role": "assistant", "content": current})

        raise RuntimeError(
            f"Design still failing after {attempts} attempt(s).\n\nLast error:\n{last_error}"
        )

    def _repair_code(
        self,
        broken_code: str,
        error: str,
        *,
        images: Sequence[Path] | None = None,
    ) -> str:
        # Truncate huge traces for the prompt
        err = error if len(error) < 4000 else error[:3500] + "\n…(truncated)…"
        self.update_worksheet(
            phase="debug",
            task_instruction=f"Fix CadQuery build error: {err.splitlines()[0][:200]}",
        )
        self._trace("Debug · sending CadQuery error back to Cursor")
        prompt = (
            f"{DEBUG_BRIEF}\n\n"
            f"{self._prompt_references(compact=True)}"
            f"Broken source:\n```python\n{broken_code}\n```\n\n"
            f"Error traceback:\n```\n{err}\n```\n\n"
            f"Overwrite {DESIGN_FILE.relative_to(ROOT)} with the corrected full build() source."
        )
        return self._run_cursor(prompt, images=images)

    def finalize_design(
        self,
        code: str,
        *,
        requirements: str | None = None,
        images: Sequence[Path] | None = None,
        max_build_attempts: int | None = None,
        max_review_rounds: int | None = None,
        on_status: Callable[[str], None] | None = None,
    ) -> DesignResult:
        """
        Build (with compile/runtime repair). In drafting mode, stop once the solid
        tessellates. In refinement mode, inspect source + multi-view renders and
        enforce the feature checklist / physics until PASS or review budget ends.
        """
        status = on_status or (lambda _s: None)
        self._status_hook = status
        req = (requirements or self.last_requirements or "").strip()
        if req:
            self.last_requirements = req

        rounds = max_review_rounds
        if rounds is None:
            rounds = int(os.getenv("DESIGN_REVIEW_ROUNDS", "3"))
        rounds = max(1, rounds)

        mode = self.design_mode or MODE_DRAFT
        self.update_worksheet(
            phase="finalize_build",
            task_instruction=req or "Finalize current design",
            notes=f"finalize in {mode} mode",
        )

        try:
            result = self.build_with_repair(
                code,
                images=images,
                max_attempts=max_build_attempts,
                on_status=on_status,
            )
            # Prove renderability with a quick mesh bind (tessellation already done)
            self.set_mesh(result.vertices, result.faces)

            if self.is_draft_mode():
                status("Drafting mode — build OK, skipping constraint review.")
                if not self.features and req:
                    self.sync_features(req, replace=True)
                self.update_worksheet(phase="draft_ready", task_instruction=req)
                status("Draft ready (compilable + renderable).")
                return result

            if not self.features and req:
                status("Extracting key features from requirements…")
                self.sync_features(req, replace=True)
            if self.features:
                status(self.features_summary())

            for round_i in range(1, rounds + 1):
                status(f"Design review (round {round_i}/{rounds})…")
                self.set_mesh(result.vertices, result.faces)
                view_paths: list[Path] = []
                try:
                    view_paths = self.render_views(REVIEW_VIEWS)
                except Exception as exc:  # noqa: BLE001 — continue review without RGB if GL fails
                    status(f"Multi-view render failed ({exc}); reviewing from source + geometry only.")
                    try:
                        self._renderer.close()
                    except Exception:
                        pass
                # Prefer freshly rendered review views; fall back to caller images
                review_images = view_paths or list(images or [])

                review = self._review_design(result, requirements=req, images=review_images)
                if review.feature_results:
                    self.features = review.feature_results
                    self._save_features()
                    status(self.features_summary())
                status(
                    "Review verdict: PASS"
                    if review.passed
                    else f"Review verdict: FAIL ({len(review.issues)} issue(s))"
                )
                if review.raw:
                    # Keep a short breadcrumb in history / UI
                    snippet = review.raw if len(review.raw) < 1200 else review.raw[:1200] + "…"
                    self.history.append({"role": "assistant", "content": f"[review]\n{snippet}"})
                    self.update_worksheet(last_review=snippet)

                if review.passed:
                    status("Design accepted after inspection.")
                    self.update_worksheet(phase="accepted", task_instruction=req)
                    return result

                if round_i >= rounds:
                    status("Review budget exhausted — returning best-effort design.")
                    self.update_worksheet(phase="review_budget_exhausted", task_instruction=req)
                    return result

                status(f"Refining from review feedback (round {round_i}/{rounds})…")
                self.update_worksheet(
                    phase="refine",
                    task_instruction=f"Refine unmet features / review actions (round {round_i})",
                )
                refined = self._refine_from_review(
                    result.code,
                    requirements=req,
                    review=review,
                    images=review_images,
                )
                result = self.build_with_repair(
                    refined,
                    images=review_images,
                    max_attempts=max_build_attempts,
                    on_status=on_status,
                )

            return result
        finally:
            self._status_hook = None

    def _review_design(
        self,
        result: DesignResult,
        *,
        requirements: str,
        images: Sequence[Path],
    ) -> ReviewOutcome:
        physics = heuristic_physics_notes(result)
        geo = geometry_summary(result.vertices, triangles=len(result.faces))
        feature_block = format_feature_list(self.features)
        self.update_worksheet(
            phase="review",
            task_instruction="Inspect design against key features + physics",
            geometry=geo + "\n" + "\n".join(physics),
        )

        if self.backend == "mock":
            # Offline: fail once if floating warning present, else pass all features
            hard = [n for n in physics if n.startswith("WARNING")]
            checked = [
                DesignFeature(
                    id=f.id,
                    text=f.text,
                    status="unmet" if hard else "met",
                    evidence=hard[0] if hard else "mock geometry checks ok",
                )
                for f in self.features
            ]
            if hard and not getattr(self, "_mock_review_failed_once", False):
                self._mock_review_failed_once = True
                raw_lines = ["FEATURES_CHECK:"]
                for f in checked:
                    raw_lines.append(f"- {f.id}: UNMET — {f.evidence}")
                raw_lines += [
                    "VERDICT: FAIL",
                    "ISSUES:",
                    *("- " + n for n in hard),
                    "ACTIONS:",
                    "- Fix floor contact / scale",
                ]
                return ReviewOutcome(
                    passed=False,
                    raw="\n".join(raw_lines),
                    issues=list(hard),
                    actions=["Fix floor contact / scale"],
                    feature_results=checked,
                )
            raw_lines = ["FEATURES_CHECK:"]
            for f in checked:
                raw_lines.append(f"- {f.id}: MET — {f.evidence}")
            raw_lines.append("VERDICT: PASS")
            return ReviewOutcome(
                passed=True,
                raw="\n".join(raw_lines),
                issues=[],
                actions=[],
                feature_results=checked,
            )

        prompt = (
            f"{REVIEW_BRIEF}\n\n"
            f"{self._prompt_references(compact=True)}"
            f"User requirements / brief:\n{requirements or '(none provided)'}\n\n"
            f"Key feature checklist (evaluate EVERY item):\n{feature_block}\n\n"
            f"Measured geometry:\n{geo}\n"
            + "\n".join(physics)
            + f"\n\nCadQuery source:\n```python\n{result.code}\n```\n\n"
            "Inspect the attached RGB renders from multiple perspectives, then output "
            "FEATURES_CHECK for each feature and VERDICT: PASS or FAIL with ISSUES/ACTIONS."
        )
        # Snapshot design file — review is read-only
        snapshot = DESIGN_FILE.read_text(encoding="utf-8") if DESIGN_FILE.exists() else None
        try:
            raw = self._send_cursor(prompt, images=images)
        finally:
            if snapshot is None:
                if DESIGN_FILE.exists():
                    DESIGN_FILE.unlink()
            elif DESIGN_FILE.read_text(encoding="utf-8") != snapshot:
                DESIGN_FILE.write_text(snapshot, encoding="utf-8")
        outcome = parse_review(raw, features=self.features)
        # If the model omitted VERDICT but physics has hard warnings, treat as fail
        if outcome.passed and any(n.startswith("WARNING") for n in physics):
            if "PASS" not in (raw or "").upper():
                outcome = ReviewOutcome(
                    passed=False,
                    raw=raw,
                    issues=outcome.issues or [n for n in physics if n.startswith("WARNING")],
                    actions=outcome.actions or ["Resolve the geometric warnings listed above."],
                    feature_results=outcome.feature_results,
                )
        return outcome

    def _refine_from_review(
        self,
        code: str,
        *,
        requirements: str,
        review: ReviewOutcome,
        images: Sequence[Path] | None,
    ) -> str:
        issues = "\n".join(f"- {i}" for i in review.issues) or "- (see review text)"
        actions = "\n".join(f"- {a}" for a in review.actions) or "- Address the listed issues"
        unmet = [f for f in (review.feature_results or self.features) if f.status == "unmet"]
        feature_block = format_feature_status(review.feature_results or self.features)
        unmet_block = (
            "\n".join(f"- {f.id}: {f.text}" + (f" ({f.evidence})" if f.evidence else "") for f in unmet)
            or "- (none marked unmet)"
        )
        if self.backend == "mock":
            fixed = self._mock_code(requirements or "chair", revision=False)
            DESIGN_FILE.write_text(fixed.rstrip() + "\n", encoding="utf-8")
            self.current_code = fixed
            return fixed

        prompt = (
            f"{REFINE_BRIEF}\n\n"
            f"{self._prompt_references(compact=True)}"
            f"User requirements / brief:\n{requirements or '(none)'}\n\n"
            f"Feature checklist status:\n{feature_block}\n\n"
            f"UNMET features to implement:\n{unmet_block}\n\n"
            f"Review ISSUES:\n{issues}\n\n"
            f"Required ACTIONS:\n{actions}\n\n"
            f"Full review text:\n{review.raw}\n\n"
            f"Current source:\n```python\n{code}\n```\n\n"
            f"Overwrite {DESIGN_FILE.relative_to(ROOT)} with the improved full build() source."
        )
        refined = self._run_cursor(prompt, images=images)
        self.current_code = refined
        self.history.append({"role": "user", "content": f"[refine] {actions}"})
        self.history.append({"role": "assistant", "content": refined})
        return refined

    def ask(
        self,
        question: str,
        *,
        geometry_summary: str = "",
        images: Sequence[Path] | None = None,
        views: Sequence[str] | None = None,
    ) -> str:
        """Answer a question about the current design without modifying it."""
        if not self.current_code:
            return "No design is loaded yet. Create a design first, then ask about it."

        self.update_worksheet(
            phase="ask",
            task_instruction=question.strip(),
            geometry=geometry_summary or None,
        )

        # Explicit images=[] means text-only. Do not auto-capture RGB on Ask —
        # follow-up image+STEP payloads stall the local Cursor agent.
        if images is not None:
            render_paths = list(images)
        elif views and self._vertices is not None:
            render_paths = self.render_views(views)
        else:
            render_paths = []

        snapshot = DESIGN_FILE.read_text(encoding="utf-8") if DESIGN_FILE.exists() else None

        if self.backend == "mock":
            self._trace("Ask · mock backend")
            answer = self._mock_ask(
                question,
                geometry_summary=geometry_summary,
                images=render_paths,
            )
        else:
            self._trace("Ask · sending question to Cursor (follow-up on same agent)")
            answer = self._ask_cursor(
                question,
                geometry_summary=geometry_summary,
                images=render_paths,
            )

        if snapshot is None:
            if DESIGN_FILE.exists():
                DESIGN_FILE.unlink()
        elif DESIGN_FILE.read_text(encoding="utf-8") != snapshot:
            DESIGN_FILE.write_text(snapshot, encoding="utf-8")

        self.history.append({"role": "user", "content": f"[ask] {question}"})
        self.history.append({"role": "assistant", "content": answer})
        return answer

    def _ask_cursor(
        self,
        question: str,
        *,
        geometry_summary: str,
        images: Sequence[Path] | None,
    ) -> str:
        prompt = (
            f"{ASK_BRIEF}\n\n"
            f"{self._prompt_references(compact=True)}"
            f"Current CadQuery source (read-only):\n```python\n{self.current_code}\n```\n\n"
        )
        if self.features:
            prompt += f"Key feature checklist:\n{format_feature_status(self.features)}\n\n"
        if geometry_summary:
            prompt += f"Measured geometry:\n{geometry_summary}\n\n"
        prompt += f"Question:\n{question}"
        return self._send_cursor(prompt, images=images)

    def _mock_ask(
        self,
        question: str,
        *,
        geometry_summary: str,
        images: Sequence[Path] | None,
    ) -> str:
        code = self.current_code or ""
        params: dict[str, float] = {}
        for names, values in re.findall(
            r"\b((?:[A-Z][A-Z0-9_]*\s*,\s*)*[A-Z][A-Z0-9_]*)\s*=\s*"
            r"((?:[0-9]+(?:\.[0-9]+)?\s*,\s*)*[0-9]+(?:\.[0-9]+)?)",
            code,
        ):
            keys = [k.strip() for k in names.split(",")]
            vals = [float(v.strip()) for v in values.split(",")]
            if len(keys) == len(vals):
                for k, v in zip(keys, vals):
                    params[k] = v

        q = question.lower()
        lines: list[str] = []
        if geometry_summary:
            lines.append(geometry_summary)

        # Mock "agent chooses camera": honor view keywords by rendering that angle
        requested = [v for v in list_views() if re.search(rf"\b{v}\b", q.replace("-", "_"))]
        if "from above" in q or "top view" in q:
            requested.append("top")
        if "from the side" in q or "side view" in q:
            requested.append("side")
        # unique preserve order
        seen: set[str] = set()
        requested = [v for v in requested if not (v in seen or seen.add(v))]
        if requested and self._vertices is not None:
            paths = self.render_views(requested)
            lines.append("Rendered for inspection: " + ", ".join(p.name for p in paths))
        elif images:
            lines.append("Using RGB renders: " + ", ".join(Path(p).name for p in images))

        def pick(*keys: str) -> list[str]:
            return [f"{k} = {params[k]:g} mm" for k in keys if k in params]

        if any(w in q for w in ("height", "tall", "how high")):
            bits = pick("SEAT_H", "TOP_H", "BACK_H")
            lines.append("Height-related parameters: " + (", ".join(bits) if bits else "see bbox Z"))
        if any(w in q for w in ("width", "wide", "how wide")):
            bits = pick("SEAT_W", "TOP_W")
            lines.append("Width-related parameters: " + (", ".join(bits) if bits else "see bbox X"))
        if any(w in q for w in ("depth", "deep", "how deep")):
            bits = pick("SEAT_D", "TOP_D")
            lines.append("Depth-related parameters: " + (", ".join(bits) if bits else "see bbox Y"))
        if "seat" in q:
            bits = pick("SEAT_W", "SEAT_D", "SEAT_T", "SEAT_H")
            if bits:
                lines.append("Seat: " + ", ".join(bits))
        if "leg" in q:
            bits = pick("LEG_W", "LEG_INSET", "INSET")
            if bits:
                lines.append("Legs: " + ", ".join(bits))
        if "back" in q:
            bits = pick("BACK_H", "BACK_T", "BACK_RECLINE", "SLATS", "GAP")
            if bits:
                lines.append("Backrest: " + ", ".join(bits))
        if "dimension" in q or "size" in q or "overall" in q or len(lines) <= (1 if geometry_summary else 0):
            if params:
                pretty = ", ".join(f"{k}={v:g}" for k, v in sorted(params.items()))
                lines.append(f"Named parameters (mm): {pretty}")

        lines.append("(Ask mode — design was not modified.)")
        return "\n".join(lines)
