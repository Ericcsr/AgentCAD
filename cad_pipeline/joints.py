#!/usr/bin/env python3
"""Parse and validate kinematic joints() from a generated design."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

JOINT_TYPES = frozenset({"fixed", "revolute", "prismatic", "continuous"})
MOVING_TYPES = frozenset({"revolute", "prismatic", "continuous"})


@dataclass
class JointSpec:
    name: str
    type: str
    parent: str
    child: str
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    origin_mm: tuple[float, float, float] | None = None
    rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)
    lower: float | None = None
    upper: float | None = None
    effort: float = 10.0
    velocity: float = 1.0

    def is_moving(self) -> bool:
        return self.type in MOVING_TYPES


def _safe_joint_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", str(name).strip())
    return cleaned.strip("_") or "joint"


def _as_vec3(value: Any, *, default: tuple[float, float, float] | None = None) -> tuple[float, float, float]:
    if value is None:
        if default is None:
            raise ValueError("expected an xyz triple")
        return default
    if isinstance(value, str):
        key = value.strip().lower()
        aliases = {
            "x": (1.0, 0.0, 0.0),
            "+x": (1.0, 0.0, 0.0),
            "-x": (-1.0, 0.0, 0.0),
            "y": (0.0, 1.0, 0.0),
            "+y": (0.0, 1.0, 0.0),
            "-y": (0.0, -1.0, 0.0),
            "z": (0.0, 0.0, 1.0),
            "+z": (0.0, 0.0, 1.0),
            "-z": (0.0, 0.0, -1.0),
        }
        if key in aliases:
            return aliases[key]
        parts = value.replace(",", " ").split()
        if len(parts) != 3:
            raise ValueError(f"axis/origin {value!r} must be 3 numbers or x/y/z")
        return (float(parts[0]), float(parts[1]), float(parts[2]))
    if isinstance(value, (int, float)):
        raise ValueError("axis/origin must be a 3-vector")
    seq = list(value)
    if len(seq) != 3:
        raise ValueError(f"axis/origin {value!r} must have 3 components")
    return (float(seq[0]), float(seq[1]), float(seq[2]))


def _normalize_axis(axis: tuple[float, float, float]) -> tuple[float, float, float]:
    vec = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-9:
        raise ValueError("joint axis cannot be a zero vector")
    unit = vec / norm
    return (float(unit[0]), float(unit[1]), float(unit[2]))


def _item_to_joint(raw: Any, index: int) -> JointSpec:
    if isinstance(raw, (tuple, list)) and len(raw) >= 3 and not isinstance(raw[0], dict):
        raw = {"type": raw[0], "parent": raw[1], "child": raw[2]}
    if not isinstance(raw, dict):
        raise ValueError(f"joints()[{index}] must be a dict or (type, parent, child)")
    jtype = str(raw.get("type") or raw.get("joint_type") or "fixed").strip().lower()
    if jtype in {"weld", "rigid", "fixed_joint"}:
        jtype = "fixed"
    if jtype in {"hinge", "revolute_joint"}:
        jtype = "revolute"
    if jtype in {"slider", "slide", "prismatic_joint"}:
        jtype = "prismatic"
    if jtype not in JOINT_TYPES:
        raise ValueError(
            f"joints()[{index}] type {jtype!r} is not one of {sorted(JOINT_TYPES)}"
        )
    parent = str(raw.get("parent") or raw.get("parent_link") or "").strip()
    child = str(raw.get("child") or raw.get("child_link") or "").strip()
    if not parent or not child:
        raise ValueError(f"joints()[{index}] needs parent and child part names")
    name = str(raw.get("name") or f"{parent}_{child}").strip()
    axis = _as_vec3(raw.get("axis"), default=(0.0, 0.0, 1.0))
    origin = raw.get("origin")
    if origin is None:
        origin = raw.get("xyz") or raw.get("origin_mm")
    origin_mm = _as_vec3(origin) if origin is not None else None
    rpy = _as_vec3(raw.get("rpy"), default=(0.0, 0.0, 0.0))
    lower = raw.get("lower")
    upper = raw.get("upper")
    return JointSpec(
        name=_safe_joint_name(name),
        type=jtype,
        parent=parent,
        child=child,
        axis=_normalize_axis(axis) if jtype in MOVING_TYPES else axis,
        origin_mm=origin_mm,
        rpy=rpy,
        lower=None if lower is None else float(lower),
        upper=None if upper is None else float(upper),
        effort=float(raw.get("effort", 10.0)),
        velocity=float(raw.get("velocity", 1.0)),
    )


def parse_joints(raw: Any) -> list[JointSpec]:
    if raw is None:
        return []
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        raise ValueError("joints() must return a list of joint dicts")
    joints = [_item_to_joint(item, i) for i, item in enumerate(raw)]
    seen: set[str] = set()
    for joint in joints:
        name = joint.name
        suffix = 2
        while name in seen:
            name = f"{joint.name}_{suffix}"
            suffix += 1
        if name != joint.name:
            joint.name = name
        seen.add(joint.name)
    return joints


def validate_joints(joints: Sequence[JointSpec], part_names: Sequence[str]) -> None:
    """Raise RuntimeError if the joint graph is invalid. Does not invent welds."""
    names = list(part_names)
    known = set(names)
    if not joints:
        return
    children: dict[str, str] = {}
    edges: list[tuple[str, str]] = []
    for joint in joints:
        if joint.parent not in known:
            raise RuntimeError(
                f"joints(): parent {joint.parent!r} is not a parts() name. "
                f"Known: {', '.join(names)}"
            )
        if joint.child not in known:
            raise RuntimeError(
                f"joints(): child {joint.child!r} is not a parts() name. "
                f"Known: {', '.join(names)}"
            )
        if joint.parent == joint.child:
            raise RuntimeError(f"joints(): {joint.name} connects a part to itself")
        if joint.child in children:
            raise RuntimeError(
                f"joints(): {joint.child!r} already has parent {children[joint.child]!r}; "
                f"URDF is a tree (one parent per part). Do not add a second weld."
            )
        children[joint.child] = joint.parent
        edges.append((joint.parent, joint.child))
        if joint.is_moving() and joint.axis == (0.0, 0.0, 0.0):
            raise RuntimeError(f"joints(): {joint.name} ({joint.type}) needs a non-zero axis")

    # Cycle check
    outgoing: dict[str, list[str]] = {n: [] for n in known}
    for parent, child in edges:
        outgoing[parent].append(child)
    state: dict[str, int] = {}  # 0 unseen, 1 visiting, 2 done

    def visit(node: str) -> None:
        mark = state.get(node, 0)
        if mark == 1:
            raise RuntimeError(
                "joints() has a cycle. URDF must be a tree — pick one parent per part."
            )
        if mark == 2:
            return
        state[node] = 1
        for nxt in outgoing.get(node, ()):
            visit(nxt)
        state[node] = 2

    for name in names:
        if state.get(name, 0) == 0:
            visit(name)


def root_links(part_names: Sequence[str], joints: Sequence[JointSpec]) -> list[str]:
    children = {j.child for j in joints}
    roots = [n for n in part_names if n not in children]
    return roots or list(part_names[:1])


def infer_origin_mm(
    parent_vertices: np.ndarray,
    child_vertices: np.ndarray,
) -> tuple[float, float, float]:
    """World-mm point for an unspecified moving joint: contact / midpoint of centers."""
    pmin, pmax = parent_vertices.min(axis=0), parent_vertices.max(axis=0)
    cmin, cmax = child_vertices.min(axis=0), child_vertices.max(axis=0)
    pc = 0.5 * (pmin + pmax)
    cc = 0.5 * (cmin + cmax)
    # If AABBs overlap or touch, use the center of the overlap slab.
    lo = np.maximum(pmin, cmin)
    hi = np.minimum(pmax, cmax)
    if np.all(hi + 1e-6 >= lo):
        mid = 0.5 * (lo + hi)
        return (float(mid[0]), float(mid[1]), float(mid[2]))
    return (float(0.5 * (pc[0] + cc[0])), float(0.5 * (pc[1] + cc[1])), float(0.5 * (pc[2] + cc[2])))


def format_joints_block(joints: Sequence[JointSpec], part_names: Sequence[str]) -> str:
    if not part_names:
        return "Kinematics: (no named parts)"
    if not joints:
        return (
            f"Kinematics: {len(part_names)} part(s), 0 joints "
            "(unrelated parts are not welded)."
        )
    lines = [f"Kinematics: {len(part_names)} part(s), {len(joints)} joint(s):"]
    for joint in joints:
        extra = ""
        if joint.is_moving():
            lim = ""
            if joint.lower is not None or joint.upper is not None:
                lim = f" limits=[{joint.lower}, {joint.upper}]"
            extra = f" axis={joint.axis}{lim}"
        lines.append(f"- {joint.name}: {joint.type}  {joint.parent} → {joint.child}{extra}")
    orphans = [n for n in part_names if n not in {j.child for j in joints} and n not in {j.parent for j in joints}]
    # roots that only appear as parent are related; true isolates have no joint at all
    related = {j.parent for j in joints} | {j.child for j in joints}
    isolates = [n for n in part_names if n not in related]
    if isolates:
        lines.append("Unrelated (no joint, not welded): " + ", ".join(isolates))
    return "\n".join(lines)
