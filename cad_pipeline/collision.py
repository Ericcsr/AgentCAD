#!/usr/bin/env python3
"""Inter-part collision check for multi-part CadQuery designs.

Face-to-face mates are allowed. Volume interpenetration is a design failure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

# Ignore coincident-face noise; catch real overlaps (e.g. a 4 mm cube).
MIN_OVERLAP_VOLUME_MM3 = 50.0
# Also fail if overlap is a meaningful fraction of the smaller part.
MIN_OVERLAP_FRACTION = 0.001
# AABB must overlap by this much on every axis before we run a boolean (mm).
AABB_MIN_OVERLAP_MM = 0.05
# Fixed/weld joints must be at least this close (mm). Larger gaps are "floating".
WELD_CONTACT_GAP_MM = 1.0


@dataclass(frozen=True)
class CollisionHit:
    part_a: str
    part_b: str
    volume_mm3: float | None
    note: str = ""

    def line(self) -> str:
        vol = (
            f"overlap ≈ {self.volume_mm3:.0f} mm³"
            if self.volume_mm3 is not None
            else "overlap (boolean inconclusive; bounding boxes interpenetrate)"
        )
        extra = f" — {self.note}" if self.note else ""
        return f"`{self.part_a}` ∩ `{self.part_b}`: {vol}{extra}"


def _ocp_shape(solid: Any) -> Any:
    obj = solid.val() if hasattr(solid, "val") else solid
    return obj.wrapped if hasattr(obj, "wrapped") else obj


def _shape_volume_mm3(solid: Any) -> float:
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps

    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(_ocp_shape(solid), props)
    return abs(float(props.Mass()))


def _aabb(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return vertices.min(axis=0), vertices.max(axis=0)


def _aabb_overlap_mm(amin: np.ndarray, amax: np.ndarray, bmin: np.ndarray, bmax: np.ndarray) -> float:
    overlap = np.minimum(amax, bmax) - np.maximum(amin, bmin)
    if np.any(overlap < AABB_MIN_OVERLAP_MM):
        return 0.0
    return float(np.min(overlap))


def _common_volume_mm3(solid_a: Any, solid_b: Any) -> float | None:
    """Volume of A ∩ B, or None if the boolean did not complete."""
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps

    algo = BRepAlgoAPI_Common(_ocp_shape(solid_a), _ocp_shape(solid_b))
    algo.SetRunParallel(True)
    algo.Build()
    if not algo.IsDone():
        return None
    common = algo.Shape()
    if common.IsNull():
        return 0.0
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(common, props)
    return abs(float(props.Mass()))


def check_part_collisions(result: Any) -> list[CollisionHit]:
    """Return overlapping named-part pairs. Single-part designs are always clean."""
    parts = getattr(result, "parts", None) or {}
    names = [n for n in parts if n]
    if len(names) < 2:
        return []

    boxes: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    volumes: dict[str, float] = {}
    for name in names:
        part = parts[name]
        verts = getattr(part, "vertices", None)
        if verts is None or len(verts) == 0:
            continue
        boxes[name] = _aabb(np.asarray(verts, dtype=np.float64))
        try:
            volumes[name] = _shape_volume_mm3(part.solid)
        except Exception:
            volumes[name] = 0.0

    hits: list[CollisionHit] = []
    ordered = [n for n in names if n in boxes]
    for i, a in enumerate(ordered):
        amin, amax = boxes[a]
        for b in ordered[i + 1 :]:
            bmin, bmax = boxes[b]
            depth = _aabb_overlap_mm(amin, amax, bmin, bmax)
            if depth <= 0.0:
                continue
            try:
                common = _common_volume_mm3(parts[a].solid, parts[b].solid)
            except Exception:
                common = None
            smaller = min(volumes.get(a, 0.0), volumes.get(b, 0.0))
            threshold = max(MIN_OVERLAP_VOLUME_MM3, MIN_OVERLAP_FRACTION * smaller)
            if common is None:
                if depth >= 1.0:
                    hits.append(
                        CollisionHit(
                            a,
                            b,
                            None,
                            note=f"bbox overlap ≥ {depth:.1f} mm on every axis",
                        )
                    )
                continue
            if common >= threshold:
                hits.append(CollisionHit(a, b, common))
    return hits


def format_collision_error(hits: Sequence[CollisionHit]) -> str:
    lines = [
        "Part collision check failed — interpenetration is a design failure.",
        "Named parts must meet at faces only; they must not occupy the same volume.",
        "Keep the same parts() keys. Translate/resize so the pairs below no longer overlap:",
        "",
    ]
    lines.extend(f"- {hit.line()}" for hit in hits)
    lines.append("")
    lines.append(
        "Do not hide the issue by merging colliding parts into one body. "
        "Face contact (zero overlap volume) is OK."
    )
    return "\n".join(lines)


def assert_no_part_collisions(result: Any) -> None:
    hits = check_part_collisions(result)
    if hits:
        raise RuntimeError(format_collision_error(hits))


def _aabb_separation_mm(
    amin: np.ndarray, amax: np.ndarray, bmin: np.ndarray, bmax: np.ndarray
) -> float:
    """Euclidean gap between two AABBs. 0 if they touch or overlap."""
    delta = np.maximum(0.0, np.maximum(amin - bmax, bmin - amax))
    return float(np.linalg.norm(delta))


def _solid_distance_mm(solid_a: Any, solid_b: Any) -> float | None:
    """Minimum distance between two solids (mm), or None if the query fails."""
    from OCP.BRepExtrema import BRepExtrema_DistShapeShape

    dist = BRepExtrema_DistShapeShape(_ocp_shape(solid_a), _ocp_shape(solid_b))
    dist.Perform()
    if not dist.IsDone():
        return None
    return abs(float(dist.Value()))


@dataclass(frozen=True)
class WeldGap:
    joint_name: str
    parent: str
    child: str
    gap_mm: float | None
    note: str = ""

    def line(self) -> str:
        gap = f"gap ≈ {self.gap_mm:.1f} mm" if self.gap_mm is not None else "not in contact"
        extra = f" — {self.note}" if self.note else ""
        return f"weld `{self.joint_name}`: `{self.parent}` — `{self.child}` {gap}{extra}"


def check_weld_contacts(result: Any) -> list[WeldGap]:
    """
    Floating welds: a fixed joint whose two parts are not in contact.

    Only *direct* fixed/weld joints are checked. Unrelated parts and parts that
    are only connected through other joints may float.
    """
    parts = getattr(result, "parts", None) or {}
    joints = getattr(result, "joints", None) or []
    hits: list[WeldGap] = []
    for joint in joints:
        if getattr(joint, "type", "") != "fixed":
            continue
        parent = parts.get(joint.parent)
        child = parts.get(joint.child)
        if parent is None or child is None:
            continue
        pverts = getattr(parent, "vertices", None)
        cverts = getattr(child, "vertices", None)
        if pverts is None or cverts is None or len(pverts) == 0 or len(cverts) == 0:
            continue
        pmin, pmax = _aabb(np.asarray(pverts, dtype=np.float64))
        cmin, cmax = _aabb(np.asarray(cverts, dtype=np.float64))
        box_gap = _aabb_separation_mm(pmin, pmax, cmin, cmax)
        if box_gap > WELD_CONTACT_GAP_MM:
            hits.append(
                WeldGap(
                    joint.name,
                    joint.parent,
                    joint.child,
                    box_gap,
                    note="bounding boxes are separated",
                )
            )
            continue
        try:
            dist = _solid_distance_mm(parent.solid, child.solid)
        except Exception:
            dist = None
        if dist is None:
            # Boxes already close; treat as in contact (collision check covers overlap).
            continue
        if dist > WELD_CONTACT_GAP_MM:
            hits.append(WeldGap(joint.name, joint.parent, joint.child, dist))
    return hits


def format_weld_gap_error(hits: Sequence[WeldGap]) -> str:
    lines = [
        "Weld contact check failed — a fixed joint requires the two parts to touch.",
        "Only directly welded pairs are checked. Unrelated or indirectly connected "
        "parts may float.",
        "Move the parts into face contact, or remove the weld from joints() if they "
        "are not actually fastened:",
        "",
    ]
    lines.extend(f"- {hit.line()}" for hit in hits)
    return "\n".join(lines)


def assert_welded_parts_in_contact(result: Any) -> None:
    hits = check_weld_contacts(result)
    if hits:
        raise RuntimeError(format_weld_gap_error(hits))


def weld_contact_notes(result: Any) -> list[str]:
    joints = getattr(result, "joints", None) or []
    welds = [j for j in joints if getattr(j, "type", "") == "fixed"]
    if not welds:
        return ["Weld contact check: skipped (no fixed joints)."]
    hits = check_weld_contacts(result)
    if not hits:
        return [f"Weld contact check: PASS ({len(welds)} fixed joint(s) in contact)."]
    notes = [f"WARNING: Weld contact FAIL ({len(hits)} floating weld(s)):"]
    notes.extend(f"  - {hit.line()}" for hit in hits)
    return notes


def assert_compile_feasibility(result: Any) -> None:
    """Compilation checks: no overlap, welds in contact, no interlocking rings."""
    from cad_pipeline.assembly import assert_assembly_feasible

    assert_no_part_collisions(result)
    assert_welded_parts_in_contact(result)
    assert_assembly_feasible(result)


def collision_notes(result: Any) -> list[str]:
    """Review / Ask context lines."""
    parts = getattr(result, "parts", None) or {}
    n = len(parts)
    if n < 2:
        return [f"Collision check: skipped (only {n} named part)."]
    hits = check_part_collisions(result)
    if not hits:
        return [f"Collision check: PASS ({n} parts, 0 overlapping pairs)."]
    notes = [f"WARNING: Collision check FAIL ({len(hits)} overlapping pair(s)):"]
    notes.extend(f"  - {hit.line()}" for hit in hits)
    return notes
