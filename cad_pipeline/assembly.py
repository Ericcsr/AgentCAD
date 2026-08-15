#!/usr/bin/env python3
"""Analytical assembly feasibility for multi-part designs.

Fails when rigid parts cannot reach the assembled pose:

- *Link* — two closed loops occupy each other's holes (chain links).
- *Captive* — a closed ring sits on a shaft / H-beam / dogbone whose profile
  is larger than the hole at *both* ends, so the ring cannot slide on.

A pin or nut that can slide off one open end is allowed. Open hooks are not
treated as shafts (not elongated along the hole axis).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from cad_pipeline.collision import (
    _aabb,
    _aabb_overlap_mm,
    _common_volume_mm3,
    _ocp_shape,
)

# Ignore thread-scale pockets and boolean specks.
MIN_HOLE_RADIUS_MM = 2.0
# Equatorial samples around a candidate hole axis.
CLOSED_RAYS = 72
# Hit distances around a ring should be similar (reject slat gaps, U-shapes).
RING_RADIUS_CV = 0.40
# Thin cap filling the hole; the other part must cross this slab.
HOLE_CAP_THICKNESS_MM = 1.2
PIERCE_VOLUME_MM3 = 1.0
# Silhouette may sit slightly outside the measured hole (tessellation).
FIT_TOL_MM = 0.75
# Occupier must be bar-like along the hole to count as a shaft / H-beam.
SHAFT_EVAL_RATIO = 1.35
SHAFT_ALIGN = 0.75


@dataclass(frozen=True)
class HoleCap:
    origin: np.ndarray
    normal: np.ndarray
    radius: float
    fit_radius: float


@dataclass(frozen=True)
class AssemblyHit:
    part_a: str
    part_b: str
    kind: str = "link"
    note: str = ""

    def line(self) -> str:
        extra = f" — {self.note}" if self.note else ""
        if self.kind == "captive":
            return (
                f"`{self.part_a}` is captive on `{self.part_b}`: both ends of the "
                f"hole are larger than the opening{extra}"
            )
        return (
            f"`{self.part_a}` and `{self.part_b}` are closed loops that thread "
            f"each other{extra}"
        )


def _as_workplane(solid: Any) -> Any:
    import cadquery as cq

    if hasattr(solid, "faces") and callable(solid.faces):
        try:
            solid.faces()
            return solid
        except Exception:
            pass
    shape = solid.val() if hasattr(solid, "val") else solid
    return cq.Workplane("XY").newObject([shape])


def _classifier(solid: Any):
    from OCP.BRepClass3d import BRepClass3d_SolidClassifier

    return BRepClass3d_SolidClassifier(_ocp_shape(solid))


def _point_state(clf: Any, point: np.ndarray, tol: float = 1e-4):
    from OCP.gp import gp_Pnt

    clf.Perform(gp_Pnt(float(point[0]), float(point[1]), float(point[2])), tol)
    return clf.State()


def _is_outside(clf: Any, point: np.ndarray) -> bool:
    from OCP.TopAbs import TopAbs_OUT

    return _point_state(clf, point) == TopAbs_OUT


def _ray_hit_distances(solid: Any, origin: np.ndarray, direction: np.ndarray) -> list[float]:
    from OCP.BRepIntCurveSurface import BRepIntCurveSurface_Inter
    from OCP.gp import gp_Dir, gp_Lin, gp_Pnt

    d = np.asarray(direction, dtype=np.float64)
    norm = float(np.linalg.norm(d))
    if norm < 1e-12:
        return []
    d = d / norm
    inter = BRepIntCurveSurface_Inter()
    inter.Init(
        _ocp_shape(solid),
        gp_Lin(
            gp_Pnt(float(origin[0]), float(origin[1]), float(origin[2])),
            gp_Dir(float(d[0]), float(d[1]), float(d[2])),
        ),
        1e-4,
    )
    hits: list[float] = []
    while inter.More():
        w = float(inter.W())
        if w > 1e-5:
            hits.append(w)
        inter.Next()
    hits.sort()
    return hits


def _orthonormal_frame(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = np.asarray(normal, dtype=np.float64)
    n = n / float(np.linalg.norm(n))
    helper = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(helper, n)
    x = x / float(np.linalg.norm(x))
    y = np.cross(n, x)
    return n, x, y


def _closed_through_hole(
    solid: Any, origin: np.ndarray, normal: np.ndarray
) -> HoleCap | None:
    """True when ±axis is open and every equatorial ray hits — a closed through-hole."""
    n, x, y = _orthonormal_frame(normal)
    if _ray_hit_distances(solid, origin, n) or _ray_hit_distances(solid, origin, -n):
        return None
    hits: list[float] = []
    for k in range(CLOSED_RAYS):
        ang = 2.0 * np.pi * k / CLOSED_RAYS
        direction = np.cos(ang) * x + np.sin(ang) * y
        dist = _ray_hit_distances(solid, origin, direction)
        if not dist:
            return None
        hits.append(dist[0])
    arr = np.asarray(hits, dtype=np.float64)
    radius = float(arr.min())
    if radius < MIN_HOLE_RADIUS_MM:
        return None
    mean = float(arr.mean())
    if mean <= 0.0 or float(arr.std()) / mean > RING_RADIUS_CV:
        return None
    return HoleCap(
        origin=np.asarray(origin, dtype=np.float64),
        normal=n,
        radius=radius * 0.85,
        fit_radius=radius,
    )


def _inner_wire_holes(solid: Any, clf: Any) -> list[HoleCap]:
    holes: list[HoleCap] = []
    try:
        faces = _as_workplane(solid).faces().vals()
    except Exception:
        return holes
    for face in faces:
        try:
            if face.geomType() != "PLANE":
                continue
            inners = face.innerWires()
            normal = np.array(face.normalAt().toTuple(), dtype=np.float64)
        except Exception:
            continue
        for wire in inners:
            try:
                sampled = wire.sample(32)
                pts = sampled[0] if isinstance(sampled, tuple) else sampled
                arr = np.array([[p.x, p.y, p.z] for p in pts], dtype=np.float64)
            except Exception:
                continue
            if len(arr) < 8:
                continue
            origin = arr.mean(axis=0)
            radius = float(np.linalg.norm(arr - origin, axis=1).mean())
            if radius < MIN_HOLE_RADIUS_MM:
                continue
            if not _is_outside(clf, origin):
                continue
            nrm = normal / float(np.linalg.norm(normal) or 1.0)
            if _ray_hit_distances(solid, origin, nrm) or _ray_hit_distances(
                solid, origin, -nrm
            ):
                continue
            holes.append(
                HoleCap(
                    origin=np.asarray(origin, dtype=np.float64),
                    normal=nrm,
                    radius=radius * 0.85,
                    fit_radius=radius,
                )
            )
    return holes


def _pca_ring_hole(solid: Any, vertices: np.ndarray, clf: Any) -> HoleCap | None:
    verts = np.asarray(vertices, dtype=np.float64)
    if len(verts) < 8:
        return None
    origin = verts.mean(axis=0)
    if not _is_outside(clf, origin):
        return None
    cov = np.cov((verts - origin).T)
    try:
        _evals, evecs = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return None
    order = np.argsort(_evals)
    # Flat ring: hole axis = thin direction. Long tube: hole axis = length.
    for idx in (int(order[0]), int(order[-1])):
        cap = _closed_through_hole(solid, origin, evecs[:, idx])
        if cap is not None:
            return cap
    return None


def holes_for_part(part: Any) -> list[HoleCap]:
    """Through-holes / ring openings of one named part (empty if genus-0)."""
    solid = getattr(part, "solid", part)
    vertices = getattr(part, "vertices", None)
    clf = _classifier(solid)
    holes = _inner_wire_holes(solid, clf)
    if not holes and vertices is not None and len(vertices) > 0:
        syn = _pca_ring_hole(solid, vertices, clf)
        if syn is not None:
            holes.append(syn)
    return holes


def _make_hole_cap_solid(hole: HoleCap):
    import cadquery as cq

    n, x, _y = _orthonormal_frame(hole.normal)
    start = hole.origin - n * (HOLE_CAP_THICKNESS_MM * 0.5)
    plane = cq.Plane(
        origin=cq.Vector(float(start[0]), float(start[1]), float(start[2])),
        xDir=cq.Vector(float(x[0]), float(x[1]), float(x[2])),
        normal=cq.Vector(float(n[0]), float(n[1]), float(n[2])),
    )
    return cq.Workplane(plane).circle(float(hole.radius)).extrude(HOLE_CAP_THICKNESS_MM)


def _disk_aabb(hole: HoleCap) -> tuple[np.ndarray, np.ndarray]:
    n = hole.normal / float(np.linalg.norm(hole.normal))
    radial = hole.radius * np.sqrt(np.maximum(0.0, 1.0 - n * n))
    axial = (HOLE_CAP_THICKNESS_MM * 0.5) * np.abs(n)
    half = radial + axial
    return hole.origin - half, hole.origin + half


def _section_has_edge(shape_a: Any, shape_b: Any) -> bool:
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Section
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopExp import TopExp_Explorer

    sec = BRepAlgoAPI_Section(_ocp_shape(shape_a), _ocp_shape(shape_b))
    sec.Build()
    if not sec.IsDone():
        return False
    explorer = TopExp_Explorer(sec.Shape(), TopAbs_EDGE)
    return bool(explorer.More())


def _occupies_hole(solid: Any, hole: HoleCap, other_aabb: tuple[np.ndarray, np.ndarray]) -> bool:
    dmin, dmax = _disk_aabb(hole)
    if _aabb_overlap_mm(dmin, dmax, other_aabb[0], other_aabb[1]) <= 0.0:
        return False
    try:
        cap = _make_hole_cap_solid(hole)
    except Exception:
        return False
    try:
        if _section_has_edge(solid, cap):
            return True
    except Exception:
        pass
    try:
        vol = _common_volume_mm3(cap, solid)
    except Exception:
        vol = None
    return vol is not None and vol >= PIERCE_VOLUME_MM3


def _max_radial_mm(points: np.ndarray, origin: np.ndarray, normal: np.ndarray) -> float:
    delta = points - origin
    axial = delta @ normal
    radial = delta - np.outer(axial, normal)
    return float(np.linalg.norm(radial, axis=1).max())


def _shaft_axis(vertices: np.ndarray) -> np.ndarray | None:
    """Long principal axis if the solid is elongated (bar / beam / bolt)."""
    verts = np.asarray(vertices, dtype=np.float64)
    if len(verts) < 8:
        return None
    cov = np.cov((verts - verts.mean(axis=0)).T)
    try:
        evals, evecs = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return None
    order = np.argsort(evals)
    mid = float(max(evals[order[-2]], 1e-12))
    if float(evals[order[-1]]) < SHAFT_EVAL_RATIO * mid:
        return None
    axis = evecs[:, order[-1]]
    return axis / float(np.linalg.norm(axis))


def _far_end_blocked(vertices: np.ndarray, hole: HoleCap, sign: float) -> bool:
    """True if the occupier's far cross-section on this side cannot pass the hole."""
    n = hole.normal / float(np.linalg.norm(hole.normal))
    s = (vertices - hole.origin) @ n
    margin = max(HOLE_CAP_THICKNESS_MM, 1.0)
    if sign > 0:
        mask = s > margin
    else:
        mask = s < -margin
    if not np.any(mask):
        return False
    s_side = s[mask]
    s_ext = float(s_side.max() if sign > 0 else s_side.min())
    slab_half = max(5.0, 0.12 * abs(s_ext))
    target = 0.88 * s_ext
    slab = vertices[np.abs(s - target) <= slab_half]
    if len(slab) < 4:
        slab = vertices[mask]
    if len(slab) < 4:
        return False
    return _max_radial_mm(slab, hole.origin, n) > hole.fit_radius + FIT_TOL_MM


def _both_ends_blocked(vertices: np.ndarray, hole: HoleCap) -> bool:
    axis = _shaft_axis(vertices)
    if axis is None:
        return False
    n = hole.normal / float(np.linalg.norm(hole.normal))
    if abs(float(np.dot(axis, n))) < SHAFT_ALIGN:
        return False
    return _far_end_blocked(vertices, hole, 1.0) and _far_end_blocked(vertices, hole, -1.0)


def check_assembly_interlocks(result: Any) -> list[AssemblyHit]:
    """Linked closed loops, or a ring captive on a double-ended oversized shaft."""
    parts = getattr(result, "parts", None) or {}
    names = [n for n in parts if n]
    if len(names) < 2:
        return []

    holes: dict[str, list[HoleCap]] = {}
    boxes: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    meshes: dict[str, np.ndarray] = {}
    for name in names:
        part = parts[name]
        verts = getattr(part, "vertices", None)
        if verts is None or len(verts) == 0:
            continue
        meshes[name] = np.asarray(verts, dtype=np.float64)
        boxes[name] = _aabb(meshes[name])
        found = holes_for_part(part)
        if found:
            holes[name] = found

    hits: list[AssemblyHit] = []
    seen: set[frozenset[str]] = set()
    for a, a_holes in holes.items():
        for b in names:
            if a == b or b not in boxes:
                continue
            occupied = [
                hole
                for hole in a_holes
                if _occupies_hole(parts[b].solid, hole, boxes[b])
            ]
            if not occupied:
                continue
            key = frozenset((a, b))
            if key in seen:
                continue
            b_holes = holes.get(b, [])
            mutual = any(
                _occupies_hole(parts[a].solid, hole, boxes[a]) for hole in b_holes
            )
            if mutual:
                seen.add(key)
                hits.append(
                    AssemblyHit(
                        a,
                        b,
                        kind="link",
                        note="rigid closed rings cannot be assembled; open a gap in one loop",
                    )
                )
                continue
            if any(_both_ends_blocked(meshes[b], hole) for hole in occupied):
                seen.add(key)
                hits.append(
                    AssemblyHit(
                        a,
                        b,
                        kind="captive",
                        note="cannot slide on from either end; open a gap or enlarge one end",
                    )
                )
    return hits


def format_assembly_error(hits: Sequence[AssemblyHit]) -> str:
    lines = [
        "Assembly check failed — the assembled pose cannot be reached from rigid parts.",
        "Closed loops that thread each other (chain links) are infeasible.",
        "A ring on a shaft / H-beam is infeasible if both ends are larger than the hole.",
        "Open a gap in the ring, drop one oversized end, or assemble before closing.",
        "A pin or bolt that can slide off one open end is OK. Face contact is OK.",
        "",
    ]
    lines.extend(f"- {hit.line()}" for hit in hits)
    return "\n".join(lines)


def assert_assembly_feasible(result: Any) -> None:
    hits = check_assembly_interlocks(result)
    if hits:
        raise RuntimeError(format_assembly_error(hits))


def assembly_notes(result: Any) -> list[str]:
    parts = getattr(result, "parts", None) or {}
    n = len(parts)
    if n < 2:
        return [f"Assembly check: skipped (only {n} named part)."]
    hits = check_assembly_interlocks(result)
    if not hits:
        return [f"Assembly check: PASS ({n} parts, no interlocking or captive rings)."]
    notes = [f"WARNING: Assembly check FAIL ({len(hits)} infeasible pair(s)):"]
    notes.extend(f"  - {hit.line()}" for hit in hits)
    return notes
