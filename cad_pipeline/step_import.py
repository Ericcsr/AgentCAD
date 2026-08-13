#!/usr/bin/env python3
"""Import STEP files, extract geometric facts, and stage them for the CAD agent."""

from __future__ import annotations

import shutil
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cadquery as cq
import numpy as np

from cad_pipeline.mesh_utils import solid_to_mesh

ROOT = Path(__file__).resolve().parent.parent
REFERENCES_DIR = ROOT / "generated" / "references"


@dataclass
class StepReference:
    """An imported STEP used as design context / constraint, not the live design."""

    name: str
    source_path: Path
    staged_path: Path
    facts_path: Path
    summary: str
    vertices: np.ndarray
    faces: np.ndarray
    bbox_min: tuple[float, float, float]
    bbox_max: tuple[float, float, float]
    volume_mm3: float
    n_solids: int
    extras: dict[str, Any] = field(default_factory=dict)

    def size_mm(self) -> tuple[float, float, float]:
        return (
            self.bbox_max[0] - self.bbox_min[0],
            self.bbox_max[1] - self.bbox_min[1],
            self.bbox_max[2] - self.bbox_min[2],
        )


def _safe_stem(path: Path) -> str:
    raw = path.stem.strip() or "reference"
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in raw).strip("_")
    return safe or "reference"


def _count_subshapes(shape: Any, kind: Any) -> int:
    from OCP.TopExp import TopExp_Explorer

    n = 0
    explorer = TopExp_Explorer(shape, kind)
    while explorer.More():
        n += 1
        explorer.Next()
    return n


def _bbox_of(shape: Any) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    from OCP.BRepBndLib import BRepBndLib
    from OCP.Bnd import Bnd_Box

    box = Bnd_Box()
    BRepBndLib.Add_s(shape, box)
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    return (float(xmin), float(ymin), float(zmin)), (float(xmax), float(ymax), float(zmax))


def _mass_props(shape: Any) -> tuple[float, float]:
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps

    vol = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, vol)
    area = GProp_GProps()
    BRepGProp.SurfaceProperties_s(shape, area)
    return float(vol.Mass()), float(area.Mass())


def _surface_stats(shape: Any, *, max_radii: int = 12) -> dict[str, Any]:
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import (
        GeomAbs_BSplineSurface,
        GeomAbs_Cone,
        GeomAbs_Cylinder,
        GeomAbs_Plane,
        GeomAbs_Sphere,
        GeomAbs_Torus,
    )
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    counts: Counter[str] = Counter()
    cyl_radii: list[float] = []
    sphere_radii: list[float] = []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = TopoDS.Face_s(explorer.Current())
        try:
            surf = BRepAdaptor_Surface(face)
            kind = surf.GetType()
        except Exception:
            counts["unknown"] += 1
            explorer.Next()
            continue
        if kind == GeomAbs_Plane:
            counts["plane"] += 1
        elif kind == GeomAbs_Cylinder:
            counts["cylinder"] += 1
            try:
                cyl_radii.append(round(float(surf.Cylinder().Radius()), 3))
            except Exception:
                pass
        elif kind == GeomAbs_Cone:
            counts["cone"] += 1
        elif kind == GeomAbs_Sphere:
            counts["sphere"] += 1
            try:
                sphere_radii.append(round(float(surf.Sphere().Radius()), 3))
            except Exception:
                pass
        elif kind == GeomAbs_Torus:
            counts["torus"] += 1
        elif kind == GeomAbs_BSplineSurface:
            counts["bspline"] += 1
        else:
            counts["other"] += 1
        explorer.Next()

    unique_cyl = sorted(set(cyl_radii))
    unique_sph = sorted(set(sphere_radii))
    return {
        "face_types": dict(counts),
        "cylinder_radii_mm": unique_cyl[:max_radii],
        "sphere_radii_mm": unique_sph[:max_radii],
    }


def _solid_summaries(workplane: Any, *, limit: int = 12) -> list[str]:
    from OCP.TopAbs import TopAbs_FACE

    lines: list[str] = []
    try:
        solids = workplane.solids().vals()
    except Exception:
        solids = []
    for i, solid in enumerate(solids[:limit], start=1):
        shape = solid.wrapped if hasattr(solid, "wrapped") else solid
        bmin, bmax = _bbox_of(shape)
        vol, area = _mass_props(shape)
        sx, sy, sz = bmax[0] - bmin[0], bmax[1] - bmin[1], bmax[2] - bmin[2]
        n_faces = _count_subshapes(shape, TopAbs_FACE)
        lines.append(
            f"  solid {i}: {sx:.1f}×{sy:.1f}×{sz:.1f} mm  "
            f"vol={vol:.0f} mm³  faces={n_faces}  "
            f"origin-bbox [({bmin[0]:.1f},{bmin[1]:.1f},{bmin[2]:.1f}) → "
            f"({bmax[0]:.1f},{bmax[1]:.1f},{bmax[2]:.1f})]"
        )
    if len(solids) > limit:
        lines.append(f"  … {len(solids) - limit} more solid(s) omitted")
    return lines


def extract_step_facts(workplane: Any, *, label: str) -> tuple[str, dict[str, Any]]:
    """Return (markdown summary, extras dict) for a CadQuery-imported STEP."""
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_SHELL, TopAbs_SOLID, TopAbs_VERTEX

    shape = workplane.val().wrapped if hasattr(workplane, "val") else workplane
    bmin, bmax = _bbox_of(shape)
    vol, area = _mass_props(shape)
    size = (bmax[0] - bmin[0], bmax[1] - bmin[1], bmax[2] - bmin[2])
    n_solids = _count_subshapes(shape, TopAbs_SOLID)
    n_shells = _count_subshapes(shape, TopAbs_SHELL)
    n_faces = _count_subshapes(shape, TopAbs_FACE)
    n_edges = _count_subshapes(shape, TopAbs_EDGE)
    n_verts = _count_subshapes(shape, TopAbs_VERTEX)
    surf = _surface_stats(shape)
    solid_lines = _solid_summaries(workplane)

    extras = {
        "bbox_min": bmin,
        "bbox_max": bmax,
        "size_mm": size,
        "volume_mm3": vol,
        "area_mm2": area,
        "n_solids": n_solids,
        "n_shells": n_shells,
        "n_faces": n_faces,
        "n_edges": n_edges,
        "n_vertices": n_verts,
        **surf,
    }

    lines = [
        f"### {label}",
        f"- bbox mm: {size[0]:.1f} × {size[1]:.1f} × {size[2]:.1f}  "
        f"(X[{bmin[0]:.1f}, {bmax[0]:.1f}] Y[{bmin[1]:.1f}, {bmax[1]:.1f}] "
        f"Z[{bmin[2]:.1f}, {bmax[2]:.1f}])",
        f"- volume: {vol:.0f} mm³   surface area: {area:.0f} mm²",
        f"- topology: {n_solids} solid(s), {n_shells} shell(s), "
        f"{n_faces} faces, {n_edges} edges, {n_verts} vertices",
    ]
    types = surf.get("face_types") or {}
    if types:
        type_txt = ", ".join(f"{k}={v}" for k, v in sorted(types.items()))
        lines.append(f"- face types: {type_txt}")
    radii = surf.get("cylinder_radii_mm") or []
    if radii:
        lines.append("- cylinder radii mm (holes/shafts): " + ", ".join(f"{r:g}" for r in radii))
    spheres = surf.get("sphere_radii_mm") or []
    if spheres:
        lines.append("- sphere radii mm: " + ", ".join(f"{r:g}" for r in spheres))
    if solid_lines:
        lines.append("- solids:")
        lines.extend(solid_lines)
    return "\n".join(lines), extras


def _unique_name(stem: str, existing: set[str]) -> str:
    if stem not in existing:
        return stem
    i = 2
    while f"{stem}_{i}" in existing:
        i += 1
    return f"{stem}_{i}"


def import_step_file(
    path: Path,
    *,
    dest_dir: Path | None = None,
    existing_names: set[str] | None = None,
) -> StepReference:
    """
    Load a STEP, tessellate it, extract facts, and copy into generated/references/.
    """
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"STEP file not found: {path}")
    if path.suffix.lower() not in {".step", ".stp"}:
        raise ValueError(f"Expected a .step / .stp file, got {path.suffix or '(no suffix)'}")

    dest_dir = (dest_dir or REFERENCES_DIR).resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)

    name = _unique_name(_safe_stem(path), existing_names or set())
    staged = dest_dir / f"{name}.step"
    facts_path = dest_dir / f"{name}.md"

    if path.resolve() != staged:
        shutil.copy2(path, staged)

    workplane = cq.importers.importStep(str(staged))
    if workplane is None:
        raise RuntimeError(f"CadQuery importStep returned None for {path}")
    try:
        _ = workplane.val()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"STEP imported but has no solid: {path}\n{exc}") from exc

    summary, extras = extract_step_facts(workplane, label=name)
    vertices, faces = solid_to_mesh(workplane)
    bmin = extras["bbox_min"]
    bmax = extras["bbox_max"]

    facts_md = "\n".join(
        [
            f"# Reference STEP `{name}`",
            "",
            f"- CadQuery: `import_reference(\"{name}\")`  (injected at runtime)",
            "- Do not open the binary STEP; facts below are already measured.",
            "",
            summary,
            "",
        ]
    )
    facts_path.write_text(facts_md, encoding="utf-8")

    return StepReference(
        name=name,
        source_path=path,
        staged_path=staged,
        facts_path=facts_path,
        summary=summary,
        vertices=vertices,
        faces=faces,
        bbox_min=bmin,
        bbox_max=bmax,
        volume_mm3=float(extras["volume_mm3"]),
        n_solids=int(extras["n_solids"]),
        extras=extras,
    )


def format_references_block(refs: list[StepReference], *, compact: bool = False) -> str:
    """Prompt / worksheet block describing imported STEP constraints.

    Never include .step/.stp paths — the local Cursor agent will try to read
    those binaries on follow-up turns and hang.
    """
    if not refs:
        return ""
    lines = [
        "Imported STEP references (design constraints — not the live CadQuery design):",
        "Use the measured facts below. Do NOT open or read any .step/.stp files.",
        "To use the exact B-rep in CadQuery, call import_reference(\"name\") "
        "(injected at runtime).",
        "Prefer matching dimensions parametrically; import the B-rep only when you need "
        "the exact shape (mate, cut, envelope).",
        "",
    ]
    for ref in refs:
        facts_rel = ref.facts_path
        try:
            facts_rel = ref.facts_path.relative_to(ROOT)
        except ValueError:
            pass
        lines.append(f"## `{ref.name}`")
        if not compact:
            lines.append(f"facts: `{facts_rel}`")
        lines.append(f'CadQuery: import_reference("{ref.name}")')
        lines.append(ref.summary)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def load_reference_solid(name: str, *, dest_dir: Path | None = None) -> Any:
    """CadQuery helper used inside generated design scripts."""
    dest_dir = dest_dir or REFERENCES_DIR
    key = str(name).strip()
    if not key:
        raise ValueError("import_reference() needs a reference name")
    candidates = [
        dest_dir / f"{key}.step",
        dest_dir / f"{key}.stp",
        dest_dir / key,
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        available = sorted(p.stem for p in dest_dir.glob("*.step")) + sorted(
            p.stem for p in dest_dir.glob("*.stp")
        )
        listing = ", ".join(available) or "(none)"
        raise FileNotFoundError(
            f"Unknown STEP reference `{key}`. Available: {listing}"
        )
    return cq.importers.importStep(str(path))
