#!/usr/bin/env python3
"""Import reference CAD/mesh files, extract facts, and stage them for the agent."""

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

# B-rep and common mesh formats accepted as design constraints.
STEP_SUFFIXES = frozenset({".step", ".stp"})
IGES_SUFFIXES = frozenset({".iges", ".igs"})
BREP_SUFFIXES = frozenset({".brep"})
MESH_SUFFIXES = frozenset({".stl", ".obj", ".ply"})
REFERENCE_SUFFIXES = STEP_SUFFIXES | IGES_SUFFIXES | BREP_SUFFIXES | MESH_SUFFIXES

KIND_LABELS = {
    "step": "STEP",
    "iges": "IGES",
    "brep": "BREP",
    "mesh": "mesh",
}


@dataclass
class StepReference:
    """An imported CAD/mesh used as design context / constraint, not the live design."""

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
    kind: str = "step"
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


def reference_kind(suffix: str) -> str:
    ext = suffix.lower()
    if ext in STEP_SUFFIXES:
        return "step"
    if ext in IGES_SUFFIXES:
        return "iges"
    if ext in BREP_SUFFIXES:
        return "brep"
    if ext in MESH_SUFFIXES:
        return "mesh"
    raise ValueError(
        f"Unsupported reference format {suffix or '(no suffix)'}. "
        f"Use STEP/STP, IGES/IGS, BREP, STL, OBJ, or PLY."
    )


def _shape_to_workplane(shape: Any) -> Any:
    wrapped = cq.Shape.cast(shape)
    return cq.Workplane("XY").newObject([wrapped])


def _import_step(path: Path) -> Any:
    workplane = cq.importers.importStep(str(path))
    if workplane is None:
        raise RuntimeError(f"CadQuery importStep returned None for {path}")
    return workplane


def _import_iges(path: Path) -> Any:
    from OCP.IGESControl import IGESControl_Reader

    reader = IGESControl_Reader()
    status = reader.ReadFile(str(path))
    if "RetDone" not in str(status):
        raise RuntimeError(f"IGES read failed ({status}) for {path}")
    reader.TransferRoots()
    shape = reader.OneShape()
    if shape.IsNull():
        raise RuntimeError(f"IGES imported but has no shape: {path}")
    return _shape_to_workplane(shape)


def _import_brep(path: Path) -> Any:
    workplane = cq.importers.importBrep(str(path))
    if workplane is None:
        raise RuntimeError(f"CadQuery importBrep returned None for {path}")
    return workplane


def _import_stl(path: Path) -> Any:
    from OCP.StlAPI import StlAPI_Reader
    from OCP.TopoDS import TopoDS_Shape

    shape = TopoDS_Shape()
    ok = StlAPI_Reader().Read(shape, str(path))
    if not ok or shape.IsNull():
        raise RuntimeError(f"STL read failed for {path}")
    return _shape_to_workplane(shape)


def _parse_obj_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    verts: list[list[float]] = []
    faces: list[list[int]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0] == "v" and len(parts) >= 4:
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f" and len(parts) >= 4:
                idx: list[int] = []
                for token in parts[1:]:
                    token = token.split("/")[0]
                    if not token:
                        continue
                    n = int(token)
                    idx.append(n - 1 if n > 0 else len(verts) + n)
                for i in range(1, len(idx) - 1):
                    faces.append([idx[0], idx[i], idx[i + 1]])
    if not verts or not faces:
        raise RuntimeError(f"OBJ has no triangles: {path}")
    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def _parse_ply_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        header = handle.readline().decode("ascii", errors="ignore").strip()
        if header.lower() != "ply":
            raise RuntimeError(f"Not a PLY file: {path}")
        fmt = "ascii"
        n_verts = 0
        n_faces = 0
        while True:
            line = handle.readline().decode("ascii", errors="ignore").strip()
            if not line:
                continue
            if line.startswith("format"):
                fmt = line.split()[1].lower()
            elif line.startswith("element vertex"):
                n_verts = int(line.split()[-1])
            elif line.startswith("element face"):
                n_faces = int(line.split()[-1])
            elif line == "end_header":
                break
        if fmt != "ascii":
            raise RuntimeError(
                f"Binary PLY is not supported ({path.name}). Export ASCII PLY or STL."
            )
    if n_verts <= 0 or n_faces <= 0:
        raise RuntimeError(f"PLY has no mesh: {path}")
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    start = 0
    for i, line in enumerate(lines):
        if line.strip() == "end_header":
            start = i + 1
            break
    verts = []
    for line in lines[start : start + n_verts]:
        cols = line.split()
        if len(cols) < 3:
            continue
        verts.append([float(cols[0]), float(cols[1]), float(cols[2])])
    faces: list[list[int]] = []
    for line in lines[start + n_verts : start + n_verts + n_faces]:
        cols = line.split()
        if not cols:
            continue
        count = int(float(cols[0]))
        idx = [int(float(c)) for c in cols[1 : 1 + count]]
        for i in range(1, len(idx) - 1):
            faces.append([idx[0], idx[i], idx[i + 1]])
    if len(verts) < 3 or not faces:
        raise RuntimeError(f"PLY has no triangles: {path}")
    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def _write_ascii_stl(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    with path.open("w", encoding="ascii") as handle:
        handle.write("solid reference\n")
        for tri in faces:
            p0, p1, p2 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
            handle.write("facet normal 0 0 0\nouter loop\n")
            for p in (p0, p1, p2):
                handle.write(f"  vertex {p[0]:.7g} {p[1]:.7g} {p[2]:.7g}\n")
            handle.write("endloop\nendfacet\n")
        handle.write("endsolid reference\n")


def _import_mesh_file(path: Path) -> Any:
    ext = path.suffix.lower()
    if ext == ".stl":
        return _import_stl(path)
    if ext == ".obj":
        vertices, faces = _parse_obj_mesh(path)
    elif ext == ".ply":
        vertices, faces = _parse_ply_mesh(path)
    else:
        raise ValueError(f"Unsupported mesh format {ext}")
    tmp = path.with_name(path.stem + ".__ref.tmp.stl")
    try:
        _write_ascii_stl(tmp, vertices, faces)
        return _import_stl(tmp)
    finally:
        if tmp.exists():
            tmp.unlink()


def load_reference_workplane(path: Path) -> Any:
    """Load a supported CAD/mesh file as a CadQuery Workplane."""
    ext = path.suffix.lower()
    kind = reference_kind(ext)
    if kind == "step":
        return _import_step(path)
    if kind == "iges":
        return _import_iges(path)
    if kind == "brep":
        return _import_brep(path)
    return _import_mesh_file(path)


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


def extract_step_facts(
    workplane: Any,
    *,
    label: str,
    kind: str = "step",
    source_suffix: str = ".step",
) -> tuple[str, dict[str, Any]]:
    """Return (markdown summary, extras dict) for an imported reference."""
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
        "kind": kind,
        "source_suffix": source_suffix,
    }

    lines = [
        f"### {label}",
        f"- format: {KIND_LABELS.get(kind, kind)} ({source_suffix or 'unknown'})",
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
    if kind == "mesh":
        lines.append(
            "- mesh reference: triangulated shell (no analytic cylinders). "
            "Copy measured bbox / extents; import_reference() returns the triangle solid."
        )
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


def import_reference_file(
    path: Path,
    *,
    dest_dir: Path | None = None,
    existing_names: set[str] | None = None,
) -> StepReference:
    """
    Load a CAD or mesh reference, tessellate it, extract facts, and copy into
    generated/references/.
    """
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Reference file not found: {path}")
    suffix = path.suffix.lower()
    kind = reference_kind(suffix)

    dest_dir = (dest_dir or REFERENCES_DIR).resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)

    name = _unique_name(_safe_stem(path), existing_names or set())
    staged = dest_dir / f"{name}{suffix}"
    facts_path = dest_dir / f"{name}.md"

    if path.resolve() != staged:
        shutil.copy2(path, staged)

    workplane = load_reference_workplane(staged)
    try:
        _ = workplane.val()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Reference imported but has no shape: {path}\n{exc}") from exc

    summary, extras = extract_step_facts(
        workplane, label=name, kind=kind, source_suffix=suffix
    )
    vertices, faces = solid_to_mesh(workplane)
    if abs(float(extras.get("volume_mm3") or 0.0)) < 1e-6 and len(faces):
        mesh_vol = _triangle_volume_mm3(vertices, faces)
        old_vol = f"- volume: {float(extras.get('volume_mm3') or 0.0):.0f} mm³"
        extras["volume_mm3"] = mesh_vol
        extras["volume_from"] = "mesh"
        summary = summary.replace(
            old_vol, f"- volume: {mesh_vol:.0f} mm³ (from mesh)", 1
        )
    bmin = extras["bbox_min"]
    bmax = extras["bbox_max"]
    label = KIND_LABELS.get(kind, kind)

    facts_md = "\n".join(
        [
            f"# Reference {label} `{name}`",
            "",
            f"- CadQuery: `import_reference(\"{name}\")`  (injected at runtime)",
            "- Do not open the binary CAD/mesh file; facts below are already measured.",
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
        kind=kind,
        extras=extras,
    )


def import_step_file(
    path: Path,
    *,
    dest_dir: Path | None = None,
    existing_names: set[str] | None = None,
) -> StepReference:
    """Alias for import_reference_file (STEP, STL, IGES, …)."""
    return import_reference_file(
        path, dest_dir=dest_dir, existing_names=existing_names
    )


def _triangle_volume_mm3(vertices: np.ndarray, faces: np.ndarray) -> float:
    tris = np.asarray(vertices, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
    vol = float(
        np.einsum("ij,ij->", np.cross(tris[:, 1], tris[:, 2]), tris[:, 0]) / 6.0
    )
    return abs(vol)


def format_references_block(refs: list[StepReference], *, compact: bool = False) -> str:
    """Prompt / worksheet block describing imported CAD/mesh constraints.

    Never include binary paths — the local Cursor agent will try to read
    those files on follow-up turns and hang.
    """
    if not refs:
        return ""
    lines = [
        "Imported CAD/mesh references (design constraints — not the live CadQuery design):",
        "Formats: STEP/STP, IGES, BREP, STL, OBJ, ASCII PLY.",
        "Use the measured facts below. Do NOT open or read any .step/.stp/.stl/.obj/"
        ".iges/.igs/.brep/.ply files.",
        "To use the exact shape in CadQuery, call import_reference(\"name\") "
        "(injected at runtime).",
        "Prefer matching dimensions parametrically; import the shape only when you need "
        "the exact geometry (mate, cut, envelope).",
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
    candidates = [dest_dir / f"{key}{ext}" for ext in sorted(REFERENCE_SUFFIXES)]
    candidates.append(dest_dir / key)
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        available = sorted(
            {p.stem for p in dest_dir.iterdir() if p.suffix.lower() in REFERENCE_SUFFIXES}
        ) if dest_dir.is_dir() else []
        listing = ", ".join(available) or "(none)"
        raise FileNotFoundError(
            f"Unknown reference `{key}`. Available: {listing}"
        )
    return load_reference_workplane(path)
