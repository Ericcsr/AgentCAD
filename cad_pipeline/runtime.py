#!/usr/bin/env python3
"""Execute generated CadQuery scripts and export STEP (with named parts)."""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cadquery as cq
import math
import numpy as np

from cad_pipeline.mesh_utils import solid_to_mesh

WHOLE_DESIGN = "__whole__"


@dataclass
class PartSpec:
    """One named solid plus its tessellation."""

    name: str
    solid: Any
    vertices: np.ndarray
    faces: np.ndarray


@dataclass
class DesignResult:
    code: str
    solid: Any
    vertices: np.ndarray
    faces: np.ndarray
    parts: dict[str, PartSpec] = field(default_factory=dict)

    def part_names(self) -> list[str]:
        return list(self.parts.keys())

    def mesh_for(self, scope: str | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return (vertices, faces) for whole design or a named part."""
        if not scope or scope == WHOLE_DESIGN or scope not in self.parts:
            return self.vertices, self.faces
        part = self.parts[scope]
        return part.vertices, part.faces

    def solid_for(self, scope: str | None = None) -> Any:
        if not scope or scope == WHOLE_DESIGN or scope not in self.parts:
            return self.solid
        return self.parts[scope].solid


def _extract_code(text: str) -> str:
    """Pull Python from a fenced markdown block if present."""
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
                return body.strip()
    return text


def _union_solids(items: list[Any]) -> Any:
    if not items:
        raise RuntimeError("No solids to union")
    solid = items[0]
    for item in items[1:]:
        solid = solid.union(item)
    return solid


def _normalize_part_map(raw: Any) -> dict[str, Any]:
    """Accept dict[str, Workplane] (or list of (name, solid))."""
    if isinstance(raw, dict):
        out: dict[str, Any] = {}
        for key, val in raw.items():
            name = str(key).strip()
            if not name or val is None:
                continue
            out[name] = val
        return out
    raise RuntimeError("parts()/build() dict must map names to CadQuery solids")


def _tessellate_parts(part_solids: dict[str, Any]) -> dict[str, PartSpec]:
    parts: dict[str, PartSpec] = {}
    for name, solid in part_solids.items():
        try:
            vertices, faces = solid_to_mesh(solid)
        except Exception as exc:
            raise RuntimeError(
                f"Tessellation failed for part {name!r}:\n{exc}\n\n{traceback.format_exc()}"
            ) from exc
        parts[name] = PartSpec(name=name, solid=solid, vertices=vertices, faces=faces)
    return parts


def run_design_code(code: str) -> DesignResult:
    """
    Execute CadQuery source.

    Supported shapes:
      - build() -> Workplane                     (single body; part name "body")
      - build() -> dict[str, Workplane]          (named parts; assembly = union)
      - parts() -> dict[str, Workplane] + build() -> assembly Workplane
    """
    cleaned = _extract_code(code)
    safe_builtins = {
        "abs": abs,
        "min": min,
        "max": max,
        "range": range,
        "len": len,
        "float": float,
        "int": int,
        "round": round,
        "enumerate": enumerate,
        "zip": zip,
        "list": list,
        "dict": dict,
        "tuple": tuple,
        "str": str,
        "True": True,
        "False": False,
        "None": None,
        "Exception": Exception,
        "ValueError": ValueError,
        "TypeError": TypeError,
        "RuntimeError": RuntimeError,
        "AttributeError": AttributeError,
        "KeyError": KeyError,
    }
    namespace: dict[str, Any] = {
        "__builtins__": safe_builtins,
        "cq": cq,
        "cadquery": cq,
        "math": math,
    }

    try:
        exec(cleaned, namespace, namespace)  # noqa: S102 — intentional CAD sandbox
    except Exception as exc:
        raise RuntimeError(f"CadQuery code failed to execute:\n{exc}\n\n{traceback.format_exc()}") from exc

    build = namespace.get("build")
    parts_fn = namespace.get("parts")
    if not callable(build):
        raise RuntimeError("Generated code must define build() -> cq.Workplane or dict of parts")

    part_solids: dict[str, Any] = {}
    assembly: Any = None

    # Prefer explicit parts() when present
    if callable(parts_fn):
        try:
            raw_parts = parts_fn()
        except Exception as exc:
            raise RuntimeError(f"parts() raised:\n{exc}\n\n{traceback.format_exc()}") from exc
        if raw_parts is None:
            raise RuntimeError("parts() returned None")
        part_solids = _normalize_part_map(raw_parts)
        try:
            assembly = build()
        except Exception as exc:
            raise RuntimeError(f"build() raised:\n{exc}\n\n{traceback.format_exc()}") from exc
        if assembly is None:
            raise RuntimeError("build() returned None")
        if isinstance(assembly, dict):
            # build() also returned parts — prefer parts() map, assembly = union
            if not part_solids:
                part_solids = _normalize_part_map(assembly)
            assembly = _union_solids(list(part_solids.values()))
    else:
        try:
            built = build()
        except Exception as exc:
            raise RuntimeError(f"build() raised:\n{exc}\n\n{traceback.format_exc()}") from exc
        if built is None:
            raise RuntimeError("build() returned None")
        if isinstance(built, dict):
            part_solids = _normalize_part_map(built)
            if not part_solids:
                raise RuntimeError("build() returned an empty parts dict")
            assembly = _union_solids(list(part_solids.values()))
        else:
            assembly = built
            part_solids = {"body": built}

    if not part_solids:
        part_solids = {"body": assembly}

    try:
        vertices, faces = solid_to_mesh(assembly)
    except Exception as exc:
        raise RuntimeError(f"Tessellation failed:\n{exc}\n\n{traceback.format_exc()}") from exc

    parts = _tessellate_parts(part_solids)
    return DesignResult(
        code=cleaned,
        solid=assembly,
        vertices=vertices,
        faces=faces,
        parts=parts,
    )


def _safe_part_filename(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name).strip("_") or "part"


def export_step(solid: Any, path: Path) -> Path:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() not in {".step", ".stp"}:
        path = path.with_suffix(".step")
    cq.exporters.export(solid, str(path))
    return path


def export_stl(
    solid: Any,
    path: Path,
    *,
    tolerance: float = 0.1,
    angular_tolerance: float = 0.1,
) -> Path:
    """Export a solid as an STL triangle mesh."""
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() != ".stl":
        path = path.with_suffix(".stl")
    cq.exporters.export(
        solid,
        str(path),
        exportType="STL",
        tolerance=tolerance,
        angularTolerance=angular_tolerance,
    )
    return path


def export_parts(
    result: DesignResult,
    directory: Path,
    *,
    basename: str = "design",
) -> list[Path]:
    """Export each named part as its own STEP file. Returns written paths."""
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, part in result.parts.items():
        safe = _safe_part_filename(name)
        path = directory / f"{basename}_{safe}.step"
        export_step(part.solid, path)
        written.append(path)
    return written


def export_parts_stl(
    result: DesignResult,
    directory: Path,
    *,
    basename: str = "design",
    tolerance: float = 0.1,
    angular_tolerance: float = 0.1,
) -> list[Path]:
    """Export each named part as its own STL mesh. Returns written paths."""
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, part in result.parts.items():
        safe = _safe_part_filename(name)
        path = directory / f"{basename}_{safe}.stl"
        export_stl(
            part.solid,
            path,
            tolerance=tolerance,
            angular_tolerance=angular_tolerance,
        )
        written.append(path)
    return written


def export_parts_stl_zip(
    result: DesignResult,
    zip_path: Path,
    *,
    basename: str = "design",
    folder_name: str | None = None,
    tolerance: float = 0.1,
    angular_tolerance: float = 0.1,
) -> tuple[Path, Path]:
    """
    Write all part STLs into a folder and package them as a zip.

    Returns (folder_path, zip_path).
    """
    import zipfile

    zip_path = zip_path.resolve()
    if zip_path.suffix.lower() != ".zip":
        zip_path = zip_path.with_suffix(".zip")
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    folder = zip_path.parent / (folder_name or f"{zip_path.stem}_stl")
    if folder.exists():
        # Replace previous export of the same name
        for old in folder.glob("*.stl"):
            old.unlink()
    folder.mkdir(parents=True, exist_ok=True)

    stl_paths = export_parts_stl(
        result,
        folder,
        basename=basename,
        tolerance=tolerance,
        angular_tolerance=angular_tolerance,
    )
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in stl_paths:
            zf.write(path, arcname=f"{folder.name}/{path.name}")
    return folder, zip_path


def save_design_script(code: str, path: Path) -> Path:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(code.rstrip() + "\n", encoding="utf-8")
    return path
