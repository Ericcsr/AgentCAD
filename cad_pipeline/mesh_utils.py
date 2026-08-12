#!/usr/bin/env python3
"""Shared mesh helpers: CadQuery/OCP solid → triangle mesh."""

from __future__ import annotations

from typing import Any

import numpy as np


def solid_to_mesh(
    solid: Any,
    linear_deflection: float = 0.5,
    angular_deflection: float = 0.4,
) -> tuple[np.ndarray, np.ndarray]:
    """Tessellate a CadQuery Workplane / Shape into (vertices, faces)."""
    from OCP.BRep import BRep_Tool
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.TopAbs import TopAbs_FACE, TopAbs_Orientation
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS

    shape = solid.val().wrapped if hasattr(solid, "val") else solid

    BRepMesh_IncrementalMesh(shape, linear_deflection, False, angular_deflection, True)

    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    offset = 0

    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = TopoDS.Face_s(explorer.Current())
        loc = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation_s(face, loc)
        if triangulation is not None:
            trsf = loc.Transformation()
            n_nodes = triangulation.NbNodes()
            for i in range(1, n_nodes + 1):
                p = triangulation.Node(i).Transformed(trsf)
                vertices.append([p.X(), p.Y(), p.Z()])

            reversed_face = face.Orientation() == TopAbs_Orientation.TopAbs_REVERSED
            for i in range(1, triangulation.NbTriangles() + 1):
                n1, n2, n3 = triangulation.Triangle(i).Get()
                if reversed_face:
                    faces.append([offset + n1 - 1, offset + n3 - 1, offset + n2 - 1])
                else:
                    faces.append([offset + n1 - 1, offset + n2 - 1, offset + n3 - 1])
            offset += n_nodes
        explorer.Next()

    if not vertices:
        raise RuntimeError("No triangulated geometry in solid")

    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int32)


def mesh_bbox_summary(vertices: np.ndarray) -> str:
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    size = maxs - mins
    return f"bbox mm: {size[0]:.0f} × {size[1]:.0f} × {size[2]:.0f}"


def geometry_summary(vertices: np.ndarray, *, triangles: int | None = None) -> str:
    """Richer measured-geometry blurb for Ask mode."""
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    size = maxs - mins
    lines = [
        f"Overall bounding box (mm): width(X)={size[0]:.1f}, depth(Y)={size[1]:.1f}, height(Z)={size[2]:.1f}",
        f"Extents: X[{mins[0]:.1f}, {maxs[0]:.1f}], Y[{mins[1]:.1f}, {maxs[1]:.1f}], Z[{mins[2]:.1f}, {maxs[2]:.1f}]",
    ]
    if triangles is not None:
        lines.append(f"Tessellation: {triangles:,} triangles, {len(vertices):,} vertices")
    return "\n".join(lines)
