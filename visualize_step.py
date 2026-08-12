#!/usr/bin/env python3
"""Visualize a STEP (.step / .stp) CAD file with an interactive 3D viewer.

Loads B-Rep geometry via CadQuery/OCP, tessellates it, and renders with VTK
(interactive window and/or PNG export).

Usage:
    python visualize_step.py models/chair.step
    python visualize_step.py models/chair.step --save preview.png
    python visualize_step.py models/chair.step --no-show --save preview.png
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


def load_step_mesh(
    path: Path,
    linear_deflection: float = 0.4,
    angular_deflection: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a STEP file and return (vertices Nx3, faces Mx3)."""
    import cadquery as cq
    from OCP.BRep import BRep_Tool
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.TopAbs import TopAbs_FACE, TopAbs_Orientation
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS

    shape = cq.importers.importStep(str(path)).val().wrapped
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
            n_tris = triangulation.NbTriangles()

            for i in range(1, n_nodes + 1):
                p = triangulation.Node(i).Transformed(trsf)
                vertices.append([p.X(), p.Y(), p.Z()])

            reversed_face = face.Orientation() == TopAbs_Orientation.TopAbs_REVERSED
            for i in range(1, n_tris + 1):
                tri = triangulation.Triangle(i)
                n1, n2, n3 = tri.Get()
                if reversed_face:
                    faces.append([offset + n1 - 1, offset + n3 - 1, offset + n2 - 1])
                else:
                    faces.append([offset + n1 - 1, offset + n2 - 1, offset + n3 - 1])

            offset += n_nodes

        explorer.Next()

    if not vertices:
        raise RuntimeError(f"No triangulated geometry found in {path}")

    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int32)


def _vtk_polydata(vertices: np.ndarray, faces: np.ndarray):
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(vertices, deep=True))

    # VTK cell array: [3, i, j, k, 3, i, j, k, ...]
    cells = np.empty((len(faces), 4), dtype=np.int64)
    cells[:, 0] = 3
    cells[:, 1:] = faces
    cell_array = vtk.vtkCellArray()
    cell_array.ImportLegacyFormat(numpy_to_vtkIdTypeArray(cells.ravel(), deep=True))

    poly = vtk.vtkPolyData()
    poly.SetPoints(points)
    poly.SetPolys(cell_array)

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(poly)
    normals.ComputePointNormalsOn()
    normals.ComputeCellNormalsOn()
    normals.Update()
    return normals.GetOutput()


def visualize(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    title: str = "STEP preview",
    save_path: Path | None = None,
    show: bool = True,
    elev: float = 18.0,
    azim: float = -55.0,
) -> None:
    """Render a triangle mesh with VTK."""
    import vtk

    poly = _vtk_polydata(vertices, faces)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    prop = actor.GetProperty()
    prop.SetColor(0.72, 0.58, 0.42)  # warm wood tone
    prop.SetSpecular(0.25)
    prop.SetSpecularPower(18)
    prop.SetAmbient(0.35)
    prop.SetDiffuse(0.75)
    prop.EdgeVisibilityOff()

    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(0.96, 0.95, 0.93)
    renderer.SetBackground2(0.88, 0.86, 0.82)
    renderer.GradientBackgroundOn()

    # Camera: three-quarter view framed on the full bbox
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    center = 0.5 * (mins + maxs)
    extent = float(np.linalg.norm(maxs - mins))
    elev_r = np.deg2rad(elev)
    azim_r = np.deg2rad(azim)
    direction = np.array(
        [
            np.cos(elev_r) * np.cos(azim_r),
            np.cos(elev_r) * np.sin(azim_r),
            np.sin(elev_r),
        ],
        dtype=np.float64,
    )
    cam_pos = center + direction * (extent * 1.35)

    camera = renderer.GetActiveCamera()
    camera.SetFocalPoint(*center)
    camera.SetPosition(*cam_pos)
    camera.SetViewUp(0, 0, 1)
    camera.SetViewAngle(30)
    renderer.ResetCameraClippingRange()

    # Prefer offscreen when only saving, or when no display is available
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    use_offscreen = (save_path is not None and not show) or not has_display

    if use_offscreen:
        # OSMesa / EGL offscreen path
        try:
            graphics_factory = vtk.vtkGraphicsFactory()
            graphics_factory.SetOffScreenOnlyMode(1)
            graphics_factory.SetUseMesaClasses(1)
        except Exception:
            pass

    window = vtk.vtkRenderWindow()
    window.AddRenderer(renderer)
    window.SetSize(1100, 860)
    window.SetWindowName(title)
    if use_offscreen or not show:
        window.SetOffScreenRendering(1)

    window.Render()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        writer = vtk.vtkPNGWriter()
        filter_img = vtk.vtkWindowToImageFilter()
        filter_img.SetInput(window)
        filter_img.SetInputBufferTypeToRGB()
        filter_img.ReadFrontBufferOff()
        filter_img.Update()
        writer.SetFileName(str(save_path))
        writer.SetInputConnection(filter_img.GetOutputPort())
        writer.Write()
        print(f"Saved preview → {save_path.resolve()}")

    if show and has_display and not use_offscreen:
        interactor = vtk.vtkRenderWindowInteractor()
        interactor.SetRenderWindow(window)
        style = vtk.vtkInteractorStyleTrackballCamera()
        interactor.SetInteractorStyle(style)
        print("Interactive viewer: drag to rotate, scroll to zoom, 'q' to quit.")
        interactor.Start()
    elif show and not has_display:
        print("No DISPLAY found — skipped interactive window (use --save for PNG).")


def mesh_stats(vertices: np.ndarray, faces: np.ndarray) -> str:
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    size = maxs - mins
    return (
        f"triangles={len(faces):,}  vertices={len(vertices):,}\n"
        f"bbox (mm): X={size[0]:.1f}  Y={size[1]:.1f}  Z={size[2]:.1f}\n"
        f"origin→max corner: ({mins[0]:.1f},{mins[1]:.1f},{mins[2]:.1f}) → "
        f"({maxs[0]:.1f},{maxs[1]:.1f},{maxs[2]:.1f})"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize a STEP CAD file")
    p.add_argument("step_file", type=Path, help="Path to .step / .stp file")
    p.add_argument("--save", type=Path, default=None, help="Save PNG preview to this path")
    p.add_argument("--no-show", action="store_true", help="Do not open interactive window")
    p.add_argument("--deflection", type=float, default=0.4, help="Mesh linear deflection (mm)")
    p.add_argument("--elev", type=float, default=22.0, help="Camera elevation (deg)")
    p.add_argument("--azim", type=float, default=-40.0, help="Camera azimuth (deg)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    path = args.step_file.expanduser().resolve()

    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        return 1
    if path.suffix.lower() not in {".step", ".stp"}:
        print(f"Expected a .step/.stp file, got: {path.suffix}", file=sys.stderr)
        return 1

    print(f"Loading {path} …")
    vertices, faces = load_step_mesh(path, linear_deflection=args.deflection)
    print(mesh_stats(vertices, faces))

    visualize(
        vertices,
        faces,
        title=path.name,
        save_path=args.save,
        show=not args.no_show,
        elev=args.elev,
        azim=args.azim,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
