#!/usr/bin/env python3
"""Offscreen RGB rendering of triangle meshes with named camera angles."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RENDER_DIR = ROOT / "generated" / "renders"

# Serialize all VTK OpenGL access — multiple windows/contexts cause VAO errors.
VTK_LOCK = threading.RLock()

VIEW_PRESETS: dict[str, tuple[float, float]] = {
    "isometric": (22.0, -40.0),
    "front": (8.0, -90.0),
    "back": (8.0, 90.0),
    "side": (8.0, 0.0),
    "left": (8.0, 180.0),
    "top": (88.0, -40.0),
    "bottom": (-88.0, -40.0),
    "three_quarter": (28.0, -55.0),
}


@dataclass
class RenderResult:
    path: Path
    view: str
    elev: float
    azim: float


def list_views() -> list[str]:
    return sorted(VIEW_PRESETS.keys())


def resolve_camera(
    view: str | None = None,
    elev: float | None = None,
    azim: float | None = None,
) -> tuple[str, float, float]:
    if elev is not None and azim is not None:
        label = view or f"custom_e{elev:g}_a{azim:g}"
        return label, float(elev), float(azim)
    name = (view or "isometric").strip().lower().replace("-", "_").replace(" ", "_")
    if name not in VIEW_PRESETS:
        known = ", ".join(list_views())
        raise ValueError(f"Unknown view {view!r}. Known: {known}, or pass elev+azim.")
    e, a = VIEW_PRESETS[name]
    return name, e, a


class MeshRenderer:
    """Standalone offscreen VTK renderer (fallback when no UI preview exists)."""

    def __init__(self, size: tuple[int, int] = (1024, 768)) -> None:
        self.size = size
        self._window = None
        self._renderer = None
        self._mapper = None
        self._actor = None
        self._last_size: tuple[int, int] | None = None

    def _ensure_pipeline(self) -> None:
        if self._window is not None:
            return
        import vtk

        self._renderer = vtk.vtkRenderer()
        self._renderer.SetBackground(0.95, 0.94, 0.91)
        self._renderer.SetBackground2(0.88, 0.85, 0.80)
        self._renderer.GradientBackgroundOn()

        self._window = vtk.vtkRenderWindow()
        self._window.SetOffScreenRendering(1)
        self._window.SetMultiSamples(0)
        self._window.AddRenderer(self._renderer)
        self._window.SetSize(*self.size)
        self._last_size = self.size

        self._mapper = vtk.vtkPolyDataMapper()
        self._actor = vtk.vtkActor()
        self._actor.SetMapper(self._mapper)
        prop = self._actor.GetProperty()
        prop.SetColor(0.72, 0.58, 0.42)
        prop.SetSpecular(0.22)
        prop.SetSpecularPower(18)
        prop.SetAmbient(0.32)
        prop.SetDiffuse(0.78)
        self._renderer.AddActor(self._actor)

    def _set_size(self, size: tuple[int, int]) -> None:
        if self._last_size == size:
            return
        self.size = size
        self._window.SetSize(*size)
        self._last_size = size

    def _set_mesh(self, vertices: np.ndarray, faces: np.ndarray) -> None:
        import vtk
        from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

        self._ensure_pipeline()
        points = vtk.vtkPoints()
        points.SetData(numpy_to_vtk(np.asarray(vertices, dtype=np.float64), deep=True))
        faces_arr = np.asarray(faces, dtype=np.int64)
        cells = np.empty((len(faces_arr), 4), dtype=np.int64)
        cells[:, 0] = 3
        cells[:, 1:] = faces_arr
        cell_array = vtk.vtkCellArray()
        cell_array.ImportLegacyFormat(numpy_to_vtkIdTypeArray(cells.ravel(), deep=True))
        poly = vtk.vtkPolyData()
        poly.SetPoints(points)
        poly.SetPolys(cell_array)
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputData(poly)
        normals.ComputePointNormalsOn()
        normals.Update()
        self._mapper.SetInputData(normals.GetOutput())

    def _aim_camera(self, vertices: np.ndarray, elev: float, azim: float) -> None:
        mins = vertices.min(axis=0)
        maxs = vertices.max(axis=0)
        center = 0.5 * (mins + maxs)
        extent = float(np.linalg.norm(maxs - mins))
        elev_r, azim_r = np.deg2rad(elev), np.deg2rad(azim)
        direction = np.array(
            [
                np.cos(elev_r) * np.cos(azim_r),
                np.cos(elev_r) * np.sin(azim_r),
                np.sin(elev_r),
            ],
            dtype=np.float64,
        )
        cam = self._renderer.GetActiveCamera()
        cam.SetFocalPoint(*center)
        cam.SetPosition(*(center + direction * max(extent, 1.0) * 1.55))
        cam.SetViewUp(0, 0, 1)
        cam.SetViewAngle(30)
        self._renderer.ResetCameraClippingRange()
        cam.Zoom(0.92)

    def render(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        *,
        view: str | None = "isometric",
        elev: float | None = None,
        azim: float | None = None,
        path: Path | None = None,
        size: tuple[int, int] | None = None,
    ) -> RenderResult:
        import vtk

        with VTK_LOCK:
            label, e, a = resolve_camera(view, elev, azim)
            self._ensure_pipeline()
            if size is not None:
                self._set_size(size)
            elif self._last_size != self.size:
                self._set_size(self.size)

            self._set_mesh(vertices, faces)
            self._aim_camera(np.asarray(vertices, dtype=np.float64), e, a)
            try:
                self._window.MakeCurrent()
            except Exception:
                pass
            self._window.Render()

            RENDER_DIR.mkdir(parents=True, exist_ok=True)
            out = path or (RENDER_DIR / f"{label}.png")
            out = out.resolve()
            out.parent.mkdir(parents=True, exist_ok=True)

            filt = vtk.vtkWindowToImageFilter()
            filt.SetInput(self._window)
            filt.SetInputBufferTypeToRGB()
            filt.ReadFrontBufferOff()
            filt.Update()
            writer = vtk.vtkPNGWriter()
            writer.SetFileName(str(out))
            writer.SetInputConnection(filt.GetOutputPort())
            writer.Write()
            return RenderResult(path=out, view=label, elev=e, azim=a)

    def render_views(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        views: Iterable[str] | None = None,
    ) -> list[RenderResult]:
        selected = list(views) if views is not None else ["isometric", "front", "side"]
        return [self.render(vertices, faces, view=view) for view in selected]

    def close(self) -> None:
        with VTK_LOCK:
            if self._window is not None:
                try:
                    self._window.Finalize()
                except Exception:
                    pass
            self._window = None
            self._renderer = None
            self._mapper = None
            self._actor = None
            self._last_size = None
