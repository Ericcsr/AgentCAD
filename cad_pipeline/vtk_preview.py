#!/usr/bin/env python3
"""Tk-embedded CAD preview: offscreen VTK rendered into a Canvas."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import vtk
from PIL import Image, ImageTk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy

from cad_pipeline.render import RENDER_DIR, VTK_LOCK, RenderResult, resolve_camera


class VtkPreview(tk.Frame):
    """Interactive 3D preview hosted inside Tk (drag orbit, scroll zoom)."""

    def __init__(
        self,
        master: tk.Misc,
        *,
        background: str = "#ebe7df",
        on_key_save: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(master, bg=background, highlightthickness=0, bd=0)
        self.on_key_save = on_key_save
        self._photo: ImageTk.PhotoImage | None = None
        self._drag_last: tuple[int, int] | None = None
        self._pending_redraw = False
        self._last_size: tuple[int, int] | None = None

        self.elev = 22.0
        self.azim = -40.0
        self.distance = 1.0
        self.center = np.zeros(3, dtype=np.float64)
        self._extent = 1.0

        self.canvas = tk.Canvas(self, bg=background, highlightthickness=0, bd=0, cursor="hand2")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", self._on_configure)
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        self.canvas.bind("<Button-4>", lambda e: self._zoom(1.1))
        self.canvas.bind("<Button-5>", lambda e: self._zoom(0.9))
        self.canvas.bind("<Control-s>", lambda _e: self._save_key())
        self.canvas.bind("<Control-S>", lambda _e: self._save_key())
        self.canvas.focus_set()

        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.95, 0.94, 0.91)
        self.renderer.SetBackground2(0.88, 0.85, 0.80)
        self.renderer.GradientBackgroundOn()

        self.render_window = vtk.vtkRenderWindow()
        self.render_window.SetOffScreenRendering(1)
        self.render_window.SetMultiSamples(0)
        self.render_window.AddRenderer(self.renderer)
        self.render_window.SetSize(640, 480)
        self._last_size = (640, 480)

        self.mapper = vtk.vtkPolyDataMapper()
        self.actor = vtk.vtkActor()
        self.actor.SetMapper(self.mapper)
        prop = self.actor.GetProperty()
        prop.SetColor(0.72, 0.58, 0.42)
        prop.SetSpecular(0.22)
        prop.SetSpecularPower(18)
        prop.SetAmbient(0.32)
        prop.SetDiffuse(0.78)
        self.renderer.AddActor(self.actor)

    def _save_key(self) -> str:
        if self.on_key_save:
            self.on_key_save()
        return "break"

    def set_mesh(self, vertices: np.ndarray, faces: np.ndarray, *, reset_camera: bool) -> None:
        with VTK_LOCK:
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
            self.mapper.SetInputData(normals.GetOutput())

            if reset_camera:
                mins = vertices.min(axis=0)
                maxs = vertices.max(axis=0)
                self.center = 0.5 * (mins + maxs)
                self._extent = float(max(np.linalg.norm(maxs - mins), 1.0))
                self.distance = self._extent * 1.55
                self.elev = 22.0
                self.azim = -40.0
        self.redraw(immediate=True)

    def _apply_camera(self) -> None:
        elev_r, azim_r = np.deg2rad(self.elev), np.deg2rad(self.azim)
        direction = np.array(
            [
                np.cos(elev_r) * np.cos(azim_r),
                np.cos(elev_r) * np.sin(azim_r),
                np.sin(elev_r),
            ],
            dtype=np.float64,
        )
        cam = self.renderer.GetActiveCamera()
        cam.SetFocalPoint(*self.center)
        cam.SetPosition(*(self.center + direction * self.distance))
        cam.SetViewUp(0, 0, 1)
        cam.SetViewAngle(30)
        self.renderer.ResetCameraClippingRange()

    def camera_angles(self) -> tuple[float, float]:
        return self.elev, self.azim

    def _ensure_size(self, w: int, h: int) -> None:
        size = (max(int(w), 2), max(int(h), 2))
        if self._last_size == size:
            return
        self.render_window.SetSize(*size)
        self._last_size = size

    def redraw(self, *, immediate: bool = False) -> None:
        if immediate:
            self._pending_redraw = False
            self._do_redraw()
            return
        if self._pending_redraw:
            return
        self._pending_redraw = True
        self.after(16, self._flush_redraw)

    def _flush_redraw(self) -> None:
        self._pending_redraw = False
        self._do_redraw()

    def _do_redraw(self) -> None:
        with VTK_LOCK:
            w = max(self.canvas.winfo_width(), 2)
            h = max(self.canvas.winfo_height(), 2)
            self._ensure_size(w, h)
            self._apply_camera()
            try:
                self.render_window.MakeCurrent()
            except Exception:
                pass
            self.render_window.Render()

            filt = vtk.vtkWindowToImageFilter()
            filt.SetInput(self.render_window)
            filt.SetInputBufferTypeToRGB()
            filt.ReadFrontBufferOff()
            filt.Update()
            vtk_image = filt.GetOutput()
            dims = vtk_image.GetDimensions()
            if dims[0] < 2 or dims[1] < 2:
                return
            arr = vtk_to_numpy(vtk_image.GetPointData().GetScalars())
            arr = arr.reshape(dims[1], dims[0], 3)
            arr = np.flipud(arr)
            image = Image.fromarray(arr.astype(np.uint8), mode="RGB")
            self._photo = ImageTk.PhotoImage(image)
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self._photo)

    def capture_png(self, path: Path) -> Path:
        path = path.resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        with VTK_LOCK:
            w = max(self.canvas.winfo_width(), 640)
            h = max(self.canvas.winfo_height(), 480)
            self._ensure_size(w, h)
            self._apply_camera()
            try:
                self.render_window.MakeCurrent()
            except Exception:
                pass
            self.render_window.Render()
            filt = vtk.vtkWindowToImageFilter()
            filt.SetInput(self.render_window)
            filt.SetInputBufferTypeToRGB()
            filt.ReadFrontBufferOff()
            filt.Update()
            writer = vtk.vtkPNGWriter()
            writer.SetFileName(str(path))
            writer.SetInputConnection(filt.GetOutputPort())
            writer.Write()
        return path

    def render_view(
        self,
        *,
        view: str | None = None,
        elev: float | None = None,
        azim: float | None = None,
        path: Path | None = None,
    ) -> RenderResult:
        """Render a named/custom camera angle without permanently moving the UI camera."""
        label, e, a = resolve_camera(view, elev, azim)
        out = path or (RENDER_DIR / f"{label}.png")
        out = out.resolve()
        out.parent.mkdir(parents=True, exist_ok=True)

        saved = (self.elev, self.azim, self.distance)
        try:
            self.elev, self.azim = e, a
            self.distance = self._extent * 1.55
            self.capture_png(out)
        finally:
            self.elev, self.azim, self.distance = saved
            self.redraw(immediate=True)
        return RenderResult(path=out, view=label, elev=e, azim=a)

    def render_views(self, views: Iterable[str] | None = None) -> list[RenderResult]:
        selected = list(views) if views is not None else ["isometric", "front", "side"]
        return [self.render_view(view=v) for v in selected]

    def close(self) -> None:
        with VTK_LOCK:
            try:
                self.render_window.Finalize()
            except Exception:
                pass

    # --- interaction --------------------------------------------------------

    def _on_configure(self, _event: tk.Event) -> None:
        self.redraw()

    def _on_press(self, event: tk.Event) -> None:
        self._drag_last = (event.x, event.y)
        self.canvas.focus_set()

    def _on_drag(self, event: tk.Event) -> None:
        if self._drag_last is None:
            return
        dx = event.x - self._drag_last[0]
        dy = event.y - self._drag_last[1]
        self._drag_last = (event.x, event.y)
        self.azim -= dx * 0.35
        self.elev = float(np.clip(self.elev + dy * 0.35, -89.0, 89.0))
        self.redraw()

    def _on_release(self, _event: tk.Event) -> None:
        self._drag_last = None

    def _on_wheel(self, event: tk.Event) -> None:
        if getattr(event, "delta", 0) > 0:
            self._zoom(0.9)
        elif getattr(event, "delta", 0) < 0:
            self._zoom(1.1)

    def _zoom(self, factor: float) -> None:
        self.distance = float(np.clip(self.distance * factor, self._extent * 0.3, self._extent * 12.0))
        self.redraw()
