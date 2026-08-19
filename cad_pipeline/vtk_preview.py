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
        self._ref_actors: list[vtk.vtkActor] = []
        self._ref_bounds: list[tuple[np.ndarray, np.ndarray]] = []
        self._part_actors: list[vtk.vtkActor] = []
        self._part_bounds: list[tuple[np.ndarray, np.ndarray]] = []
        self._part_actor_names: list[str] = []
        self._part_actors_by_name: dict[str, vtk.vtkActor] = {}

    def _save_key(self) -> str:
        if self.on_key_save:
            self.on_key_save()
        return "break"

    def _polydata(self, vertices: np.ndarray, faces: np.ndarray) -> vtk.vtkPolyData:
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
        return normals.GetOutput()

    def _style_actor(self, actor: vtk.vtkActor, color: tuple[float, float, float]) -> None:
        prop = actor.GetProperty()
        prop.SetColor(*color)
        prop.SetSpecular(0.22)
        prop.SetSpecularPower(18)
        prop.SetAmbient(0.32)
        prop.SetDiffuse(0.78)
        prop.EdgeVisibilityOn()
        prop.SetEdgeColor(
            max(0.0, color[0] * 0.45),
            max(0.0, color[1] * 0.45),
            max(0.0, color[2] * 0.45),
        )
        prop.SetLineWidth(1.0)

    def _clear_part_actors(self) -> None:
        for actor in self._part_actors:
            self.renderer.RemoveActor(actor)
        self._part_actors = []
        self._part_bounds = []
        self._part_actor_names = []
        self._part_actors_by_name = {}

    def set_mesh(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        *,
        reset_camera: bool,
        color: tuple[float, float, float] | None = None,
    ) -> None:
        rgb = color or (0.72, 0.58, 0.42)
        with VTK_LOCK:
            self._clear_part_actors()
            self.mapper.SetInputData(self._polydata(vertices, faces))
            self._style_actor(self.actor, rgb)
            self.actor.SetVisibility(True)
            if reset_camera:
                mins, maxs = self._combined_bounds(vertices)
                self.center = 0.5 * (mins + maxs)
                self._extent = float(max(np.linalg.norm(maxs - mins), 1.0))
                self.distance = self._extent * 1.55
                self.elev = 22.0
                self.azim = -40.0
        self.redraw(immediate=True)

    def set_part_meshes(
        self,
        meshes: Iterable[
            tuple[str, np.ndarray, np.ndarray, tuple[float, float, float]]
            | tuple[np.ndarray, np.ndarray, tuple[float, float, float]]
        ],
        *,
        reset_camera: bool = False,
    ) -> None:
        """Draw each named part as its own colored actor (whole-design / URDF view)."""
        with VTK_LOCK:
            self._clear_part_actors()
            self.actor.SetVisibility(False)
            all_pts: list[np.ndarray] = []
            for item in meshes:
                if len(item) == 4:
                    name, vertices, faces, color = item
                else:
                    vertices, faces, color = item
                    name = f"part_{len(self._part_actors)}"
                verts = np.asarray(vertices, dtype=np.float64)
                faces_arr = np.asarray(faces, dtype=np.int64)
                if len(verts) == 0 or len(faces_arr) == 0:
                    continue
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputData(self._polydata(verts, faces_arr))
                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                self._style_actor(actor, color)
                self.renderer.AddActor(actor)
                self._part_actors.append(actor)
                self._part_actor_names.append(str(name))
                self._part_actors_by_name[str(name)] = actor
                self._part_bounds.append((verts.min(axis=0), verts.max(axis=0)))
                all_pts.append(verts)
            if reset_camera and all_pts:
                stacked = np.concatenate(all_pts, axis=0)
                mins, maxs = self._combined_bounds(stacked)
                self.center = 0.5 * (mins + maxs)
                self._extent = float(max(np.linalg.norm(maxs - mins), 1.0))
                self.distance = self._extent * 1.55
                self.elev = 22.0
                self.azim = -40.0
        self.redraw(immediate=True)

    def set_part_transforms(
        self,
        transforms: dict[str, np.ndarray] | None,
        *,
        redraw: bool = True,
    ) -> None:
        """Apply 4×4 world mm poses to named part actors (identity if omitted)."""
        with VTK_LOCK:
            for name, actor in self._part_actors_by_name.items():
                mat = None if not transforms else transforms.get(name)
                if mat is None:
                    actor.SetUserTransform(None)
                    continue
                vtk_mat = vtk.vtkMatrix4x4()
                arr = np.asarray(mat, dtype=np.float64)
                for i in range(4):
                    for j in range(4):
                        vtk_mat.SetElement(i, j, float(arr[i, j]))
                xf = vtk.vtkTransform()
                xf.SetMatrix(vtk_mat)
                actor.SetUserTransform(xf)
        if redraw:
            self.redraw(immediate=True)

    def set_reference_meshes(
        self,
        meshes: Iterable[tuple[np.ndarray, np.ndarray]],
        *,
        reset_camera: bool = False,
    ) -> None:
        """Show imported STEP solids as a translucent overlay (constraint context)."""
        with VTK_LOCK:
            for actor in self._ref_actors:
                self.renderer.RemoveActor(actor)
            self._ref_actors = []
            self._ref_bounds = []
            for vertices, faces in meshes:
                verts = np.asarray(vertices, dtype=np.float64)
                faces_arr = np.asarray(faces, dtype=np.int64)
                if len(verts) == 0 or len(faces_arr) == 0:
                    continue
                points = vtk.vtkPoints()
                points.SetData(numpy_to_vtk(verts, deep=True))
                cells = np.empty((len(faces_arr), 4), dtype=np.int64)
                cells[:, 0] = 3
                cells[:, 1:] = faces_arr
                cell_array = vtk.vtkCellArray()
                cell_array.ImportLegacyFormat(numpy_to_vtkIdTypeArray(cells.ravel(), deep=True))
                poly = vtk.vtkPolyData()
                poly.SetPoints(points)
                poly.SetPolys(cell_array)
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputData(poly)
                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                prop = actor.GetProperty()
                prop.SetColor(0.28, 0.45, 0.62)
                prop.SetOpacity(0.28)
                prop.SetAmbient(0.35)
                prop.SetDiffuse(0.7)
                prop.EdgeVisibilityOn()
                prop.SetEdgeColor(0.18, 0.32, 0.48)
                prop.SetLineWidth(1.0)
                self.renderer.AddActor(actor)
                self._ref_actors.append(actor)
                self._ref_bounds.append((verts.min(axis=0), verts.max(axis=0)))
            if reset_camera and self.mapper.GetInput() is not None:
                try:
                    pdata = self.mapper.GetInput()
                    bounds = pdata.GetBounds()
                    design = np.array(
                        [[bounds[0], bounds[2], bounds[4]], [bounds[1], bounds[3], bounds[5]]],
                        dtype=np.float64,
                    )
                    mins, maxs = self._combined_bounds(design)
                    self.center = 0.5 * (mins + maxs)
                    self._extent = float(max(np.linalg.norm(maxs - mins), 1.0))
                    self.distance = self._extent * 1.55
                except Exception:
                    pass
        self.redraw(immediate=True)

    def _combined_bounds(self, vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mins = np.asarray(vertices, dtype=np.float64).min(axis=0)
        maxs = np.asarray(vertices, dtype=np.float64).max(axis=0)
        for bmin, bmax in self._part_bounds:
            mins = np.minimum(mins, bmin)
            maxs = np.maximum(maxs, bmax)
        for bmin, bmax in self._ref_bounds:
            mins = np.minimum(mins, bmin)
            maxs = np.maximum(maxs, bmax)
        return mins, maxs

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
