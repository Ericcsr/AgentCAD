#!/usr/bin/env python3
"""Interactive design studio: merged 3D preview + Agent/Ask chat in one window."""

from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Callable

from cad_pipeline.agent import DesignAgent
from cad_pipeline.mesh_utils import geometry_summary, mesh_bbox_summary
from cad_pipeline.render import RENDER_DIR
from cad_pipeline.runtime import (
    WHOLE_DESIGN,
    DesignResult,
    export_parts,
    export_parts_stl_zip,
    export_step,
    export_stl,
    save_design_script,
)
from cad_pipeline.ui_dialogs import ask_save_name
from cad_pipeline.ui_scale import (
    apply_scaling,
    colors,
    fonts,
    place_window,
    scaled,
    studio_default_size,
)
from cad_pipeline.vtk_preview import VtkPreview
from cad_pipeline.versioning import VersionMeta

MODE_AGENT = "agent"
MODE_ASK = "ask"


class DesignStudio:
    """Single-window studio: 3D preview (left) + language chat (right)."""

    def __init__(
        self,
        agent: DesignAgent,
        initial: DesignResult,
        *,
        models_dir: Path,
        on_status: Callable[[str], None] | None = None,
    ) -> None:
        self.agent = agent
        self.result = initial
        self.models_dir = models_dir
        self.on_status = on_status or (lambda _s: None)
        self._busy = False
        self._closed = False
        self.mode = MODE_AGENT
        self.view_scope = WHOLE_DESIGN
        self.edit_scope = WHOLE_DESIGN

        self.root = tk.Tk()
        apply_scaling(self.root)
        c = colors(self.root)
        self.root.title("AI CAD Studio")
        self.root.configure(bg=c["bg"])
        win_w, win_h, min_w, min_h = studio_default_size(self.root)
        place_window(
            self.root,
            width=win_w,
            height=win_h,
            x=scaled(self.root, 24),
            y=scaled(self.root, 24),
            min_width=min_w,
            min_height=min_h,
        )

        self._build_layout()
        self.root.update_idletasks()
        # Re-apply after layout so minsize/geometry match real chrome
        win_w, win_h, min_w, min_h = studio_default_size(self.root)
        need_w = max(win_w, self.root.winfo_reqwidth() + scaled(self.root, 24))
        place_window(
            self.root,
            width=need_w,
            height=win_h,
            x=scaled(self.root, 24),
            y=scaled(self.root, 24),
            min_width=min_w,
            min_height=min_h,
        )

        self.preview.set_mesh(initial.vertices, initial.faces, reset_camera=True)
        self.agent.set_mesh(initial.vertices, initial.faces)
        # One OpenGL context only — agent renders go through the preview on the UI thread
        self.agent.set_host_renderer(self.preview.render_view)
        self.agent.set_ui_marshal(lambda fn: self.root.after(0, fn))
        self._refresh_parts_ui()
        if self.result.part_names():
            self._log_system(
                "Parts: " + ", ".join(self.result.part_names())
                + " — use View to isolate a part; Edit scope to target Agent revisions."
            )
        self._log_system(
            "Design ready. Drag in the 3D view to orbit · scroll to zoom · "
            "Agent revises · Ask answers · Ctrl+S saves.\n"
            f"Mode: {self.agent.mode_label()} — "
            + (
                "stops once CAD builds and renders (fast iteration)."
                if self.agent.is_draft_mode()
                else "checks every key feature / physics constraint after each build."
            )
        )
        self._log_system(mesh_bbox_summary(initial.vertices))
        if self.agent.features:
            self._log_system(self.agent.features_summary())
        self._log_system(
            "Context worksheet: generated/context_worksheet.md "
            "(auto-recovers a new agent after Cursor API / context crashes)."
        )
        self._commit_version(
            initial,
            label="Initial design",
            note="Studio opened with first compilable design",
        )
        self._refresh_version_ui()

        self.root.bind_all("<Control-s>", self._on_save)
        self.root.bind_all("<Control-S>", self._on_save)
        self.root.bind_all("<Control-z>", self._on_rollback_shortcut)
        self.root.bind_all("<Control-Z>", self._on_rollback_shortcut)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # --- layout -------------------------------------------------------------

    def _build_layout(self) -> None:
        f = fonts(self.root)
        c = colors(self.root)
        pad = scaled(self.root, 14)

        outer = ttk.Frame(self.root, padding=pad)
        outer.pack(fill=tk.BOTH, expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(1, weight=1)

        # Header — two rows so Mode/History controls aren't clipped on narrower widths
        header = ttk.Frame(outer)
        header.grid(row=0, column=0, sticky="ew", pady=(0, scaled(self.root, 10)))
        header.columnconfigure(1, weight=1)

        title_row = ttk.Frame(header)
        title_row.grid(row=0, column=0, columnspan=2, sticky="ew")
        title_row.columnconfigure(1, weight=1)
        ttk.Label(title_row, text="AI CAD", style="Brand.TLabel", font=f["brand"]).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(title_row, text="Studio", style="Title.TLabel", font=f["title"]).grid(
            row=0, column=1, sticky="w", padx=(scaled(self.root, 8), 0)
        )
        self.status = ttk.Label(title_row, text="Agent · Ready", style="Status.TLabel")
        self.status.grid(row=0, column=2, sticky="e")

        controls = ttk.Frame(header)
        controls.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(scaled(self.root, 8), 0))
        controls.columnconfigure(1, weight=1)

        mode_wrap = ttk.Frame(controls)
        mode_wrap.grid(row=0, column=0, sticky="w")
        ttk.Label(mode_wrap, text="Mode", style="Hint.TLabel").pack(
            side=tk.LEFT, padx=(0, scaled(self.root, 6))
        )
        self.mode_var = tk.StringVar(
            value="draft" if self.agent.is_draft_mode() else "refine"
        )
        self.draft_btn = ttk.Radiobutton(
            mode_wrap,
            text="Drafting",
            value="draft",
            variable=self.mode_var,
            command=self._on_mode_toggle,
        )
        self.draft_btn.pack(side=tk.LEFT, padx=(0, scaled(self.root, 4)))
        self.refine_btn = ttk.Radiobutton(
            mode_wrap,
            text="Refinement",
            value="refine",
            variable=self.mode_var,
            command=self._on_mode_toggle,
        )
        self.refine_btn.pack(side=tk.LEFT)

        hist_wrap = ttk.Frame(controls)
        hist_wrap.grid(row=0, column=1, sticky="e")
        ttk.Label(hist_wrap, text="History", style="Hint.TLabel").pack(
            side=tk.LEFT, padx=(0, scaled(self.root, 6))
        )
        self.version_var = tk.StringVar(value="")
        self.version_combo = ttk.Combobox(
            hist_wrap,
            textvariable=self.version_var,
            state="readonly",
            width=36,
        )
        self.version_combo.pack(side=tk.LEFT, padx=(0, scaled(self.root, 6)))
        self.rollback_btn = ttk.Button(hist_wrap, text="Rollback", command=self._on_rollback)
        self.rollback_btn.pack(side=tk.LEFT)

        self._version_labels: list[str] = []
        self._version_ids: list[str] = []

        # Split: 3D | chat
        split = ttk.Panedwindow(outer, orient=tk.HORIZONTAL)
        split.grid(row=1, column=0, sticky="nsew")

        left = ttk.Frame(split, padding=(0, 0, scaled(self.root, 8), 0))
        right = ttk.Frame(split, padding=(scaled(self.root, 8), 0, 0, 0))
        split.add(left, weight=3)
        split.add(right, weight=2)

        left.columnconfigure(0, weight=1)
        left.rowconfigure(2, weight=1)

        ttk.Label(left, text="3D preview · drag to orbit · scroll to zoom", style="Hint.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, scaled(self.root, 6))
        )

        view_row = ttk.Frame(left)
        view_row.grid(row=1, column=0, sticky="ew", pady=(0, scaled(self.root, 6)))
        ttk.Label(view_row, text="View", style="Hint.TLabel").pack(
            side=tk.LEFT, padx=(0, scaled(self.root, 6))
        )
        self.view_scope_var = tk.StringVar(value="Whole design")
        self.view_scope_combo = ttk.Combobox(
            view_row,
            textvariable=self.view_scope_var,
            state="readonly",
            width=22,
        )
        self.view_scope_combo.pack(side=tk.LEFT)
        self.view_scope_combo.bind("<<ComboboxSelected>>", self._on_view_scope_changed)
        ttk.Button(view_row, text="Export part STEP", command=self._on_export_part).pack(
            side=tk.LEFT, padx=(scaled(self.root, 8), 0)
        )
        ttk.Button(view_row, text="Export part STL", command=self._on_export_part_stl).pack(
            side=tk.LEFT, padx=(scaled(self.root, 6), 0)
        )
        ttk.Button(view_row, text="Export all STEP", command=self._on_export_all_parts).pack(
            side=tk.LEFT, padx=(scaled(self.root, 6), 0)
        )
        ttk.Button(view_row, text="ZIP all STLs", command=self._on_export_all_stl_zip).pack(
            side=tk.LEFT, padx=(scaled(self.root, 6), 0)
        )

        preview_border = tk.Frame(left, bg=c["border"], bd=0, highlightthickness=0)
        preview_border.grid(row=2, column=0, sticky="nsew")
        self.preview = VtkPreview(
            preview_border,
            background=c["surface"],
            on_key_save=lambda: self._on_save(),
        )
        self.preview.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)

        # Right chat column
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)
        right.rowconfigure(1, weight=0)

        chat_wrap = tk.Frame(right, bg=c["border"], highlightthickness=0, bd=0)
        chat_wrap.grid(row=0, column=0, sticky="nsew")
        chat_inner = tk.Frame(chat_wrap, bg=c["surface"], bd=0)
        chat_inner.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)
        chat_inner.rowconfigure(0, weight=1)
        chat_inner.columnconfigure(0, weight=1)

        self.log = tk.Text(
            chat_inner,
            wrap=tk.WORD,
            height=8,
            state=tk.DISABLED,
            font=f["chat"],
            background=c["surface"],
            foreground=c["ink"],
            relief=tk.FLAT,
            borderwidth=0,
            highlightthickness=0,
            spacing1=3,
            spacing3=8,
            padx=scaled(self.root, 12),
            pady=scaled(self.root, 10),
            cursor="arrow",
        )
        self.log.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(chat_inner, orient=tk.VERTICAL, command=self.log.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.log.configure(yscrollcommand=scroll.set)
        self._configure_log_tags()

        composer = ttk.Frame(right)
        composer.grid(row=1, column=0, sticky="ew", pady=(scaled(self.root, 10), 0))
        composer.columnconfigure(0, weight=1)
        composer.rowconfigure(0, weight=1)
        # Tall enough for notebook chrome + edit scope + prompt + buttons on HiDPI
        composer.configure(height=scaled(self.root, 340))
        composer.grid_propagate(False)

        self.notebook = ttk.Notebook(composer)
        self.notebook.grid(row=0, column=0, sticky="nsew")
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        tab_pad = scaled(self.root, 10)
        self.agent_tab = ttk.Frame(self.notebook, padding=tab_pad)
        self.ask_tab = ttk.Frame(self.notebook, padding=tab_pad)
        self.notebook.add(self.agent_tab, text="Agent")
        self.notebook.add(self.ask_tab, text="Ask")

        self._build_composer_tab(
            self.agent_tab,
            hint="Describe a change to the model",
            placeholder="make the backrest taller",
            primary_label="Apply",
            prompt_attr="agent_prompt",
            button_attr="agent_btn",
            with_edit_scope=True,
        )
        self._build_composer_tab(
            self.ask_tab,
            hint="Ask about dimensions, parts, materials…",
            placeholder="What are the overall dimensions?",
            primary_label="Ask",
            prompt_attr="ask_prompt",
            button_attr="ask_btn",
            with_edit_scope=False,
        )

        ttk.Label(
            right,
            text="Ctrl+Enter to send · Ctrl+S to save STEP",
            style="Hint.TLabel",
            font=f["small"],
        ).grid(row=2, column=0, sticky="w", pady=(scaled(self.root, 8), 0))

        self.prompt = self.agent_prompt

        # Set a sensible initial sash position after widgets map
        def _place_sash() -> None:
            try:
                total = split.winfo_width()
                if total > 100:
                    split.sashpos(0, int(total * 0.58))
            except tk.TclError:
                pass

        self.root.after(50, _place_sash)

    def _build_composer_tab(
        self,
        tab: ttk.Frame,
        *,
        hint: str,
        placeholder: str,
        primary_label: str,
        prompt_attr: str,
        button_attr: str,
        with_edit_scope: bool = False,
    ) -> None:
        f = fonts(self.root)
        c = colors(self.root)
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(2 if with_edit_scope else 1, weight=1)

        row_i = 0
        ttk.Label(tab, text=hint, style="Hint.TLabel", font=f["small"]).grid(
            row=row_i, column=0, sticky="w"
        )
        row_i += 1

        if with_edit_scope:
            scope_row = ttk.Frame(tab)
            scope_row.grid(row=row_i, column=0, sticky="ew", pady=(scaled(self.root, 4), 0))
            ttk.Label(scope_row, text="Edit scope", style="Hint.TLabel").pack(
                side=tk.LEFT, padx=(0, scaled(self.root, 6))
            )
            self.edit_scope_var = tk.StringVar(value="Whole design")
            self.edit_scope_combo = ttk.Combobox(
                scope_row,
                textvariable=self.edit_scope_var,
                state="readonly",
                width=22,
            )
            self.edit_scope_combo.pack(side=tk.LEFT)
            self.edit_scope_combo.bind("<<ComboboxSelected>>", self._on_edit_scope_changed)
            row_i += 1

        box = tk.Frame(tab, bg=c["border"], bd=0, height=scaled(self.root, 110))
        box.grid(
            row=row_i,
            column=0,
            sticky="nsew",
            pady=(scaled(self.root, 6), scaled(self.root, 8)),
        )
        box.pack_propagate(False)
        prompt = tk.Text(
            box,
            height=4,
            wrap=tk.WORD,
            font=f["body"],
            undo=True,
            background=c["surface"],
            foreground=c["ink"],
            insertbackground=c["ink"],
            relief=tk.FLAT,
            borderwidth=0,
            highlightthickness=0,
            padx=scaled(self.root, 10),
            pady=scaled(self.root, 8),
        )
        prompt.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)
        prompt.insert("1.0", placeholder)
        prompt.bind("<Control-Return>", lambda _e: self._on_submit())
        setattr(self, prompt_attr, prompt)
        row_i += 1

        # Buttons stay on a non-shrinking row below the prompt
        row = ttk.Frame(tab)
        row.grid(row=row_i, column=0, sticky="ew")
        primary = ttk.Button(row, text=primary_label, style="Accent.TButton", command=self._on_submit)
        primary.pack(side=tk.LEFT)
        setattr(self, button_attr, primary)
        ttk.Button(row, text="Save STEP", command=self._on_save).pack(
            side=tk.LEFT, padx=(scaled(self.root, 8), 0)
        )

    def _configure_log_tags(self) -> None:
        f = fonts(self.root)
        c = colors(self.root)
        family = f["family"]
        size = f["chat"][1]
        self.log.tag_configure("user", foreground=c["user"], font=(family, size, "bold"))
        self.log.tag_configure("assistant", foreground=c["assistant"], font=(family, size))
        self.log.tag_configure("ask", foreground=c["ask"], font=(family, size))
        self.log.tag_configure("system", foreground=c["system"], font=(family, max(10, size - 1)))
        self.log.tag_configure("error", foreground=c["error"], font=(family, size))
        self.log.tag_configure("gap", spacing3=scaled(self.root, 10))

    def _on_mode_toggle(self) -> None:
        mode = self.mode_var.get()
        self.agent.set_design_mode(mode)
        if self.agent.is_draft_mode():
            self._log_system(
                "Mode → Drafting: Agent stops once CAD compiles and renders "
                "(no feature / physics review loop)."
            )
        else:
            self._log_system(
                "Mode → Refinement: Agent checks each key feature and physics "
                "constraint after every successful build."
            )
        label = "Ask" if self.mode == MODE_ASK else "Agent"
        self.status.configure(text=f"{label} · {self.agent.mode_label()}")

    def _on_tab_changed(self, _event: object | None = None) -> None:
        tab_text = self.notebook.tab(self.notebook.select(), "text").strip()
        if tab_text == "Ask":
            self.mode = MODE_ASK
            self.prompt = self.ask_prompt
            self.status.configure(text=f"Ask · {self.agent.mode_label()}")
        else:
            self.mode = MODE_AGENT
            self.prompt = self.agent_prompt
            self.status.configure(text=f"Agent · {self.agent.mode_label()}")

    # --- mesh / capture -----------------------------------------------------

    def _scope_label(self, scope: str) -> str:
        if scope == WHOLE_DESIGN:
            return "Whole design"
        return scope

    def _scope_from_label(self, label: str) -> str:
        text = (label or "").strip()
        if not text or text.lower() in {"whole design", "whole", "assembly", "all"}:
            return WHOLE_DESIGN
        return text

    def _part_labels(self) -> list[str]:
        names = self.result.part_names() if self.result else []
        return ["Whole design"] + names

    def _refresh_parts_ui(self) -> None:
        labels = self._part_labels()
        if hasattr(self, "view_scope_combo"):
            self.view_scope_combo.configure(values=labels)
            current = self._scope_label(self.view_scope)
            self.view_scope_var.set(current if current in labels else "Whole design")
            self.view_scope = self._scope_from_label(self.view_scope_var.get())
        if hasattr(self, "edit_scope_combo"):
            self.edit_scope_combo.configure(values=labels)
            current = self._scope_label(self.edit_scope)
            self.edit_scope_var.set(current if current in labels else "Whole design")
            self.edit_scope = self._scope_from_label(self.edit_scope_var.get())

    def _apply_result(self, result: DesignResult, *, reset_camera: bool) -> None:
        self.result = result
        verts, faces = result.mesh_for(self.view_scope)
        self.agent.set_mesh(result.vertices, result.faces)
        self._set_mesh(verts, faces, reset_camera=reset_camera)
        self._refresh_parts_ui()

    def _on_view_scope_changed(self, _event: object | None = None) -> None:
        self.view_scope = self._scope_from_label(self.view_scope_var.get())
        verts, faces = self.result.mesh_for(self.view_scope)
        self._set_mesh(verts, faces, reset_camera=True)
        label = self._scope_label(self.view_scope)
        self._log_system(f"Viewing · {label}\n{mesh_bbox_summary(verts)}")

    def _on_edit_scope_changed(self, _event: object | None = None) -> None:
        self.edit_scope = self._scope_from_label(self.edit_scope_var.get())
        self._log_system(f"Edit scope · {self._scope_label(self.edit_scope)}")

    def _set_mesh(self, vertices, faces, *, reset_camera: bool) -> None:
        self.preview.set_mesh(vertices, faces, reset_camera=reset_camera)

    def _geometry_context(self) -> str:
        verts, faces = self.result.mesh_for(self.view_scope)
        base = geometry_summary(verts, triangles=len(faces))
        names = ", ".join(self.result.part_names()) or "(none)"
        return (
            f"Viewing: {self._scope_label(self.view_scope)}\n"
            f"Parts: {names}\n"
            f"{base}"
        )

    def _capture_current_view(self) -> Path | None:
        try:
            RENDER_DIR.mkdir(parents=True, exist_ok=True)
            return self.preview.capture_png(RENDER_DIR / "current.png")
        except Exception:
            return None

    def _prepare_agent_images(self) -> list[Path]:
        self.agent.set_mesh(self.result.vertices, self.result.faces)
        paths: list[Path] = []
        current = self._capture_current_view()
        if current is not None:
            paths.append(current)
        # Reuse the same preview OpenGL context (not a second MeshRenderer)
        for result in self.preview.render_views(("isometric", "front", "side")):
            paths.append(result.path)
        seen: set[str] = set()
        unique: list[Path] = []
        for p in paths:
            key = str(p.resolve())
            if key not in seen:
                seen.add(key)
                unique.append(p)
        return unique

    # --- logging / busy -----------------------------------------------------

    def _append_log(self, message: str, tag: str) -> None:
        self.log.configure(state=tk.NORMAL)
        self.log.insert(tk.END, message.rstrip() + "\n", (tag, "gap"))
        self.log.see(tk.END)
        self.log.configure(state=tk.DISABLED)
        self.on_status(message)

    def _log_system(self, message: str) -> None:
        self._append_log(message, "system")

    def _log_user(self, message: str) -> None:
        self._append_log(message, "user")

    def _log_assistant(self, message: str) -> None:
        self._append_log(message, "assistant")

    def _log_ask(self, message: str) -> None:
        self._append_log(message, "ask")

    def _log_error(self, message: str) -> None:
        self._append_log(message, "error")

    def _set_busy(self, busy: bool, status: str = "") -> None:
        self._busy = busy
        state = tk.DISABLED if busy else tk.NORMAL
        self.agent_btn.configure(state=state)
        self.ask_btn.configure(state=state)
        self.agent_prompt.configure(state=state)
        self.ask_prompt.configure(state=state)
        for btn in (
            getattr(self, "draft_btn", None),
            getattr(self, "refine_btn", None),
            getattr(self, "rollback_btn", None),
            getattr(self, "version_combo", None),
        ):
            if btn is not None:
                try:
                    btn.configure(state=state if btn is not self.version_combo else ("readonly" if not busy else "disabled"))
                except tk.TclError:
                    pass
        try:
            self.notebook.state(["!disabled"] if not busy else ["disabled"])
        except tk.TclError:
            pass
        if status:
            self.status.configure(text=status)

    # --- actions ------------------------------------------------------------

    def _on_submit(self) -> None:
        if self.mode == MODE_ASK:
            self._on_ask()
        else:
            self._on_revise()

    def _on_revise(self) -> None:
        if self._busy:
            return
        instruction = self.agent_prompt.get("1.0", tk.END).strip()
        if not instruction:
            messagebox.showinfo(
                "Revision",
                "Enter a language instruction to revise the design.",
                parent=self.root,
            )
            return

        self._set_busy(True, "Agent · Working…")
        self._log_user(f"You · {instruction}")
        if self.edit_scope != WHOLE_DESIGN:
            self._log_system(f"Edit scope · part `{self.edit_scope}`")
        images = self._prepare_agent_images()
        self._log_system("Sent RGB renders to agent: " + ", ".join(p.name for p in images))

        def work() -> None:
            err: str | None = None
            new_result: DesignResult | None = None
            try:
                code = self.agent.revise(
                    instruction,
                    images=images,
                    scope=self.edit_scope,
                )

                def status(msg: str) -> None:
                    self.root.after(0, lambda m=msg: self._log_system(m))

                # Build (+ full review only in Refinement mode)
                new_result = self.agent.finalize_design(
                    code,
                    images=images,
                    on_status=status,
                )
            except Exception as exc:  # noqa: BLE001
                err = str(exc)

            def done() -> None:
                if self._closed:
                    return
                self._set_busy(False, "Agent · Ready")
                if err:
                    self._log_error(f"Error · {err}")
                    messagebox.showerror("Design failed", err, parent=self.root)
                    return
                assert new_result is not None
                self._apply_result(new_result, reset_camera=False)
                self._log_assistant(
                    f"Updated design\n{mesh_bbox_summary(new_result.vertices)}"
                )
                if new_result.part_names():
                    self._log_system("Parts: " + ", ".join(new_result.part_names()))
                if self.agent.features:
                    self._log_system(self.agent.features_summary())
                self._commit_version(
                    new_result,
                    label=instruction.strip()[:60] or "Revision",
                    note=f"After Agent revision: {instruction.strip()[:200]}",
                )
                self.agent_prompt.delete("1.0", tk.END)
                self.status.configure(text=f"Agent · {self.agent.mode_label()}")

            self.root.after(0, done)

        threading.Thread(target=work, daemon=True).start()

    def _on_ask(self) -> None:
        if self._busy:
            return
        question = self.ask_prompt.get("1.0", tk.END).strip()
        if not question:
            messagebox.showinfo(
                "Ask",
                "Enter a question about the current design.",
                parent=self.root,
            )
            return

        self._set_busy(True, "Ask · Thinking…")
        self._log_user(f"You · {question}")
        geo = self._geometry_context()
        images = self._prepare_agent_images()
        self._log_system("Sent RGB renders to agent: " + ", ".join(p.name for p in images))

        def work() -> None:
            err: str | None = None
            answer: str | None = None
            try:
                answer = self.agent.ask(question, geometry_summary=geo, images=images)
            except Exception as exc:  # noqa: BLE001
                err = str(exc)

            def done() -> None:
                if self._closed:
                    return
                self._set_busy(False, "Ask · Ready")
                if err:
                    self._log_error(f"Error · {err}")
                    messagebox.showerror("Ask failed", err, parent=self.root)
                    return
                assert answer is not None
                self._log_ask(answer)
                self.ask_prompt.delete("1.0", tk.END)

            self.root.after(0, done)

        threading.Thread(target=work, daemon=True).start()

    def _chat_log_text(self) -> str:
        try:
            return self.log.get("1.0", tk.END)
        except tk.TclError:
            return ""

    def _set_chat_log_text(self, text: str) -> None:
        self.log.configure(state=tk.NORMAL)
        self.log.delete("1.0", tk.END)
        if text:
            self.log.insert(tk.END, text.rstrip() + "\n")
        self.log.see(tk.END)
        self.log.configure(state=tk.DISABLED)

    def _commit_version(
        self,
        result: DesignResult,
        *,
        label: str,
        note: str = "",
    ) -> VersionMeta | None:
        try:
            meta = self.agent.commit_version(
                result,
                label=label,
                note=note,
                chat_log=self._chat_log_text(),
            )
            self._refresh_version_ui(select_id=meta.id)
            self._log_system(f"Version saved · {meta.short_label()}")
            return meta
        except Exception as exc:  # noqa: BLE001
            self._log_error(f"Version save failed · {exc}")
            return None

    def _refresh_version_ui(self, select_id: str | None = None) -> None:
        versions = self.agent.list_versions()
        self._version_ids = [v.id for v in versions]
        self._version_labels = [v.short_label() for v in versions]
        self.version_combo.configure(values=self._version_labels)
        current = select_id or self.agent.version_store().current_id()
        if current and current in self._version_ids:
            idx = self._version_ids.index(current)
            self.version_var.set(self._version_labels[idx])
        elif self._version_labels:
            self.version_var.set(self._version_labels[-1])
        else:
            self.version_var.set("")

    def _selected_version_id(self) -> str | None:
        label = self.version_var.get().strip()
        if not label or label not in self._version_labels:
            return None
        return self._version_ids[self._version_labels.index(label)]

    def _on_rollback_shortcut(self, _event: object | None = None) -> str:
        if self._busy or self._closed:
            return "break"
        versions = self.agent.list_versions()
        current = self.agent.version_store().current_id()
        if len(versions) < 2:
            self._log_system("Rollback · no earlier version")
            return "break"
        ids = [v.id for v in versions]
        if current in ids and ids.index(current) > 0:
            target = ids[ids.index(current) - 1]
        else:
            target = ids[-2]
        self.version_var.set(self._version_labels[ids.index(target)])
        self._on_rollback()
        return "break"

    def _on_rollback(self) -> None:
        if self._busy or self._closed:
            return
        version_id = self._selected_version_id()
        if not version_id:
            messagebox.showinfo(
                "Rollback",
                "Select a version from History first.",
                parent=self.root,
            )
            return
        current = self.agent.version_store().current_id()
        if version_id == current:
            messagebox.showinfo(
                "Rollback",
                "That version is already current.",
                parent=self.root,
            )
            return
        ok = messagebox.askyesno(
            "Rollback design",
            f"Restore {version_id}?\n\n"
            "This restores the CAD design, feature list, and chat transcript "
            "from that snapshot, and clears newer versions and the Cursor agent "
            "conversation.",
            parent=self.root,
        )
        if not ok:
            return

        self._set_busy(True, "Rolling back…")

        def work() -> None:
            err: str | None = None
            restored: DesignResult | None = None
            snap_chat = ""
            meta_label = version_id
            try:
                restored, snap = self.agent.restore_version(version_id, truncate_newer=True)
                snap_chat = snap.chat_log
                meta_label = snap.meta.short_label()
            except Exception as exc:  # noqa: BLE001
                err = str(exc)

            def done() -> None:
                if self._closed:
                    return
                self._set_busy(False, f"Agent · {self.agent.mode_label()}")
                if err:
                    self._log_error(f"Rollback failed · {err}")
                    messagebox.showerror("Rollback failed", err, parent=self.root)
                    return
                assert restored is not None
                self.view_scope = WHOLE_DESIGN
                self.edit_scope = WHOLE_DESIGN
                self._apply_result(restored, reset_camera=True)
                # Sync mode radios with restored design_mode
                self.mode_var.set("draft" if self.agent.is_draft_mode() else "refine")
                if snap_chat.strip():
                    self._set_chat_log_text(snap_chat)
                self._log_system(
                    f"Rolled back to {meta_label}\n{mesh_bbox_summary(restored.vertices)}"
                )
                if restored.part_names():
                    self._log_system("Parts: " + ", ".join(restored.part_names()))
                if self.agent.features:
                    self._log_system(self.agent.features_summary())
                self._refresh_version_ui(select_id=version_id)

            self.root.after(0, done)

        threading.Thread(target=work, daemon=True).start()

    def _on_export_part(self) -> None:
        if self._busy or self._closed:
            return
        scope = self.view_scope
        if scope == WHOLE_DESIGN:
            messagebox.showinfo(
                "Export part STEP",
                "Select a specific part in View first (not Whole design).",
                parent=self.root,
            )
            return
        suggested = f"{scope}.step"
        path = ask_save_name(
            self.root,
            self.models_dir,
            title="Export part STEP",
            prompt="Enter a name for the STEP file (without path):",
            initialvalue=suggested,
            extensions=(".step", ".stp"),
            default_ext=".step",
        )
        if path is None:
            return
        try:
            export_step(self.result.solid_for(scope), path)
            self._log_system(f"Exported part `{scope}` → {path.name}")
            messagebox.showinfo("Exported", f"Wrote part `{scope}`:\n{path}", parent=self.root)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Export failed", str(exc), parent=self.root)

    def _on_export_part_stl(self) -> None:
        if self._busy or self._closed:
            return
        scope = self.view_scope
        if scope == WHOLE_DESIGN:
            messagebox.showinfo(
                "Export part STL",
                "Select a specific part in View first (not Whole design).",
                parent=self.root,
            )
            return
        suggested = f"{scope}.stl"
        path = ask_save_name(
            self.root,
            self.models_dir,
            title="Export part STL",
            prompt="Enter a name for the STL mesh (without path):",
            initialvalue=suggested,
            extensions=(".stl",),
            default_ext=".stl",
        )
        if path is None:
            return
        try:
            export_stl(self.result.solid_for(scope), path)
            self._log_system(f"Exported part STL `{scope}` → {path.name}")
            messagebox.showinfo("Exported", f"Wrote part mesh `{scope}`:\n{path}", parent=self.root)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Export failed", str(exc), parent=self.root)

    def _on_export_all_parts(self) -> None:
        if self._busy or self._closed:
            return
        if not self.result.part_names():
            messagebox.showinfo("Export all STEP", "No named parts in this design.", parent=self.root)
            return
        try:
            paths = export_parts(self.result, self.models_dir, basename="design")
            listing = "\n".join(p.name for p in paths)
            self._log_system(f"Exported {len(paths)} part STEP file(s)")
            messagebox.showinfo("Exported parts", f"Wrote:\n{listing}", parent=self.root)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Export failed", str(exc), parent=self.root)

    def _on_export_all_stl_zip(self) -> None:
        if self._busy or self._closed:
            return
        if not self.result.part_names():
            messagebox.showinfo("ZIP all STLs", "No named parts in this design.", parent=self.root)
            return
        path = ask_save_name(
            self.root,
            self.models_dir,
            title="ZIP all STLs",
            prompt="Enter a name for the ZIP archive (without path):",
            initialvalue="design_parts_stl.zip",
            extensions=(".zip",),
            default_ext=".zip",
        )
        if path is None:
            return
        try:
            folder, zip_path = export_parts_stl_zip(
                self.result,
                path,
                basename="design",
            )
            listing = "\n".join(p.name for p in sorted(folder.glob("*.stl")))
            self._log_system(f"Exported STL folder + zip → {folder.name}/ and {zip_path.name}")
            messagebox.showinfo(
                "Exported STLs",
                f"Wrote folder:\n{folder}\n\nZip:\n{zip_path}\n\nMeshes:\n{listing}",
                parent=self.root,
            )
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Export failed", str(exc), parent=self.root)

    def _on_save(self, _event: object | None = None) -> str:
        if self._busy or self._closed:
            return "break"
        path = ask_save_name(self.root, self.models_dir)
        if path is None:
            return "break"
        try:
            # Always save the full assembly assembly for Ctrl+S / Save STEP
            export_step(self.result.solid, path)
            script_path = path.with_suffix(".py")
            save_design_script(self.result.code, script_path)
            self._log_system(f"Saved {path.name} and {script_path.name}")
            mode_label = "Ask" if self.mode == MODE_ASK else "Agent"
            self.status.configure(text=f"{mode_label} · Saved")
            messagebox.showinfo("Saved", f"Wrote:\n{path}\n{script_path}", parent=self.root)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Save failed", str(exc), parent=self.root)
        return "break"

    def _on_close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.agent.close()
        except Exception:
            pass
        try:
            self.preview.close()
        except Exception:
            pass
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()
