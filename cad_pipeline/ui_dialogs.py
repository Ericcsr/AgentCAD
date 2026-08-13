#!/usr/bin/env python3
"""Tk dialogs: initial language prompt, STEP import, and save name."""

from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk

from cad_pipeline.ui_scale import apply_scaling, colors, fit_window, fonts, scaled


@dataclass
class InitialBrief:
    prompt: str
    step_paths: list[Path] = field(default_factory=list)


def ask_open_step_paths(
    parent: tk.Misc,
    *,
    initialdir: Path | None = None,
    title: str = "Import STEP model",
) -> list[Path]:
    """Pick one or more STEP files. Returns an empty list if cancelled."""
    raw = filedialog.askopenfilenames(
        parent=parent,
        title=title,
        initialdir=str(initialdir) if initialdir else None,
        filetypes=[
            ("STEP models", "*.step *.stp *.STEP *.STP"),
            ("All files", "*.*"),
        ],
    )
    paths: list[Path] = []
    seen: set[str] = set()
    for item in raw or ():
        path = Path(item).expanduser().resolve()
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        paths.append(path)
    return paths


def ask_initial_prompt(parent: tk.Tk | None = None) -> InitialBrief | None:
    """Modal window asking for the initial design brief. Returns None if cancelled."""
    owns_root = parent is None
    root = parent or tk.Tk()
    if owns_root:
        apply_scaling(root)
    c = colors(root)
    root.title("AI CAD Design")
    root.configure(bg=c["bg"])

    result: dict[str, InitialBrief | None] = {"brief": None}
    step_paths: list[Path] = []
    f = fonts(root)
    c = colors(root)

    # Grid layout so the button row stays visible even when the text area shrinks
    frame = ttk.Frame(root, padding=scaled(root, 22))
    frame.pack(fill=tk.BOTH, expand=True)
    frame.columnconfigure(0, weight=1)
    frame.rowconfigure(3, weight=1)

    ttk.Label(frame, text="AI CAD", style="Brand.TLabel", font=f["brand"]).grid(
        row=0, column=0, sticky="w"
    )
    ttk.Label(
        frame,
        text="What should we design?",
        style="Title.TLabel",
        font=f["title"],
    ).grid(row=1, column=0, sticky="w", pady=(scaled(root, 4), scaled(root, 6)))
    ttk.Label(
        frame,
        text="e.g. a modern wooden dining chair with four slats and a slight recline. "
        "Optionally load an existing STEP as a size / mating constraint.",
        style="Hint.TLabel",
        font=f["small"],
        wraplength=scaled(root, 640),
    ).grid(row=2, column=0, sticky="ew", pady=(0, scaled(root, 12)))

    box = tk.Frame(frame, bg=c["border"], bd=0)
    box.grid(row=3, column=0, sticky="nsew")
    text = tk.Text(
        box,
        height=8,
        wrap=tk.WORD,
        font=f["body"],
        undo=True,
        background=c["surface"],
        foreground=c["ink"],
        insertbackground=c["ink"],
        relief=tk.FLAT,
        borderwidth=0,
        highlightthickness=0,
        padx=scaled(root, 10),
        pady=scaled(root, 8),
    )
    text.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)
    text.focus_set()

    ref_row = ttk.Frame(frame)
    ref_row.grid(row=4, column=0, sticky="ew", pady=(scaled(root, 10), 0))
    ref_row.columnconfigure(1, weight=1)
    ttk.Label(ref_row, text="STEP refs", style="Hint.TLabel").grid(row=0, column=0, sticky="nw")
    ref_var = tk.StringVar(value="None — optional constraint models")
    ref_label = ttk.Label(
        ref_row,
        textvariable=ref_var,
        style="Hint.TLabel",
        font=f["small"],
        wraplength=scaled(root, 480),
        justify=tk.LEFT,
    )
    ref_label.grid(row=0, column=1, sticky="ew", padx=(scaled(root, 8), scaled(root, 8)))

    def refresh_ref_label() -> None:
        if not step_paths:
            ref_var.set("None — optional constraint models")
            return
        names = ", ".join(p.name for p in step_paths)
        ref_var.set(f"{len(step_paths)} file(s): {names}")

    def add_steps() -> None:
        models_dir = Path(__file__).resolve().parent.parent / "models"
        picked = ask_open_step_paths(root, initialdir=models_dir if models_dir.is_dir() else None)
        for path in picked:
            if path not in step_paths:
                step_paths.append(path)
        refresh_ref_label()

    def clear_steps() -> None:
        step_paths.clear()
        refresh_ref_label()

    ttk.Button(ref_row, text="Load STEP…", command=add_steps).grid(row=0, column=2, sticky="e")
    ttk.Button(ref_row, text="Clear", command=clear_steps).grid(
        row=0, column=3, sticky="e", padx=(scaled(root, 6), 0)
    )

    btn_row = ttk.Frame(frame)
    btn_row.grid(row=5, column=0, sticky="ew", pady=(scaled(root, 14), 0))

    def submit(_event: object | None = None) -> None:
        prompt = text.get("1.0", tk.END).strip()
        if not prompt:
            messagebox.showwarning("Empty prompt", "Please enter a design description.", parent=root)
            return
        result["brief"] = InitialBrief(prompt=prompt, step_paths=list(step_paths))
        root.quit()

    def cancel() -> None:
        result["brief"] = None
        root.quit()

    ttk.Button(btn_row, text="Cancel", command=cancel).pack(side=tk.RIGHT)
    ttk.Button(btn_row, text="Design", style="Accent.TButton", command=submit).pack(
        side=tk.RIGHT, padx=(0, scaled(root, 8))
    )
    root.bind("<Control-Return>", submit)
    root.protocol("WM_DELETE_WINDOW", cancel)

    fit_window(
        root,
        min_width=scaled(root, 720),
        min_height=scaled(root, 580),
        pad=scaled(root, 28),
    )
    try:
        root.deiconify()
    except tk.TclError:
        pass
    root.mainloop()
    brief = result["brief"]
    if owns_root:
        try:
            root.destroy()
        except tk.TclError:
            pass
    else:
        for child in list(root.winfo_children()):
            try:
                child.destroy()
            except tk.TclError:
                pass
        if brief is not None:
            try:
                root.withdraw()
            except tk.TclError:
                pass
    return brief


def ask_save_name(
    parent: tk.Misc,
    models_dir: Path,
    *,
    title: str = "Save STEP",
    prompt: str = "Enter a name for the STEP file (without path):",
    initialvalue: str = "design.step",
    extensions: tuple[str, ...] = (".step", ".stp"),
    default_ext: str = ".step",
) -> Path | None:
    """Ask for an export filename and return the full path, or None if cancelled."""
    models_dir.mkdir(parents=True, exist_ok=True)
    name = simpledialog.askstring(
        title,
        prompt,
        parent=parent,
        initialvalue=initialvalue,
    )
    if not name:
        return None
    name = name.strip()
    lower = name.lower()
    if not any(lower.endswith(ext) for ext in extensions):
        name += default_ext
    name = Path(name).name
    return models_dir / name
