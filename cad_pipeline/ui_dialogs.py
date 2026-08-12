#!/usr/bin/env python3
"""Tk dialogs: initial language prompt and STEP save name."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import messagebox, simpledialog, ttk

from cad_pipeline.ui_scale import apply_scaling, colors, fit_window, fonts, scaled


def ask_initial_prompt(parent: tk.Tk | None = None) -> str | None:
    """Modal window asking for the initial design brief. Returns None if cancelled."""
    owns_root = parent is None
    root = parent or tk.Tk()
    if owns_root:
        apply_scaling(root)
        c = colors(root)
        root.title("AI CAD Design")
        root.configure(bg=c["bg"])

    result: dict[str, str | None] = {"prompt": None}
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
        text="e.g. a modern wooden dining chair with four slats and a slight recline",
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

    btn_row = ttk.Frame(frame)
    btn_row.grid(row=4, column=0, sticky="ew", pady=(scaled(root, 14), 0))

    def submit(_event: object | None = None) -> None:
        prompt = text.get("1.0", tk.END).strip()
        if not prompt:
            messagebox.showwarning("Empty prompt", "Please enter a design description.", parent=root)
            return
        result["prompt"] = prompt
        root.quit()

    def cancel() -> None:
        result["prompt"] = None
        root.quit()

    ttk.Button(btn_row, text="Cancel", command=cancel).pack(side=tk.RIGHT)
    ttk.Button(btn_row, text="Design", style="Accent.TButton", command=submit).pack(
        side=tk.RIGHT, padx=(0, scaled(root, 8))
    )
    root.bind("<Control-Return>", submit)
    root.protocol("WM_DELETE_WINDOW", cancel)

    if owns_root:
        fit_window(
            root,
            min_width=scaled(root, 720),
            min_height=scaled(root, 520),
            pad=scaled(root, 28),
        )
        root.mainloop()
        try:
            root.destroy()
        except tk.TclError:
            pass
    return result["prompt"]


def ask_save_name(parent: tk.Misc, models_dir: Path) -> Path | None:
    """Ask for a STEP filename and return the full path, or None if cancelled."""
    models_dir.mkdir(parents=True, exist_ok=True)
    name = simpledialog.askstring(
        "Save STEP",
        "Enter a name for the STEP file (without path):",
        parent=parent,
        initialvalue="design.step",
    )
    if not name:
        return None
    name = name.strip()
    if not name.lower().endswith((".step", ".stp")):
        name += ".step"
    name = Path(name).name
    return models_dir / name
