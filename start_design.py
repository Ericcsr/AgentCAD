#!/usr/bin/env python3
"""
AI CAD Design — interactive language → STEP pipeline.

Flow:
  1. Popup asks for a language design brief (optional: load existing STEP constraints)
  2. Cursor agent writes CadQuery code and builds the solid
  3. Interactive VTK studio opens for review + language revisions
  4. Ctrl+S saves the STEP (and matching .py) under models/

Usage:
  export CURSOR_API_KEY=...   # https://cursor.com/dashboard/integrations
  python start_design.py
  python start_design.py --model grok-4.6
  python start_design.py --model claude
  python start_design.py --model openai
  python start_design.py --fast          # Cursor model fast mode
  DESIGN_FAST=1 python start_design.py   # same via env

  DESIGN_LLM=mock python start_design.py   # offline demo (no API key)

Optional env (.env supported):
  CURSOR_API_KEY         required for Cursor designs
  DESIGN_MODEL           default grok-4.5 (aliases: grok-4.6, claude, openai, …)
  DESIGN_LLM             auto|cursor|mock  (auto → cursor if key present)
  DESIGN_FAST            1/true to enable Cursor model fast mode
  DESIGN_MODE            draft|refine (default draft — skip constraint review)
  DESIGN_DEBUG_RETRIES   build→debug cycles on CadQuery errors (default 5)
  DESIGN_REVIEW_ROUNDS   inspect→refine cycles after a successful build (default 3)
"""

from __future__ import annotations

import argparse
import faulthandler
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Callable

faulthandler.enable()

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cad_pipeline.models import DEFAULT_MODEL, model_help_text, resolve_model
from cad_pipeline.ui_dialogs import ask_initial_prompt
from cad_pipeline.ui_scale import apply_scaling, colors, fit_window, fonts, scaled


def _clear_root(root: tk.Tk) -> None:
    for child in list(root.winfo_children()):
        try:
            child.destroy()
        except tk.TclError:
            pass


def _show_progress(root: tk.Tk, message: str = "Designing…") -> tuple[Callable[[str], None], Callable[[], None]]:
    """Reuse the same Tk root — a second Tk() after VTK is loaded segfaults on Linux."""
    _clear_root(root)
    f = fonts(root)
    c = colors(root)
    root.title("AI CAD Design")
    root.configure(bg=c["bg"])
    frame = ttk.Frame(root, padding=scaled(root, 22))
    frame.pack(fill=tk.BOTH, expand=True)
    ttk.Label(frame, text="AI CAD", style="Brand.TLabel", font=f["brand"]).pack(anchor=tk.W)
    label = ttk.Label(frame, text=message, font=f["body"], wraplength=scaled(root, 480))
    label.pack(anchor=tk.W, pady=(scaled(root, 6), 0))
    bar = ttk.Progressbar(frame, mode="indeterminate")
    bar.pack(fill=tk.X, pady=(scaled(root, 14), 0))
    bar.start(12)

    def set_message(text: str) -> None:
        try:
            label.configure(text=text)
            root.update_idletasks()
        except tk.TclError:
            pass

    def close() -> None:
        try:
            bar.stop()
        except tk.TclError:
            pass
        try:
            root.quit()
        except tk.TclError:
            pass

    root.protocol("WM_DELETE_WINDOW", close)
    fit_window(
        root,
        min_width=scaled(root, 560),
        min_height=scaled(root, 180),
        pad=scaled(root, 20),
    )
    root.deiconify()
    return set_message, close


def run_pipeline(*, fast: bool | None = None, model: str | None = None) -> int:
    models_dir = ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Create Tk before importing VTK/OCP. Destroying one Tk() and creating
    # another after vtk is loaded is a common Linux segfault.
    root = tk.Tk()
    root.withdraw()
    apply_scaling(root)

    from cad_pipeline.agent import DesignAgent
    from cad_pipeline.studio import DesignStudio

    brief = ask_initial_prompt(root)
    if not brief:
        try:
            root.destroy()
        except tk.TclError:
            pass
        print("Cancelled.")
        return 0
    prompt = brief.prompt
    step_paths = list(brief.step_paths)

    agent = DesignAgent()
    if fast is not None:
        agent.fast = fast
    if model is not None:
        agent.model = resolve_model(model)
    print(
        f"LLM backend: {agent.backend}  model={agent.model}  "
        f"fast={'on' if agent.fast else 'off'}  "
        f"mode={agent.mode_label()}",
        flush=True,
    )
    if step_paths:
        print("STEP references:", ", ".join(p.name for p in step_paths), flush=True)

    set_message, close_progress = _show_progress(root, "Generating initial design…")
    # VTK OpenGL must run on the Tk thread — marshal agent renders here until Studio takes over
    agent.set_ui_marshal(lambda fn: root.after(0, fn))

    def _progress_status(msg: str) -> None:
        text = msg[2:].strip() if msg.startswith("~ ") else msg
        if len(text) > 90:
            text = "…" + text[-89:]
        root.after(0, lambda m=text: set_message(m))

    agent.set_status_hook(_progress_status)
    holder: dict = {"result": None, "error": None}

    def work() -> None:
        try:
            if step_paths:
                root.after(0, lambda: set_message("Reading STEP reference(s)…"))
                for path in step_paths:
                    agent.add_step_reference(path)
            root.after(0, lambda: set_message("Asking design agent…"))
            code = agent.generate(prompt)
            result = agent.finalize_design(
                code,
                requirements=prompt,
                on_status=_progress_status,
            )
            holder["result"] = result
        except Exception as exc:  # noqa: BLE001 — surfaced in UI
            holder["error"] = str(exc)
        finally:
            agent.set_status_hook(None)
            try:
                root.after(0, close_progress)
            except tk.TclError:
                close_progress()

    threading.Thread(target=work, daemon=True).start()
    root.mainloop()
    agent.set_ui_marshal(None)

    def _shutdown_agent() -> None:
        try:
            agent.close()
        except Exception:
            pass

    if holder["error"]:
        _shutdown_agent()
        try:
            if root.winfo_exists():
                messagebox.showerror("Design failed", holder["error"], parent=root)
                root.destroy()
        except tk.TclError:
            pass
        print(holder["error"], file=sys.stderr)
        return 1

    if holder["result"] is None or not root.winfo_exists():
        _shutdown_agent()
        try:
            root.destroy()
        except tk.TclError:
            pass
        print("Cancelled.")
        return 0

    studio = DesignStudio(agent, holder["result"], models_dir=models_dir, root=root)
    try:
        studio.run()
    finally:
        _shutdown_agent()
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI CAD Design — language → STEP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=model_help_text(),
    )
    parser.add_argument(
        "-m",
        "--model",
        metavar="ID",
        help=(
            "Cursor SDK model or alias "
            f"(default: {DEFAULT_MODEL} / DESIGN_MODEL). "
            "Examples: grok-4.5, grok-4.6, claude, claude-opus, openai, composer"
        ),
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print model aliases and exit",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Enable Cursor model fast mode (ModelSelection param fast=true)",
    )
    parser.add_argument(
        "--no-fast",
        action="store_true",
        help="Disable Cursor model fast mode even if DESIGN_FAST is set",
    )
    args = parser.parse_args()
    if args.list_models:
        print(model_help_text())
        raise SystemExit(0)
    fast: bool | None = None
    if args.fast:
        fast = True
    elif args.no_fast:
        fast = False
    raise SystemExit(run_pipeline(fast=fast, model=args.model))


if __name__ == "__main__":
    main()
