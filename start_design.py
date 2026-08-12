#!/usr/bin/env python3
"""
AI CAD Design — interactive language → STEP pipeline.

Flow:
  1. Popup asks for a language design brief
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
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Callable

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cad_pipeline.agent import DesignAgent
from cad_pipeline.models import DEFAULT_MODEL, model_help_text, resolve_model
from cad_pipeline.studio import DesignStudio
from cad_pipeline.ui_dialogs import ask_initial_prompt
from cad_pipeline.ui_scale import apply_scaling, colors, fit_window, fonts, scaled


def _show_progress(message: str = "Designing…") -> tuple[tk.Tk, Callable[[str], None], Callable[[], None]]:
    root = tk.Tk()
    apply_scaling(root)
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

    fit_window(
        root,
        min_width=scaled(root, 560),
        min_height=scaled(root, 180),
        pad=scaled(root, 20),
    )
    return root, set_message, close


def run_pipeline(*, fast: bool | None = None, model: str | None = None) -> int:
    models_dir = ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    prompt = ask_initial_prompt()
    if not prompt:
        print("Cancelled.")
        return 0

    agent = DesignAgent()
    if fast is not None:
        agent.fast = fast
    if model is not None:
        agent.model = resolve_model(model)
    print(
        f"LLM backend: {agent.backend}  model={agent.model}  "
        f"fast={'on' if agent.fast else 'off'}  "
        f"mode={agent.mode_label()}"
    )

    progress, set_message, close_progress = _show_progress("Generating initial design…")
    # VTK OpenGL must run on the Tk thread — marshal agent renders here until Studio takes over
    agent.set_ui_marshal(lambda fn: progress.after(0, fn))
    holder: dict = {"result": None, "error": None}

    def work() -> None:
        try:
            progress.after(0, lambda: set_message("Asking design agent…"))
            code = agent.generate(prompt)
            result = agent.finalize_design(
                code,
                requirements=prompt,
                on_status=lambda msg: progress.after(0, lambda m=msg: set_message(m)),
            )
            holder["result"] = result
        except Exception as exc:  # noqa: BLE001 — surfaced in UI
            holder["error"] = str(exc)
        finally:
            progress.after(0, close_progress)

    threading.Thread(target=work, daemon=True).start()
    progress.mainloop()
    try:
        progress.destroy()
    except tk.TclError:
        pass
    # Progress window is gone; Studio will install its own marshal + host renderer
    agent.set_ui_marshal(None)

    if holder["error"]:
        try:
            agent.close()
        except Exception:
            pass
        err_root = tk.Tk()
        err_root.withdraw()
        messagebox.showerror("Design failed", holder["error"], parent=err_root)
        err_root.destroy()
        print(holder["error"], file=sys.stderr)
        return 1

    studio = DesignStudio(agent, holder["result"], models_dir=models_dir)
    try:
        studio.run()
    finally:
        try:
            agent.close()
        except Exception:
            pass
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
