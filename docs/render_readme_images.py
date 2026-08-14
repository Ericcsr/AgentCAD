#!/usr/bin/env python3
"""Render README gallery images (offscreen VTK + optional studio screenshot)."""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("DESIGN_UI_SCALE", "1.15")
os.environ.setdefault("DESIGN_LLM", "mock")

from cad_pipeline.agent import MOCK_CHAIR, MOCK_TABLE, DesignAgent
from cad_pipeline.mesh_utils import solid_to_mesh
from cad_pipeline.render import MeshRenderer
from cad_pipeline.runtime import run_design_code
from cad_pipeline.step_import import load_reference_solid

OUT = ROOT / "docs" / "images"


def _exec_build(code: str):
    import cadquery as cq

    ns = {
        "__builtins__": {
            "abs": abs,
            "min": min,
            "max": max,
            "range": range,
            "len": len,
            "float": float,
            "int": int,
            "round": round,
            "enumerate": enumerate,
            "zip": zip,
            "list": list,
            "dict": dict,
            "tuple": tuple,
            "str": str,
            "True": True,
            "False": False,
            "None": None,
        },
        "cq": cq,
        "cadquery": cq,
        "math": math,
        "import_reference": load_reference_solid,
    }
    exec(code, ns, ns)  # noqa: S102
    build = ns.get("build")
    if not callable(build):
        raise RuntimeError("no build()")
    return build()


def tessellate(code: str):
    return solid_to_mesh(_exec_build(code))


def render_mesh(name: str, vertices, faces, *, view: str = "isometric", size=(1280, 960)):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.png"
    renderer = MeshRenderer(size=size)
    try:
        renderer.render(vertices, faces, view=view, path=path, size=size)
    finally:
        renderer.close()
    print(f"wrote {path}  ({path.stat().st_size // 1024} KB)")
    return path


def capture_studio(chair_result) -> Path | None:
    import tkinter as tk

    from PIL import ImageGrab

    from cad_pipeline.studio import DesignStudio

    agent = DesignAgent()
    agent.backend = "mock"
    agent.fast = True
    studio = DesignStudio(agent, chair_result, models_dir=ROOT / "models")
    studio.root.geometry("1600x960+24+24")
    studio._log_user("You · a wooden dining chair with four slats and a slight recline")
    studio._log_assistant("Updated design\nbbox mm: 450 × 420 × 869")
    studio._log_system(
        "Parts: seat, leg_fl, leg_fr, leg_bl, leg_br, stretcher_l, stretcher_r, stretcher_f, backrest"
    )
    studio._log_system("Kinematics: 9 parts, 8 joint(s) — seat is the root; legs and backrest are fixed.")
    studio._log_user("You · make the backrest taller")
    studio._log_assistant("Updated design\nbbox mm: 450 × 420 × 920")
    studio._log_system("Feasibility · PASS (9 parts)")
    studio.agent_prompt.delete("1.0", "end")
    studio.agent_prompt.insert("1.0", "add a small lumbar curve")

    path = OUT / "studio.png"
    grabbed: dict = {}

    def shoot() -> None:
        try:
            studio.preview.redraw(immediate=True)
            studio.root.update()
            studio.root.update_idletasks()
            studio.root.lift()
            studio.root.focus_force()
            x = studio.root.winfo_rootx()
            y = studio.root.winfo_rooty()
            w = studio.root.winfo_width()
            h = studio.root.winfo_height()
            img = ImageGrab.grab(bbox=(x, y, x + w, y + h))
            img.save(path)
            grabbed["ok"] = True
            print(f"wrote {path}  ({path.stat().st_size // 1024} KB)")
        except Exception as exc:  # noqa: BLE001
            grabbed["error"] = str(exc)
            print(f"studio screenshot failed: {exc}")
        finally:
            studio.root.destroy()

    studio.root.after(1800, shoot)
    studio.root.mainloop()
    return path if grabbed.get("ok") else None


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    studio_only = "--studio-only" in sys.argv

    chair = run_design_code(MOCK_CHAIR)
    if not studio_only:
        render_mesh("chair", chair.vertices, chair.faces)

        table = run_design_code(MOCK_TABLE)
        render_mesh("table", table.vertices, table.faces)

        nut = tessellate((ROOT / "models" / "nut_bolt.py").read_text(encoding="utf-8"))
        render_mesh("nut_bolt", *nut)

        gripper_src = ROOT / "generated" / "current_design.py"
        if gripper_src.exists() and "def build" in gripper_src.read_text(encoding="utf-8"):
            try:
                render_mesh("gripper", *tessellate(gripper_src.read_text(encoding="utf-8")))
            except Exception as exc:  # noqa: BLE001
                print(f"skip gripper: {exc}")

        prop_src = ROOT / "models" / "propeller2.py"
        if prop_src.exists():
            try:
                print("tessellating propeller (slow)…")
                render_mesh("propeller", *tessellate(prop_src.read_text(encoding="utf-8")))
            except Exception as exc:  # noqa: BLE001
                print(f"skip propeller: {exc}")

    if os.environ.get("DISPLAY"):
        capture_studio(chair)
    else:
        print("no DISPLAY — skip studio screenshot")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
