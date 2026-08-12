#!/usr/bin/env python3
"""HiDPI scaling + shared visual theme for Tk windows."""

from __future__ import annotations

import os
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk

# Calm workshop palette — ink on warm paper, wood accent
COLORS = {
    "bg": "#f3f1ec",
    "surface": "#fffcf7",
    "surface_alt": "#ebe7df",
    "border": "#d4cfc4",
    "ink": "#1f1c18",
    "muted": "#6b645a",
    "accent": "#8a5a2b",
    "accent_hover": "#6f4620",
    "accent_soft": "#f0e4d4",
    "user": "#1f1c18",
    "assistant": "#3d342c",
    "ask": "#2f4a3c",
    "system": "#6b645a",
    "error": "#8b2e2e",
    "status_bg": "#e7e1d6",
}


def screen_metrics(root: tk.Misc) -> tuple[int, int, float]:
    """Return (width_px, height_px, scale_factor)."""
    w = int(root.winfo_screenwidth())
    h = int(root.winfo_screenheight())
    env = os.getenv("DESIGN_UI_SCALE")
    if env:
        try:
            return w, h, max(1.0, float(env))
        except ValueError:
            pass
    # Keep 4K readable without ballooning titles
    if w >= 3500 or (w * w + h * h) ** 0.5 >= 4000:
        scale = 1.55
    elif w >= 2500:
        scale = 1.3
    elif w >= 1900:
        scale = 1.12
    else:
        scale = 1.0
    return w, h, scale


def _pick_ui_font(root: tk.Misc) -> str:
    # Prefer fontconfig-backed families (often absent from tkfont.families() on Linux)
    candidates = (
        "IBM Plex Sans",
        "Source Sans 3",
        "Source Sans Pro",
        "Inter",
        "Ubuntu",
        "Liberation Sans",
        "DejaVu Sans",
        "Noto Sans",
        "FreeSans",
        "Cantarell",
        "Segoe UI",
    )
    available = {name.lower() for name in tkfont.families()}
    for name in candidates:
        if name.lower() in available:
            return name
    for name in candidates:
        try:
            f = tkfont.Font(root=root, family=name, size=12)
            if f.measure("Ag") > 0:
                actual = f.actual("family")
                if actual and actual.lower() not in {"fixed", "nil", "clean"}:
                    return name
        except tk.TclError:
            continue
    return "DejaVu Sans"


def apply_scaling(root: tk.Tk) -> float:
    """Apply Tk scaling, fonts, and ttk theme. Returns scale factor."""
    _w, _h, scale = screen_metrics(root)
    try:
        current = float(root.tk.call("tk", "scaling"))
        # Mild bump only — avoid compounding into huge chrome
        root.tk.call("tk", "scaling", max(current, 1.2) * min(scale, 1.35) / 1.15)
    except tk.TclError:
        pass

    family = _pick_ui_font(root)
    # Tight hierarchy: title only ~2pt above body
    body = max(12, int(round(12.5 * scale)))
    base = max(11, body - 1)
    title = body + 2
    small = max(10, base - 1)
    chat = body

    root.configure(bg=COLORS["bg"])
    root.option_add("*Font", f"{{{family}}} {base}")
    root.option_add("*Background", COLORS["bg"])
    root.option_add("*Foreground", COLORS["ink"])
    root.option_add("*Text.Font", f"{{{family}}} {body}")
    root.option_add("*Text.Background", COLORS["surface"])
    root.option_add("*Text.Foreground", COLORS["ink"])
    root.option_add("*Text.Relief", "flat")
    root.option_add("*Text.HighlightThickness", 1)
    root.option_add("*Text.HighlightColor", COLORS["accent"])
    root.option_add("*Text.HighlightBackground", COLORS["border"])
    root.option_add("*Entry.Background", COLORS["surface"])

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    style.configure(".", font=(family, base), background=COLORS["bg"], foreground=COLORS["ink"])
    style.configure("TFrame", background=COLORS["bg"])
    style.configure("Card.TFrame", background=COLORS["surface"], relief="flat")
    style.configure("TLabel", background=COLORS["bg"], foreground=COLORS["ink"], font=(family, base))
    style.configure(
        "Title.TLabel",
        background=COLORS["bg"],
        foreground=COLORS["ink"],
        font=(family, title, "bold"),
    )
    style.configure(
        "Brand.TLabel",
        background=COLORS["bg"],
        foreground=COLORS["accent"],
        font=(family, small, "bold"),
    )
    style.configure(
        "Hint.TLabel",
        background=COLORS["bg"],
        foreground=COLORS["muted"],
        font=(family, small),
    )
    style.configure(
        "Status.TLabel",
        background=COLORS["status_bg"],
        foreground=COLORS["muted"],
        font=(family, small),
        padding=(10, 6),
    )
    style.configure(
        "TButton",
        font=(family, base),
        padding=(14, 8),
        background=COLORS["surface_alt"],
        foreground=COLORS["ink"],
        bordercolor=COLORS["border"],
        lightcolor=COLORS["surface_alt"],
        darkcolor=COLORS["border"],
        focuscolor=COLORS["accent_soft"],
    )
    style.map(
        "TButton",
        background=[("active", COLORS["accent_soft"]), ("disabled", COLORS["surface_alt"])],
        foreground=[("disabled", COLORS["muted"])],
    )
    style.configure(
        "Accent.TButton",
        font=(family, base, "bold"),
        padding=(16, 8),
        background=COLORS["accent"],
        foreground="#fffaf3",
        bordercolor=COLORS["accent"],
        lightcolor=COLORS["accent"],
        darkcolor=COLORS["accent_hover"],
    )
    style.map(
        "Accent.TButton",
        background=[("active", COLORS["accent_hover"]), ("disabled", COLORS["border"])],
        foreground=[("disabled", COLORS["muted"])],
    )
    style.configure(
        "TNotebook",
        background=COLORS["bg"],
        borderwidth=0,
        tabmargins=(0, 0, 0, 0),
    )
    style.configure(
        "TNotebook.Tab",
        font=(family, base),
        padding=(18, 8),
        background=COLORS["surface_alt"],
        foreground=COLORS["muted"],
        bordercolor=COLORS["border"],
        lightcolor=COLORS["surface_alt"],
        darkcolor=COLORS["border"],
    )
    style.map(
        "TNotebook.Tab",
        background=[("selected", COLORS["surface"]), ("active", COLORS["accent_soft"])],
        foreground=[("selected", COLORS["ink"]), ("active", COLORS["ink"])],
    )
    style.configure(
        "TScrollbar",
        background=COLORS["surface_alt"],
        troughcolor=COLORS["bg"],
        bordercolor=COLORS["bg"],
        arrowcolor=COLORS["muted"],
    )
    style.configure(
        "TProgressbar",
        background=COLORS["accent"],
        troughcolor=COLORS["surface_alt"],
        bordercolor=COLORS["border"],
        lightcolor=COLORS["accent"],
        darkcolor=COLORS["accent"],
    )

    root._ui_scale = scale  # type: ignore[attr-defined]
    root._ui_fonts = {  # type: ignore[attr-defined]
        "family": family,
        "base": (family, base),
        "title": (family, title, "bold"),
        "body": (family, body),
        "small": (family, small),
        "mono": (family, small),
        "chat": (family, chat),
        "brand": (family, small, "bold"),
    }
    root._ui_colors = COLORS  # type: ignore[attr-defined]
    return scale


def fonts(root: tk.Misc) -> dict[str, tuple]:
    stored = getattr(root, "_ui_fonts", None)
    if stored:
        return stored
    return {
        "family": "sans-serif",
        "base": ("sans-serif", 12),
        "title": ("sans-serif", 14, "bold"),
        "body": ("sans-serif", 13),
        "small": ("sans-serif", 11),
        "mono": ("sans-serif", 11),
        "chat": ("sans-serif", 13),
        "brand": ("sans-serif", 11, "bold"),
    }


def colors(root: tk.Misc | None = None) -> dict[str, str]:
    if root is not None:
        stored = getattr(root, "_ui_colors", None)
        if stored:
            return stored
    return COLORS


def scaled(root: tk.Misc, value: float) -> int:
    scale = float(getattr(root, "_ui_scale", 1.35))
    return int(round(value * scale))


def place_window(
    root: tk.Tk,
    *,
    width: int,
    height: int,
    x: int | None = None,
    y: int | None = None,
    min_width: int | None = None,
    min_height: int | None = None,
) -> None:
    root.update_idletasks()
    sw = max(1, int(root.winfo_screenwidth()))
    sh = max(1, int(root.winfo_screenheight()))
    # Leave margin for window chrome / taskbars
    max_w = max(400, sw - 48)
    max_h = max(300, sh - 96)
    width = max(320, min(int(width), max_w))
    height = max(240, min(int(height), max_h))
    if x is None:
        x = max(16, (sw - width) // 14)
    if y is None:
        y = max(16, (sh - height) // 12)
    mw = int(min_width) if min_width else None
    mh = int(min_height) if min_height else None
    if mw is not None and mh is not None:
        # Never set minsize larger than the actual geometry (clips content)
        mw = min(mw, width)
        mh = min(mh, height)
        root.minsize(mw, mh)
    root.geometry(f"{width}x{height}+{int(x)}+{int(y)}")
    root.update_idletasks()


def fit_window(
    root: tk.Tk,
    *,
    min_width: int,
    min_height: int,
    pad: int = 24,
    max_screen_frac: tuple[float, float] = (0.92, 0.90),
) -> None:
    """
    Size the window to at least the requested mins and the widget req size,
    without exceeding a fraction of the screen.
    """
    root.update_idletasks()
    sw = max(1, int(root.winfo_screenwidth()))
    sh = max(1, int(root.winfo_screenheight()))
    need_w = max(min_width, int(root.winfo_reqwidth()) + pad)
    need_h = max(min_height, int(root.winfo_reqheight()) + pad)
    width = min(need_w, max(400, int(sw * max_screen_frac[0])))
    height = min(need_h, max(300, int(sh * max_screen_frac[1])))
    place_window(
        root,
        width=width,
        height=height,
        min_width=min(min_width, width),
        min_height=min(min_height, height),
    )


def studio_default_size(root: tk.Misc) -> tuple[int, int, int, int]:
    """Return (width, height, min_width, min_height) for the main studio."""
    sw = max(1, int(root.winfo_screenwidth()))
    sh = max(1, int(root.winfo_screenheight()))
    # Prefer a large fraction of the display so header controls aren't clipped
    width = max(scaled(root, 1480), int(sw * 0.88))
    height = max(scaled(root, 900), int(sh * 0.82))
    width = min(width, sw - 48)
    height = min(height, sh - 96)
    min_w = min(scaled(root, 1100), width)
    min_h = min(scaled(root, 720), height)
    return width, height, min_w, min_h
