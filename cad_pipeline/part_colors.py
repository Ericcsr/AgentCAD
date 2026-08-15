#!/usr/bin/env python3
"""Assign contrasting preview colors so contacting parts do not share a hue."""

from __future__ import annotations

from typing import Any

import numpy as np

from cad_pipeline.collision import WELD_CONTACT_GAP_MM, _aabb, _aabb_separation_mm

# Distinct workshop tones — adjacent parts in the contact graph get different slots.
PART_PALETTE: tuple[tuple[float, float, float], ...] = (
    (0.76, 0.58, 0.38),  # oak
    (0.32, 0.50, 0.68),  # steel blue
    (0.66, 0.36, 0.34),  # terracotta
    (0.42, 0.58, 0.40),  # sage
    (0.78, 0.64, 0.28),  # mustard
    (0.50, 0.40, 0.60),  # plum
    (0.30, 0.50, 0.50),  # teal
    (0.58, 0.42, 0.26),  # walnut
    (0.62, 0.58, 0.52),  # stone
    (0.78, 0.52, 0.48),  # clay
    (0.38, 0.44, 0.58),  # indigo
    (0.52, 0.62, 0.36),  # olive
)

DEFAULT_PART_COLOR = PART_PALETTE[0]


def _contact_graph(result: Any) -> dict[str, set[str]]:
    parts = getattr(result, "parts", None) or {}
    names = [n for n in parts if n]
    graph: dict[str, set[str]] = {n: set() for n in names}
    boxes: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in names:
        verts = getattr(parts[name], "vertices", None)
        if verts is None or len(verts) == 0:
            continue
        boxes[name] = _aabb(np.asarray(verts, dtype=np.float64))
    ordered = [n for n in names if n in boxes]
    for i, a in enumerate(ordered):
        amin, amax = boxes[a]
        for b in ordered[i + 1 :]:
            bmin, bmax = boxes[b]
            if _aabb_separation_mm(amin, amax, bmin, bmax) <= WELD_CONTACT_GAP_MM:
                graph[a].add(b)
                graph[b].add(a)
    for joint in getattr(result, "joints", None) or []:
        a, b = getattr(joint, "parent", ""), getattr(joint, "child", "")
        if a in graph and b in graph and a != b:
            graph[a].add(b)
            graph[b].add(a)
    return graph


def assign_part_colors(result: Any) -> dict[str, tuple[float, float, float]]:
    """Greedy graph coloring: contacting / jointed parts get different palette slots."""
    parts = getattr(result, "parts", None) or {}
    names = list(parts.keys())
    if not names:
        return {}
    if len(names) == 1:
        return {names[0]: DEFAULT_PART_COLOR}

    graph = _contact_graph(result)
    order = sorted(names, key=lambda n: (-len(graph.get(n, ())), n))
    index: dict[str, int] = {}
    n_colors = len(PART_PALETTE)
    for name in order:
        used = {index[nb] for nb in graph.get(name, ()) if nb in index}
        choice = 0
        while choice in used:
            choice += 1
        index[name] = choice % n_colors
        # If we wrapped into a neighbor's color, pick the least-conflicting slot
        if index[name] in used and n_colors > 1:
            for cand in range(n_colors):
                if cand not in used:
                    index[name] = cand
                    break
    return {name: PART_PALETTE[index[name]] for name in names}
