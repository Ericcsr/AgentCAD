#!/usr/bin/env python3
"""Forward kinematics for URDF-style joints() on an assembled DesignResult.

Meshes are stored in assembled world millimetres. Joint values are offsets from
that pose: revolute in radians, prismatic in millimetres.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from cad_pipeline.joints import JointSpec, infer_origin_mm, root_links

_IDENTITY = np.eye(4, dtype=np.float64)


def link_world_origins_mm(result: Any) -> dict[str, tuple[float, float, float]]:
    """World-mm origin of each link frame (world-aligned). Roots stay at 0."""
    names = result.part_names()
    joints = list(getattr(result, "joints", None) or [])
    world: dict[str, tuple[float, float, float]] = {
        name: (0.0, 0.0, 0.0) for name in root_links(names, joints)
    }
    pending = list(joints)
    guard = 0
    while pending and guard < len(pending) + 2:
        guard += 1
        leftover: list[JointSpec] = []
        for joint in pending:
            if joint.parent not in world:
                leftover.append(joint)
                continue
            parent_w = np.asarray(world[joint.parent], dtype=np.float64)
            if joint.origin_mm is not None:
                child_w = np.asarray(joint.origin_mm, dtype=np.float64)
            elif joint.is_moving():
                child_w = np.asarray(
                    infer_origin_mm(
                        result.parts[joint.parent].vertices,
                        result.parts[joint.child].vertices,
                    ),
                    dtype=np.float64,
                )
            else:
                child_w = parent_w
            world[joint.child] = (float(child_w[0]), float(child_w[1]), float(child_w[2]))
        if len(leftover) == len(pending):
            break
        pending = leftover
    for name in names:
        world.setdefault(name, (0.0, 0.0, 0.0))
    return world


def moving_joints(result: Any) -> list[JointSpec]:
    return [j for j in (getattr(result, "joints", None) or []) if j.is_moving()]


def joint_limits(joint: JointSpec) -> tuple[float, float]:
    """Return (lower, upper) in internal units: radians or millimetres."""
    if joint.type == "prismatic":
        lo = -50.0 if joint.lower is None else float(joint.lower)
        hi = 50.0 if joint.upper is None else float(joint.upper)
    elif joint.type == "continuous":
        lo = -math.pi if joint.lower is None else float(joint.lower)
        hi = math.pi if joint.upper is None else float(joint.upper)
    else:
        lo = -math.pi if joint.lower is None else float(joint.lower)
        hi = math.pi if joint.upper is None else float(joint.upper)
    if hi < lo:
        lo, hi = hi, lo
    if abs(hi - lo) < 1e-9:
        hi = lo + 1e-3
    return lo, hi


def _skew(axis: np.ndarray) -> np.ndarray:
    x, y, z = (float(axis[0]), float(axis[1]), float(axis[2]))
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def _rot_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    k = np.asarray(axis, dtype=np.float64)
    n = float(np.linalg.norm(k))
    if n < 1e-12 or abs(angle) < 1e-15:
        return np.eye(3, dtype=np.float64)
    k = k / n
    k_x = _skew(k)
    c, s = math.cos(angle), math.sin(angle)
    return np.eye(3, dtype=np.float64) + s * k_x + (1.0 - c) * (k_x @ k_x)


def _homog(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = rotation
    mat[:3, 3] = translation
    return mat


def _joint_motion(joint: JointSpec, origin: np.ndarray, value: float) -> np.ndarray:
    axis = np.asarray(joint.axis, dtype=np.float64)
    n = float(np.linalg.norm(axis))
    if n < 1e-12:
        return _IDENTITY.copy()
    axis = axis / n
    if joint.type == "prismatic":
        return _homog(np.eye(3, dtype=np.float64), axis * float(value))
    rotation = _rot_axis_angle(axis, float(value))
    translation = origin - rotation @ origin
    return _homog(rotation, translation)


def forward_kinematics(
    result: Any,
    values: Mapping[str, float] | None = None,
) -> dict[str, np.ndarray]:
    """
    4×4 world transforms (mm) for each named part, relative to the assembled pose.

    `values` maps joint name → radians (revolute/continuous) or mm (prismatic).
    Missing joints are 0 (assembled pose).
    """
    names = list(result.part_names())
    joints: list[JointSpec] = list(getattr(result, "joints", None) or [])
    poses = {name: _IDENTITY.copy() for name in names}
    if not joints:
        return poses

    origins = link_world_origins_mm(result)
    vals = values or {}
    pending = list(joints)
    guard = 0
    while pending and guard < len(pending) + 2:
        guard += 1
        leftover: list[JointSpec] = []
        for joint in pending:
            if joint.parent not in poses:
                leftover.append(joint)
                continue
            origin = np.asarray(origins.get(joint.child, (0.0, 0.0, 0.0)), dtype=np.float64)
            q = float(vals.get(joint.name, 0.0)) if joint.is_moving() else 0.0
            local = _joint_motion(joint, origin, q)
            poses[joint.child] = poses[joint.parent] @ local
        if len(leftover) == len(pending):
            break
        pending = leftover
    return poses
