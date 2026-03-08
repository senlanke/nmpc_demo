from __future__ import annotations

from typing import Iterable

import numpy as np

from .geometry import Capsule, CapsuleDistanceResult

try:
    import mujoco
except Exception:  # pragma: no cover
    mujoco = None


def quat_wxyz_to_rot(quat_wxyz) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    w, x, y, z = quat
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotation_from_z(direction) -> np.ndarray:
    z_axis = np.asarray(direction, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(z_axis))
    if norm <= 1.0e-12:
        return np.eye(3, dtype=np.float64)
    z_axis = z_axis / norm
    helper = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(helper, z_axis))) > 0.9:
        helper = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    x_axis = np.cross(helper, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    return np.column_stack((x_axis, y_axis, z_axis))


def add_geom(scene, geom_type, pos, mat, size, rgba) -> None:
    if mujoco is None or scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        geom_type,
        np.asarray(size, dtype=np.float64),
        np.asarray(pos, dtype=np.float64),
        np.asarray(mat, dtype=np.float64),
        np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def add_capsule(scene, capsule: Capsule, rgba) -> None:
    center = 0.5 * (capsule.point0 + capsule.point1)
    axis = capsule.point1 - capsule.point0
    half_length = 0.5 * float(np.linalg.norm(axis))
    rotation = rotation_from_z(axis)
    add_geom(
        scene,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        pos=center,
        mat=rotation.reshape(9),
        size=np.array([capsule.radius, half_length, 0.0], dtype=np.float64),
        rgba=np.asarray(rgba, dtype=np.float32),
    )


def add_sphere(scene, pos, radius: float, rgba) -> None:
    add_geom(
        scene,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        pos=np.asarray(pos, dtype=np.float64),
        mat=np.eye(3, dtype=np.float64).reshape(9),
        size=np.array([radius, 0.0, 0.0], dtype=np.float64),
        rgba=np.asarray(rgba, dtype=np.float32),
    )


def add_segment(scene, point0, point1, radius: float, rgba) -> None:
    p0 = np.asarray(point0, dtype=np.float64).reshape(3)
    p1 = np.asarray(point1, dtype=np.float64).reshape(3)
    direction = p1 - p0
    length = float(np.linalg.norm(direction))
    if length <= 1.0e-12:
        add_sphere(scene, p0, radius=radius, rgba=rgba)
        return
    center = 0.5 * (p0 + p1)
    rotation = rotation_from_z(direction)
    add_geom(
        scene,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        pos=center,
        mat=rotation.reshape(9),
        size=np.array([radius, 0.5 * length, 0.0], dtype=np.float64),
        rgba=np.asarray(rgba, dtype=np.float32),
    )


def render_overlay(
    scene,
    robot_capsules: Iterable[Capsule],
    obstacle_capsule: Capsule,
    nearest_result: CapsuleDistanceResult | None,
) -> None:
    if mujoco is None:
        return
    for capsule in robot_capsules:
        add_capsule(scene, capsule, rgba=(0.98, 0.24, 0.24, 0.32))
    add_capsule(scene, obstacle_capsule, rgba=(0.98, 0.55, 0.18, 0.75))
    if nearest_result is not None:
        add_segment(
            scene,
            nearest_result.point_a,
            nearest_result.point_b,
            radius=0.0025,
            rgba=(0.15, 0.95, 0.25, 0.95),
        )
        add_sphere(scene, nearest_result.point_a, radius=0.006, rgba=(0.1, 1.0, 0.1, 1.0))
        add_sphere(scene, nearest_result.point_b, radius=0.006, rgba=(1.0, 0.95, 0.1, 1.0))
