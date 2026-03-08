from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

from .geometry import Capsule


def _package_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _config_path() -> Path:
    return _package_root() / "config" / "xarm6_capsules.yaml"


def _rot_x(angle: float) -> np.ndarray:
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]],
        dtype=np.float64,
    )


def _rot_z(angle: float) -> np.ndarray:
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    return np.array(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _transform_mdh(alpha: float, a: float, d: float, theta_offset: float, q: float) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _rot_x(alpha)
    translate = np.eye(4, dtype=np.float64)
    translate[:3, 3] = np.array([a, 0.0, d], dtype=np.float64)
    rotate = np.eye(4, dtype=np.float64)
    rotate[:3, :3] = _rot_z(theta_offset + q)
    return transform @ translate @ rotate


def _rot_z_correction(mdh_theta_offset: np.ndarray, link_index: int) -> np.ndarray:
    angle = -float(np.sum(mdh_theta_offset[: int(link_index) + 1]))
    return _rot_z(angle)


@dataclass(frozen=True)
class CapsuleDefinition:
    link_index: int
    body_name: str
    center_local: np.ndarray
    axis_local: np.ndarray
    radius: float
    length: float


@dataclass(frozen=True)
class XArm6Config:
    mdh_alpha: np.ndarray
    mdh_a: np.ndarray
    mdh_d: np.ndarray
    mdh_theta_offset: np.ndarray
    joint_lower: np.ndarray
    joint_upper: np.ndarray
    capsules: tuple[CapsuleDefinition, ...]


def _as_f64_array(name: str, value, expected_len: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size != expected_len:
        raise ValueError(f"{name} must have length {expected_len}, got {arr.size}")
    return arr


@lru_cache(maxsize=1)
def load_xarm6_config() -> XArm6Config:
    config_path = _config_path()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    mdh = raw["mdh"]
    joint_limits = raw["joint_limits"]
    mdh_alpha = _as_f64_array("mdh.alpha", mdh["alpha"], 6)
    mdh_a = _as_f64_array("mdh.a", mdh["a"], 6)
    mdh_d = _as_f64_array("mdh.d", mdh["d"], 6)
    mdh_theta_offset = _as_f64_array("mdh.theta_offset", mdh["theta_offset"], 6)
    joint_lower = _as_f64_array("joint_limits.lower", joint_limits["lower"], 6)
    joint_upper = _as_f64_array("joint_limits.upper", joint_limits["upper"], 6)

    capsules = []
    for item in raw["capsules"]:
        link_index = int(item["link_index"])
        center_local = _as_f64_array(f"capsules[{link_index}].center_local", item["center_local"], 3)
        axis_local = _as_f64_array(f"capsules[{link_index}].axis_local", item["axis_local"], 3)
        axis_local = axis_local / np.linalg.norm(axis_local)
        correction = _rot_z_correction(mdh_theta_offset, link_index)
        center_local = correction @ center_local
        axis_local = correction @ axis_local
        axis_local = axis_local / np.linalg.norm(axis_local)
        capsules.append(
            CapsuleDefinition(
                link_index=link_index,
                body_name=str(item["body_name"]),
                center_local=center_local,
                axis_local=axis_local,
                radius=float(item["radius"]),
                length=float(item["length"]),
            )
        )

    return XArm6Config(
        mdh_alpha=mdh_alpha,
        mdh_a=mdh_a,
        mdh_d=mdh_d,
        mdh_theta_offset=mdh_theta_offset,
        joint_lower=joint_lower,
        joint_upper=joint_upper,
        capsules=tuple(capsules),
    )


class XArm6CapsuleModel:
    def __init__(self) -> None:
        self.config = load_xarm6_config()

    def clamp_q(self, q) -> np.ndarray:
        q_arr = np.asarray(q, dtype=np.float64).reshape(6)
        return np.minimum(np.maximum(q_arr, self.config.joint_lower), self.config.joint_upper)

    def forward_link_transforms(self, q) -> list[np.ndarray]:
        q_arr = self.clamp_q(q)
        transforms = []
        current = np.eye(4, dtype=np.float64)
        for idx in range(6):
            current = current @ _transform_mdh(
                float(self.config.mdh_alpha[idx]),
                float(self.config.mdh_a[idx]),
                float(self.config.mdh_d[idx]),
                float(self.config.mdh_theta_offset[idx]),
                float(q_arr[idx]),
            )
            transforms.append(current.copy())
        return transforms

    def compute_all_capsules(self, q) -> list[Capsule]:
        transforms = self.forward_link_transforms(q)
        capsules = []
        for definition in self.config.capsules:
            transform = transforms[definition.link_index]
            rotation = transform[:3, :3]
            translation = transform[:3, 3]
            center_world = rotation @ definition.center_local + translation
            axis_world = rotation @ definition.axis_local
            axis_world = axis_world / np.linalg.norm(axis_world)
            half_length = 0.5 * float(definition.length)
            point0 = center_world - axis_world * half_length
            point1 = center_world + axis_world * half_length
            capsules.append(
                Capsule.from_segment_endpoints(
                    p0=point0,
                    p1=point1,
                    radius=definition.radius,
                    body_name=definition.body_name,
                )
            )
        return capsules

    def compute_active_capsules(self, q) -> list[Capsule]:
        return [capsule for capsule in self.compute_all_capsules(q) if capsule.body_name != "link1"]
