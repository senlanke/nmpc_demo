from __future__ import annotations

from dataclasses import dataclass

import numpy as np


EPS = 1.0e-12


@dataclass(frozen=True)
class Capsule:
    point0: np.ndarray
    point1: np.ndarray
    radius: float
    body_name: str

    @classmethod
    def from_segment_endpoints(
        cls,
        p0: np.ndarray,
        p1: np.ndarray,
        radius: float,
        body_name: str,
    ) -> "Capsule":
        return cls(
            point0=np.asarray(p0, dtype=np.float64).reshape(3),
            point1=np.asarray(p1, dtype=np.float64).reshape(3),
            radius=float(radius),
            body_name=str(body_name),
        )

    @property
    def axis(self) -> np.ndarray:
        return self.point1 - self.point0

    @property
    def center(self) -> np.ndarray:
        return 0.5 * (self.point0 + self.point1)

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.axis))


@dataclass(frozen=True)
class CapsuleDistanceResult:
    distance: float
    centerline_distance: float
    point_a: np.ndarray
    point_b: np.ndarray
    segment_point_a: np.ndarray
    segment_point_b: np.ndarray


def _segment_closest_points(
    p1: np.ndarray,
    q1: np.ndarray,
    p2: np.ndarray,
    q2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    a = float(d1 @ d1)
    e = float(d2 @ d2)
    f = float(d2 @ r)

    if a <= EPS and e <= EPS:
        return p1.copy(), p2.copy()
    if a <= EPS:
        s = 0.0
        t = np.clip(f / e, 0.0, 1.0)
    else:
        c = float(d1 @ r)
        if e <= EPS:
            t = 0.0
            s = np.clip(-c / a, 0.0, 1.0)
        else:
            b = float(d1 @ d2)
            denom = a * e - b * b
            if denom > EPS:
                s = np.clip((b * f - c * e) / denom, 0.0, 1.0)
            else:
                s = 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = np.clip(-c / a, 0.0, 1.0)
            elif t > 1.0:
                t = 1.0
                s = np.clip((b - c) / a, 0.0, 1.0)

    c1 = p1 + d1 * s
    c2 = p2 + d2 * t
    return c1, c2


def capsule_capsule_distance(capsule_a: Capsule, capsule_b: Capsule) -> CapsuleDistanceResult:
    seg_a, seg_b = _segment_closest_points(
        capsule_a.point0,
        capsule_a.point1,
        capsule_b.point0,
        capsule_b.point1,
    )
    delta = seg_b - seg_a
    centerline_distance = float(np.linalg.norm(delta))
    if centerline_distance > EPS:
        direction = delta / centerline_distance
        point_a = seg_a + direction * capsule_a.radius
        point_b = seg_b - direction * capsule_b.radius
    else:
        direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        point_a = seg_a.copy()
        point_b = seg_b.copy()
    surface_distance = max(0.0, centerline_distance - capsule_a.radius - capsule_b.radius)
    return CapsuleDistanceResult(
        distance=float(surface_distance),
        centerline_distance=centerline_distance,
        point_a=np.asarray(point_a, dtype=np.float64),
        point_b=np.asarray(point_b, dtype=np.float64),
        segment_point_a=np.asarray(seg_a, dtype=np.float64),
        segment_point_b=np.asarray(seg_b, dtype=np.float64),
    )
