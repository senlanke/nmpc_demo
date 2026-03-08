from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from nmpc_demo.geometry import Capsule, capsule_capsule_distance


def test_capsule_capsule_distance_for_parallel_separated_capsules():
    capsule_a = Capsule.from_segment_endpoints(
        p0=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        p1=np.array([0.0, 0.0, 1.0], dtype=np.float64),
        radius=0.1,
        body_name="a",
    )
    capsule_b = Capsule.from_segment_endpoints(
        p0=np.array([0.5, 0.0, 0.0], dtype=np.float64),
        p1=np.array([0.5, 0.0, 1.0], dtype=np.float64),
        radius=0.1,
        body_name="b",
    )

    result = capsule_capsule_distance(capsule_a, capsule_b)

    assert np.isclose(result.distance, 0.3, atol=1e-6)
    assert result.point_a.shape == (3,)
    assert result.point_b.shape == (3,)


def test_capsule_capsule_distance_is_non_negative_when_overlapping():
    capsule_a = Capsule.from_segment_endpoints(
        p0=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        p1=np.array([0.0, 0.0, 1.0], dtype=np.float64),
        radius=0.2,
        body_name="a",
    )
    capsule_b = Capsule.from_segment_endpoints(
        p0=np.array([0.1, 0.0, 0.0], dtype=np.float64),
        p1=np.array([0.1, 0.0, 1.0], dtype=np.float64),
        radius=0.2,
        body_name="b",
    )

    result = capsule_capsule_distance(capsule_a, capsule_b)

    assert np.isclose(result.distance, 0.0, atol=1e-9)
    assert result.centerline_distance <= capsule_a.radius + capsule_b.radius
