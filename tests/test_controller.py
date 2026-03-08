from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from nmpc_demo.constants import START_Q
from nmpc_demo.controller import DemoNmpcController
from nmpc_demo.geometry import Capsule
from nmpc_demo.xarm6_kinematics import XArm6CapsuleModel


def _shifted_capsule(source: Capsule, shift_xyz) -> Capsule:
    shift = np.asarray(shift_xyz, dtype=np.float64).reshape(3)
    return Capsule.from_segment_endpoints(
        p0=source.point0 + shift,
        p1=source.point1 + shift,
        radius=source.radius,
        body_name="obstacle",
    )


def test_controller_holds_start_q_when_obstacle_is_far():
    model = XArm6CapsuleModel()
    controller = DemoNmpcController(model=model, horizon=4, dt=0.04, max_iter=8)
    obstacle = Capsule.from_segment_endpoints(
        p0=np.array([2.0, 0.0, 0.0], dtype=np.float64),
        p1=np.array([2.0, 0.0, 0.3], dtype=np.float64),
        radius=0.05,
        body_name="obstacle",
    )

    result = controller.solve(q_now=START_Q, obstacle=obstacle)

    assert result.command.shape == (6,)
    assert np.linalg.norm(result.command) < 1.0e-3


def test_controller_moves_when_obstacle_intrudes_near_active_capsule():
    model = XArm6CapsuleModel()
    controller = DemoNmpcController(model=model, horizon=4, dt=0.04, max_iter=8)
    active_capsules = model.compute_active_capsules(START_Q)
    obstacle = _shifted_capsule(active_capsules[0], [0.03, 0.0, 0.0])

    result = controller.solve(q_now=START_Q, obstacle=obstacle)

    assert result.command.shape == (6,)
    assert np.linalg.norm(result.command) > 1.0e-3


def test_controller_uses_stronger_default_obstacle_settings():
    model = XArm6CapsuleModel()
    controller = DemoNmpcController(model=model)

    assert controller.clearance == 0.2
    assert controller.w_obstacle == 3000.0
