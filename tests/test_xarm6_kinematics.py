from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from nmpc_demo.constants import START_Q
from nmpc_demo.xarm6_kinematics import XArm6CapsuleModel


def test_active_capsules_exclude_link1_and_keep_five_links():
    model = XArm6CapsuleModel()

    capsules = model.compute_active_capsules(START_Q)

    assert len(capsules) == 5
    assert [capsule.body_name for capsule in capsules] == ["link2", "link3", "link4", "link5", "link6"]
