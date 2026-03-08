from pathlib import Path

import mujoco


def test_scene_contains_mocap_obstacle_body():
    scene_path = Path(__file__).resolve().parents[1] / "scene.xml"

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "obstacle_mocap")

    assert body_id >= 0
    assert model.body_mocapid[body_id] >= 0
