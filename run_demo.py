from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
import sys
import time

import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from nmpc_demo.constants import START_Q
from nmpc_demo.controller import DemoNmpcController
from nmpc_demo.geometry import Capsule, CapsuleDistanceResult, capsule_capsule_distance
from nmpc_demo.rendering import quat_wxyz_to_rot, render_overlay
from nmpc_demo.xarm6_kinematics import XArm6CapsuleModel


SCENE_PATH = PROJECT_ROOT / "scene.xml"
OBSTACLE_BODY_NAME = "obstacle_mocap"
OBSTACLE_GEOM_NAME = "obstacle_capsule"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Standalone xArm6 NMPC dynamic obstacle avoidance demo.")
    parser.add_argument("--backend", choices=("render", "headless"), default="render")
    parser.add_argument("--sim-time", type=float, default=5.0)
    parser.add_argument("--control-dt", type=float, default=0.04)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--max-iter", type=int, default=12)
    parser.add_argument("--ftol", type=float, default=1.0e-4)
    parser.add_argument("--max-joint-vel", type=float, default=0.8)
    parser.add_argument("--clearance", type=float, default=0.10)
    return parser.parse_args(argv)


def compute_total_loops(sim_time: float, control_dt: float) -> int | None:
    if float(sim_time) <= 0.0:
        return None
    return max(1, int(np.ceil(float(sim_time) / float(control_dt))))


def load_model_and_data():
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)
    data.qpos[:6] = START_Q
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)
    return model, data


def build_obstacle_capsule(model, data) -> Capsule:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, OBSTACLE_BODY_NAME)
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, OBSTACLE_GEOM_NAME)
    if body_id < 0 or geom_id < 0:
        raise RuntimeError("Obstacle mocap body or geom not found in scene.")
    mocap_id = int(model.body_mocapid[body_id])
    if mocap_id < 0:
        raise RuntimeError("Obstacle body is not backed by mocap.")
    center = np.asarray(data.mocap_pos[mocap_id], dtype=np.float64).reshape(3)
    quat = np.asarray(data.mocap_quat[mocap_id], dtype=np.float64).reshape(4)
    rotation = quat_wxyz_to_rot(quat)
    axis = rotation[:, 2]
    radius = float(model.geom_size[geom_id, 0])
    half_length = float(model.geom_size[geom_id, 1])
    point0 = center - axis * half_length
    point1 = center + axis * half_length
    return Capsule.from_segment_endpoints(
        p0=point0,
        p1=point1,
        radius=radius,
        body_name=OBSTACLE_GEOM_NAME,
    )


def compute_nearest_distance(robot_capsules, obstacle_capsule: Capsule) -> tuple[float, CapsuleDistanceResult | None]:
    min_distance = float("inf")
    best_result = None
    for capsule in robot_capsules:
        result = capsule_capsule_distance(capsule, obstacle_capsule)
        if result.distance < min_distance:
            min_distance = result.distance
            best_result = result
    return min_distance, best_result


def run(argv=None) -> int:
    args = parse_args(argv)
    model, data = load_model_and_data()
    capsule_model = XArm6CapsuleModel()
    controller = DemoNmpcController(
        model=capsule_model,
        horizon=args.horizon,
        dt=args.control_dt,
        max_iter=args.max_iter,
        ftol=args.ftol,
        max_joint_vel=args.max_joint_vel,
        clearance=args.clearance,
    )

    sim_dt = float(model.opt.timestep)
    control_substeps = max(1, int(round(float(args.control_dt) / sim_dt)))
    total_loops = compute_total_loops(sim_time=args.sim_time, control_dt=args.control_dt)

    viewer_context = nullcontext(None)
    if args.backend == "render":
        from mujoco import viewer as mujoco_viewer

        viewer_context = mujoco_viewer.launch_passive(model, data)

    predicted_min_distances = []
    actual_min_distances = []

    with viewer_context as viewer:
        loop_iter = range(total_loops) if total_loops is not None else iter(int, 1)
        for _ in loop_iter:
            if viewer is not None and not viewer.is_running():
                break

            obstacle_capsule = build_obstacle_capsule(model, data)
            result = controller.solve(q_now=np.asarray(data.qpos[:6], dtype=np.float64), obstacle=obstacle_capsule)
            predicted_min_distances.append(result.predicted_min_distance)
            data.ctrl[:] = result.command
            for _ in range(control_substeps):
                mujoco.mj_step(model, data)

            robot_capsules = capsule_model.compute_active_capsules(np.asarray(data.qpos[:6], dtype=np.float64))
            actual_min_distance, nearest_result = compute_nearest_distance(robot_capsules, obstacle_capsule)
            actual_min_distances.append(actual_min_distance)

            if viewer is not None:
                with viewer.lock():
                    viewer.user_scn.ngeom = 0
                    render_overlay(viewer.user_scn, robot_capsules, obstacle_capsule, nearest_result)
                viewer.sync()
            else:
                time.sleep(0.0)

    final_q_error_norm = float(np.linalg.norm(np.asarray(data.qpos[:6], dtype=np.float64) - START_Q))
    predicted_min_distance = float(min(predicted_min_distances)) if predicted_min_distances else float("nan")
    actual_min_distance = float(min(actual_min_distances)) if actual_min_distances else float("nan")
    print(f"predicted_min_distance_m={predicted_min_distance:.6f}")
    print(f"actual_min_distance_m={actual_min_distance:.6f}")
    print(f"final_q_error_norm={final_q_error_norm:.6f}")
    return 0


def main(argv=None) -> int:
    return run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
