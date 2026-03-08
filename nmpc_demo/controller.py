from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from scipy.optimize import minimize

from .constants import START_Q
from .geometry import Capsule, capsule_capsule_distance
from .xarm6_kinematics import XArm6CapsuleModel


@dataclass(frozen=True)
class NmpcResult:
    command: np.ndarray
    solved: bool
    solve_time_s: float
    cost: float
    predicted_min_distance: float


class DemoNmpcController:
    def __init__(
        self,
        model: XArm6CapsuleModel,
        horizon: int = 6,
        dt: float = 0.04,
        max_iter: int = 15,
        ftol: float = 1.0e-4,
        max_joint_vel: float = 0.8,
        clearance: float = 0.20,
    ) -> None:
        self.model = model
        self.horizon = int(horizon)
        self.dt = float(dt)
        self.max_iter = int(max_iter)
        self.ftol = float(ftol)
        self.max_joint_vel = float(max_joint_vel)
        self.clearance = float(clearance)
        self.q_target = np.asarray(START_Q, dtype=np.float64).reshape(6)
        self._u_warm = np.zeros((self.horizon, 6), dtype=np.float64)
        self.bounds = [
            (-self.max_joint_vel, self.max_joint_vel)
            for _ in range(self.horizon)
            for _ in range(6)
        ]
        self.w_goal = 40.0
        self.w_terminal = 120.0
        self.w_ctrl = 0.05
        self.w_dctrl = 0.2
        self.w_joint_limit = 500.0
        self.w_obstacle = 3000.0

    def _joint_limit_penalty(self, q: np.ndarray) -> float:
        low = np.maximum(self.model.config.joint_lower - q, 0.0)
        high = np.maximum(q - self.model.config.joint_upper, 0.0)
        violation = low + high
        return float(violation @ violation)

    def _obstacle_penalty(self, q: np.ndarray, obstacle: Capsule) -> tuple[float, float]:
        capsules = self.model.compute_active_capsules(q)
        min_distance = float("inf")
        penalty = 0.0
        for capsule in capsules:
            result = capsule_capsule_distance(capsule, obstacle)
            min_distance = min(min_distance, result.distance)
            intrusion = max(0.0, self.clearance - result.distance)
            penalty += intrusion * intrusion
        return penalty, min_distance

    def _rollout_cost(self, flat_u: np.ndarray, q0: np.ndarray, obstacle: Capsule) -> float:
        us = flat_u.reshape(self.horizon, 6)
        q = np.asarray(q0, dtype=np.float64).reshape(6).copy()
        total = 0.0
        prev_u = None
        for step in range(self.horizon):
            u = us[step]
            q = self.model.clamp_q(q + self.dt * u)
            e = q - self.q_target
            total += self.w_goal * float(e @ e)
            total += self.w_ctrl * float(u @ u)
            total += self.w_joint_limit * self._joint_limit_penalty(q)
            obstacle_penalty, _ = self._obstacle_penalty(q, obstacle)
            total += self.w_obstacle * obstacle_penalty
            if prev_u is not None:
                du = u - prev_u
                total += self.w_dctrl * float(du @ du)
            prev_u = u
        e_terminal = q - self.q_target
        total += self.w_terminal * float(e_terminal @ e_terminal)
        return float(total)

    def _predicted_min_distance(self, q0: np.ndarray, us: np.ndarray, obstacle: Capsule) -> float:
        q = np.asarray(q0, dtype=np.float64).reshape(6).copy()
        min_distance = float("inf")
        for u in us.reshape(self.horizon, 6):
            q = self.model.clamp_q(q + self.dt * u)
            _, step_min_distance = self._obstacle_penalty(q, obstacle)
            min_distance = min(min_distance, step_min_distance)
        return min_distance

    def solve(self, q_now, obstacle: Capsule) -> NmpcResult:
        q0 = self.model.clamp_q(q_now)
        x0 = self._u_warm.reshape(-1)
        t0 = time.time()
        result = minimize(
            self._rollout_cost,
            x0,
            args=(q0, obstacle),
            method="L-BFGS-B",
            bounds=self.bounds,
            options={"maxiter": self.max_iter, "ftol": self.ftol},
        )
        solve_time_s = time.time() - t0
        if result.success:
            us = np.asarray(result.x, dtype=np.float64).reshape(self.horizon, 6)
            self._u_warm = np.vstack([us[1:], us[-1:]])
            command = us[0]
            cost = float(result.fun)
        else:
            self._u_warm[:] = 0.0
            command = np.zeros(6, dtype=np.float64)
            cost = float(self._rollout_cost(np.zeros(self.horizon * 6, dtype=np.float64), q0, obstacle))
            us = np.zeros((self.horizon, 6), dtype=np.float64)
        predicted_min_distance = self._predicted_min_distance(q0, us, obstacle)
        return NmpcResult(
            command=np.asarray(command, dtype=np.float64),
            solved=bool(result.success),
            solve_time_s=float(solve_time_s),
            cost=cost,
            predicted_min_distance=float(predicted_min_distance),
        )
