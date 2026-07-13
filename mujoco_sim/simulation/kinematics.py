"""MuJoCo GP7 kinematics used by the handoff planner.

The solver enumerates seeded numerical branches and FK-verifies every result.
It is not analytically complete like IKFast, but it never accepts an
unverified solution and exposes that limitation in planning reports.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .workcell import ARM_JOINTS, WorkcellSim


def rotation_log(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=float)
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))
    if theta < 1e-8:
        return 0.5 * np.array([R[2, 1] - R[1, 2],
                               R[0, 2] - R[2, 0],
                               R[1, 0] - R[0, 1]])
    if np.pi - theta < 1e-5:
        # Stable axis extraction around pi.
        axis = np.sqrt(np.maximum((np.diag(R) + 1.0) / 2.0, 0.0))
        axis[0] = np.copysign(axis[0], R[2, 1] - R[1, 2] or 1.0)
        axis[1] = np.copysign(axis[1], R[0, 2] - R[2, 0] or 1.0)
        axis[2] = np.copysign(axis[2], R[1, 0] - R[0, 1] or 1.0)
        norm = np.linalg.norm(axis)
        return theta * axis / max(norm, 1e-12)
    return theta / (2.0 * np.sin(theta)) * np.array([
        R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def pose_matrix(position, rotation) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = position
    return T


@dataclass(frozen=True)
class IKResult:
    q: np.ndarray
    position_error: float
    rotation_error: float
    iterations: int


class GP7Kinematics:
    def __init__(self, sim: WorkcellSim):
        self.sim = sim
        self.model, self.data = sim.model, sim.data
        self.qadr = {}
        self.dadr = {}
        self.lower = {}
        self.upper = {}
        for robot in ("A", "B"):
            joints = [self.model.joint(f"{robot}_{name}") for name in ARM_JOINTS]
            self.qadr[robot] = np.array([j.qposadr[0] for j in joints], dtype=int)
            self.dadr[robot] = np.array([j.dofadr[0] for j in joints], dtype=int)
            self.lower[robot] = np.array([j.range[0] for j in joints])
            self.upper[robot] = np.array([j.range[1] for j in joints])

    def set_q(self, robot: str, q) -> None:
        q = np.asarray(q, dtype=float)
        self.data.qpos[self.qadr[robot]] = q
        mujoco.mj_forward(self.model, self.data)

    def get_q(self, robot: str) -> np.ndarray:
        return self.data.qpos[self.qadr[robot]].copy()

    def fk(self, robot: str, q=None) -> np.ndarray:
        if q is not None:
            self.set_q(robot, q)
        site = self.data.site(f"{robot}_tcp")
        return pose_matrix(site.xpos.copy(), site.xmat.reshape(3, 3).copy())

    def jacobian(self, robot: str, q=None) -> np.ndarray:
        if q is not None:
            self.set_q(robot, q)
        jp = np.zeros((3, self.model.nv))
        jr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jp, jr,
                          self.model.site(f"{robot}_tcp").id)
        return np.vstack((jp[:, self.dadr[robot]], jr[:, self.dadr[robot]]))

    def pose_error(self, current: np.ndarray, target: np.ndarray) -> np.ndarray:
        # Spatial/world-frame error, consistent with mj_jacSite.
        translation = target[:3, 3] - current[:3, 3]
        rotation = rotation_log(target[:3, :3] @ current[:3, :3].T)
        return np.concatenate((translation, rotation))

    def solve(self, robot: str, target: np.ndarray, seed=None,
              position_tolerance: float = 7e-4,
              rotation_tolerance: float = np.radians(0.35),
              max_iterations: int = 250, damping: float = 2e-3,
              step_limit: float = 0.18) -> IKResult | None:
        lo, hi = self.lower[robot], self.upper[robot]
        q = np.clip(self.get_q(robot) if seed is None else np.asarray(seed, float), lo, hi)
        for iteration in range(max_iterations):
            current = self.fk(robot, q)
            error = self.pose_error(current, target)
            pe, re = np.linalg.norm(error[:3]), np.linalg.norm(error[3:])
            if pe <= position_tolerance and re <= rotation_tolerance:
                return IKResult(q.copy(), float(pe), float(re), iteration)
            J = self.jacobian(robot)
            dq = J.T @ np.linalg.solve(J @ J.T + damping * damping * np.eye(6), error)
            scale = min(1.0, step_limit / max(np.max(np.abs(dq)), 1e-12))
            q = np.clip(q + scale * dq, lo, hi)
        return None

    def branch_seeds(self, robot: str, target: np.ndarray,
                     random_restarts: int, rng: np.random.Generator) -> list[np.ndarray]:
        base = self.data.body(f"{robot}_base")
        local = base.xmat.reshape(3, 3).T @ (target[:3, 3] - base.xpos)
        aim = np.arctan2(local[1], local[0])
        seeds = [self.get_q(robot)]
        for shoulder in (aim, aim + np.pi):
            for elbow in (-0.8, 0.8):
                for wrist in (-1.2, 1.2):
                    seeds.append(np.array([shoulder, -0.3, elbow, 0.0, wrist, 0.0]))
        seeds.extend(rng.uniform(self.lower[robot], self.upper[robot])
                     for _ in range(random_restarts))
        return [np.clip(seed, self.lower[robot], self.upper[robot]) for seed in seeds]

    def solutions(self, robot: str, target: np.ndarray, random_restarts: int = 18,
                  max_solutions: int = 8, rng=None, **solve_kwargs) -> list[IKResult]:
        rng = rng or np.random.default_rng(0)
        results = []
        for seed in self.branch_seeds(robot, target, random_restarts, rng):
            result = self.solve(robot, target, seed=seed, **solve_kwargs)
            if result is None:
                continue
            if all(np.max(np.abs(result.q - existing.q)) > 0.04 for existing in results):
                results.append(result)
                if len(results) >= max_solutions:
                    break
        return results

    def within_limits(self, robot: str, q, margin: float) -> bool:
        q = np.asarray(q)
        return bool(np.all(q >= self.lower[robot] + margin)
                    and np.all(q <= self.upper[robot] - margin))

    def normalized_limit_margin(self, robot: str, q) -> float:
        q = np.asarray(q)
        span = self.upper[robot] - self.lower[robot]
        return float(np.min(2.0 * np.minimum(q - self.lower[robot],
                                            self.upper[robot] - q) / span))

    def singular_values(self, robot: str, q) -> np.ndarray:
        return np.linalg.svd(self.jacobian(robot, q), compute_uv=False)

    def manipulability(self, robot: str, q) -> float:
        return float(np.prod(self.singular_values(robot, q)))

    def penalized_manipulability(self, robot: str, q, k: float = 100.0) -> float:
        q = np.asarray(q)
        lo, hi = self.lower[robot], self.upper[robot]
        product = np.prod((q - lo) * (hi - q) / np.square(hi - lo))
        return self.manipulability(robot, q) * (1.0 - np.exp(-k * max(product, 0.0)))

    def directional_manipulability(self, robot: str, q, direction6) -> float:
        J = self.jacobian(robot, q)
        u = np.asarray(direction6, dtype=float)
        u /= max(np.linalg.norm(u), 1e-12)
        gram = J @ J.T
        return float(1.0 / np.sqrt(max(u @ np.linalg.pinv(gram) @ u, 1e-18)))

    def calibrate_manipulability(self, robot: str, percentile: float = 3.0,
                                 samples: int = 300, seed: int = 13) -> float:
        rng = np.random.default_rng(seed)
        values = [self.manipulability(robot, rng.uniform(self.lower[robot], self.upper[robot]))
                  for _ in range(samples)]
        return float(np.percentile(values, percentile))
