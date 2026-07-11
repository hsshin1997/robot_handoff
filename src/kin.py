"""Kinematics + collision layer shared by offline grasp checks and the
online handoff oracle. ONE IK path, ONE collision path — everything in
handoff.py must go through these functions.

Conventions:
  - Poses are 4x4 numpy transforms, meters, world frame unless suffixed.
  - "flange" = tool0. Grasps map part<->flange, so all IK targets here are
    flange poses.
  - Functions that take q SET the robot to q (kinematic reset). Callers own
    the scene state; nothing here restores it.
"""
from __future__ import annotations

import numpy as np
import pybullet as p

from scene import Scene, Robot, _pose_from_T

# ---------- transform helpers ----------


def T_from_pos_quat(pos, quat) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = np.reshape(p.getMatrixFromQuaternion(quat), (3, 3))
    T[:3, 3] = pos
    return T


def pos_quat_from_T(T) -> tuple[list[float], list[float]]:
    return _pose_from_T(T)


def inv_T(T) -> np.ndarray:
    T = np.asarray(T, dtype=float)
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def rot_angle(Ra, Rb) -> float:
    """Geodesic angle (rad) between two rotation matrices."""
    tr = np.trace(Ra.T @ Rb)
    return float(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0)))


# ---------- FK / IK ----------

POS_TOL = 1e-3     # m   : IK acceptance
ROT_TOL = 0.01     # rad : IK acceptance (~0.6 deg)


def fk(robot: Robot, q) -> np.ndarray:
    """T_world_flange at q. Sets the robot to q."""
    robot.set_q(q)
    ls = p.getLinkState(robot.body, robot.flange_link, computeForwardKinematics=1)
    return T_from_pos_quat(ls[4], ls[5])


def _ik_once(robot: Robot, T, pos, quat, seed, pos_tol, rot_tol, refine: int = 6):
    """One seeded damped-LS solve with iterative refinement (re-solving from
    the previous iterate is the standard trick to force pybullet's IK to
    converge on orientation). Returns a verified q or None."""
    lo, hi = robot.lower, robot.upper
    q = np.asarray(seed, dtype=float)
    for _ in range(refine):
        robot.set_q(q)
        sol = p.calculateInverseKinematics(
            robot.body, robot.flange_link, pos, quat,
            lowerLimits=lo.tolist(), upperLimits=hi.tolist(),
            jointRanges=(hi - lo).tolist(), restPoses=q.tolist(),
            maxNumIterations=200, residualThreshold=1e-6,
        )
        q = np.clip(np.array(sol[:6]), lo, hi)
        Tq = fk(robot, q)
        if (np.linalg.norm(Tq[:3, 3] - T[:3, 3]) < pos_tol
                and rot_angle(Tq[:3, :3], T[:3, :3]) < rot_tol):
            return q
    return None


def _branch_seeds(robot: Robot, T) -> list[np.ndarray]:
    """Canonical seeds covering the 8 kinematic branch classes of a 6R arm
    (shoulder front/back x elbow up/down x wrist flip), aimed at the target.
    Systematic enumeration replaces analytic branch solutions; every result
    is FK-verified, so completeness is heuristic but correctness is not."""
    base_pos, base_quat = p.getBasePositionAndOrientation(robot.body)
    Tb = T_from_pos_quat(base_pos, base_quat)
    local = inv_T(Tb) @ np.asarray(T, dtype=float)
    aim = float(np.arctan2(local[1, 3], local[0, 3]))   # q1 toward the target
    seeds = []
    for s1 in (aim, aim + np.pi):
        for elbow in (0.8, -0.8):
            for wrist in (-1.2, 1.2):
                q = np.array([s1, -0.3, elbow, 0.0, wrist, 0.0])
                seeds.append(np.clip(q, robot.lower, robot.upper))
    return seeds


def ik_solutions(robot: Robot, T_world_flange, seed=None, restarts: int = 12,
                 max_solutions: int = 8, pos_tol: float = POS_TOL,
                 rot_tol: float = ROT_TOL,
                 rng: np.random.Generator | None = None) -> list[np.ndarray]:
    """Distinct verified IK solutions (different branches) for a flange pose.
    Seeds: caller's seed, the 8 canonical branch classes aimed at the target,
    then random restarts. The oracle filters results by limits/collision —
    never take only the first branch. Deterministic for a fixed rng."""
    T = np.asarray(T_world_flange, dtype=float)
    pos, quat = _pose_from_T(T)
    lo, hi = robot.lower, robot.upper
    rng = rng or np.random.default_rng(0)

    seeds = [np.asarray(seed, dtype=float) if seed is not None else robot.get_q()]
    seeds += _branch_seeds(robot, T)
    seeds += [rng.uniform(lo, hi) for _ in range(restarts)]

    sols: list[np.ndarray] = []
    for s in seeds:
        q = _ik_once(robot, T, pos, quat, s, pos_tol, rot_tol)
        if q is None:
            continue
        if all(np.max(np.abs(q - prev)) > 0.05 for prev in sols):
            sols.append(q)
            if len(sols) >= max_solutions:
                break
    return sols


def ik(robot: Robot, T_world_flange, seed=None, restarts: int = 12,
       pos_tol: float = POS_TOL, rot_tol: float = ROT_TOL,
       rng: np.random.Generator | None = None):
    """First verified IK solution, or None. Same solver as ik_solutions."""
    sols = ik_solutions(robot, T_world_flange, seed=seed, restarts=restarts,
                        max_solutions=1, pos_tol=pos_tol, rot_tol=rot_tol, rng=rng)
    return sols[0] if sols else None


def jacobian(robot: Robot, q) -> np.ndarray:
    """6x6 geometric Jacobian at the flange (rows: v, omega)."""
    robot.set_q(q)
    zero = [0.0] * 6
    jt, jr = p.calculateJacobian(robot.body, robot.flange_link, [0, 0, 0],
                                 list(np.asarray(q, dtype=float)), zero, zero)
    return np.vstack([np.array(jt), np.array(jr)])


def manipulability(robot: Robot, q) -> float:
    """Yoshikawa's measure w = sqrt(det(J J^T)). 0 at a singularity."""
    J = jacobian(robot, q)
    return float(np.sqrt(max(np.linalg.det(J @ J.T), 0.0)))


def calibrate_w_min(robot: Robot, percentile: float = 5.0, n: int = 300,
                    seed: int = 0) -> float:
    """w_min as a percentile of w over random in-limit configs (the doc's
    calibration rule — absolute w values aren't comparable across robots)."""
    rng = np.random.default_rng(seed)
    ws = [manipulability(robot, rng.uniform(robot.lower, robot.upper))
          for _ in range(n)]
    return float(np.percentile(ws, percentile))


def within_limits(robot: Robot, q, margin: float = 0.0) -> bool:
    q = np.asarray(q, dtype=float)
    return bool(np.all(q >= robot.lower + margin) and np.all(q <= robot.upper - margin))


def limit_margin(robot: Robot, q) -> float:
    """Normalized worst-joint distance to a limit, in [0, 1] (0.5 = centered)."""
    q = np.asarray(q, dtype=float)
    span = robot.upper - robot.lower
    return float(np.min(np.minimum(q - robot.lower, robot.upper - q) / span))


# ---------- collision ----------


class CollisionChecker:
    """All collision queries for the cell. Whitelists:
    - each robot's base link vs the workcell (pedestal mount contact);
    - self-collision pairs that are adjacent or already in contact at home;
    - the part vs the holding robot's gripper link (fingers around the part).
    """

    def __init__(self, scene: Scene, clearance: float = 0.002):
        self.s = scene
        self.clearance = clearance
        # env = workcell collision body + floor plane + pcb/bin/nest/stands
        self.env_ids = [b for b in scene.workcell_ids] + list(scene.fixture_ids)
        self.acm = {r.body: self._self_acm(r) for r in (scene.robotA, scene.robotB)}

    # -- self-collision allowed-pair matrix --

    def _self_acm(self, robot: Robot) -> set[tuple[int, int]]:
        pairs: set[tuple[int, int]] = set()
        # adjacent (parent-child) pairs
        for i in range(p.getNumJoints(robot.body)):
            parent = p.getJointInfo(robot.body, i)[16]
            pairs.add((min(parent, i), max(parent, i)))
        # pairs already near contact at the (known-good) home pose
        home = self.s.cfg.get(f"home_q{robot.name}", self.s.cfg.get("home_q", [0.0] * 6))
        robot.set_q(home)
        for pt in p.getClosestPoints(robot.body, robot.body, 0.01):
            a, b = pt[3], pt[4]
            if a != b:
                pairs.add((min(a, b), max(a, b)))
        return pairs

    # -- primitive queries (True = collision) --

    def self_collision(self, robot: Robot) -> bool:
        for pt in p.getClosestPoints(robot.body, robot.body, self.clearance):
            a, b = pt[3], pt[4]
            if a != b and (min(a, b), max(a, b)) not in self.acm[robot.body]:
                return True
        return False

    def robot_env_collision(self, robot: Robot) -> bool:
        for env in self.env_ids:
            for pt in p.getClosestPoints(robot.body, env, self.clearance):
                if pt[3] == -1 and env == self.s.workcell_collision_id:
                    continue  # base vs pedestal mount
                return True
        return False

    def robot_robot_collision(self) -> bool:
        return len(p.getClosestPoints(self.s.robotA.body, self.s.robotB.body,
                                      self.clearance)) > 0

    def part_collision(self, finger_ok: tuple[Robot, ...] = ()) -> bool:
        """Part vs environment and vs both robots. Contact with the gripper
        links of robots in `finger_ok` is allowed (fingers around the part —
        pass both robots at the co-grasp instant)."""
        part = self.s.part_id
        for env in self.env_ids:
            if p.getClosestPoints(part, env, self.clearance):
                return True
        ok_ids = {r.body: (r.link_index["gripper"], r.flange_link, r.tcp_link)
                  for r in finger_ok}
        for r in (self.s.robotA, self.s.robotB):
            for pt in p.getClosestPoints(part, r.body, self.clearance):
                if pt[4] in ok_ids.get(r.body, ()):
                    continue
                return True
        return False

    # -- part placement --

    def place_part(self, holder: Robot, T_flange_part) -> None:
        """Kinematically pose the part on holder's flange (no constraint)."""
        ls = p.getLinkState(holder.body, holder.flange_link)
        rp, rq = _pose_from_T(np.asarray(T_flange_part, dtype=float))
        wp, wq = p.multiplyTransforms(ls[4], ls[5], rp, rq)
        p.resetBasePositionAndOrientation(self.s.part_id, wp, wq)

    # -- the one full-state check the oracle uses --

    def check_state(self, qA=None, qB=None, holder: str | None = "A",
                    T_flange_part=None, finger_ok: tuple[str, ...] | None = None,
                    ) -> tuple[bool, str]:
        """Set both robots (None = leave as-is), pose the part on the holder,
        and run every check. Returns (collision_free, reason).
        finger_ok: robot names whose gripper may touch the part (default: just
        the holder; pass ("A", "B") at the co-grasp instant)."""
        if qA is not None:
            self.s.robotA.set_q(qA)
        if qB is not None:
            self.s.robotB.set_q(qB)
        by_name = {"A": self.s.robotA, "B": self.s.robotB, None: None}
        hold = by_name[holder]
        if hold is not None:
            if T_flange_part is None:
                T_flange_part = np.asarray(self.s.cfg["T_flangeA_part"], dtype=float)
            self.place_part(hold, T_flange_part)
        allowed = tuple(by_name[n] for n in (finger_ok if finger_ok is not None
                                             else (holder,) if holder else ()))

        for r in (self.s.robotA, self.s.robotB):
            if self.self_collision(r):
                return False, f"self_collision_{r.name}"
            if self.robot_env_collision(r):
                return False, f"env_collision_{r.name}"
        if self.robot_robot_collision():
            return False, "robot_robot_collision"
        # part check runs when someone holds it, or when finger_ok is passed
        # explicitly (part fixed in the world, e.g. resting on the nest while
        # a robot reaches for it)
        if (hold is not None or finger_ok is not None) and self.part_collision(allowed):
            return False, "part_collision"
        return True, "ok"

    def set_part_world(self, T_world_part) -> None:
        """Pin the part at a world pose (e.g. resting on the nest)."""
        pos, quat = _pose_from_T(np.asarray(T_world_part, dtype=float))
        p.resetBasePositionAndOrientation(self.s.part_id, pos, quat)

    def min_clearance(self, holder: str | None = "A", dmax: float = 0.05) -> float:
        """Minimum clearance (m) over all non-whitelisted pairs at the current
        state, capped at dmax. For scoring feasible candidates."""
        s = self.s
        hold = {"A": s.robotA, "B": s.robotB, None: None}[holder]
        best = dmax

        def scan(a, b, skip=None):
            nonlocal best
            for pt in p.getClosestPoints(a, b, dmax):
                if skip and skip(pt):
                    continue
                best = min(best, pt[8])

        for r in (s.robotA, s.robotB):
            for env in self.env_ids:
                scan(r.body, env,
                     skip=lambda pt, e=env: pt[3] == -1 and e == s.workcell_collision_id)
        scan(s.robotA.body, s.robotB.body)
        for env in self.env_ids:
            scan(s.part_id, env)
        for r in (s.robotA, s.robotB):
            grip = (r.link_index["gripper"], r.flange_link, r.tcp_link)
            scan(s.part_id, r.body,
                 skip=lambda pt, rr=r: hold is not None and rr.body == hold.body
                 and pt[4] in grip)
        return max(best, 0.0)
