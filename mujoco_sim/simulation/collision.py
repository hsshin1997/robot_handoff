"""Collision, semantic-contact, and swept-path checks for the planning scene.

The part is not allowed to contact an entire robot link merely because that
robot is its holder.  Only gripper collision components may touch a held part,
and even those contacts have a bounded penetration allowance.  This is an
important distinction for dual-arm handoff: allowing ``link_6`` (as the old
prototype did) can hide a wrist/part collision.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .kinematics import GP7Kinematics
from ..planner.motion import (JointRRTConnect, MotionPlanResult,
                              MotionPlannerConfig, validate_edge)
from .workcell import WorkcellSim


_PART_GEOM = "part_collision"
_PART_CHUNK_PREFIX = f"{_PART_GEOM}_"


def is_part_collision_geom(name: str) -> bool:
    """Return whether *name* is the canonical part geom or a numbered chunk.

    Exact visual-CAD preprocessing may split one part mesh into multiple MuJoCo
    assets/geoms.  Restricting the suffix to decimal digits avoids accidentally
    granting part-contact semantics to unrelated names such as
    ``part_collision_fixture``.
    """
    return (name == _PART_GEOM
            or (name.startswith(_PART_CHUNK_PREFIX)
                and name[len(_PART_CHUNK_PREFIX):].isdigit()))


def _semantic_geom_name(name: str) -> str:
    return _PART_GEOM if is_part_collision_geom(name) else name


def _semantic_pair(name1: str, name2: str) -> tuple[str, str]:
    """Canonicalize mesh chunks for phase-contact policy matching."""
    return tuple(sorted((_semantic_geom_name(name1),
                         _semantic_geom_name(name2))))


def _other_than_part(name1: str, name2: str) -> str | None:
    """Return the non-part name when exactly one geom is part geometry."""
    first_part = is_part_collision_geom(name1)
    second_part = is_part_collision_geom(name2)
    if first_part == second_part:
        return None
    return name2 if first_part else name1


@dataclass(frozen=True)
class CollisionResult:
    free: bool
    reason: str = "ok"
    pair: tuple[str, str] | None = None
    penetration: float = 0.0


@dataclass(frozen=True)
class AllowedContact:
    """One semantic geom pair and its maximum tolerated penetration."""

    geom1: str
    geom2: str
    max_penetration_m: float = 0.00075

    @property
    def pair(self) -> tuple[str, str]:
        # A canonical allowance such as
        # ("part_collision", "reorientation_surface") applies to every
        # automatically generated part_collision_<n> chunk.
        return _semantic_pair(self.geom1, self.geom2)

    @staticmethod
    def _matches_pattern(pattern: str, name: str) -> bool:
        return (name.startswith(pattern[:-1]) if pattern.endswith("*")
                else _semantic_geom_name(name) == _semantic_geom_name(pattern))

    def matches_names(self, name1: str, name2: str) -> bool:
        """Match exact semantic names or a trailing-``*`` prefix pattern."""
        return bool(
            (self._matches_pattern(self.geom1, name1)
             and self._matches_pattern(self.geom2, name2))
            or
            (self._matches_pattern(self.geom1, name2)
             and self._matches_pattern(self.geom2, name1))
        )

    def allows(self, name1: str, name2: str, penetration_m: float) -> bool:
        return (self.matches_names(name1, name2)
                and penetration_m <= self.max_penetration_m)


@dataclass(frozen=True)
class CollisionPolicy:
    """Contacts expected for one manipulation phase.

    ``part_holders`` is a transitional convenience for static grippers.  It
    permits only ``<robot>_gripper_collision_*`` contacts with the part, never
    arbitrary wrist/link contacts.  Articulated models should pass explicit
    pad geom names in ``allowed_contacts``.
    """

    part_holders: tuple[str, ...] = ()
    allowed_contacts: tuple[AllowedContact, ...] = ()
    holder_contact_penetration_m: float = 0.00075


class SceneCollisionChecker:
    def __init__(self, sim: WorkcellSim, kinematics: GP7Kinematics,
                 edge_max_joint_step: float = 0.035,
                 clearance_margin_m: float = 0.0):
        self.sim, self.kin = sim, kinematics
        self.model, self.data = sim.model, sim.data
        self.edge_max_joint_step = float(edge_max_joint_step)
        if not np.isfinite(self.edge_max_joint_step) or self.edge_max_joint_step <= 0:
            raise ValueError("edge_max_joint_step must be positive and finite")
        self.clearance_margin_m = float(clearance_margin_m)
        if (not np.isfinite(self.clearance_margin_m)
                or self.clearance_margin_m < 0.0):
            raise ValueError("clearance_margin_m must be finite and non-negative")
        if self.clearance_margin_m > 0.0:
            # MuJoCo combines the two geom margins. Split the required pairwise
            # clearance equally and set gap=margin so positive-distance
            # broadphase contacts are visible to this checker but exert no
            # physical force. Actual penetration still activates contact.
            active = ((self.model.geom_contype != 0)
                      | (self.model.geom_conaffinity != 0))
            per_geom = 0.5 * self.clearance_margin_m
            self.model.geom_margin[active] = np.maximum(
                self.model.geom_margin[active], per_geom)
            self.model.geom_gap[active] = np.maximum(
                self.model.geom_gap[active], self.model.geom_margin[active])
        self._baseline = self._contact_pairs()

    @staticmethod
    def _same_robot_pair(name1: str, name2: str) -> bool:
        return any(name1.startswith(f"{robot}_")
                   and name2.startswith(f"{robot}_")
                   for robot in ("A", "B"))

    def _contact_pairs(self):
        mujoco.mj_forward(self.model, self.data)
        pairs = set()
        for contact in self.data.contact[:self.data.ncon]:
            names = tuple(sorted((self.model.geom(contact.geom1).name,
                                  self.model.geom(contact.geom2).name)))
            pairs.add(names)
        return pairs

    def _adjacent(self, geom1: int, geom2: int) -> bool:
        body1, body2 = self.model.geom_bodyid[geom1], self.model.geom_bodyid[geom2]
        return (body1 == body2 or self.model.body_parentid[body1] == body2
                or self.model.body_parentid[body2] == body1)

    @staticmethod
    def _legacy_policy(allowed_part_holders: tuple[str, ...],
                       allowed_geom_pairs: tuple[tuple, ...]) -> CollisionPolicy:
        contacts = []
        for entry in allowed_geom_pairs:
            if len(entry) == 2:
                contacts.append(AllowedContact(entry[0], entry[1]))
            elif len(entry) == 3:
                contacts.append(AllowedContact(entry[0], entry[1], entry[2]))
            else:
                raise ValueError(
                    "allowed geom contact must be (geom1, geom2[, max_penetration_m])"
                )
        return CollisionPolicy(tuple(allowed_part_holders), tuple(contacts))

    def _allowed(self, contact, allowed_part_holders: tuple[str, ...] = (),
                 allowed_geom_pairs: tuple[tuple, ...] = (),
                 policy: CollisionPolicy | None = None) -> bool:
        policy = policy or self._legacy_policy(allowed_part_holders,
                                                allowed_geom_pairs)
        name1, name2 = self.model.geom(contact.geom1).name, self.model.geom(contact.geom2).name
        exact_pair = tuple(sorted((name1, name2)))
        penetration = float(max(0.0, -contact.dist))
        for allowance in policy.allowed_contacts:
            if allowance.allows(name1, name2, penetration):
                return True
        # Manufacturer link geometry may have nominal self gaps smaller than
        # the cell-wide environmental clearance. Ignore only positive-distance
        # broadphase margin contacts within one robot; real self penetration
        # remains a collision unless the bodies are adjacent below.
        if contact.dist >= 0.0 and self._same_robot_pair(name1, name2):
            return True
        if exact_pair in self._baseline and self._adjacent(contact.geom1, contact.geom2):
            return True
        if self._adjacent(contact.geom1, contact.geom2):
            return True
        other = _other_than_part(name1, name2)
        if other is not None:
            if any(other.startswith(f"{robot}_gripper_collision_")
                   for robot in policy.part_holders):
                return penetration <= policy.holder_contact_penetration_m
        return False

    def _inspect_current(self, policy: CollisionPolicy,
                         run_forward: bool = True) -> CollisionResult:
        if run_forward:
            mujoco.mj_forward(self.model, self.data)
        worst: CollisionResult | None = None
        for contact in self.data.contact[:self.data.ncon]:
            if self._allowed(contact, policy=policy):
                continue
            name1 = self.model.geom(contact.geom1).name
            name2 = self.model.geom(contact.geom2).name
            result = CollisionResult(False, "collision", (name1, name2),
                                     float(max(0.0, -contact.dist)))
            if worst is None or result.penetration > worst.penetration:
                worst = result
        return CollisionResult(True) if worst is None else worst

    def check_current(self, policy: CollisionPolicy | None = None,
                      allowed_part_holders: tuple[str, ...] = (),
                      allowed_geom_pairs: tuple[tuple, ...] = ()) -> CollisionResult:
        """Inspect the live execution state without changing weld ownership."""
        return self._inspect_current(policy or self._legacy_policy(
            allowed_part_holders, allowed_geom_pairs), run_forward=True)

    def check(self, qA, qB, X_world_part: np.ndarray,
              allowed_part_holders: tuple[str, ...] = (),
              allowed_geom_pairs: tuple[tuple, ...] = (),
              policy: CollisionPolicy | None = None) -> CollisionResult:
        # Update the coupled planning state and run exactly one forward pass.
        # Calling the public setters separately used to perform three full
        # mj_forward calls per sample, dominating swept-collision CT.
        self.sim.release_part()
        self.data.qpos[self.kin.qadr["A"]] = np.asarray(qA, dtype=float)
        self.data.qpos[self.kin.qadr["B"]] = np.asarray(qB, dtype=float)
        transform = np.asarray(X_world_part, dtype=float)
        if transform.shape != (4, 4):
            raise ValueError("X_world_part must be a 4x4 transform")
        quaternion = np.empty(4)
        mujoco.mju_mat2Quat(quaternion, transform[:3, :3].ravel())
        address = int(self.model.joint("part_free").qposadr[0])
        self.data.qpos[address:address + 3] = transform[:3, 3]
        self.data.qpos[address + 3:address + 7] = quaternion
        dof = int(self.model.joint("part_free").dofadr[0])
        self.data.qvel[dof:dof + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)
        return self._inspect_current(policy or self._legacy_policy(
            allowed_part_holders, allowed_geom_pairs), run_forward=False)

    def _dense_edge(self, q_from, q_to) -> list[np.ndarray]:
        q_from, q_to = np.asarray(q_from, float), np.asarray(q_to, float)
        count = max(1, int(np.ceil(np.max(np.abs(q_to - q_from)) /
                                   self.edge_max_joint_step)))
        return [(1.0 - alpha) * q_from + alpha * q_to
                for alpha in np.linspace(0.0, 1.0, count + 1)]

    def path(self, robot: str, q_from, q_to, other_q, grasp_part_tcp: np.ndarray,
             steps: int, allowed_part_holders: tuple[str, ...],
             allowed_geom_pairs: tuple[tuple, ...] = ()) -> tuple[bool, list[np.ndarray], str]:
        """Adaptively check a held-part joint edge; grasp is ``^P T_E``.

        ``steps`` is accepted for API compatibility but validity resolution is
        governed by ``edge_max_joint_step`` so a long move cannot silently
        receive the same eight samples as a short move.
        """
        waypoints = []
        inverse_grasp = np.linalg.inv(grasp_part_tcp)

        def in_collision(q):
            X_part = self.kin.fk(robot, q) @ inverse_grasp
            qA, qB = (q, other_q) if robot == "A" else (other_q, q)
            result = self.check(qA, qB, X_part, allowed_part_holders,
                                allowed_geom_pairs)
            return not result.free

        edge = validate_edge(q_from, q_to, in_collision,
                             self.edge_max_joint_step)
        if not edge.valid:
            return False, waypoints, f"collision_at:{edge.collision_state}"
        for q in self._dense_edge(q_from, q_to):
            waypoints.append(q.copy())
        return True, waypoints, "ok"

    def path_fixed_part(self, robot: str, q_from, q_to, other_q,
                        X_world_part: np.ndarray, steps: int,
                        allowed_part_holders: tuple[str, ...],
                        allowed_geom_pairs: tuple[tuple, ...] = ()) -> tuple[bool, list[np.ndarray], str]:
        waypoints = []

        def in_collision(q):
            qA, qB = (q, other_q) if robot == "A" else (other_q, q)
            result = self.check(qA, qB, X_world_part, allowed_part_holders,
                                allowed_geom_pairs)
            return not result.free

        edge = validate_edge(q_from, q_to, in_collision,
                             self.edge_max_joint_step)
        if not edge.valid:
            return False, waypoints, f"collision_at:{edge.collision_state}"
        for q in self._dense_edge(q_from, q_to):
            waypoints.append(q.copy())
        return True, waypoints, "ok"

    def plan_motion(self, robot: str, q_from, q_to, other_q,
                    *, grasp_part_tcp: np.ndarray | None = None,
                    fixed_part_pose: np.ndarray | None = None,
                    allowed_part_holders: tuple[str, ...] = (),
                    allowed_geom_pairs: tuple[tuple, ...] = (),
                    config: MotionPlannerConfig | None = None) -> MotionPlanResult:
        """Run bounded RRT-Connect with the complete MuJoCo state in the gate.

        Exactly one of ``grasp_part_tcp`` (part moves rigidly with ``robot``)
        and ``fixed_part_pose`` (part rests in the scene) must be supplied.
        """
        if (grasp_part_tcp is None) == (fixed_part_pose is None):
            raise ValueError("provide exactly one of grasp_part_tcp or fixed_part_pose")
        inverse_grasp = (None if grasp_part_tcp is None
                         else np.linalg.inv(np.asarray(grasp_part_tcp, float)))

        def in_collision(q):
            X_part = (np.asarray(fixed_part_pose, float) if inverse_grasp is None
                      else self.kin.fk(robot, q) @ inverse_grasp)
            qA, qB = (q, other_q) if robot == "A" else (other_q, q)
            return not self.check(qA, qB, X_part, allowed_part_holders,
                                  allowed_geom_pairs).free

        cfg = config or MotionPlannerConfig(edge_max_step=self.edge_max_joint_step)
        return JointRRTConnect(self.kin.lower[robot], self.kin.upper[robot],
                               in_collision, cfg).plan(q_from, q_to)

    def execution_waypoints(self, sparse_path) -> list[np.ndarray]:
        """Densify a validated sparse planner path for deterministic replay."""
        sparse = np.asarray(sparse_path, dtype=float)
        if sparse.ndim != 2 or sparse.shape[1] != 6:
            raise ValueError("sparse path must have shape (N, 6)")
        output = [sparse[0].copy()]
        for left, right in zip(sparse[:-1], sparse[1:]):
            output.extend(self._dense_edge(left, right)[1:])
        return output

    def minimum_clearance(self, geom_groups: tuple[str, ...] = ("A_", "B_", "part_"),
                          distance_cap: float = 0.05,
                          policy: CollisionPolicy | None = None,
                          ignore_same_robot: bool = True) -> float:
        policy = policy or CollisionPolicy()
        ids = [i for i in range(self.model.ngeom)
               if self.model.geom(i).name.startswith(geom_groups)
               and self.model.geom_contype[i] != 0]
        others = [i for i in range(self.model.ngeom)
                  if self.model.geom_conaffinity[i] != 0]
        best = distance_cap
        for i in ids:
            for j in others:
                if i >= j or self._adjacent(i, j):
                    continue
                name_i, name_j = self.model.geom(i).name, self.model.geom(j).name
                if ignore_same_robot and any(
                        name_i.startswith(f"{robot}_")
                        and name_j.startswith(f"{robot}_")
                        for robot in ("A", "B")):
                    # Mounting-neighbor gaps are not environmental clearance;
                    # actual self penetrations are still rejected by check().
                    continue
                if any(allowance.matches_names(name_i, name_j)
                       for allowance in policy.allowed_contacts):
                    continue
                other = _other_than_part(name_i, name_j)
                if other is not None:
                    if any(other.startswith(f"{robot}_gripper_collision_")
                           for robot in policy.part_holders):
                        continue
                distance = mujoco.mj_geomDistance(self.model, self.data, i, j,
                                                  distance_cap, None)
                if distance >= 0:
                    best = min(best, float(distance))
        return best


def obb_separation(T_a: np.ndarray, half_a, T_b: np.ndarray, half_b) -> float:
    """SAT signed separation: positive gap, negative overlap estimate."""
    A, B = T_a[:3, :3], T_b[:3, :3]
    delta = T_b[:3, 3] - T_a[:3, 3]
    axes = [A[:, i] for i in range(3)] + [B[:, i] for i in range(3)]
    axes += [np.cross(A[:, i], B[:, j]) for i in range(3) for j in range(3)]
    separations = []
    for axis in axes:
        norm = np.linalg.norm(axis)
        if norm < 1e-9:
            continue
        axis = axis / norm
        radius_a = np.sum(np.abs(A.T @ axis) * half_a)
        radius_b = np.sum(np.abs(B.T @ axis) * half_b)
        separations.append(abs(delta @ axis) - radius_a - radius_b)
    return float(max(separations))


__all__ = [
    "AllowedContact",
    "CollisionPolicy",
    "CollisionResult",
    "SceneCollisionChecker",
    "is_part_collision_geom",
    "obb_separation",
]
