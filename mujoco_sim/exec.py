"""Transactional execution state machine for planned handoffs.

The current gripper CAD is a non-articulated visual mesh. Consequently capture,
aperture, and force checks are explicit *virtual interlocks*, and ownership is
transferred atomically between ideal welds. The state machine is structured so
real gripper/force predicates can replace these functions without changing the
planner or sequence.
"""
from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field

import mujoco
import numpy as np

from .collision import CollisionPolicy, is_part_collision_geom
from .kinematics import GP7Kinematics
from .planning import (REORIENTATION_CONTACTS, DirectHandoffPlan,
                       HandoffPlanner, RegraspPlan)
from .se3 import inverse, so3_geodesic
from .sim import WorkcellSim

GP7_VMAX = np.radians([375, 315, 410, 550, 550, 1000])


class UnexpectedCollision(RuntimeError):
    def __init__(self, pair, penetration):
        super().__init__(f"unexpected collision {pair}, penetration={penetration:.6f} m")
        self.pair = pair
        self.penetration = float(penetration)


class PipelineState(enum.Enum):
    OWNED_BY_A = "owned_by_A"
    PLACED_FOR_REORIENTATION = "placed_for_reorientation"
    REORIENTED_REPICK = "reoriented_repick_by_A"
    A_AT_HANDOFF = "A_at_handoff"
    B_AT_PREHANDOFF = "B_at_prehandoff"
    B_CAPTURE_VERIFIED = "B_capture_verified_virtual"
    OWNED_BY_B = "owned_by_B"
    A_CLEAR = "A_clear"
    SCANNED = "scanned_virtual_exact"
    AT_PREINSERT = "at_preinsert"
    INSERTED = "inserted_virtual_geometry"
    COMPLETE = "complete"
    ABORTED = "aborted"


@dataclass
class ExecutionEvent:
    state: PipelineState
    timestamp_s: float
    detail: dict = field(default_factory=dict)


@dataclass
class ExecutionResult:
    success: bool
    outcome: str
    events: list[ExecutionEvent]
    measured_g_B: np.ndarray | None = None
    limitations: tuple[str, ...] = (
        "virtual capture predicate: articulated gripper CAD unavailable",
        "atomic ideal-weld ownership transfer: no physical dual grasp",
        "virtual exact scanner measurement: sensor model unavailable",
        "geometric insertion target: PCB hole/contact model unavailable",
    )


class PipelineExecutor:
    def __init__(self, sim: WorkcellSim | None = None,
                 planner: HandoffPlanner | None = None,
                 viewer=None, realtime: bool = False):
        self.sim = sim or WorkcellSim()
        self.kin = GP7Kinematics(self.sim)
        self.planner = planner or HandoffPlanner(self.sim)
        self.cfg = self.planner.cfg["execution"]
        self.events = []
        self.started = time.perf_counter()
        self.owner = "A"
        self.owner_grasp = self.planner.g_A_start.copy()
        self.commanded_q = {robot: self.sim.arm_qpos(robot) for robot in ("A", "B")}
        self.viewer = viewer
        self.realtime = realtime
        self._render_stride = 10
        self.fixed_part_pose = None

    def _event(self, state, **detail):
        self.events.append(ExecutionEvent(state, time.perf_counter() - self.started, detail))

    def _move(self, robot: str, target, speed_fraction: float,
              minimum_time: float = 0.10,
              allowed_part_holders: tuple[str, ...] | None = None,
              allowed_geom_pairs: tuple[tuple, ...] = ()):
        start = self.sim.arm_qpos(robot)
        target = np.asarray(target, dtype=float)
        duration = max(float(np.max(np.abs(target - start) /
                                    (GP7_VMAX * speed_fraction))), minimum_time)
        steps = max(2, int(duration / self.sim.model.opt.timestep))
        for index in range(steps):
            u = (index + 1) / steps
            blend = u * u * (3.0 - 2.0 * u)
            self.commanded_q[robot] = start + blend * (target - start)
            self.sim.set_arm_qpos(robot, self.commanded_q[robot], hold=True)
            other = "B" if robot == "A" else "A"
            self.sim.set_arm_qpos(other, self.commanded_q[other], hold=True)
            mujoco.mj_step(self.sim.model, self.sim.data)
            # The current scene has placeholder inertias/servo gains. Clamp to
            # the commanded trajectory so execution validates the planned
            # kinematics rather than those uncalibrated dynamics.
            self.sim.set_arm_qpos(robot, self.commanded_q[robot], hold=True)
            self.sim.set_arm_qpos(other, self.commanded_q[other], hold=True)
            if self.owner is not None:
                X_part = self.kin.fk(self.owner) @ inverse(self.owner_grasp)
                self.sim.set_part_world(X_part)
            elif self.fixed_part_pose is not None:
                self.sim.set_part_world(self.fixed_part_pose)
            holders = ((self.owner,) if allowed_part_holders is None
                       and self.owner is not None else
                       (() if allowed_part_holders is None else allowed_part_holders))
            collision = self.planner.collision.check_current(
                allowed_part_holders=holders,
                allowed_geom_pairs=allowed_geom_pairs,
            )
            if not collision.free:
                raise UnexpectedCollision(collision.pair, collision.penetration)
            if self.viewer is not None and index % self._render_stride == 0:
                self.viewer.sync()
                if self.realtime:
                    time.sleep(self.sim.model.opt.timestep * self._render_stride)
        self.commanded_q[robot] = target.copy()
        self.sim.set_arm_qpos(robot, target, hold=True)
        if self.owner is not None:
            self.sim.set_part_world(self.kin.fk(self.owner) @ inverse(self.owner_grasp))
        elif self.fixed_part_pose is not None:
            self.sim.set_part_world(self.fixed_part_pose)
        if self.viewer is not None:
            self.viewer.sync()

    def _follow(self, robot, trajectory, speed,
                allowed_part_holders: tuple[str, ...] | None = None,
                allowed_geom_pairs: tuple[tuple, ...] = ()):
        for q in trajectory[1:]:
            self._move(robot, q, speed,
                       allowed_part_holders=allowed_part_holders,
                       allowed_geom_pairs=allowed_geom_pairs)

    def _virtual_capture(self, plan: DirectHandoffPlan):
        X_actual = self.sim.part_pose()
        translation = np.linalg.norm(X_actual[:3, 3] - plan.X_handoff[:3, 3])
        rotation = so3_geodesic(X_actual[:3, :3], plan.X_handoff[:3, :3])
        # Capture region is limited by configured calibration 3-sigma plus a
        # small ideal-compliance allowance.
        tolerance = (self.planner.cfg["gates"]["calibration_translation_3sigma_m"]
                     + 0.002)
        return translation <= tolerance and rotation <= np.radians(2.0), translation, rotation

    def _guard_clear(self):
        # No wrist-force sensor model yet. Reject unexpected non-baseline
        # penetrations as a conservative proxy for a contact spike.
        mujoco.mj_forward(self.sim.model, self.sim.data)
        unexpected = []
        for contact in self.sim.data.contact[:self.sim.data.ncon]:
            names = (self.sim.model.geom(contact.geom1).name,
                     self.sim.model.geom(contact.geom2).name)
            if self.planner.collision._allowed(
                    contact, policy=CollisionPolicy(part_holders=("A", "B"))):
                continue
            if (not any(is_part_collision_geom(name) for name in names)
                    and contact.dist < -0.003):
                unexpected.append(names)
        return not unexpected, unexpected

    def execute_direct(
        self,
        plan: DirectHandoffPlan,
        initial_allowed_geom_pairs: tuple[tuple, ...] = (),
    ) -> ExecutionResult:
        speed = self.cfg["speed_fraction"]
        approach = self.cfg["approach_speed_fraction"]
        self._event(PipelineState.OWNED_BY_A)
        try:
            self._follow(
                "A", plan.trajectories["A_current_to_pre"], speed, ("A",),
                initial_allowed_geom_pairs)
            self._follow("A", plan.trajectories["A_approach"], approach, ("A",))
            self._event(PipelineState.A_AT_HANDOFF)

            self._follow("B", plan.trajectories["B_current_to_pre"], speed, ("A",))
            self._event(PipelineState.B_AT_PREHANDOFF)
            self._follow("B", plan.trajectories["B_approach"], approach, ("A", "B"))
            guard_ok, contacts = self._guard_clear()
            if not guard_ok:
                self._event(PipelineState.ABORTED, reason="force_guard_proxy", contacts=contacts)
                return ExecutionResult(False, "guarded_approach_abort_A_retains_part", self.events)

            captured, translation, rotation = self._virtual_capture(plan)
            if not captured:
                self._event(PipelineState.ABORTED, reason="capture_region",
                            translation_m=translation, rotation_rad=rotation)
                return ExecutionResult(False, "virtual_capture_failed_A_retains_part", self.events)
            self._event(PipelineState.B_CAPTURE_VERIFIED,
                        translation_m=translation, rotation_rad=rotation,
                        aperture="unavailable_static_gripper")
            self.sim.step_for(min(self.cfg["co_grasp_dwell_s"], 0.299))

            # Atomic call: grasp_part disables A and enables B without stepping
            # through an unowned physical state.
            self.sim.grasp_part("B")
            self.owner = "B"
            self.owner_grasp = plan.g_B.copy()
            self._event(PipelineState.OWNED_BY_B)
            self._follow("A", plan.trajectories["A_retreat"], approach, ("A", "B"))
            self._event(PipelineState.A_CLEAR)

            self._follow("B", plan.trajectories["B_to_scanner"], speed, ("B",))
            X_part = self.sim.part_pose()
            X_tcp = self.kin.fk("B")
            measured_g = inverse(X_part) @ X_tcp
            self._event(PipelineState.SCANNED, mode="virtual_exact",
                        translation_update_m=float(np.linalg.norm(
                            measured_g[:3, 3] - plan.g_B[:3, 3])),
                        rotation_update_rad=so3_geodesic(measured_g[:3, :3], plan.g_B[:3, :3]))

            # Recompute every downstream TCP target from the measured grasp,
            # even when the virtual exact scan produces zero correction.
            previous_q = self.sim.arm_qpos("B")
            qA_clear = self.sim.arm_qpos("A")
            for index, (placement_name, X_insert) in enumerate(self.planner.insertion_poses):
                X_pre = self.planner._preinsert_pose(X_insert)
                pre = self.planner._solutions("B", X_pre @ measured_g, seed=previous_q)
                insert = self.planner._solutions(
                    "B", X_insert @ measured_g, seed=pre[0].q if pre else previous_q)
                if not pre or not insert:
                    self._event(PipelineState.ABORTED, reason="post_scan_retarget_ik",
                                index=index)
                    return ExecutionResult(False, "post_scan_retarget_failed", self.events,
                                           measured_g)
                correction_ok, _, _ = self.planner._correction_ok(
                    measured_g, insert[0].q, X_insert)
                transit_ok, transit, transit_reason = self.planner.collision.path(
                    "B", previous_q, pre[0].q, qA_clear, measured_g,
                    self.planner.steps, ("B",))
                insert_ok, insertion_path, insert_reason = self.planner.collision.path(
                    "B", pre[0].q, insert[0].q, qA_clear, measured_g,
                    self.planner.steps, ("B",), (("part_collision", "pcb_board"),))
                if not correction_ok or not transit_ok or not insert_ok:
                    self._event(PipelineState.ABORTED, reason="post_scan_retarget_gate",
                                index=index, transit=transit_reason, insertion=insert_reason)
                    return ExecutionResult(False, "post_scan_retarget_failed", self.events,
                                           measured_g)
                # Kinematic validation above mutates MjData and disables welds;
                # restore the live transaction before commanding the paths.
                self.sim.set_arm_qpos("A", qA_clear)
                self.sim.set_arm_qpos("B", previous_q)
                self.commanded_q["A"] = qA_clear.copy()
                self.commanded_q["B"] = previous_q.copy()
                self.sim.set_part_world(self.kin.fk("B", previous_q) @ inverse(measured_g))
                self.sim.grasp_part("B")
                self._follow("B", transit, speed, ("B",))
                self._event(PipelineState.AT_PREINSERT, index=index)
                self._follow("B", insertion_path,
                             self.cfg["insertion_speed_fraction"], ("B",),
                             (("part_collision", "pcb_board"),))
                self._event(PipelineState.INSERTED, index=index,
                            contact_policy="geometry_only_no_holes")
                previous_q = insert[0].q
            self.sim.grasp_part("B")
            self._event(PipelineState.COMPLETE)
            return ExecutionResult(True, "pipeline_complete", self.events, measured_g)
        except UnexpectedCollision as error:
            self._event(PipelineState.ABORTED, reason="continuous_collision_monitor",
                        pair=error.pair, penetration_m=error.penetration)
            return ExecutionResult(False, "unexpected_collision_abort", self.events)
        except Exception as error:  # preserve state and return structured fault
            self._event(PipelineState.ABORTED, reason=type(error).__name__, message=str(error))
            return ExecutionResult(False, "execution_exception", self.events)

    def execute_regrasp(self, plan: RegraspPlan) -> ExecutionResult:
        speed = self.cfg["speed_fraction"]
        try:
            self._event(PipelineState.OWNED_BY_A, branch="reorientation")
            support_contact = REORIENTATION_CONTACTS
            self._follow("A", plan.trajectories["A_to_place"], speed, ("A",),
                         support_contact)
            self.sim.release_part()
            self.owner = None
            self.sim.set_part_world(plan.X_place)
            self.fixed_part_pose = plan.X_place.copy()
            self._event(PipelineState.PLACED_FOR_REORIENTATION,
                        placement=plan.placement_name)
            self._follow("A", plan.trajectories["A_place_to_repick"], speed, ("A",),
                         support_contact)
            self.sim.grasp_part("A")
            self.owner = "A"
            self.owner_grasp = plan.g_A_after.copy()
            self.fixed_part_pose = None
            self._event(PipelineState.REORIENTED_REPICK,
                        placement=plan.placement_name)
            # Re-measurement on the flat surface is virtual exact in this model.
            return self.execute_direct(
                plan.direct, initial_allowed_geom_pairs=REORIENTATION_CONTACTS)
        except UnexpectedCollision as error:
            self._event(PipelineState.ABORTED, reason="continuous_collision_monitor",
                        pair=error.pair, penetration_m=error.penetration)
            return ExecutionResult(False, "unexpected_collision_abort", self.events)
        except Exception as error:
            self._event(PipelineState.ABORTED, reason=type(error).__name__,
                        message=str(error))
            return ExecutionResult(False, "reorientation_execution_exception", self.events)


# Compatibility name for callers of the discarded prototype.
DynamicExecutor = PipelineExecutor
