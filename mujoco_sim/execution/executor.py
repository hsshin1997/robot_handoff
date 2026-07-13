"""Transactional execution state machine for planned handoffs.

The current gripper CAD is a non-articulated visual mesh. Consequently capture,
aperture, and force checks are explicit *virtual interlocks*, and ownership is
transferred atomically between ideal welds. The state machine is structured so
real gripper/force predicates can replace these functions without changing the
planner or sequence.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from ..simulation.collision import CollisionPolicy, is_part_collision_geom
from ..diagnostics.artifacts import DebugArtifactRecorder
from .schedule import (build_direct_operation_graph,
                       build_regrasp_operation_graph, schedule_operations)
from .types import (ExecutionEvent, ExecutionResult, PipelineState,
                    UnexpectedCollision)
from ..simulation.kinematics import GP7Kinematics
from ..simulation.contact_policies import (REORIENTATION_CONTACTS,
                                           insertion_contacts)
from ..planner.validation import validate_direct_plan, validate_regrasp_plan
from ..planner.planner import HandoffPlanner
from ..planner.types import DirectHandoffPlan, RegraspPlan
from ..core.profiling import HierarchicalProfiler
from ..core.se3 import inverse, so3_geodesic
from ..simulation.workcell import WorkcellSim
from .timing import JointVelocityTimingModel

GP7_VMAX = np.radians([375, 315, 410, 550, 550, 1000])


class PipelineExecutor:
    def __init__(self, sim: WorkcellSim | None = None,
                 planner: HandoffPlanner | None = None,
                 viewer=None, realtime: bool = False,
                 playback_speed: float = 1.0,
                 recorder: DebugArtifactRecorder | Any | None = None,
                 log_root: str | Path | None = None,
                 strict_debug: bool = False):
        """Create an executor.

        Debug capture is opt-in through ``recorder`` or ``log_root``. No
        contact serialization or offscreen rendering occurs on the default
        path. Recorder failures are isolated from execution unless
        ``strict_debug`` is explicitly true.
        """
        if recorder is not None and log_root is not None:
            raise ValueError("pass recorder or log_root, not both")
        self.sim = sim or WorkcellSim()
        self.kin = GP7Kinematics(self.sim)
        self.planner = planner or HandoffPlanner(self.sim)
        if self.planner.sim is not self.sim:
            raise ValueError("PipelineExecutor planner and sim must share one WorkcellSim")
        self.cfg = self.planner.cfg["execution"]
        self.events = []
        self.started: float | None = None
        self.owner = "A"
        self.owner_grasp = self.planner.g_A_start.copy()
        self.commanded_q = {robot: self.sim.arm_qpos(robot) for robot in ("A", "B")}
        self.viewer = viewer
        self.realtime = realtime
        self.playback_speed = float(playback_speed)
        if not np.isfinite(self.playback_speed) or self.playback_speed <= 0.0:
            raise ValueError("playback_speed must be positive and finite")
        # Keep GUI synchronization near 100 Hz in wall time. At accelerated
        # playback, syncing every simulation 10 ms would oversample the window
        # and make rendering, rather than physics, the bottleneck.
        self._render_stride = max(1, int(round(10 * self.playback_speed)))
        self._steps_since_viewer_sync = 0
        self._modeled_since_viewer_sync_s = 0.0
        self.estimated_robot_time_s = 0.0
        self.timing_model = JointVelocityTimingModel(GP7_VMAX)
        self.profiler = HierarchicalProfiler("execution")
        self._schedule_summary = None
        self.fixed_part_pose = None
        self.strict_debug = bool(strict_debug)
        self.recorder = recorder
        self._debug_errors: list[str] = []
        if log_root is not None:
            try:
                self.recorder = DebugArtifactRecorder(
                    log_root, strict=self.strict_debug)
            except Exception as error:
                message = f"debug recorder initialization: {type(error).__name__}: {error}"
                if self.strict_debug:
                    raise
                self._debug_errors.append(message)
                self.recorder = None
        self._active_policy = self._policy(("A",), ())
        self._plan_metadata: dict[str, Any] = {"branch": "unassigned"}

    def _elapsed(self) -> float:
        return (0.0 if self.started is None else
                time.perf_counter() - self.started)

    def _start_run(self, operations) -> None:
        """Start timing at execution, not at executor construction."""
        self.events.clear()
        self.estimated_robot_time_s = 0.0
        self._steps_since_viewer_sync = 0
        self._modeled_since_viewer_sync_s = 0.0
        self.profiler.reset()
        self.started = time.perf_counter()
        self._schedule_summary = schedule_operations(
            operations, allow_parallel=False)

    @contextmanager
    def _stage(self, name: str):
        with self.profiler.span(name):
            yield

    def _policy(self, holders: tuple[str, ...], allowed_geom_pairs: tuple[tuple, ...]):
        # Use the collision checker's exact legacy adapter so debug labels and
        # the live safety decision cannot drift apart.
        return self.planner.collision._legacy_policy(
            tuple(holders), tuple(allowed_geom_pairs))

    @staticmethod
    def _direct_plan_metadata(plan: DirectHandoffPlan) -> dict[str, Any]:
        return {
            "branch": "direct",
            "receiver_grasp": plan.grasp_name_B,
            "world_handoff_part_pose": plan.X_handoff,
            "sender_grasp": plan.g_A,
            "receiver_grasp_transform": plan.g_B,
            "score": plan.score,
            "trajectory_waypoints": {
                name: len(path) for name, path in plan.trajectories.items()
            },
        }

    @classmethod
    def _regrasp_plan_metadata(cls, plan: RegraspPlan) -> dict[str, Any]:
        return {
            "branch": "reorientation_then_direct",
            "placement": plan.placement_name,
            "world_placement_part_pose": plan.X_place,
            "sender_grasp_after_repick": plan.g_A_after,
            "trajectory_waypoints": {
                name: len(path) for name, path in plan.trajectories.items()
            },
            "direct": cls._direct_plan_metadata(plan.direct),
        }

    def _execution_debug_metadata(self) -> dict[str, Any]:
        return {
            "owner": self.owner,
            "owner_grasp": self.owner_grasp,
            "fixed_part_pose": self.fixed_part_pose,
            "commanded_q": self.commanded_q,
            "estimated_robot_time_s": self.estimated_robot_time_s,
            "wall_elapsed_s": self._elapsed(),
        }

    def _event(self, state, *, policy: CollisionPolicy | None = None,
               step_name: str | None = None, **detail):
        if self.started is None:
            self.started = time.perf_counter()
        event = ExecutionEvent(
            state, self._elapsed(), detail,
            self.estimated_robot_time_s)
        self.events.append(event)
        if self.recorder is None:
            return event
        try:
            with self.profiler.span("diagnostic_io"):
                self.recorder.record(
                    step_name or state.value,
                    self.sim,
                    event=event,
                    plan_metadata=self._plan_metadata,
                    execution_metadata=self._execution_debug_metadata(),
                    collision_checker=self.planner.collision,
                    policy=policy or self._active_policy,
                )
        except Exception as error:
            message = f"{state.value}: {type(error).__name__}: {error}"
            self._debug_errors.append(message)
            if self.strict_debug:
                raise
        return event

    def _result(self, success: bool, outcome: str,
                measured_g_B: np.ndarray | None = None) -> ExecutionResult:
        recorder_errors = tuple(getattr(self.recorder, "errors", ()))
        run_dir = getattr(self.recorder, "run_dir", None)
        insertion_has_cad = bool(
            self.planner.project.manifest["insertion"].get("collision_cad"))
        limitations = (
            "virtual capture predicate: articulated gripper scene adapter unavailable",
            "atomic ideal-weld ownership transfer: no physical dual grasp",
            "virtual exact scanner measurement: sensor model unavailable",
            (
                "fixture collision CAD is checked geometrically, but seating force/contact "
                "calibration is not physically certified"
                if insertion_has_cad else
                "geometric insertion target: PCB hole/contact model unavailable"
            ),
            "cycle-time estimate excludes unmodeled gripper, scanner, PLC, "
            "and controller acceleration/settling delays",
        )
        stage_timings = []
        previous_wall = 0.0
        previous_robot = 0.0
        for event in self.events:
            label = event.state.value
            if event.detail.get("phase"):
                label = f"{label}:{event.detail['phase']}"
            stage_timings.append({
                "completed_state": event.state.value,
                "label": label,
                "estimated_robot_duration_s": (
                    event.estimated_robot_time_s - previous_robot),
                "observed_wall_duration_s": event.timestamp_s - previous_wall,
                "estimated_robot_elapsed_s": event.estimated_robot_time_s,
                "observed_wall_elapsed_s": event.timestamp_s,
            })
            previous_wall = event.timestamp_s
            previous_robot = event.estimated_robot_time_s
        wall_elapsed = self._elapsed()
        schedule = self._schedule_summary
        return ExecutionResult(
            success, outcome, self.events, measured_g_B,
            limitations=limitations,
            debug_run_dir=str(run_dir) if run_dir is not None else None,
            debug_errors=tuple(self._debug_errors) + recorder_errors,
            estimated_cycle_time_s=self.estimated_robot_time_s,
            executed_modeled_time_s=self.estimated_robot_time_s,
            planned_modeled_makespan_s=(
                0.0 if schedule is None else schedule.modeled_makespan_s),
            timing_estimate_complete=(
                False if schedule is None else schedule.estimate_complete),
            unmodeled_operations=(
                () if schedule is None else schedule.unmodeled_operations),
            wall_elapsed_s=wall_elapsed,
            stage_timings=tuple(stage_timings),
            operation_schedule=(None if schedule is None else {
                "modeled_makespan_s": schedule.modeled_makespan_s,
                "operation_work_s": schedule.operation_work_s,
                "critical_path": schedule.critical_path,
                "resource_busy_s": schedule.resource_busy_s,
                "estimate_complete": schedule.estimate_complete,
                "unmodeled_operations": schedule.unmodeled_operations,
                "operations": schedule.operations,
                "schedule": schedule.schedule,
                "concurrency_enabled": schedule.concurrency_enabled,
            }),
        )

    def _finalize_profile(self, result: ExecutionResult) -> ExecutionResult:
        result.profile_spans = self.profiler.report()
        result.wall_elapsed_s = self._elapsed()
        return result

    def _sync_viewer(self, *, force: bool = False) -> None:
        """Synchronize/pause using cumulative simulation steps across edges."""
        if self.viewer is None:
            self._steps_since_viewer_sync = 0
            self._modeled_since_viewer_sync_s = 0.0
            return
        if (not force
                and self._steps_since_viewer_sync < self._render_stride):
            return
        if self._steps_since_viewer_sync <= 0:
            return
        elapsed_modeled = self._modeled_since_viewer_sync_s
        with self.profiler.span("viewer_sync"):
            self.viewer.sync()
        if self.realtime:
            with self.profiler.span("pacing_wait"):
                time.sleep(elapsed_modeled / self.playback_speed)
        self._steps_since_viewer_sync = 0
        self._modeled_since_viewer_sync_s = 0.0

    def _move(self, robot: str, target, speed_fraction: float,
              minimum_time: float = 0.10,
              allowed_part_holders: tuple[str, ...] | None = None,
              allowed_geom_pairs: tuple[tuple, ...] = (),
              flush_viewer: bool = True):
        start = self.sim.arm_qpos(robot)
        target = np.asarray(target, dtype=float)
        duration = self.timing_model.edge_duration(
            start, target, speed_fraction, minimum_time_s=minimum_time)
        steps = max(2, int(np.ceil(duration / self.sim.model.opt.timestep)))
        modeled_step_s = duration / steps
        holders = ((self.owner,) if allowed_part_holders is None
                   and self.owner is not None else
                   (() if allowed_part_holders is None else allowed_part_holders))
        policy = self._policy(tuple(holders), allowed_geom_pairs)
        self._active_policy = policy
        with self.profiler.span("simulation_collision"):
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
                collision = self.planner.collision.check_current(
                    policy=policy,
                )
                # Modeled robot time follows geometric path travel, not the
                # number of collision or viewer samples.
                self.estimated_robot_time_s += modeled_step_s
                self._modeled_since_viewer_sync_s += modeled_step_s
                self._steps_since_viewer_sync += 1
                if not collision.free:
                    raise UnexpectedCollision(collision.pair, collision.penetration)
                self._sync_viewer()
        self.commanded_q[robot] = target.copy()
        self.sim.set_arm_qpos(robot, target, hold=True)
        if self.owner is not None:
            self.sim.set_part_world(self.kin.fk(self.owner) @ inverse(self.owner_grasp))
        elif self.fixed_part_pose is not None:
            self.sim.set_part_world(self.fixed_part_pose)
        if flush_viewer:
            self._sync_viewer(force=True)

    def _follow(self, robot, trajectory, speed,
                allowed_part_holders: tuple[str, ...] | None = None,
                allowed_geom_pairs: tuple[tuple, ...] = ()):
        for q in trajectory[1:]:
            # Dense path points are collision samples, not separate 100 ms
            # robot commands. Replay them continuously and apply timing from
            # joint distance/velocity, then flush the viewer once per path.
            self._move(robot, q, speed, minimum_time=0.0,
                       allowed_part_holders=allowed_part_holders,
                       allowed_geom_pairs=allowed_geom_pairs,
                       flush_viewer=False)
        self._sync_viewer(force=True)

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

    def _stage_sender_to_handoff(
        self, plan: DirectHandoffPlan, speed: float, approach: float,
        initial_allowed_geom_pairs: tuple[tuple, ...],
    ) -> None:
        """Move A from its current state to the checked handoff pose."""
        self._follow(
            "A", plan.trajectories["A_current_to_pre"], speed, ("A",),
            initial_allowed_geom_pairs)
        self._follow("A", plan.trajectories["A_approach"], approach, ("A",))
        self._event(
            PipelineState.A_AT_HANDOFF,
            policy=self._policy(("A",), ()))

    def _stage_receiver_to_handoff(
        self, plan: DirectHandoffPlan, speed: float, approach: float,
    ) -> ExecutionResult | None:
        """Move B into co-grasp and apply the pre-transfer guard."""
        self._follow("B", plan.trajectories["B_current_to_pre"], speed, ("A",))
        self._event(
            PipelineState.B_AT_PREHANDOFF,
            policy=self._policy(("A",), ()))
        self._follow("B", plan.trajectories["B_approach"], approach, ("A", "B"))
        guard_ok, contacts = self._guard_clear()
        if guard_ok:
            return None
        self._event(
            PipelineState.ABORTED,
            policy=self._policy(("A", "B"), ()),
            reason="force_guard_proxy", contacts=contacts)
        return self._result(False, "guarded_approach_abort_A_retains_part")

    def _stage_capture_and_transfer(
        self, plan: DirectHandoffPlan,
    ) -> ExecutionResult | None:
        """Validate virtual capture and atomically transfer ownership to B."""
        captured, translation, rotation = self._virtual_capture(plan)
        if not captured:
            self._event(
                PipelineState.ABORTED,
                policy=self._policy(("A", "B"), ()),
                reason="capture_region", translation_m=translation,
                rotation_rad=rotation)
            return self._result(False, "virtual_capture_failed_A_retains_part")
        self._event(
            PipelineState.B_CAPTURE_VERIFIED,
            policy=self._policy(("A", "B"), ()),
            translation_m=translation, rotation_rad=rotation,
            aperture="unavailable_static_gripper")
        dwell = min(self.cfg["co_grasp_dwell_s"], 0.299)
        dwell_steps = max(0, round(dwell / self.sim.model.opt.timestep))
        policy = self._policy(("A", "B"), ())
        self._active_policy = policy
        timestep = self.sim.model.opt.timestep
        with self.profiler.span("co_grasp_dwell"):
            for _ in range(dwell_steps):
                # Hold both commanded witnesses throughout the dwell. A bulk
                # mj_step previously let placeholder servo dynamics move A,
                # changing retreat time and skipping continuous monitoring.
                for robot in ("A", "B"):
                    self.sim.set_arm_qpos(
                        robot, self.commanded_q[robot], hold=True)
                mujoco.mj_step(self.sim.model, self.sim.data)
                for robot in ("A", "B"):
                    self.sim.set_arm_qpos(
                        robot, self.commanded_q[robot], hold=True)
                self.sim.set_part_world(
                    self.kin.fk("A") @ inverse(self.owner_grasp))
                collision = self.planner.collision.check_current(policy=policy)
                self.estimated_robot_time_s += timestep
                self._modeled_since_viewer_sync_s += timestep
                self._steps_since_viewer_sync += 1
                if not collision.free:
                    raise UnexpectedCollision(
                        collision.pair, collision.penetration)
                self._sync_viewer()
        self._sync_viewer(force=True)

        # Atomic call: grasp_part disables A and enables B without stepping
        # through an unowned physical state.
        self.sim.grasp_part("B")
        self.owner = "B"
        self.owner_grasp = plan.g_B.copy()
        self._event(
            PipelineState.OWNED_BY_B,
            policy=self._policy(("A", "B"), ()))
        return None

    def _stage_sender_retreat(
        self, plan: DirectHandoffPlan, approach: float,
    ) -> None:
        """Retreat A while retaining co-grasp contact semantics along the path."""
        self._follow("A", plan.trajectories["A_retreat"], approach, ("A", "B"))
        self._event(
            PipelineState.A_CLEAR,
            policy=self._policy(("B",), ()))

    def _stage_scan(self, plan: DirectHandoffPlan, speed: float) -> np.ndarray:
        """Move B to the scanner and return the measured part-to-TCP grasp."""
        self._follow("B", plan.trajectories["B_to_scanner"], speed, ("B",))
        X_part = self.sim.part_pose()
        X_tcp = self.kin.fk("B")
        measured_g = inverse(X_part) @ X_tcp
        self._event(
            PipelineState.SCANNED,
            policy=self._policy(("B",), ()), mode="virtual_exact",
            translation_update_m=float(np.linalg.norm(
                measured_g[:3, 3] - plan.g_B[:3, 3])),
            rotation_update_rad=so3_geodesic(
                measured_g[:3, :3], plan.g_B[:3, :3]))
        return measured_g

    def _stage_sender_park_after_scan(
        self, plan: DirectHandoffPlan, speed: float,
    ) -> None:
        """Put A in the park state used to certify downstream trajectories."""
        self._follow(
            "A", plan.trajectories["A_scanner_clear_to_park"], speed,
            ("B",))
        self._event(
            PipelineState.A_CLEAR,
            policy=self._policy(("B",), ()),
            phase="parked_for_insertion")

    def _stage_insertions(
        self, plan: DirectHandoffPlan, measured_g: np.ndarray, speed: float,
    ) -> ExecutionResult | None:
        """Retarget and execute all configured pre-insert/insert stages."""
        previous_q = self.sim.arm_qpos("B")
        qA_clear = self.sim.arm_qpos("A")
        insertion_pair = insertion_contacts(self.planner.project)
        insertion_policy_name = (
            "exact_fixture_zero_penetration_semantic_contact"
            if self.planner.project.manifest["insertion"].get("collision_cad")
            else "placeholder_virtual_aperture_10um_ring_tolerance")
        # The present scanner is explicitly virtual-exact. Reuse the already
        # collision-checked downstream trajectories when the measurement is
        # numerically the planned grasp. The former implementation discarded
        # an RRT path and retried only a straight edge, causing execution to
        # reject its own valid plan.
        nominal_measurement = (
            np.linalg.norm(measured_g[:3, 3] - plan.g_B[:3, 3]) <= 1e-9
            and so3_geodesic(measured_g[:3, :3], plan.g_B[:3, :3]) <= 1e-9)
        for index, (placement_name, X_insert) in enumerate(
                self.planner.insertion_poses):
            if nominal_measurement:
                pre_q = plan.downstream.q_preinsert[index]
                insert_q = plan.downstream.q_insert[index]
                transit = plan.downstream.trajectories[
                    f"scanner_to_{placement_name}_pre"]
                insertion_path = plan.downstream.trajectories[
                    f"{placement_name}_insert"]
                correction_ok = transit_ok = insert_ok = True
                transit_reason = insert_reason = "verified_planned_path"
            else:
                X_pre = self.planner._preinsert_pose(X_insert)
                pre = self.planner._solutions(
                    "B", X_pre @ measured_g, seed=previous_q)
                insert = self.planner._solutions(
                    "B", X_insert @ measured_g,
                    seed=pre[0].q if pre else previous_q)
                if not pre or not insert:
                    self._event(
                        PipelineState.ABORTED,
                        policy=self._policy(("B",), ()),
                        reason="post_scan_retarget_ik", index=index,
                        target=placement_name)
                    return self._result(
                        False, "post_scan_retarget_failed", measured_g)
                pre_q, insert_q = pre[0].q, insert[0].q
                correction_ok, _, _ = self.planner._correction_ok(
                    measured_g, insert_q, X_insert)
                transit_ok, transit, transit_reason = self.planner.collision.path(
                    "B", previous_q, pre_q, qA_clear, measured_g,
                    self.planner.steps, ("B",))
                insert_ok, insertion_path, insert_reason = (
                    self.planner.collision.path(
                        "B", pre_q, insert_q, qA_clear, measured_g,
                        self.planner.steps, ("B",), insertion_pair))
            if not correction_ok or not transit_ok or not insert_ok:
                self._event(
                    PipelineState.ABORTED,
                    policy=self._policy(("B",), insertion_pair),
                    reason="post_scan_retarget_gate", index=index,
                    target=placement_name, transit=transit_reason,
                    insertion=insert_reason)
                return self._result(
                    False, "post_scan_retarget_failed", measured_g)
            # Kinematic validation above mutates MjData and disables welds;
            # restore the live transaction before commanding the paths.
            self.sim.set_arm_qpos("A", qA_clear)
            self.sim.set_arm_qpos("B", previous_q)
            self.commanded_q["A"] = qA_clear.copy()
            self.commanded_q["B"] = previous_q.copy()
            self.sim.set_part_world(
                self.kin.fk("B", previous_q) @ inverse(measured_g))
            self.sim.grasp_part("B")
            self._follow("B", transit, speed, ("B",))
            self._event(
                PipelineState.AT_PREINSERT,
                policy=self._policy(("B",), ()),
                step_name=f"{PipelineState.AT_PREINSERT.value}_{index:02d}",
                index=index, target=placement_name)
            self._follow(
                "B", insertion_path, self.cfg["insertion_speed_fraction"],
                ("B",), insertion_pair)
            self._event(
                PipelineState.INSERTED,
                policy=self._policy(("B",), insertion_pair),
                step_name=f"{PipelineState.INSERTED.value}_{index:02d}",
                index=index, target=placement_name,
                contact_policy=insertion_policy_name)
            previous_q = insert_q
        return None

    def _execute_direct_stages(
        self, plan: DirectHandoffPlan,
        initial_allowed_geom_pairs: tuple[tuple, ...] = (),
    ) -> ExecutionResult:
        speed = self.cfg["speed_fraction"]
        approach = self.cfg["approach_speed_fraction"]
        self._event(
            PipelineState.OWNED_BY_A,
            policy=self._policy(("A",), initial_allowed_geom_pairs))
        try:
            with self._stage("sender_to_handoff"):
                self._stage_sender_to_handoff(
                    plan, speed, approach, initial_allowed_geom_pairs)
            with self._stage("receiver_to_handoff"):
                failure = self._stage_receiver_to_handoff(
                    plan, speed, approach)
            if failure is not None:
                return failure
            with self._stage("capture_and_transfer"):
                failure = self._stage_capture_and_transfer(plan)
            if failure is not None:
                return failure
            with self._stage("sender_retreat"):
                self._stage_sender_retreat(plan, approach)
            with self._stage("receiver_to_scanner"):
                measured_g = self._stage_scan(plan, speed)
            with self._stage("sender_park"):
                self._stage_sender_park_after_scan(plan, speed)
            with self._stage("insertion_sequence"):
                failure = self._stage_insertions(plan, measured_g, speed)
            if failure is not None:
                return failure
            self.sim.grasp_part("B")
            self._event(
                PipelineState.COMPLETE,
                policy=self._policy(
                    ("B",), insertion_contacts(self.planner.project)))
            return self._result(True, "pipeline_complete", measured_g)
        except UnexpectedCollision as error:
            self._event(
                PipelineState.ABORTED,
                reason="continuous_collision_monitor", pair=error.pair,
                penetration_m=error.penetration)
            return self._result(False, "unexpected_collision_abort")
        except Exception as error:  # preserve state and return structured fault
            self._event(
                PipelineState.ABORTED,
                reason=type(error).__name__, message=str(error))
            return self._result(False, "execution_exception")

    def execute_direct(
        self,
        plan: DirectHandoffPlan,
        initial_allowed_geom_pairs: tuple[tuple, ...] = (),
    ) -> ExecutionResult:
        """Execute a direct plan through independently debuggable stages."""
        validate_direct_plan(plan, q_start=self.planner.q_start)
        operations = build_direct_operation_graph(
            plan, self.cfg, self.timing_model)
        self._start_run(operations)
        self._plan_metadata = self._direct_plan_metadata(plan)
        result = self._execute_direct_stages(plan, initial_allowed_geom_pairs)
        return self._finalize_profile(result)

    def _stage_reorientation_place_and_repick(
        self, plan: RegraspPlan, speed: float,
    ) -> None:
        """Place on the support, release, approach the new grasp, and re-pick."""
        support_contact = REORIENTATION_CONTACTS
        self._follow(
            "A", plan.trajectories["A_to_place"], speed, ("A",),
            support_contact)
        self.sim.release_part()
        self.owner = None
        self.sim.set_part_world(plan.X_place)
        self.fixed_part_pose = plan.X_place.copy()
        self._event(
            PipelineState.PLACED_FOR_REORIENTATION,
            policy=self._policy((), support_contact),
            placement=plan.placement_name)
        self._follow(
            "A", plan.trajectories["A_place_to_repick"], speed, ("A",),
            support_contact)
        self.sim.grasp_part("A")
        self.owner = "A"
        self.owner_grasp = plan.g_A_after.copy()
        self.fixed_part_pose = None
        self._event(
            PipelineState.REORIENTED_REPICK,
            policy=self._policy(("A",), support_contact),
            placement=plan.placement_name)

    def execute_regrasp(self, plan: RegraspPlan) -> ExecutionResult:
        validate_regrasp_plan(plan, q_start=self.planner.q_start)
        speed = self.cfg["speed_fraction"]
        operations = build_regrasp_operation_graph(
            plan, self.cfg, self.timing_model)
        self._start_run(operations)
        self._plan_metadata = self._regrasp_plan_metadata(plan)
        try:
            self._event(
                PipelineState.OWNED_BY_A,
                policy=self._policy(("A",), ()), branch="reorientation")
            with self._stage("reorientation_place_and_repick"):
                self._stage_reorientation_place_and_repick(plan, speed)
            # Re-measurement on the flat surface is virtual exact in this model.
            result = self._execute_direct_stages(
                plan.direct, initial_allowed_geom_pairs=REORIENTATION_CONTACTS)
        except UnexpectedCollision as error:
            self._event(
                PipelineState.ABORTED,
                reason="continuous_collision_monitor", pair=error.pair,
                penetration_m=error.penetration)
            result = self._result(False, "unexpected_collision_abort")
        except Exception as error:
            self._event(
                PipelineState.ABORTED, reason=type(error).__name__,
                message=str(error))
            result = self._result(False, "reorientation_execution_exception")
        return self._finalize_profile(result)


# Compatibility name for callers of the discarded prototype.
DynamicExecutor = PipelineExecutor
