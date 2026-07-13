"""Explicit robot-operation graph and modeled critical-path schedule."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np

from ..planner.types import DirectHandoffPlan, RegraspPlan
from .timing import JointVelocityTimingModel


@dataclass(frozen=True)
class RobotOperation:
    operation_id: str
    label: str
    kind: str
    resources: tuple[str, ...]
    predecessors: tuple[str, ...]
    duration_s: float | None
    trajectory_key: str | None = None
    timing_model: str | None = None
    concurrency_certificate_id: str | None = None

    def __post_init__(self) -> None:
        if not self.operation_id or not self.label or not self.kind:
            raise ValueError("operation id, label, and kind must be non-empty")
        if len(set(self.resources)) != len(self.resources):
            raise ValueError("operation resources must be unique")
        if len(set(self.predecessors)) != len(self.predecessors):
            raise ValueError("operation predecessors must be unique")
        if self.operation_id in self.predecessors:
            raise ValueError("operation cannot depend on itself")
        if self.duration_s is not None:
            value = float(self.duration_s)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError("operation duration must be non-negative and finite")
            object.__setattr__(self, "duration_s", value)


@dataclass(frozen=True)
class ScheduledOperation:
    operation_id: str
    start_s: float
    end_s: float


@dataclass(frozen=True)
class RobotScheduleSummary:
    modeled_makespan_s: float
    operation_work_s: float
    critical_path: tuple[str, ...]
    resource_busy_s: dict[str, float]
    estimate_complete: bool
    unmodeled_operations: tuple[str, ...]
    operations: tuple[dict[str, Any], ...]
    schedule: tuple[dict[str, Any], ...]
    concurrency_enabled: bool


def schedule_operations(
    operations: Iterable[RobotOperation],
    *,
    allow_parallel: bool = False,
) -> RobotScheduleSummary:
    """List-schedule a dependency DAG with resource serialization.

    Parallel scheduling is fail-closed: overlapping operations must carry the
    same non-empty coordinated-collision certificate.  Current plans therefore
    use the serial mode until a dual-arm planner supplies such a certificate.
    """
    values = tuple(operations)
    by_id = {item.operation_id: item for item in values}
    if len(by_id) != len(values):
        raise ValueError("operation IDs must be unique")
    for item in values:
        missing = set(item.predecessors) - set(by_id)
        if missing:
            raise ValueError(
                f"operation {item.operation_id!r} has missing predecessors: "
                f"{sorted(missing)}")

    pending = set(by_id)
    completed: dict[str, ScheduledOperation] = {}
    paths: dict[str, tuple[str, ...]] = {}
    resource_available: dict[str, float] = {}
    serial_available = 0.0
    while pending:
        ready = sorted(
            item_id for item_id in pending
            if all(parent in completed for parent in by_id[item_id].predecessors))
        if not ready:
            raise ValueError("operation dependency graph contains a cycle")
        for item_id in ready:
            item = by_id[item_id]
            dependency_end = max(
                (completed[parent].end_s for parent in item.predecessors),
                default=0.0)
            resource_end = max(
                (resource_available.get(resource, 0.0)
                 for resource in item.resources), default=0.0)
            start = max(dependency_end, resource_end)
            if not allow_parallel:
                start = max(start, serial_available)
            duration = 0.0 if item.duration_s is None else item.duration_s
            end = start + duration

            if allow_parallel:
                for scheduled in completed.values():
                    if scheduled.end_s <= start or end <= scheduled.start_s:
                        continue
                    other = by_id[scheduled.operation_id]
                    if set(item.resources) & set(other.resources):
                        continue
                    certificate = item.concurrency_certificate_id
                    if (not certificate
                            or certificate != other.concurrency_certificate_id):
                        raise ValueError(
                            "parallel operations require one shared coordinated "
                            "collision certificate")

            completed[item_id] = ScheduledOperation(item_id, start, end)
            if item.predecessors:
                parent = max(item.predecessors,
                             key=lambda name: completed[name].end_s)
                paths[item_id] = paths[parent] + (item_id,)
            else:
                paths[item_id] = (item_id,)
            for resource in item.resources:
                resource_available[resource] = end
            serial_available = max(serial_available, end)
            pending.remove(item_id)

    makespan = max((item.end_s for item in completed.values()), default=0.0)
    terminal = max(completed, key=lambda name: completed[name].end_s,
                   default=None)
    busy: dict[str, float] = {}
    for item in values:
        if item.duration_s is None:
            continue
        for resource in item.resources:
            busy[resource] = busy.get(resource, 0.0) + item.duration_s
    unmodeled = tuple(item.operation_id for item in values
                      if item.duration_s is None)
    return RobotScheduleSummary(
        modeled_makespan_s=float(makespan),
        operation_work_s=float(sum(item.duration_s or 0.0 for item in values)),
        critical_path=() if terminal is None else paths[terminal],
        resource_busy_s=dict(sorted(busy.items())),
        estimate_complete=not unmodeled,
        unmodeled_operations=unmodeled,
        operations=tuple(asdict(item) for item in values),
        schedule=tuple(asdict(completed[item.operation_id]) for item in values),
        concurrency_enabled=bool(allow_parallel),
    )


def _trajectory_operation(
    operation_id: str,
    label: str,
    robot: str,
    predecessor: str | None,
    trajectory_key: str,
    trajectories,
    speed: float,
    timing: JointVelocityTimingModel,
) -> RobotOperation:
    analysis = timing.analyze(trajectories[trajectory_key], speed)
    return RobotOperation(
        operation_id, label, "joint_trajectory", (f"arm:{robot}",),
        () if predecessor is None else (predecessor,), analysis.duration_s,
        trajectory_key, analysis.timing_model)


def build_direct_operation_graph(
    plan: DirectHandoffPlan,
    execution_config: dict,
    timing: JointVelocityTimingModel,
    *,
    prefix: str = "",
    predecessor: str | None = None,
) -> tuple[RobotOperation, ...]:
    """Build the currently certified serial direct-handoff operation graph."""
    speed = float(execution_config["speed_fraction"])
    approach = float(execution_config["approach_speed_fraction"])
    insert_speed = float(execution_config["insertion_speed_fraction"])
    key = lambda name: f"{prefix}{name}"
    operations: list[RobotOperation] = []

    def motion(name, label, robot, parent, trajectory_key, fraction,
               trajectories=plan.trajectories):
        operation = _trajectory_operation(
            key(name), label, robot, parent, trajectory_key, trajectories,
            fraction, timing)
        operations.append(operation)
        return operation.operation_id

    last = motion("a_transit", "Robot A transit to pre-handoff", "A",
                  predecessor, "A_current_to_pre", speed)
    last = motion("a_approach", "Robot A handoff approach", "A", last,
                  "A_approach", approach)
    last = motion("b_transit", "Robot B transit to pre-handoff", "B", last,
                  "B_current_to_pre", speed)
    last = motion("b_approach", "Robot B handoff approach", "B", last,
                  "B_approach", approach)
    dwell = min(float(execution_config["co_grasp_dwell_s"]), 0.299)
    transfer_id = key("capture_transfer")
    operations.append(RobotOperation(
        transfer_id, "Capture verification and ownership transfer", "transfer",
        ("arm:A", "arm:B", "part"), (last,), dwell,
        timing_model="configured_virtual_dwell_v1"))
    last = motion("a_retreat", "Robot A retreat", "A", transfer_id,
                  "A_retreat", approach)
    last = motion("b_scanner", "Robot B transit to scanner", "B", last,
                  "B_to_scanner", speed)
    scanner_id = key("scanner_measurement")
    operations.append(RobotOperation(
        scanner_id, "Scanner measurement", "sensor", ("scanner", "part"),
        (last,), None))
    last = motion("a_park", "Robot A clear to insertion park", "A",
                  scanner_id, "A_scanner_clear_to_park", speed)

    insertion_names = [
        name[:-len("_insert")]
        for name in plan.downstream.trajectories
        if name.endswith("_insert")]
    for index, target_name in enumerate(insertion_names):
        transit_key = f"scanner_to_{target_name}_pre"
        # For multiple insertions the plan key retains the historical scanner
        # prefix even though the path starts at the previous insertion.
        last = motion(
            f"insert_{index:02d}_transit", f"Transit to {target_name}", "B",
            last, transit_key, speed, plan.downstream.trajectories)
        last = motion(
            f"insert_{index:02d}_descent", f"Insert at {target_name}", "B",
            last, f"{target_name}_insert", insert_speed,
            plan.downstream.trajectories)
    return tuple(operations)


def build_regrasp_operation_graph(
    plan: RegraspPlan,
    execution_config: dict,
    timing: JointVelocityTimingModel,
) -> tuple[RobotOperation, ...]:
    speed = float(execution_config["speed_fraction"])
    place = _trajectory_operation(
        "stage_place", "Place part on reorientation stage", "A", None,
        "A_to_place", plan.trajectories, speed, timing)
    release = RobotOperation(
        "stage_release", "Release part on stage", "gripper_command",
        ("gripper:A", "part"), (place.operation_id,), None)
    repick = _trajectory_operation(
        "stage_repick", "Robot A re-pick", "A", release.operation_id,
        "A_place_to_repick", plan.trajectories, speed, timing)
    close = RobotOperation(
        "stage_capture", "Close Robot A gripper", "gripper_command",
        ("gripper:A", "part"), (repick.operation_id,), None)
    direct = build_direct_operation_graph(
        plan.direct, execution_config, timing, prefix="direct_",
        predecessor=close.operation_id)
    return (place, release, repick, close) + direct


__all__ = [
    "RobotOperation",
    "RobotScheduleSummary",
    "ScheduledOperation",
    "build_direct_operation_graph",
    "build_regrasp_operation_graph",
    "schedule_operations",
]
