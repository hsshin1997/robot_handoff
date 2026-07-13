"""Trajectory timing and future parallel-scheduling invariants."""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from mujoco_sim.execution_schedule import (RobotOperation,
                                            build_direct_operation_graph,
                                            schedule_operations)  # noqa: E402
from mujoco_sim.trajectory_timing import JointVelocityTimingModel  # noqa: E402


def timing_model():
    return JointVelocityTimingModel(np.ones(6))


def test_timing_is_invariant_to_collinear_collision_densification():
    model = timing_model()
    coarse = np.vstack((np.zeros(6), np.array([0.1, 0, 0, 0, 0, 0])))
    dense = np.zeros((101, 6))
    dense[:, 0] = np.linspace(0.0, 0.1, len(dense))
    first = model.analyze(coarse, 0.25)
    second = model.analyze(dense, 0.25)
    assert np.isclose(first.duration_s, 0.6, atol=1e-12)
    assert np.isclose(second.duration_s, first.duration_s, atol=1e-12)
    assert second.waypoint_count == 101


def test_timing_counts_real_detours_but_not_stationary_samples():
    model = timing_model()
    direct = np.zeros((2, 6)); direct[1, 0] = 0.2
    detour = np.zeros((4, 6))
    detour[1, 0] = 0.1
    detour[2, 0] = -0.1
    detour[3, 0] = 0.2
    stationary = np.vstack((direct[0], direct[0], direct[1], direct[1]))
    assert model.analyze(detour, 1.0).duration_s > model.analyze(
        direct, 1.0).duration_s
    assert np.isclose(model.analyze(stationary, 1.0).duration_s,
                      model.analyze(direct, 1.0).duration_s)


def test_edge_duration_enforces_velocity_fraction_and_explicit_minimum():
    model = timing_model()
    q0 = np.zeros(6); q1 = np.zeros(6); q1[2] = 0.2
    assert np.isclose(model.edge_duration(q0, q1, 0.5), 0.6)
    assert model.edge_duration(q0, q0, 1.0, minimum_time_s=0.1) == 0.1


def test_parallel_schedule_requires_coordinated_collision_certificate():
    uncertified = (
        RobotOperation("a", "A", "move", ("arm:A",), (), 3.0),
        RobotOperation("b", "B", "move", ("arm:B",), (), 2.0),
    )
    serial = schedule_operations(uncertified)
    assert serial.modeled_makespan_s == 5.0
    try:
        schedule_operations(uncertified, allow_parallel=True)
    except ValueError as error:
        assert "coordinated collision certificate" in str(error)
    else:
        raise AssertionError("uncertified dual-arm overlap was accepted")

    certified = tuple(RobotOperation(
        item.operation_id, item.label, item.kind, item.resources,
        item.predecessors, item.duration_s,
        concurrency_certificate_id="dual-arm-path-001")
        for item in uncertified)
    parallel = schedule_operations(certified, allow_parallel=True)
    assert parallel.modeled_makespan_s == 3.0
    assert parallel.operation_work_s == 5.0


def test_scheduler_rejects_cycles_and_marks_unmodeled_devices():
    try:
        schedule_operations((
            RobotOperation("a", "A", "move", (), ("b",), 1.0),
            RobotOperation("b", "B", "move", (), ("a",), 1.0),
        ))
    except ValueError as error:
        assert "cycle" in str(error)
    else:
        raise AssertionError("dependency cycle was accepted")

    result = schedule_operations((
        RobotOperation("scan", "Scan", "sensor", ("scanner",), (), None),
    ))
    assert not result.estimate_complete
    assert result.unmodeled_operations == ("scan",)


def test_direct_plan_builds_explicit_resource_operation_graph():
    trajectory = [np.zeros(6), np.full(6, 0.1)]
    direct_paths = {
        key: trajectory for key in (
            "A_current_to_pre", "A_approach", "B_current_to_pre",
            "B_approach", "A_retreat", "B_to_scanner",
            "A_scanner_clear_to_park")
    }
    downstream_paths = {
        "scanner_to_pcb_pre": trajectory,
        "pcb_insert": trajectory,
    }
    plan = SimpleNamespace(
        trajectories=direct_paths,
        downstream=SimpleNamespace(
            trajectories=downstream_paths, q_insert=[np.zeros(6)]),
    )
    cfg = {
        "speed_fraction": 1.0,
        "approach_speed_fraction": 0.5,
        "insertion_speed_fraction": 0.25,
        "co_grasp_dwell_s": 0.2,
    }
    operations = build_direct_operation_graph(plan, cfg, timing_model())
    ids = [item.operation_id for item in operations]
    assert ids == [
        "a_transit", "a_approach", "b_transit", "b_approach",
        "capture_transfer", "a_retreat", "b_scanner",
        "scanner_measurement", "a_park", "insert_00_transit",
        "insert_00_descent",
    ]
    schedule = schedule_operations(operations)
    assert schedule.critical_path == tuple(ids)
    assert not schedule.estimate_complete
    assert schedule.unmodeled_operations == ("scanner_measurement",)


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"PASS  {test.__name__}")
    print(f"\n{len(tests)}/{len(tests)} passed")
