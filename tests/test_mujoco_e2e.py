"""Slow release-gate acceptance tests for both production branches.

These tests intentionally require a feasible plan and successful execution;
"no solution" is a failure, never a skipped or passing outcome.  They use the
checked-in content-addressed offline policies. Cold-cache equivalence belongs
to the separate nightly qualification tier.
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from mujoco_sim.exec import PipelineExecutor, PipelineState  # noqa: E402
from mujoco_sim.plan_validation import (validate_direct_plan,
                                        validate_regrasp_plan)  # noqa: E402
from mujoco_sim.planning import HandoffPlanner  # noqa: E402
from mujoco_sim.sim import WorkcellSim  # noqa: E402
from mujoco_sim.visualize_reorientation_demo import build_demo  # noqa: E402


def _restore_start(sim, planner):
    sim.set_arm_qpos("A", planner.q_start["A"])
    sim.set_arm_qpos("B", planner.q_start["B"])
    sim.apply_active_grasp()


def test_direct_current_workcell_plans_and_executes_to_complete():
    sim = WorkcellSim()
    planner = HandoffPlanner(sim)
    report = planner.plan(allow_regrasp=False, return_best=False)
    assert report.feasible and report.direct is not None
    assert report.regrasp is None
    validate_direct_plan(report.direct, q_start=planner.q_start)
    assert report.stage_timings
    assert report.bottlenecks

    _restore_start(sim, planner)
    result = PipelineExecutor(sim, planner).execute_direct(report.direct)
    assert result.success and result.outcome == "pipeline_complete"
    assert result.events[-1].state is PipelineState.COMPLETE
    assert all(event.state is not PipelineState.ABORTED for event in result.events)
    assert [event.state for event in result.events] == [
        PipelineState.OWNED_BY_A,
        PipelineState.A_AT_HANDOFF,
        PipelineState.B_AT_PREHANDOFF,
        PipelineState.B_CAPTURE_VERIFIED,
        PipelineState.OWNED_BY_B,
        PipelineState.A_CLEAR,
        PipelineState.SCANNED,
        PipelineState.A_CLEAR,
        PipelineState.AT_PREINSERT,
        PipelineState.INSERTED,
        PipelineState.COMPLETE,
    ]
    assert result.profile_spans
    assert result.operation_schedule is not None
    assert result.planned_modeled_makespan_s > 0.0
    assert not result.timing_estimate_complete
    # The executor and schedule use the same geometric timing model. The
    # configured dwell is quantized by at most one simulation step.
    assert abs(result.executed_modeled_time_s
               - result.planned_modeled_makespan_s) <= (
                   sim.model.opt.timestep + 1e-6)


def test_forced_stage_route_executes_reorientation_before_direct_handoff():
    sim, planner, plan, bad_grasp = build_demo()
    validate_regrasp_plan(plan, q_start=planner.q_start)
    executor = PipelineExecutor(sim, planner)
    executor.owner_grasp = bad_grasp.copy()
    result = executor.execute_regrasp(plan)
    assert result.success and result.events[-1].state is PipelineState.COMPLETE
    states = [event.state for event in result.events]
    assert states.index(PipelineState.PLACED_FOR_REORIENTATION) < states.index(
        PipelineState.REORIENTED_REPICK)
    assert states.index(PipelineState.REORIENTED_REPICK) < states.index(
        PipelineState.A_AT_HANDOFF)
    assert PipelineState.ABORTED not in states
    assert any(item["path"].startswith(
        "execution.reorientation_place_and_repick")
        for item in result.profile_spans)


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"PASS  {test.__name__}")
    print(f"\n{len(tests)}/{len(tests)} passed")
