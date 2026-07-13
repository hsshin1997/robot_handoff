"""Cached-plan structure and trajectory endpoint regression tests."""
from __future__ import annotations

from dataclasses import replace
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from mujoco_sim.planner.validation import (PlanValidationError,
                                        validate_direct_plan,
                                        validate_regrasp_plan)  # noqa: E402
from mujoco_sim.planner.codec import (deserialize_direct, deserialize_regrasp,
                                   serialize_direct,
                                   serialize_regrasp)  # noqa: E402
from mujoco_sim.planner.types import (DirectHandoffPlan, DownstreamWitness,
                                       RegraspPlan, ScoreBreakdown)  # noqa: E402


def _path(start, end):
    return [np.asarray(start, float).copy(), np.asarray(end, float).copy()]


def valid_direct():
    zero = np.zeros(6)
    a_pre = np.full(6, 0.1)
    a_handoff = np.full(6, 0.2)
    a_retreat = np.full(6, 0.3)
    b_pre = np.full(6, -0.1)
    b_handoff = np.full(6, -0.2)
    b_scanner = np.full(6, -0.3)
    b_insert_pre = np.full(6, -0.4)
    b_insert = np.full(6, -0.5)
    downstream = DownstreamWitness(
        "surface_001", np.eye(4), b_scanner,
        [b_insert_pre], [b_insert], [[b_insert.copy()]],
        {
            "scanner_to_pcb_pre": _path(b_scanner, b_insert_pre),
            "pcb_insert": _path(b_insert_pre, b_insert),
        },
        0.5, 0.1,
    )
    return DirectHandoffPlan(
        np.eye(4), np.eye(4), "surface_001", np.eye(4),
        a_handoff, b_handoff, a_pre, b_pre, a_retreat, downstream,
        {
            "A_current_to_pre": _path(zero, a_pre),
            "A_approach": _path(a_pre, a_handoff),
            "B_current_to_pre": _path(zero, b_pre),
            "B_approach": _path(b_pre, b_handoff),
            "A_retreat": _path(a_handoff, a_retreat),
            "B_to_scanner": _path(b_handoff, b_scanner),
            "A_scanner_clear_to_park": _path(a_retreat, zero),
        },
        ScoreBreakdown(0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
    )


def test_valid_direct_plan_checks_every_named_endpoint():
    plan = valid_direct()
    assert validate_direct_plan(
        plan, q_start={"A": np.zeros(6), "B": np.zeros(6)}) is plan


def test_direct_plan_rejects_missing_or_disconnected_trajectory():
    plan = valid_direct()
    paths = dict(plan.trajectories)
    paths.pop("A_retreat")
    try:
        validate_direct_plan(replace(plan, trajectories=paths))
    except PlanValidationError as error:
        assert "missing trajectories" in str(error)
    else:
        raise AssertionError("missing trajectory was accepted")

    paths = dict(plan.trajectories)
    paths["B_approach"] = _path(np.ones(6), plan.qB_handoff)
    try:
        validate_direct_plan(replace(plan, trajectories=paths))
    except PlanValidationError as error:
        assert "start does not match" in str(error)
    else:
        raise AssertionError("disconnected trajectory was accepted")


def test_direct_plan_rejects_nonfinite_transform_and_joint_values():
    plan = valid_direct()
    bad_transform = plan.X_handoff.copy(); bad_transform[0, 3] = np.nan
    for bad in (replace(plan, X_handoff=bad_transform),
                replace(plan, qA_pre=np.full(6, np.inf))):
        try:
            validate_direct_plan(bad)
        except PlanValidationError:
            pass
        else:
            raise AssertionError("non-finite plan value was accepted")


def test_reorientation_plan_connects_start_place_repick_and_direct():
    direct = valid_direct()
    q_place = np.full(6, 0.05)
    q_repick = np.full(6, 0.08)
    direct_paths = dict(direct.trajectories)
    direct_paths["A_current_to_pre"] = _path(q_repick, direct.qA_pre)
    direct = replace(direct, trajectories=direct_paths)
    plan = RegraspPlan(
        "stable_0001", np.eye(4), np.eye(4), np.eye(4),
        q_place, q_repick, direct,
        {
            "A_to_place": _path(np.zeros(6), q_place),
            "A_place_to_repick": _path(q_place, q_repick),
        },
    )
    assert validate_regrasp_plan(
        plan, q_start={"A": np.zeros(6), "B": np.zeros(6)}) is plan

    decoded = deserialize_regrasp(serialize_regrasp(plan))
    validate_regrasp_plan(
        decoded, q_start={"A": np.zeros(6), "B": np.zeros(6)})
    decoded.X_place[0, 3] = 10.0
    assert plan.X_place[0, 3] == 0.0


def test_direct_cache_codec_round_trip_returns_independent_arrays():
    original = valid_direct()
    decoded = deserialize_direct(serialize_direct(original))
    validate_direct_plan(
        decoded, q_start={"A": np.zeros(6), "B": np.zeros(6)})
    assert decoded is not original
    assert np.array_equal(decoded.X_handoff, original.X_handoff)
    decoded.X_handoff[0, 3] = 4.0
    decoded.trajectories["A_approach"][0][0] = 3.0
    assert original.X_handoff[0, 3] == 0.0
    assert original.trajectories["A_approach"][0][0] == original.qA_pre[0]


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"PASS  {test.__name__}")
    print(f"\n{len(tests)}/{len(tests)} passed")
