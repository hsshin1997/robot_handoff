"""Deterministic tests for hierarchical bottleneck profiling."""
from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from mujoco_sim.profiling import (HierarchicalProfiler,
                                  profile_lookup)  # noqa: E402


class ManualClock:
    def __init__(self):
        self.value = 0

    def __call__(self):
        return self.value

    def advance(self, nanoseconds: int):
        self.value += nanoseconds


def test_nested_spans_reconcile_inclusive_and_self_wall_time():
    wall = ManualClock()
    cpu = ManualClock()
    profile = HierarchicalProfiler(
        "planning", wall_clock_ns=wall, cpu_clock_ns=cpu)
    with profile.span("query"):
        wall.advance(2_000_000)
        cpu.advance(1_000_000)
        with profile.span("ik"):
            wall.advance(5_000_000)
            cpu.advance(4_000_000)
        wall.advance(3_000_000)
        cpu.advance(2_000_000)

    metrics = profile_lookup(profile.report())
    root = metrics["planning.query"]
    child = metrics["planning.query.ik"]
    assert root["wall_total_s"] == 0.010
    assert root["wall_self_s"] == 0.005
    assert root["cpu_total_s"] == 0.007
    assert child["wall_total_s"] == child["wall_self_s"] == 0.005


def test_repeated_paths_aggregate_calls_maxima_and_failures():
    wall = ManualClock()
    cpu = ManualClock()
    profile = HierarchicalProfiler(
        wall_clock_ns=wall, cpu_clock_ns=cpu)
    with profile.span("candidate"):
        wall.advance(2_000_000)
    try:
        with profile.span("candidate"):
            wall.advance(7_000_000)
            raise RuntimeError("expected")
    except RuntimeError:
        pass
    metric = profile.report()[0]
    assert metric["calls"] == 2
    assert metric["failures"] == 1
    assert metric["wall_total_s"] == 0.009
    assert metric["wall_max_s"] == 0.007


def test_reset_and_bottleneck_order_are_deterministic():
    wall = ManualClock()
    cpu = ManualClock()
    profile = HierarchicalProfiler(
        wall_clock_ns=wall, cpu_clock_ns=cpu)
    with profile.span("small"):
        wall.advance(1_000_000)
    with profile.span("large"):
        wall.advance(3_000_000)
    assert [item["path"] for item in profile.bottlenecks()] == [
        "large", "small"]
    profile.reset()
    assert profile.report() == ()


def test_profile_lookup_rejects_duplicate_paths():
    duplicate = [{"path": "same"}, {"path": "same"}]
    try:
        profile_lookup(duplicate)
    except ValueError as error:
        assert "duplicate profile path" in str(error)
    else:
        raise AssertionError("duplicate profile path was accepted")


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"PASS  {test.__name__}")
    print(f"\n{len(tests)}/{len(tests)} passed")
