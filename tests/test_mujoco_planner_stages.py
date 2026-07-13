"""Fast behavioral contracts for independently modularized planner stages."""
from __future__ import annotations

from collections import Counter
import os
import sys
from types import SimpleNamespace

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from mujoco_sim.planner_stages import (DirectHandoffSearch,
                                       DownstreamCertifier)  # noqa: E402


def _pose(x):
    value = np.eye(4)
    value[0, 3] = x
    return value


def _plan(score):
    return SimpleNamespace(score=SimpleNamespace(total=float(score)))


def test_direct_search_warm_first_returns_first_feasible_candidate():
    poses = [_pose(0), _pose(1), _pose(2)]
    witnesses = [SimpleNamespace(grasp_name="a"),
                 SimpleNamespace(grasp_name="b")]
    calls = []

    def evaluate(pose, grasp, witness, stats, *, fast, warm_only=False):
        calls.append((pose[0, 3], witness.grasp_name, fast, warm_only))
        return _plan(1.0) if (pose[0, 3], witness.grasp_name) == (1, "a") else None

    stage = DirectHandoffSearch(lambda: poses, evaluate)
    result, count = stage.search(
        np.eye(4), witnesses, Counter(), return_best=False)
    assert result is not None
    assert count == 3
    assert calls == [
        (0, "a", True, True),
        (0, "b", True, True),
        (1, "a", True, True),
    ]


def test_direct_search_exhaustive_fallback_preserves_completeness():
    poses = [_pose(0), _pose(1)]
    witness = SimpleNamespace(grasp_name="only")
    calls = []

    def evaluate(pose, grasp, downstream, stats, *, fast, warm_only=False):
        calls.append((pose[0, 3], fast, warm_only))
        if not fast and pose[0, 3] == 1:
            return _plan(2.0)
        return None

    result, count = DirectHandoffSearch(lambda: poses, evaluate).search(
        np.eye(4), [witness], Counter(), return_best=False)
    assert result.score.total == 2.0
    assert count == 4  # two warm probes plus two complete probes
    assert calls[-1] == (1, False, False)


def test_direct_best_mode_skips_warm_pass_and_selects_highest_score():
    poses = [_pose(0), _pose(1), _pose(2)]

    def evaluate(pose, grasp, witness, stats, *, fast, warm_only=False):
        assert not fast and not warm_only
        return _plan([0.2, 0.9, 0.4][int(pose[0, 3])])

    result, count = DirectHandoffSearch(lambda: poses, evaluate).search(
        np.eye(4), [SimpleNamespace()], Counter(), return_best=True)
    assert count == 3
    assert result.score.total == 0.9


class FakeKinematics:
    lower = {"B": np.full(6, -2.0)}
    upper = {"B": np.full(6, 2.0)}

    @staticmethod
    def penalized_manipulability(robot, q):
        return 0.75


class FakeDownstreamRuntime:
    def __init__(self, *, correction_ok=True, path_ok=True):
        self.g_B_candidates = [("surface_000", np.eye(4))]
        self.q_start = {"A": np.zeros(6), "B": np.zeros(6)}
        self.cfg = {"downstream": {"wrist_dither_deg": 1.0}}
        self.limit_margin = 0.1
        self.kin = FakeKinematics()
        self._correction_result = correction_ok
        self._path_result = path_ok
        self.path_calls = []

    @property
    def X_scanner(self):
        return _pose(0.1)

    @property
    def insertion_poses(self):
        return [("pcb_0", _pose(0.2))]

    @staticmethod
    def _preinsert_pose(X_insert):
        return _pose(X_insert[0, 3] - 0.05)

    @staticmethod
    def _solutions(robot, target, seed=None):
        q = np.zeros(6)
        q[0] = target[0, 3]
        return [SimpleNamespace(q=q)]

    @staticmethod
    def _config_ok(robot, q):
        return True

    def _correction_ok(self, grasp, q_insert, X_insert):
        return self._correction_result, [q_insert.copy()], 0.12

    def _held_path(self, *args, **kwargs):
        self.path_calls.append((args, kwargs))
        return self._path_result, [args[1].copy(), args[2].copy()], "ok"


def test_downstream_certifier_builds_complete_named_witness():
    runtime = FakeDownstreamRuntime()
    stats = Counter()
    contacts = (("part_collision", "fixture", 0.0),)
    witnesses = DownstreamCertifier(runtime, contacts).certify(stats)
    assert len(witnesses) == 1
    witness = witnesses[0]
    assert witness.grasp_name == "surface_000"
    assert set(witness.trajectories) == {
        "scanner_to_pcb_0_pre", "pcb_0_insert"}
    assert witness.quality == 0.75
    assert witness.sigma_min == 0.12
    assert runtime.path_calls[-1][0][-1] == contacts
    assert not stats


def test_downstream_certifier_rejects_failed_correction_without_motion():
    runtime = FakeDownstreamRuntime(correction_ok=False)
    stats = Counter()
    witnesses = DownstreamCertifier(runtime, ()).certify(stats)
    assert witnesses == []
    assert stats["downstream_rejected"] == 1
    assert runtime.path_calls == []


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"PASS  {test.__name__}")
    print(f"\n{len(tests)}/{len(tests)} passed")
