"""Focused regressions for handoff and stable-placement state semantics."""
from __future__ import annotations

from collections import Counter
import os
import sys
import tempfile
from types import SimpleNamespace

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import mujoco_sim.planning as planning_module
from mujoco_sim.geometry_grasps import TriangleMesh
from mujoco_sim.planning import (
    DirectHandoffPlan,
    HandoffPlanner,
    ScoreBreakdown,
    StablePlacementWitness,
    _normalized_placement_robustness,
)


def _box_mesh(size=(2.0, 1.0, 0.5)) -> TriangleMesh:
    half = 0.5 * np.asarray(size, dtype=float)
    vertices = np.array([
        [-half[0], -half[1], -half[2]],
        [+half[0], -half[1], -half[2]],
        [+half[0], +half[1], -half[2]],
        [-half[0], +half[1], -half[2]],
        [-half[0], -half[1], +half[2]],
        [+half[0], -half[1], +half[2]],
        [+half[0], +half[1], +half[2]],
        [-half[0], +half[1], +half[2]],
    ])
    faces = np.array([
        [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [3, 7, 6], [3, 6, 2],
        [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5],
    ])
    return TriangleMesh.from_triangles(vertices[faces])


def test_unseeded_ik_restarts_are_target_keyed_and_cache_history_independent():
    class FakeKinematics:
        def __init__(self):
            self.reset = None

        def set_q(self, robot, q):
            self.reset = (robot, np.asarray(q).copy())

        def solutions(self, robot, target, restarts, maximum, rng, **kwargs):
            return [rng.random(8)]

    def make_planner(unrelated_draws):
        planner = HandoffPlanner.__new__(HandoffPlanner)
        planner.pos_tol = 1e-4
        planner.rot_tol = 1e-4
        planner.restarts = 18
        planner.max_solutions = 8
        planner.q_start = {"A": np.arange(6.0), "B": -np.arange(6.0)}
        planner._ik_cache = {}
        planner._seed_ik_cache = {}
        planner.kin = FakeKinematics()
        # This legacy shared RNG state must have no influence on target IK.
        planner.rng = np.random.default_rng(7)
        planner.rng.random(unrelated_draws)
        return planner

    target = np.eye(4)
    target[:3, 3] = [0.41, -0.12, 0.73]
    first = make_planner(0)
    second = make_planner(1000)
    result_1 = first._solutions("B", target)
    result_2 = second._solutions("B", target)

    assert np.array_equal(result_1[0], result_2[0])
    assert first.kin.reset[0] == second.kin.reset[0] == "B"
    assert np.array_equal(first.kin.reset[1], first.q_start["B"])
    assert np.array_equal(second.kin.reset[1], second.q_start["B"])


def test_candidate_measures_clearance_in_explicit_cograsp_state():
    class FakeCollision:
        def __init__(self):
            self.current = None
            self.events = []
            self.minimum_state = None

        def check(self, qA, qB, X_part, holders):
            self.current = (
                "co_grasp", np.asarray(qA).copy(), np.asarray(qB).copy(),
                np.asarray(X_part).copy(), tuple(holders))
            self.events.append("check")
            return SimpleNamespace(free=True)

        def minimum_clearance(self, *, policy):
            self.events.append("minimum_clearance")
            self.minimum_state = self.current
            return 0.030

    planner = HandoffPlanner.__new__(HandoffPlanner)
    planner.cfg = {
        "handoff_search": {
            "prehandoff_distance_m": 0.04,
            "retreat_distance_m": 0.06,
        },
        "gates": {
            "minimum_clearance_m": 0.005,
            "calibration_translation_3sigma_m": 0.001,
        },
    }
    planner.q_start = {"A": np.zeros(6), "B": np.zeros(6)}
    insertion_target = SimpleNamespace(T_W_P_insert=np.eye(4))
    planner.project = SimpleNamespace(
        insertion_targets=lambda: [insertion_target],
        region=lambda name: SimpleNamespace(center=np.zeros(3)),
    )
    planner.collision = FakeCollision()
    qA = np.linspace(0.1, 0.6, 6)
    qB = np.linspace(-0.6, -0.1, 6)
    planner._gripper_compatibility = lambda gA, gB: (True, 0.040)
    planner._reach_lookup = lambda robot, target: True
    planner._config_ok = lambda robot, q: True
    planner._backoff_target = lambda target, distance: np.asarray(target).copy()
    planner._solutions = lambda robot, target, seed=None: [
        SimpleNamespace(q=(qA if robot == "A" else qB).copy())]

    def path_query(*args, **kwargs):
        # Reproduce the shared-MjData side effect of real path checking.
        planner.collision.current = ("path_terminal",)
        planner.collision.events.append("path")
        return True, [np.zeros(6), np.ones(6)], "ok"

    planner._held_path = path_query
    planner._fixed_path = path_query
    planner._score = lambda X, qa, qb, downstream, clearance: ScoreBreakdown(
        0.5, 0.5, clearance, 0.5, 0.5, 1.0)

    X_handoff = np.eye(4)
    downstream = SimpleNamespace(
        grasp_name="receiver", grasp=np.eye(4), q_scanner=np.ones(6))
    plan = planner._candidate(
        X_handoff, np.eye(4), downstream, Counter(), fast=False)

    assert plan is not None
    assert planner.collision.events[-2:] == ["check", "minimum_clearance"]
    label, measured_qA, measured_qB, measured_X, holders = (
        planner.collision.minimum_state)
    assert label == "co_grasp"
    assert np.array_equal(measured_qA, qA)
    assert np.array_equal(measured_qB, qB)
    assert np.array_equal(measured_X, X_handoff)
    assert holders == ("A", "B")


def test_scale_aware_support_policy_and_cached_robustness_witnesses():
    mesh = _box_mesh()
    part_scale = float(np.linalg.norm(mesh.extent))
    fraction = 0.15

    class FakeProject:
        solver = {
            "geometry": {
                "minimum_support_margin_mesh_fraction": fraction,
            },
            "planning": {"reorientation_yaw_samples": 2},
        }
        manifest = {"active_task": {"part": "synthetic-box"}}
        active_part_path = __file__

        @staticmethod
        def support_region():
            return SimpleNamespace(T_W_N=np.eye(4), size=np.array([10.0, 10.0]))

    planner = HandoffPlanner.__new__(HandoffPlanner)
    planner.project = FakeProject()
    planner.part_geometry = SimpleNamespace(
        artifact_fingerprint="synthetic-box-prepared-si")
    planner.part_mesh = mesh
    with tempfile.TemporaryDirectory() as directory:
        planner.cache_dir = directory
        first = list(planner.stable_placement_witnesses())
        second = list(planner.stable_placement_witnesses())
        pairs = list(planner.stable_placements())

    # A 15%-of-diagonal margin removes the four narrow box supports. The two
    # broad supports each retain both requested stage yaws.
    assert len(first) == len(second) == len(pairs) == 4
    minimum = fraction * part_scale
    for first_item, cached_item, pair in zip(first, second, pairs):
        assert first_item.support_margin >= minimum - 1e-12
        assert np.isclose(first_item.minimum_support_margin, minimum)
        assert first_item.support_area > 0.0
        assert first_item.edge_clearance > 0.0
        assert 0.0 < first_item.support_robustness <= 1.0
        assert 0.0 < first_item.stage_robustness <= 1.0
        assert np.isclose(first_item.robustness, min(
            first_item.support_robustness, first_item.stage_robustness))
        assert cached_item.support_margin == first_item.support_margin
        assert cached_item.support_area == first_item.support_area
        assert cached_item.edge_clearance == first_item.edge_clearance
        assert cached_item.robustness == first_item.robustness
        assert pair[0] == first_item.name
        assert np.array_equal(pair[1], first_item.T_W_P)
        assert pair[1].flags.writeable
        assert not first_item.T_W_P.flags.writeable


def test_placement_robustness_is_scale_invariant_and_reaches_task_edges():
    scores = _normalized_placement_robustness(0.1, 0.2, 1.0, 2.0)
    scaled = _normalized_placement_robustness(1.0, 2.0, 10.0, 20.0)
    assert np.allclose(scores, (0.2, 0.2, 0.2))
    assert np.allclose(scaled, scores)

    witness = StablePlacementWitness(
        name="stable_test",
        T_W_P=np.eye(4),
        support_margin=0.1,
        support_area=0.2,
        edge_clearance=0.2,
        probability_proxy=0.5,
        minimum_support_margin=0.01,
        part_scale=1.0,
        stage_scale=2.0,
        support_robustness=scores[0],
        stage_robustness=scores[1],
        robustness=scores[2],
    )
    q = np.zeros(6)
    template = DirectHandoffPlan(
        X_handoff=np.eye(4),
        g_A=np.eye(4),
        grasp_name_B="receiver",
        g_B=np.eye(4),
        qA_handoff=q.copy(),
        qB_handoff=q.copy(),
        qA_pre=q.copy(),
        qB_pre=q.copy(),
        qA_retreat=q.copy(),
        downstream=SimpleNamespace(),
        trajectories={"A_approach": [q.copy()], "B_approach": [q.copy()]},
        score=ScoreBreakdown(0.5, 0.5, 0.5, 0.5, 0.5, 1.0),
    )
    planner = HandoffPlanner.__new__(HandoffPlanner)
    planner.cfg = {"regrasp": {"enabled": True}}
    planner.project = SimpleNamespace(
        solver={"planning": {
            "reorientation_goal_grasp_limit": 1,
            "reorientation_direct_goal_limit": 1,
            "max_reorientation_hops": 1,
        }},
        T_tcp_part_start=np.eye(4),
    )
    planner.g_B_candidates = []
    planner.g_A_start = np.eye(4)
    planner.q_start = {"A": q.copy(), "B": q.copy()}
    planner.stable_placement_witnesses = lambda: iter((witness,))
    planner.search_direct = lambda grasp, stats, return_best=False: (
        template, 1, [])
    planner._solutions = lambda robot, target, seed=None: [
        SimpleNamespace(q=np.ones(6))]
    planner._config_ok = lambda robot, values: True
    planner._held_path = lambda *args, **kwargs: (
        True, [np.zeros(6), np.ones(6)], "ok")
    planner._fixed_path = lambda *args, **kwargs: (
        True, [np.zeros(6), np.ones(6)], "ok")

    captured_edges = []

    class CapturingTaskGraph:
        def __init__(self, initial, receivers, direct_edges, placement_edges):
            captured_edges.extend(placement_edges)

        @staticmethod
        def plan(initial_class, max_reorientation_hops):
            return SimpleNamespace(success=True, steps=(
                SimpleNamespace(kind="place", target=witness.name),
                SimpleNamespace(kind="regrasp", target="nominal"),
            ))

    original = planning_module.TaskGraph
    planning_module.TaskGraph = CapturingTaskGraph
    try:
        result = planner._search_regrasp_core(Counter())
    finally:
        planning_module.TaskGraph = original

    assert result is not None
    assert len(captured_edges) == 2
    assert all(np.isclose(edge.robustness, witness.robustness)
               for edge in captured_edges)
    assert all(not np.isclose(edge.robustness, 1.0) for edge in captured_edges)


if __name__ == "__main__":
    tests = [
        value for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
        print(f"PASS  {test.__name__}")
    print(f"\n{len(tests)}/{len(tests)} passed")
