"""Tests for the MuJoCo-independent backward handoff task graph."""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from mujoco_sim.planner.task_graph import (
    DirectCoGraspEdge,
    InitialGraspClass,
    PlacementGraspEdge,
    TaskGraph,
)


def _edge(giver, receiver, cost, robustness, edge_id=None):
    return DirectCoGraspEdge(giver, receiver, cost, robustness, edge_id)


def _placement(placement, grasp, cost, robustness, edge_id=None):
    return PlacementGraspEdge(placement, grasp, cost, robustness, edge_id)


def test_direct_is_hard_preference_even_when_reorientation_is_cheaper():
    graph = TaskGraph(
        initial_classes=["A_initial"],
        insertion_feasible_receiver_grasps=["B_insert"],
        direct_edges=[
            _edge("A_initial", "B_insert", 8.0, 0.2, "slow_direct"),
            _edge("A_repick", "B_insert", 0.1, 0.9, "after_reorient"),
        ],
        placement_edges=[
            _placement("table", "A_initial", 0.1, 0.95),
            _placement("table", "A_repick", 0.1, 0.95),
        ],
    )
    plan = graph.plan("A_initial", max_reorientation_hops=3)
    assert plan.success
    assert plan.mode == "direct"
    assert plan.reason == "direct_co_grasp_available"
    assert [step.kind for step in plan.steps] == ["handoff"]
    assert plan.steps[0].edge_id == "slow_direct"
    assert plan.total_cost == 8.0


def test_backward_reorientation_ends_in_insertion_feasible_direct_handoff():
    graph = TaskGraph(
        initial_classes=[InitialGraspClass.singleton("A_center", "center_class")],
        insertion_feasible_receiver_grasps=["B_pin_aligned"],
        direct_edges=[
            # This tempting edge must be ignored: the B grasp cannot insert.
            _edge("A_center", "B_not_insertable", 0.01, 1.0, "invalid_downstream"),
            _edge("A_end", "B_pin_aligned", 0.7, 0.8, "valid_handoff"),
        ],
        placement_edges=[
            _placement("flat_surface", "A_center", 0.4, 0.75, "place_center"),
            _placement("flat_surface", "A_end", 0.6, 0.85, "pick_end"),
        ],
    )
    plan = graph.plan("center_class", max_reorientation_hops=1)
    assert plan.success
    assert plan.mode == "reorientation"
    assert plan.reason == "reorientation_path_found"
    assert plan.initial_grasp == "A_center"
    assert plan.receiver_grasp == "B_pin_aligned"
    assert plan.reorientation_hops == 1
    assert [(step.kind, step.source, step.target) for step in plan.steps] == [
        ("place", "A_center", "flat_surface"),
        ("regrasp", "flat_surface", "A_end"),
        ("handoff", "A_end", "B_pin_aligned"),
    ]
    # The re-picked A grasp, rather than the original center grasp, must be
    # the source of an insertion-feasible direct co-grasp edge.
    assert plan.steps[-2].target == plan.steps[-1].source
    assert plan.steps[-1].target in graph.insertion_feasible_receiver_grasps
    assert plan.total_cost == 1.7
    assert plan.bottleneck_robustness == 0.75


def test_minimum_cycle_cost_then_bottleneck_robustness_breaks_ties():
    graph = TaskGraph(
        initial_classes=["g0"],
        insertion_feasible_receiver_grasps=["b"],
        direct_edges=[
            _edge("g_low", "b", 1.0, 0.9, "low_route_handoff"),
            _edge("g_high", "b", 1.0, 0.9, "high_route_handoff"),
            _edge("g_expensive", "b", 1.0, 1.0, "expensive_handoff"),
        ],
        placement_edges=[
            _placement("p_low", "g0", 0.5, 0.40),
            _placement("p_low", "g_low", 0.5, 0.95),
            _placement("p_high", "g0", 0.5, 0.80),
            _placement("p_high", "g_high", 0.5, 0.85),
            _placement("p_expensive", "g0", 0.8, 1.0),
            _placement("p_expensive", "g_expensive", 0.8, 1.0),
        ],
    )
    plan = graph.plan("g0")
    assert plan.mode == "reorientation"
    assert plan.total_cost == 2.0
    assert plan.bottleneck_robustness == 0.8
    assert plan.steps[0].target == "p_high"
    assert plan.steps[-1].source == "g_high"


def test_lower_cycle_cost_can_use_more_hops_within_bound():
    graph = TaskGraph(
        initial_classes=["start"],
        insertion_feasible_receiver_grasps=["receiver"],
        direct_edges=[
            _edge("one_hop", "receiver", 3.0, 0.9),
            _edge("two_hop", "receiver", 0.1, 0.9),
        ],
        placement_edges=[
            _placement("p_one", "start", 2.0, 0.9),
            _placement("p_one", "one_hop", 2.0, 0.9),
            _placement("p_a", "start", 0.1, 0.9),
            _placement("p_a", "middle", 0.1, 0.9),
            _placement("p_b", "middle", 0.1, 0.9),
            _placement("p_b", "two_hop", 0.1, 0.9),
        ],
    )
    plan = graph.plan("start", max_reorientation_hops=2)
    assert plan.mode == "reorientation"
    assert plan.reorientation_hops == 2
    assert plan.total_cost == 0.5
    assert [step.kind for step in plan.steps] == [
        "place", "regrasp", "place", "regrasp", "handoff"
    ]
    bounded = graph.plan("start", max_reorientation_hops=1)
    assert bounded.reorientation_hops == 1
    assert bounded.steps[-1].source == "one_hop"


def test_hop_bound_and_empty_downstream_return_explicit_reasons():
    edges = [
        _placement("p", "start", 0.2, 0.8),
        _placement("p", "repick", 0.2, 0.8),
    ]
    bounded = TaskGraph(
        ["start"], ["receiver"],
        [_edge("repick", "receiver", 0.2, 0.8)], edges)
    disabled = bounded.plan("start", max_reorientation_hops=0)
    assert not disabled.success
    assert disabled.reason == "no_direct_path_and_reorientation_disabled"

    disconnected = bounded.plan("start", max_reorientation_hops=1)
    assert disconnected.success
    impossible = TaskGraph(
        ["other"], ["receiver"],
        [_edge("repick", "receiver", 0.2, 0.8)], edges)
    failure = impossible.plan("other", max_reorientation_hops=3)
    assert not failure.success
    assert failure.reason == "no_path_within_reorientation_hop_bound"

    no_downstream = TaskGraph(["start"], [], [], edges).plan("start")
    assert not no_downstream.success
    assert no_downstream.reason == "no_insertion_feasible_receiver_grasps"


def test_coverage_certificate_separates_all_categories_and_is_exact():
    graph = TaskGraph(
        initial_classes={
            "class_direct": ["g_direct"],
            "class_reorientation": ["g_before"],
            "class_uncovered": ["g_missing"],
        },
        insertion_feasible_receiver_grasps=["b_insert"],
        direct_edges=[
            _edge("g_direct", "b_insert", 1.0, 0.9),
            _edge("g_after", "b_insert", 1.0, 0.9),
        ],
        placement_edges=[
            _placement("surface", "g_before", 0.5, 0.8),
            _placement("surface", "g_after", 0.5, 0.8),
        ],
    )
    fraction = 2.0 / 3.0
    report = graph.coverage_report(target_fraction=fraction)
    assert report.direct_classes == ("class_direct",)
    assert report.reorientation_classes == ("class_reorientation",)
    assert report.uncovered_classes == ("class_uncovered",)
    assert set(report.covered_classes) == {"class_direct", "class_reorientation"}
    assert report.covered_count == 2
    assert report.total_count == 3
    assert report.fraction == fraction
    assert report.certified

    full_target = graph.coverage_report(target_fraction=1.0)
    assert not full_target.certified
    assert graph.coverage_report(target_fraction=0.25).certified
    encoded = json.dumps(report.to_dict(), sort_keys=True)
    assert '"certified": true' in encoded
    assert '"reorientation_classes"' in encoded


def test_multi_representative_class_still_applies_direct_first_rule():
    graph = TaskGraph(
        [InitialGraspClass("symmetric_class", ("g_sym0", "g_sym1"))],
        ["receiver"],
        [
            _edge("g_sym1", "receiver", 4.0, 0.5),
            _edge("g_repick", "receiver", 0.1, 1.0),
        ],
        [
            _placement("surface", "g_sym0", 0.1, 1.0),
            _placement("surface", "g_repick", 0.1, 1.0),
        ],
    )
    plan = graph.plan("symmetric_class")
    assert plan.mode == "direct"
    assert plan.initial_grasp == "g_sym1"


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS  {test.__name__}")
    print(f"\n{len(tests)}/{len(tests)} passed")
