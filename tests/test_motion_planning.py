"""Unit tests for the standalone bounded joint-space motion planner."""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from mujoco_sim.planner.motion import (
    JointRRTConnect,
    MotionPlannerConfig,
    PlanningReason,
    plan_joint_path,
    validate_edge,
)


def _wall_with_upper_gap(q: np.ndarray) -> bool:
    """A vertical wall that can only be passed above y=0.72."""

    return bool(0.45 <= q[0] <= 0.55 and q[1] <= 0.72)


def _obstacle_config(seed: int = 7) -> MotionPlannerConfig:
    return MotionPlannerConfig(
        extension_step=0.12,
        edge_max_step=0.01,
        goal_bias=0.1,
        max_nodes=1_000,
        max_iterations=5_000,
        timeout_s=2.0,
        shortcut_attempts=100,
        seed=seed,
    )


def test_adaptive_edge_validation_detects_interior_collision():
    # Both endpoints are free; a checker that only inspected them would miss
    # this narrow obstacle at the middle of the edge.
    collision = lambda q: bool(abs(q[0] - 0.5) <= 0.005)
    result = validate_edge([0.0], [1.0], collision, max_joint_step=0.02)
    assert not result.valid
    assert result.collision_state is not None
    assert np.isclose(result.collision_state[0], 0.5)
    assert result.subdivisions >= 1
    assert result.collision_queries >= 3


def test_direct_edge_is_attempted_before_sampling_even_with_zero_time_budget():
    result = plan_joint_path(
        [0.1, 0.2],
        [0.9, 0.8],
        [0.0, 0.0],
        [1.0, 1.0],
        lambda _q: False,
        MotionPlannerConfig(timeout_s=0.0, seed=123),
    )
    assert result.success
    assert result.reason is PlanningReason.DIRECT_PATH
    assert np.allclose(result.path, [[0.1, 0.2], [0.9, 0.8]])
    assert result.stats.direct_edge_attempted
    assert result.stats.direct_edge_valid
    assert result.stats.random_samples == 0


def test_rrt_connect_finds_collision_free_detour_and_shortcuts_it():
    result = JointRRTConnect(
        [0.0, 0.0], [1.0, 1.0], _wall_with_upper_gap, _obstacle_config()
    ).plan([0.1, 0.5], [0.9, 0.5])

    assert result.success
    assert result.reason is PlanningReason.RRT_CONNECTED
    assert np.allclose(result.path[0], [0.1, 0.5])
    assert np.allclose(result.path[-1], [0.9, 0.5])
    assert np.max(result.path[:, 1]) > 0.72
    for q_from, q_to in zip(result.path[:-1], result.path[1:]):
        assert validate_edge(q_from, q_to, _wall_with_upper_gap, 0.01).valid

    stats = result.stats
    assert not stats.direct_edge_valid
    assert stats.start_tree_nodes + stats.goal_tree_nodes == stats.nodes_created
    assert stats.raw_waypoints >= stats.final_waypoints
    assert stats.raw_path_length + 1e-12 >= stats.final_path_length
    assert stats.shortcut_attempts > 0
    assert stats.shortcuts_accepted > 0


def test_seed_makes_search_and_smoothing_reproducible():
    planner_a = JointRRTConnect(
        [0.0, 0.0], [1.0, 1.0], _wall_with_upper_gap, _obstacle_config(seed=91)
    )
    planner_b = JointRRTConnect(
        [0.0, 0.0], [1.0, 1.0], _wall_with_upper_gap, _obstacle_config(seed=91)
    )
    first = planner_a.plan([0.1, 0.5], [0.9, 0.5])
    second = planner_b.plan([0.1, 0.5], [0.9, 0.5])

    assert first.reason is second.reason
    assert np.array_equal(first.path, second.path)
    assert first.stats.iterations == second.stats.iterations
    assert first.stats.nodes_created == second.stats.nodes_created
    assert first.stats.collision_queries == second.stats.collision_queries
    assert first.stats.shortcuts_accepted == second.stats.shortcuts_accepted


def test_endpoint_failures_have_specific_structured_reasons():
    config = MotionPlannerConfig(timeout_s=1.0)
    planner = JointRRTConnect([0.0], [1.0], lambda q: bool(q[0] > 0.8), config)

    outside_start = planner.plan([-0.1], [0.5])
    assert not outside_start.success
    assert outside_start.path is None
    assert outside_start.reason is PlanningReason.START_OUT_OF_BOUNDS

    outside_goal = planner.plan([0.1], [1.1])
    assert outside_goal.reason is PlanningReason.GOAL_OUT_OF_BOUNDS

    collision_start = JointRRTConnect(
        [0.0], [1.0], lambda q: bool(q[0] < 0.2), config
    ).plan([0.1], [0.5])
    assert collision_start.reason is PlanningReason.START_IN_COLLISION

    collision_goal = planner.plan([0.1], [0.9])
    assert collision_goal.reason is PlanningReason.GOAL_IN_COLLISION


def test_node_and_time_budgets_terminate_blocked_searches():
    # This wall completely separates the two roots.
    barrier = lambda q: bool(0.45 <= q[0] <= 0.55)

    node_limited = plan_joint_path(
        [0.1, 0.5],
        [0.9, 0.5],
        [0.0, 0.0],
        [1.0, 1.0],
        barrier,
        MotionPlannerConfig(
            edge_max_step=0.01,
            max_nodes=2,
            max_iterations=100,
            timeout_s=1.0,
        ),
    )
    assert not node_limited.success
    assert node_limited.reason is PlanningReason.NODE_BUDGET_EXCEEDED
    assert node_limited.stats.nodes_created == 2

    time_limited = plan_joint_path(
        [0.1, 0.5],
        [0.9, 0.5],
        [0.0, 0.0],
        [1.0, 1.0],
        barrier,
        MotionPlannerConfig(
            edge_max_step=0.01,
            max_nodes=100,
            max_iterations=100,
            timeout_s=0.0,
        ),
    )
    assert not time_limited.success
    assert time_limited.reason is PlanningReason.TIME_BUDGET_EXCEEDED


def test_invalid_problem_returns_invalid_input_instead_of_partial_search():
    result = plan_joint_path(
        [0.0, 0.0],
        [1.0, 1.0],
        [0.0],
        [1.0],
        lambda _q: False,
    )
    assert not result.success
    assert result.reason is PlanningReason.INVALID_INPUT
    assert result.stats.collision_queries == 0


if __name__ == "__main__":
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
        print(f"PASS  {test.__name__}")
    print(f"\n{len(tests)}/{len(tests)} passed")
