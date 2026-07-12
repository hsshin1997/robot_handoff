"""Deterministic, collision-aware motion planning in bounded joint space.

This module deliberately has no dependency on the handoff planner or MuJoCo.
The caller supplies a collision predicate with the following contract::

    in_collision(q) -> bool

``True`` means that the joint configuration is invalid.  This makes the
planner usable for a single robot, a coupled two-robot configuration, or any
other bounded configuration space for which the caller can answer collision
queries.

Edges are validated by recursively bisecting them until every joint changes by
at most ``edge_max_step``.  Consequently, motion validity does not depend on a
fixed number of interpolation samples or on the length of the requested edge.
The guarantee is discrete (as all sampled collision checking is): obstacles
that are thinner than the configured joint-space resolution can still be
missed, so ``edge_max_step`` must be derived from the collision model's motion
resolution.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Sequence

import numpy as np


ArrayLike = Sequence[float] | np.ndarray
CollisionCallback = Callable[[np.ndarray], bool]


class PlanningReason(str, Enum):
    """Machine-readable terminal reason for a planning request."""

    DIRECT_PATH = "direct_path"
    RRT_CONNECTED = "rrt_connected"
    INVALID_INPUT = "invalid_input"
    START_OUT_OF_BOUNDS = "start_out_of_bounds"
    GOAL_OUT_OF_BOUNDS = "goal_out_of_bounds"
    START_IN_COLLISION = "start_in_collision"
    GOAL_IN_COLLISION = "goal_in_collision"
    TIME_BUDGET_EXCEEDED = "time_budget_exceeded"
    NODE_BUDGET_EXCEEDED = "node_budget_exceeded"
    ITERATION_BUDGET_EXCEEDED = "iteration_budget_exceeded"


@dataclass(frozen=True)
class MotionPlannerConfig:
    """Configuration for :class:`JointRRTConnect`.

    ``extension_step`` and ``edge_max_step`` may be scalars or one value per
    joint.  Extension steering scales the entire delta so that no component
    exceeds its extension step; it therefore preserves the sampled direction.
    ``edge_max_step`` has the same per-joint interpretation during recursive
    edge validation.

    ``max_nodes`` includes the start and goal roots.  ``max_iterations`` is a
    deterministic guard for searches in which every extension is rejected and
    therefore the node budget cannot be consumed.  If omitted, it is derived
    from ``max_nodes``.
    """

    extension_step: float | tuple[float, ...] = 0.20
    edge_max_step: float | tuple[float, ...] = 0.025
    goal_bias: float = 0.10
    max_nodes: int = 5_000
    max_iterations: int | None = None
    timeout_s: float = 2.0
    shortcut_attempts: int = 100
    seed: int = 0


@dataclass
class MotionPlanningStats:
    """Search and validation measurements for one planning request."""

    seed: int
    elapsed_s: float = 0.0
    iterations: int = 0
    random_samples: int = 0
    nodes_created: int = 0
    start_tree_nodes: int = 0
    goal_tree_nodes: int = 0
    collision_queries: int = 0
    collision_cache_hits: int = 0
    edge_queries: int = 0
    edge_subdivisions: int = 0
    rejected_edges: int = 0
    connect_steps: int = 0
    direct_edge_attempted: bool = False
    direct_edge_valid: bool = False
    shortcut_attempts: int = 0
    shortcuts_accepted: int = 0
    raw_waypoints: int = 0
    final_waypoints: int = 0
    raw_path_length: float = 0.0
    final_path_length: float = 0.0


@dataclass(frozen=True)
class MotionPlanResult:
    """A path, or an explicit reason that no path was returned."""

    path: np.ndarray | None
    reason: PlanningReason
    message: str
    stats: MotionPlanningStats

    @property
    def success(self) -> bool:
        return self.reason in {
            PlanningReason.DIRECT_PATH,
            PlanningReason.RRT_CONNECTED,
        }


@dataclass(frozen=True)
class EdgeValidationResult:
    """Detailed result from a standalone adaptive edge query."""

    valid: bool
    collision_state: np.ndarray | None
    collision_queries: int
    cache_hits: int
    subdivisions: int


class _CollisionOracle:
    """Memoize deterministic configuration collision queries."""

    def __init__(self, callback: CollisionCallback, stats: MotionPlanningStats | None = None):
        if not callable(callback):
            raise TypeError("in_collision must be callable")
        self.callback = callback
        self.stats = stats
        self.cache: dict[bytes, bool] = {}
        self.queries = 0
        self.cache_hits = 0

    @staticmethod
    def _key(q: np.ndarray) -> bytes:
        # All planner-owned states are float64.  Include the shape so distinct
        # vector dimensions cannot share a key in a reused oracle.
        value = np.ascontiguousarray(q, dtype=np.float64)
        return np.asarray(value.shape, dtype=np.int64).tobytes() + value.tobytes()

    def in_collision(self, q: np.ndarray) -> bool:
        key = self._key(q)
        if key in self.cache:
            self.cache_hits += 1
            if self.stats is not None:
                self.stats.collision_cache_hits += 1
            return self.cache[key]
        # A copy prevents an ill-behaved callback from modifying a tree node.
        result = bool(self.callback(np.asarray(q, dtype=float).copy()))
        self.cache[key] = result
        self.queries += 1
        if self.stats is not None:
            self.stats.collision_queries += 1
        return result


def _step_vector(value: float | Sequence[float], dimension: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        array = np.full(dimension, float(array))
    if array.shape != (dimension,):
        raise ValueError(f"{name} must be a positive scalar or a length-{dimension} vector")
    if not np.all(np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError(f"{name} must contain finite positive values")
    return array


def _adaptive_edge_check(
    q_from: np.ndarray,
    q_to: np.ndarray,
    oracle: _CollisionOracle,
    max_step: np.ndarray,
    stats: MotionPlanningStats | None = None,
) -> tuple[bool, np.ndarray | None, int]:
    """Recursively bisect an edge, returning its first sampled collision."""

    if stats is not None:
        stats.edge_queries += 1

    if oracle.in_collision(q_from):
        if stats is not None:
            stats.rejected_edges += 1
        return False, q_from.copy(), 0
    if oracle.in_collision(q_to):
        if stats is not None:
            stats.rejected_edges += 1
        return False, q_to.copy(), 0

    subdivisions = 0

    def recurse(left: np.ndarray, right: np.ndarray) -> tuple[bool, np.ndarray | None]:
        nonlocal subdivisions
        if np.all(np.abs(right - left) <= max_step):
            return True, None
        midpoint = 0.5 * (left + right)
        subdivisions += 1
        if stats is not None:
            stats.edge_subdivisions += 1
        if oracle.in_collision(midpoint):
            return False, midpoint.copy()
        valid, collision = recurse(left, midpoint)
        if not valid:
            return False, collision
        return recurse(midpoint, right)

    valid, collision = recurse(q_from, q_to)
    if not valid and stats is not None:
        stats.rejected_edges += 1
    return valid, collision, subdivisions


def validate_edge(
    q_from: ArrayLike,
    q_to: ArrayLike,
    in_collision: CollisionCallback,
    max_joint_step: float | Sequence[float],
) -> EdgeValidationResult:
    """Validate one edge with adaptive recursive subdivision.

    This public helper is useful for validating or replaying returned paths.
    It checks both endpoints and all recursively required midpoints.  The
    collision callback returns ``True`` for collision.
    """

    start = np.asarray(q_from, dtype=float)
    goal = np.asarray(q_to, dtype=float)
    if start.ndim != 1 or goal.shape != start.shape or start.size == 0:
        raise ValueError("edge endpoints must be non-empty vectors with equal shape")
    if not np.all(np.isfinite(start)) or not np.all(np.isfinite(goal)):
        raise ValueError("edge endpoints must be finite")
    step = _step_vector(max_joint_step, start.size, "max_joint_step")
    oracle = _CollisionOracle(in_collision)
    valid, collision, subdivisions = _adaptive_edge_check(start, goal, oracle, step)
    return EdgeValidationResult(
        valid=valid,
        collision_state=collision,
        collision_queries=oracle.queries,
        cache_hits=oracle.cache_hits,
        subdivisions=subdivisions,
    )


@dataclass
class _Tree:
    nodes: list[np.ndarray]
    parents: list[int]
    root_is_start: bool

    @classmethod
    def with_root(cls, q: np.ndarray, root_is_start: bool) -> "_Tree":
        return cls([q.copy()], [-1], root_is_start)

    def add(self, q: np.ndarray, parent: int) -> int:
        self.nodes.append(q.copy())
        self.parents.append(parent)
        return len(self.nodes) - 1

    def root_path(self, index: int) -> list[np.ndarray]:
        reverse_path: list[np.ndarray] = []
        while index >= 0:
            reverse_path.append(self.nodes[index])
            index = self.parents[index]
        return [q.copy() for q in reversed(reverse_path)]


class _ExtendState(Enum):
    TRAPPED = 0
    ADVANCED = 1
    REACHED = 2
    BUDGET = 3


def _path_length(path: Sequence[np.ndarray] | np.ndarray) -> float:
    if len(path) < 2:
        return 0.0
    values = np.asarray(path, dtype=float)
    return float(np.linalg.norm(np.diff(values, axis=0), axis=1).sum())


class JointRRTConnect:
    """Bidirectional RRT-Connect for a bounded joint configuration space."""

    def __init__(
        self,
        lower_bounds: ArrayLike,
        upper_bounds: ArrayLike,
        in_collision: CollisionCallback,
        config: MotionPlannerConfig | None = None,
    ):
        self.lower = np.asarray(lower_bounds, dtype=float)
        self.upper = np.asarray(upper_bounds, dtype=float)
        self.in_collision = in_collision
        self.config = config or MotionPlannerConfig()

    def _validated_problem(self, start: ArrayLike, goal: ArrayLike):
        start_array = np.asarray(start, dtype=float)
        goal_array = np.asarray(goal, dtype=float)
        if self.lower.ndim != 1 or self.lower.size == 0:
            raise ValueError("joint bounds must be non-empty vectors")
        if self.upper.shape != self.lower.shape:
            raise ValueError("lower and upper joint bounds must have equal shape")
        if start_array.shape != self.lower.shape or goal_array.shape != self.lower.shape:
            raise ValueError("start and goal must match the joint-bound dimension")
        arrays = (self.lower, self.upper, start_array, goal_array)
        if not all(np.all(np.isfinite(array)) for array in arrays):
            raise ValueError("bounds, start, and goal must be finite")
        if np.any(self.lower > self.upper):
            raise ValueError("every lower joint bound must be <= its upper bound")
        if not callable(self.in_collision):
            raise ValueError("in_collision must be callable")

        dimension = self.lower.size
        extension = _step_vector(self.config.extension_step, dimension, "extension_step")
        edge_step = _step_vector(self.config.edge_max_step, dimension, "edge_max_step")
        if not 0.0 <= self.config.goal_bias <= 1.0:
            raise ValueError("goal_bias must be in [0, 1]")
        if self.config.max_nodes < 2:
            raise ValueError("max_nodes must be at least 2 (the two roots)")
        if self.config.max_iterations is not None and self.config.max_iterations < 0:
            raise ValueError("max_iterations must be non-negative")
        if not np.isfinite(self.config.timeout_s) or self.config.timeout_s < 0.0:
            raise ValueError("timeout_s must be finite and non-negative")
        if self.config.shortcut_attempts < 0:
            raise ValueError("shortcut_attempts must be non-negative")
        return start_array, goal_array, extension, edge_step

    def _failure(
        self,
        reason: PlanningReason,
        message: str,
        stats: MotionPlanningStats,
        begin: float,
        start_tree: _Tree | None = None,
        goal_tree: _Tree | None = None,
    ) -> MotionPlanResult:
        stats.elapsed_s = time.monotonic() - begin
        if start_tree is not None:
            stats.start_tree_nodes = len(start_tree.nodes)
        if goal_tree is not None:
            stats.goal_tree_nodes = len(goal_tree.nodes)
        return MotionPlanResult(None, reason, message, stats)

    def plan(self, start: ArrayLike, goal: ArrayLike) -> MotionPlanResult:
        """Plan from ``start`` to ``goal`` within the configured budgets."""

        begin = time.monotonic()
        stats = MotionPlanningStats(seed=int(self.config.seed))
        try:
            start, goal, extension_step, edge_step = self._validated_problem(start, goal)
        except (TypeError, ValueError) as error:
            return self._failure(
                PlanningReason.INVALID_INPUT, str(error), stats, begin
            )

        tolerance = 64.0 * np.finfo(float).eps
        if np.any(start < self.lower - tolerance) or np.any(start > self.upper + tolerance):
            return self._failure(
                PlanningReason.START_OUT_OF_BOUNDS,
                "start configuration is outside the joint bounds",
                stats,
                begin,
            )
        if np.any(goal < self.lower - tolerance) or np.any(goal > self.upper + tolerance):
            return self._failure(
                PlanningReason.GOAL_OUT_OF_BOUNDS,
                "goal configuration is outside the joint bounds",
                stats,
                begin,
            )

        oracle = _CollisionOracle(self.in_collision, stats)
        if oracle.in_collision(start):
            return self._failure(
                PlanningReason.START_IN_COLLISION,
                "start configuration is in collision",
                stats,
                begin,
            )
        if oracle.in_collision(goal):
            return self._failure(
                PlanningReason.GOAL_IN_COLLISION,
                "goal configuration is in collision",
                stats,
                begin,
            )

        # A direct edge is both the fastest common case and an important way to
        # avoid stochastic search latency for unobstructed robot motions.
        stats.direct_edge_attempted = True
        direct_valid, _, _ = _adaptive_edge_check(start, goal, oracle, edge_step, stats)
        stats.direct_edge_valid = direct_valid
        if direct_valid:
            if np.array_equal(start, goal):
                path = start.reshape(1, -1).copy()
            else:
                path = np.vstack((start, goal))
            length = _path_length(path)
            stats.nodes_created = 2
            stats.start_tree_nodes = 1
            stats.goal_tree_nodes = 1
            stats.raw_waypoints = stats.final_waypoints = len(path)
            stats.raw_path_length = stats.final_path_length = length
            stats.elapsed_s = time.monotonic() - begin
            return MotionPlanResult(
                path,
                PlanningReason.DIRECT_PATH,
                "start and goal are connected by a collision-free edge",
                stats,
            )

        start_tree = _Tree.with_root(start, True)
        goal_tree = _Tree.with_root(goal, False)
        stats.nodes_created = 2
        active, other = start_tree, goal_tree
        rng = np.random.default_rng(self.config.seed)
        span = self.upper - self.lower
        metric_scale = np.where(span > 0.0, span, 1.0)
        max_iterations = (
            self.config.max_iterations
            if self.config.max_iterations is not None
            else max(1_000, 10 * self.config.max_nodes)
        )

        def budget_reason() -> PlanningReason | None:
            if time.monotonic() - begin >= self.config.timeout_s:
                return PlanningReason.TIME_BUDGET_EXCEEDED
            if stats.nodes_created >= self.config.max_nodes:
                return PlanningReason.NODE_BUDGET_EXCEEDED
            return None

        def nearest(tree: _Tree, target: np.ndarray) -> int:
            nodes = np.asarray(tree.nodes)
            distances = np.linalg.norm((nodes - target) / metric_scale, axis=1)
            return int(np.argmin(distances))

        def steer(q_from: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, bool]:
            delta = target - q_from
            ratio = float(np.max(np.abs(delta) / extension_step))
            if ratio <= 1.0:
                return target.copy(), True
            return q_from + delta / ratio, False

        def extend(tree: _Tree, target: np.ndarray) -> tuple[_ExtendState, int | None]:
            if budget_reason() is not None:
                return _ExtendState.BUDGET, None
            near_index = nearest(tree, target)
            q_near = tree.nodes[near_index]
            q_new, reaches_target = steer(q_near, target)
            # A target already in the tree counts as reached without creating a
            # duplicate node.
            if np.array_equal(q_new, q_near):
                return _ExtendState.REACHED, near_index
            edge_valid, _, _ = _adaptive_edge_check(
                q_near, q_new, oracle, edge_step, stats
            )
            if not edge_valid:
                return _ExtendState.TRAPPED, None
            if stats.nodes_created >= self.config.max_nodes:
                return _ExtendState.BUDGET, None
            new_index = tree.add(q_new, near_index)
            stats.nodes_created += 1
            return (
                _ExtendState.REACHED if reaches_target else _ExtendState.ADVANCED,
                new_index,
            )

        connection: tuple[_Tree, int, _Tree, int] | None = None
        terminal_reason: PlanningReason | None = None

        for iteration in range(max_iterations):
            stats.iterations = iteration + 1
            terminal_reason = budget_reason()
            if terminal_reason is not None:
                break

            stats.random_samples += 1
            if rng.random() < self.config.goal_bias:
                sample = other.nodes[0].copy()
            else:
                sample = rng.uniform(self.lower, self.upper)

            state, active_index = extend(active, sample)
            if state is _ExtendState.BUDGET:
                terminal_reason = budget_reason() or PlanningReason.NODE_BUDGET_EXCEEDED
                break
            if state is not _ExtendState.TRAPPED and active_index is not None:
                target = active.nodes[active_index]
                while True:
                    terminal_reason = budget_reason()
                    if terminal_reason is not None:
                        break
                    stats.connect_steps += 1
                    connect_state, other_index = extend(other, target)
                    if connect_state is _ExtendState.REACHED and other_index is not None:
                        connection = (active, active_index, other, other_index)
                        break
                    if connect_state in {_ExtendState.TRAPPED, _ExtendState.BUDGET}:
                        if connect_state is _ExtendState.BUDGET:
                            terminal_reason = budget_reason() or PlanningReason.NODE_BUDGET_EXCEEDED
                        break
                if connection is not None or terminal_reason is not None:
                    break

            # Alternating tree growth avoids systematic preference for either
            # endpoint and is deterministic for a fixed seed.
            active, other = other, active
        else:
            terminal_reason = PlanningReason.ITERATION_BUDGET_EXCEEDED

        if connection is None:
            terminal_reason = terminal_reason or PlanningReason.ITERATION_BUDGET_EXCEEDED
            messages = {
                PlanningReason.TIME_BUDGET_EXCEEDED: "motion-planning time budget was exhausted",
                PlanningReason.NODE_BUDGET_EXCEEDED: "motion-planning node budget was exhausted",
                PlanningReason.ITERATION_BUDGET_EXCEEDED: (
                    "motion-planning iteration budget was exhausted"
                ),
            }
            return self._failure(
                terminal_reason,
                messages[terminal_reason],
                stats,
                begin,
                start_tree,
                goal_tree,
            )

        first_tree, first_index, second_tree, second_index = connection
        first_branch = first_tree.root_path(first_index)
        second_branch = second_tree.root_path(second_index)
        if first_tree.root_is_start:
            raw_path = first_branch + list(reversed(second_branch[:-1]))
        else:
            raw_path = second_branch + list(reversed(first_branch[:-1]))

        stats.raw_waypoints = len(raw_path)
        stats.raw_path_length = _path_length(raw_path)
        smoothed = self._shortcut(raw_path, oracle, edge_step, rng, stats, budget_reason)
        path = np.asarray(smoothed, dtype=float)
        stats.final_waypoints = len(path)
        stats.final_path_length = _path_length(path)
        stats.start_tree_nodes = len(start_tree.nodes)
        stats.goal_tree_nodes = len(goal_tree.nodes)
        stats.elapsed_s = time.monotonic() - begin
        return MotionPlanResult(
            path,
            PlanningReason.RRT_CONNECTED,
            "bidirectional RRT-Connect joined the start and goal trees",
            stats,
        )

    def _shortcut(
        self,
        path: Sequence[np.ndarray],
        oracle: _CollisionOracle,
        edge_step: np.ndarray,
        rng: np.random.Generator,
        stats: MotionPlanningStats,
        budget_reason: Callable[[], PlanningReason | None],
    ) -> list[np.ndarray]:
        result = [np.asarray(q, dtype=float).copy() for q in path]
        for _ in range(self.config.shortcut_attempts):
            if len(result) <= 2 or budget_reason() is not None:
                break
            stats.shortcut_attempts += 1
            indices = np.sort(rng.choice(len(result), size=2, replace=False))
            left, right = int(indices[0]), int(indices[1])
            if right <= left + 1:
                continue
            valid, _, _ = _adaptive_edge_check(
                result[left], result[right], oracle, edge_step, stats
            )
            if valid:
                result = result[: left + 1] + result[right:]
                stats.shortcuts_accepted += 1
        return result


# A descriptive alias for users who prefer the algorithm name over the joint-
# space emphasis.  Keeping one implementation avoids divergent behavior.
RRTConnectPlanner = JointRRTConnect


def plan_joint_path(
    start: ArrayLike,
    goal: ArrayLike,
    lower_bounds: ArrayLike,
    upper_bounds: ArrayLike,
    in_collision: CollisionCallback,
    config: MotionPlannerConfig | None = None,
) -> MotionPlanResult:
    """Functional wrapper around :class:`JointRRTConnect`."""

    return JointRRTConnect(
        lower_bounds, upper_bounds, in_collision, config
    ).plan(start, goal)


__all__ = [
    "CollisionCallback",
    "EdgeValidationResult",
    "JointRRTConnect",
    "MotionPlanResult",
    "MotionPlannerConfig",
    "MotionPlanningStats",
    "PlanningReason",
    "RRTConnectPlanner",
    "plan_joint_path",
    "validate_edge",
]
