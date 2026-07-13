"""Direct-first backward task graph for handoff and reorientation.

The graph is deliberately independent of MuJoCo, robot models, and geometry.
Continuous planners populate it with already-validated feasibility edges:

* a :class:`DirectCoGraspEdge` joins an A grasp to an insertion-feasible B
  grasp and represents a collision-free co-grasp/handoff;
* a :class:`PlacementGraspEdge` joins a stable placement to an A grasp and
  represents a feasible place or pick motion at that placement.

An A reorientation transition ``g0 -> placement -> g1`` exists only when both
``(placement, g0)`` and ``(placement, g1)`` edges exist.  Searching backward
from insertion-feasible B grasps guarantees that the final re-picked A grasp
has an actual direct co-grasp edge to an insertion-valid B grasp.

Selection is lexicographic:

1. Any direct plan from the initial grasp class is preferred, regardless of
   the cost of a reorientation alternative.
2. Otherwise minimize summed cycle-time cost over bounded reorientation paths.
3. Break equal-cost ties by fewer reorientation hops, then by maximum
   bottleneck robustness, then by stable ID ordering.

All IDs are generic hashable values.  Geometry, collision, IK, and trajectory
checks therefore remain cleanly separated from this discrete certificate.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
import json
from typing import Any, Literal

import numpy as np


NodeId = Hashable
PlanMode = Literal["direct", "reorientation"]
StepKind = Literal["place", "regrasp", "handoff"]


def _validate_id(value: NodeId, name: str) -> NodeId:
    if value is None:
        raise ValueError(f"{name} cannot be None")
    try:
        hash(value)
    except TypeError as error:
        raise TypeError(f"{name} must be hashable") from error
    return value


def _stable_id(value: NodeId) -> str:
    """Total deterministic ordering for heterogeneous hashable IDs."""
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}:{value!r}"


def _metric(value: float, name: str, *, strictly_positive: bool = False) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if result < 0.0 or (strictly_positive and result <= 0.0):
        qualifier = "positive" if strictly_positive else "non-negative"
        raise ValueError(f"{name} must be {qualifier}")
    return result


def _json_id(value: NodeId) -> Any:
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return {"repr": repr(value), "type": type(value).__qualname__}
    return value


@dataclass(frozen=True)
class InitialGraspClass:
    """An admissible initial condition and its interchangeable grasp IDs.

    Use singleton classes when the actual initial grasp is known exactly.  A
    multi-ID class is useful for symmetry-equivalent or discretized grasps for
    which the continuous layer may select any representative.
    """

    class_id: NodeId
    grasp_ids: tuple[NodeId, ...]

    def __post_init__(self) -> None:
        class_id = _validate_id(self.class_id, "class_id")
        grasps = tuple(self.grasp_ids)
        if not grasps:
            raise ValueError("an initial grasp class cannot be empty")
        for grasp in grasps:
            _validate_id(grasp, "grasp_id")
        if len(set(grasps)) != len(grasps):
            raise ValueError("grasp IDs within a class must be unique")
        object.__setattr__(self, "class_id", class_id)
        object.__setattr__(self, "grasp_ids", grasps)

    @classmethod
    def singleton(
        cls,
        grasp_id: NodeId,
        class_id: NodeId | None = None,
    ) -> "InitialGraspClass":
        return cls(grasp_id if class_id is None else class_id, (grasp_id,))

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_id": _json_id(self.class_id),
            "grasp_ids": [_json_id(value) for value in self.grasp_ids],
        }


@dataclass(frozen=True)
class DirectCoGraspEdge:
    """Validated direct handoff from one A grasp to one B grasp."""

    giver_grasp: NodeId
    receiver_grasp: NodeId
    cost: float
    robustness: float
    edge_id: NodeId | None = None

    def __post_init__(self) -> None:
        _validate_id(self.giver_grasp, "giver_grasp")
        _validate_id(self.receiver_grasp, "receiver_grasp")
        cost = _metric(self.cost, "cost")
        robustness = _metric(self.robustness, "robustness")
        edge_id = self.edge_id
        if edge_id is not None:
            _validate_id(edge_id, "edge_id")
        object.__setattr__(self, "cost", cost)
        object.__setattr__(self, "robustness", robustness)

    @property
    def stable_id(self) -> str:
        if self.edge_id is not None:
            return _stable_id(self.edge_id)
        return f"{_stable_id(self.giver_grasp)}->{_stable_id(self.receiver_grasp)}"


@dataclass(frozen=True)
class PlacementGraspEdge:
    """Validated feasibility of placing/picking a grasp at a placement.

    The cost and robustness apply to one traversal.  Reorientation through a
    placement uses two edges: the current grasp's place edge and the new
    grasp's pick edge.
    """

    placement_id: NodeId
    grasp_id: NodeId
    cost: float
    robustness: float
    edge_id: NodeId | None = None

    def __post_init__(self) -> None:
        _validate_id(self.placement_id, "placement_id")
        _validate_id(self.grasp_id, "grasp_id")
        cost = _metric(self.cost, "cost")
        robustness = _metric(self.robustness, "robustness")
        if self.edge_id is not None:
            _validate_id(self.edge_id, "edge_id")
        object.__setattr__(self, "cost", cost)
        object.__setattr__(self, "robustness", robustness)

    @property
    def stable_id(self) -> str:
        if self.edge_id is not None:
            return _stable_id(self.edge_id)
        return f"{_stable_id(self.placement_id)}<->{_stable_id(self.grasp_id)}"


@dataclass(frozen=True)
class TaskStep:
    """One explicit operation in a selected task sequence."""

    kind: StepKind
    source: NodeId
    target: NodeId
    cost: float
    robustness: float
    edge_id: NodeId | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("place", "regrasp", "handoff"):
            raise ValueError(f"unknown task step kind {self.kind!r}")
        _validate_id(self.source, "step source")
        _validate_id(self.target, "step target")
        object.__setattr__(self, "cost", _metric(self.cost, "step cost"))
        object.__setattr__(self, "robustness",
                           _metric(self.robustness, "step robustness"))
        if self.edge_id is not None:
            _validate_id(self.edge_id, "step edge_id")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "source": _json_id(self.source),
            "target": _json_id(self.target),
            "cost": self.cost,
            "robustness": self.robustness,
            "edge_id": None if self.edge_id is None else _json_id(self.edge_id),
        }


@dataclass(frozen=True)
class TaskPlan:
    """Successful sequence or an explicit search failure."""

    success: bool
    reason: str
    initial_class: NodeId
    initial_grasp: NodeId | None = None
    receiver_grasp: NodeId | None = None
    mode: PlanMode | None = None
    steps: tuple[TaskStep, ...] = ()
    total_cost: float | None = None
    bottleneck_robustness: float | None = None
    reorientation_hops: int = 0

    def __post_init__(self) -> None:
        _validate_id(self.initial_class, "initial_class")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("reason must be a non-empty string")
        steps = tuple(self.steps)
        object.__setattr__(self, "steps", steps)
        if self.success:
            if self.mode not in ("direct", "reorientation"):
                raise ValueError("a successful plan needs a mode")
            if self.initial_grasp is None or self.receiver_grasp is None:
                raise ValueError("a successful plan needs initial and receiver grasps")
            _validate_id(self.initial_grasp, "initial_grasp")
            _validate_id(self.receiver_grasp, "receiver_grasp")
            if not steps or steps[-1].kind != "handoff":
                raise ValueError("a successful plan must end with handoff")
            if self.total_cost is None or self.bottleneck_robustness is None:
                raise ValueError("a successful plan needs cost and robustness")
            cost = _metric(self.total_cost, "total_cost")
            robustness = _metric(self.bottleneck_robustness,
                                 "bottleneck_robustness")
            if not np.isclose(cost, sum(step.cost for step in steps), atol=1e-12):
                raise ValueError("total_cost does not equal the step sum")
            if not np.isclose(robustness,
                              min(step.robustness for step in steps), atol=1e-12):
                raise ValueError("bottleneck_robustness does not match the steps")
            expected_hops = sum(step.kind == "place" for step in steps)
            if self.reorientation_hops != expected_hops:
                raise ValueError("reorientation_hops does not match the sequence")
            if self.mode == "direct" and self.reorientation_hops != 0:
                raise ValueError("a direct plan cannot contain reorientation")
            if self.mode == "reorientation" and self.reorientation_hops <= 0:
                raise ValueError("a reorientation plan needs at least one placement")
            object.__setattr__(self, "total_cost", cost)
            object.__setattr__(self, "bottleneck_robustness", robustness)
        else:
            if self.mode is not None or steps:
                raise ValueError("a failed plan cannot contain a mode or steps")
            if self.total_cost is not None or self.bottleneck_robustness is not None:
                raise ValueError("a failed plan cannot contain path metrics")
            if self.reorientation_hops != 0:
                raise ValueError("a failed plan cannot report completed hops")

    @property
    def covered(self) -> bool:
        return self.success

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "reason": self.reason,
            "initial_class": _json_id(self.initial_class),
            "initial_grasp": (None if self.initial_grasp is None
                              else _json_id(self.initial_grasp)),
            "receiver_grasp": (None if self.receiver_grasp is None
                               else _json_id(self.receiver_grasp)),
            "mode": self.mode,
            "steps": [step.to_dict() for step in self.steps],
            "total_cost": self.total_cost,
            "bottleneck_robustness": self.bottleneck_robustness,
            "reorientation_hops": self.reorientation_hops,
        }


@dataclass(frozen=True)
class CoverageReport:
    """Coverage certificate over all configured admissible initial classes."""

    plans: tuple[TaskPlan, ...]
    direct_classes: tuple[NodeId, ...]
    reorientation_classes: tuple[NodeId, ...]
    uncovered_classes: tuple[NodeId, ...]
    covered_classes: tuple[NodeId, ...]
    fraction: float
    target_fraction: float
    certified: bool

    def __post_init__(self) -> None:
        plans = tuple(self.plans)
        direct = tuple(self.direct_classes)
        reorientation = tuple(self.reorientation_classes)
        uncovered = tuple(self.uncovered_classes)
        covered = tuple(self.covered_classes)
        class_ids = tuple(plan.initial_class for plan in plans)
        if len(set(class_ids)) != len(class_ids):
            raise ValueError("coverage plans must have unique initial classes")
        if set(direct) & set(reorientation) or set(covered) & set(uncovered):
            raise ValueError("coverage categories must be disjoint")
        if set(covered) != set(direct) | set(reorientation):
            raise ValueError("covered classes must equal direct union reorientation")
        if set(class_ids) != set(covered) | set(uncovered):
            raise ValueError("coverage categories must include every plan")
        expected = 1.0 if not plans else len(covered) / len(plans)
        if not np.isclose(self.fraction, expected, atol=1e-15, rtol=0.0):
            raise ValueError("incorrect coverage fraction")
        target = float(self.target_fraction)
        if not np.isfinite(target) or not 0.0 <= target <= 1.0:
            raise ValueError("target_fraction must lie in [0, 1]")
        if bool(self.certified) != (float(self.fraction) >= target):
            raise ValueError("certified must mean fraction >= target_fraction")
        object.__setattr__(self, "plans", plans)
        object.__setattr__(self, "direct_classes", direct)
        object.__setattr__(self, "reorientation_classes", reorientation)
        object.__setattr__(self, "uncovered_classes", uncovered)
        object.__setattr__(self, "covered_classes", covered)
        object.__setattr__(self, "fraction", float(self.fraction))
        object.__setattr__(self, "target_fraction", target)
        object.__setattr__(self, "certified", bool(self.certified))

    @property
    def covered_count(self) -> int:
        return len(self.covered_classes)

    @property
    def total_count(self) -> int:
        return len(self.plans)

    def to_dict(self) -> dict[str, Any]:
        return {
            "plans": [plan.to_dict() for plan in self.plans],
            "direct_classes": [_json_id(value) for value in self.direct_classes],
            "reorientation_classes": [
                _json_id(value) for value in self.reorientation_classes
            ],
            "uncovered_classes": [
                _json_id(value) for value in self.uncovered_classes
            ],
            "covered_classes": [_json_id(value) for value in self.covered_classes],
            "fraction": self.fraction,
            "target_fraction": self.target_fraction,
            "certified": self.certified,
        }


@dataclass(frozen=True)
class _Transition:
    previous_grasp: NodeId
    placement_id: NodeId
    next_grasp: NodeId
    place_edge: PlacementGraspEdge
    pick_edge: PlacementGraspEdge

    @property
    def signature(self) -> tuple[str, ...]:
        return (
            _stable_id(self.previous_grasp),
            _stable_id(self.placement_id),
            _stable_id(self.next_grasp),
            self.place_edge.stable_id,
            self.pick_edge.stable_id,
        )


def _normalize_initial_classes(
    initial_classes: (
        Mapping[NodeId, Iterable[NodeId]]
        | Iterable[InitialGraspClass | NodeId]
    ),
) -> tuple[InitialGraspClass, ...]:
    if isinstance(initial_classes, Mapping):
        normalized = []
        for class_id, values in initial_classes.items():
            if isinstance(values, (str, bytes)):
                grasps = (values,)
            else:
                try:
                    grasps = tuple(values)
                except TypeError:
                    grasps = (values,)  # type: ignore[arg-type]
            normalized.append(InitialGraspClass(class_id, grasps))
    else:
        normalized = []
        for value in initial_classes:
            if isinstance(value, InitialGraspClass):
                normalized.append(value)
            else:
                normalized.append(InitialGraspClass.singleton(value))
    if not normalized:
        raise ValueError("at least one admissible initial grasp class is required")
    class_ids = [item.class_id for item in normalized]
    if len(set(class_ids)) != len(class_ids):
        raise ValueError("initial class IDs must be unique")
    return tuple(sorted(normalized, key=lambda item: _stable_id(item.class_id)))


class TaskGraph:
    """Immutable direct-first handoff/reorientation feasibility graph."""

    def __init__(
        self,
        initial_classes: (
            Mapping[NodeId, Iterable[NodeId]]
            | Iterable[InitialGraspClass | NodeId]
        ),
        insertion_feasible_receiver_grasps: Iterable[NodeId],
        direct_edges: Iterable[DirectCoGraspEdge],
        placement_edges: Iterable[PlacementGraspEdge],
    ) -> None:
        self.initial_classes = _normalize_initial_classes(initial_classes)
        self._class_by_id = {item.class_id: item for item in self.initial_classes}
        insertion = tuple(insertion_feasible_receiver_grasps)
        for grasp in insertion:
            _validate_id(grasp, "insertion-feasible receiver grasp")
        if len(set(insertion)) != len(insertion):
            raise ValueError("insertion-feasible receiver grasps must be unique")
        self.insertion_feasible_receiver_grasps = tuple(
            sorted(insertion, key=_stable_id))
        self._insertion_set = set(insertion)

        raw_direct = tuple(direct_edges)
        raw_placement = tuple(placement_edges)
        if not all(isinstance(edge, DirectCoGraspEdge) for edge in raw_direct):
            raise TypeError("direct_edges must contain DirectCoGraspEdge values")
        if not all(isinstance(edge, PlacementGraspEdge) for edge in raw_placement):
            raise TypeError("placement_edges must contain PlacementGraspEdge values")
        self.direct_edges = tuple(sorted(raw_direct, key=lambda edge: (
            _stable_id(edge.giver_grasp), _stable_id(edge.receiver_grasp),
            edge.cost, -edge.robustness, edge.stable_id)))
        self.placement_edges = tuple(sorted(raw_placement, key=lambda edge: (
            _stable_id(edge.placement_id), _stable_id(edge.grasp_id),
            edge.cost, -edge.robustness, edge.stable_id)))

        self._valid_direct = tuple(
            edge for edge in self.direct_edges
            if edge.receiver_grasp in self._insertion_set
        )
        by_placement: dict[NodeId, list[PlacementGraspEdge]] = defaultdict(list)
        by_grasp: dict[NodeId, list[PlacementGraspEdge]] = defaultdict(list)
        for edge in self.placement_edges:
            by_placement[edge.placement_id].append(edge)
            by_grasp[edge.grasp_id].append(edge)
        self._by_placement = {
            key: tuple(sorted(values, key=lambda edge: (
                _stable_id(edge.grasp_id), edge.cost, -edge.robustness,
                edge.stable_id)))
            for key, values in by_placement.items()
        }
        self._by_grasp = {
            key: tuple(sorted(values, key=lambda edge: (
                _stable_id(edge.placement_id), edge.cost, -edge.robustness,
                edge.stable_id)))
            for key, values in by_grasp.items()
        }

    def _direct_plan(
        self,
        initial: InitialGraspClass,
    ) -> TaskPlan | None:
        candidates = [
            edge for edge in self._valid_direct
            if edge.giver_grasp in set(initial.grasp_ids)
        ]
        if not candidates:
            return None
        edge = min(candidates, key=lambda item: (
            item.cost, -item.robustness,
            _stable_id(item.giver_grasp), _stable_id(item.receiver_grasp),
            item.stable_id))
        step = TaskStep(
            kind="handoff", source=edge.giver_grasp,
            target=edge.receiver_grasp, cost=edge.cost,
            robustness=edge.robustness, edge_id=edge.edge_id)
        return TaskPlan(
            success=True,
            reason="direct_co_grasp_available",
            initial_class=initial.class_id,
            initial_grasp=edge.giver_grasp,
            receiver_grasp=edge.receiver_grasp,
            mode="direct",
            steps=(step,),
            total_cost=edge.cost,
            bottleneck_robustness=edge.robustness,
            reorientation_hops=0,
        )

    @staticmethod
    def _steps_for(
        transitions: Sequence[_Transition],
        direct: DirectCoGraspEdge,
    ) -> tuple[TaskStep, ...]:
        steps: list[TaskStep] = []
        for transition in transitions:
            steps.append(TaskStep(
                kind="place",
                source=transition.previous_grasp,
                target=transition.placement_id,
                cost=transition.place_edge.cost,
                robustness=transition.place_edge.robustness,
                edge_id=transition.place_edge.edge_id,
            ))
            steps.append(TaskStep(
                kind="regrasp",
                source=transition.placement_id,
                target=transition.next_grasp,
                cost=transition.pick_edge.cost,
                robustness=transition.pick_edge.robustness,
                edge_id=transition.pick_edge.edge_id,
            ))
        steps.append(TaskStep(
            kind="handoff",
            source=direct.giver_grasp,
            target=direct.receiver_grasp,
            cost=direct.cost,
            robustness=direct.robustness,
            edge_id=direct.edge_id,
        ))
        return tuple(steps)

    def _backward_reorientation_candidates(
        self,
        initial: InitialGraspClass,
        max_reorientation_hops: int,
    ) -> list[TaskPlan]:
        initial_set = set(initial.grasp_ids)
        found: list[TaskPlan] = []

        def visit(
            current_grasp: NodeId,
            direct: DirectCoGraspEdge,
            transitions: tuple[_Transition, ...],
            visited_grasps: frozenset[NodeId],
        ) -> None:
            hops = len(transitions)
            if current_grasp in initial_set and hops > 0:
                steps = self._steps_for(transitions, direct)
                found.append(TaskPlan(
                    success=True,
                    reason="reorientation_path_found",
                    initial_class=initial.class_id,
                    initial_grasp=current_grasp,
                    receiver_grasp=direct.receiver_grasp,
                    mode="reorientation",
                    steps=steps,
                    total_cost=sum(step.cost for step in steps),
                    bottleneck_robustness=min(step.robustness for step in steps),
                    reorientation_hops=hops,
                ))
                # Non-negative costs mean appending a cycle cannot improve this
                # same initial state, so do not expand it further.
                return
            if hops >= max_reorientation_hops:
                return

            # Backward expansion: current is the grasp re-picked from a
            # placement.  Any other feasible grasp at that same placement can
            # be the preceding grasp used to place the part.
            for pick_edge in self._by_grasp.get(current_grasp, ()):
                for place_edge in self._by_placement.get(pick_edge.placement_id, ()):
                    previous = place_edge.grasp_id
                    if previous == current_grasp or previous in visited_grasps:
                        continue
                    transition = _Transition(
                        previous_grasp=previous,
                        placement_id=pick_edge.placement_id,
                        next_grasp=current_grasp,
                        place_edge=place_edge,
                        pick_edge=pick_edge,
                    )
                    visit(
                        previous,
                        direct,
                        (transition,) + transitions,
                        visited_grasps | {previous},
                    )

        for direct in self._valid_direct:
            visit(
                direct.giver_grasp,
                direct,
                (),
                frozenset({direct.giver_grasp}),
            )
        return found

    @staticmethod
    def _plan_signature(plan: TaskPlan) -> tuple[str, ...]:
        return tuple(
            f"{step.kind}:{_stable_id(step.source)}->{_stable_id(step.target)}:"
            f"{_stable_id(step.edge_id) if step.edge_id is not None else ''}"
            for step in plan.steps
        )

    def plan(
        self,
        initial_class: NodeId,
        *,
        max_reorientation_hops: int = 2,
    ) -> TaskPlan:
        """Plan one admissible class with direct handoff as a hard preference."""
        if (not isinstance(max_reorientation_hops, (int, np.integer))
                or int(max_reorientation_hops) < 0):
            raise ValueError("max_reorientation_hops must be a non-negative integer")
        try:
            initial = self._class_by_id[initial_class]
        except (KeyError, TypeError):
            return TaskPlan(
                success=False,
                reason="unknown_initial_class",
                initial_class=initial_class,
            )
        if not self._insertion_set:
            return TaskPlan(
                success=False,
                reason="no_insertion_feasible_receiver_grasps",
                initial_class=initial.class_id,
            )
        direct = self._direct_plan(initial)
        if direct is not None:
            return direct
        if max_reorientation_hops == 0:
            return TaskPlan(
                success=False,
                reason="no_direct_path_and_reorientation_disabled",
                initial_class=initial.class_id,
            )
        candidates = self._backward_reorientation_candidates(
            initial, int(max_reorientation_hops))
        if not candidates:
            return TaskPlan(
                success=False,
                reason="no_path_within_reorientation_hop_bound",
                initial_class=initial.class_id,
            )
        return min(candidates, key=lambda item: (
            float(item.total_cost),
            item.reorientation_hops,
            -float(item.bottleneck_robustness),
            self._plan_signature(item),
            _stable_id(item.initial_grasp),
            _stable_id(item.receiver_grasp),
        ))

    def coverage_report(
        self,
        *,
        max_reorientation_hops: int = 2,
        target_fraction: float = 1.0,
    ) -> CoverageReport:
        """Plan every admissible class and issue an exact coverage certificate."""
        target = float(target_fraction)
        if not np.isfinite(target) or not 0.0 <= target <= 1.0:
            raise ValueError("target_fraction must lie in [0, 1]")
        plans = tuple(self.plan(
            initial.class_id,
            max_reorientation_hops=max_reorientation_hops,
        ) for initial in self.initial_classes)
        direct = tuple(plan.initial_class for plan in plans if plan.mode == "direct")
        reorientation = tuple(
            plan.initial_class for plan in plans if plan.mode == "reorientation")
        uncovered = tuple(plan.initial_class for plan in plans if not plan.success)
        covered = direct + reorientation
        fraction = 1.0 if not plans else len(covered) / len(plans)
        return CoverageReport(
            plans=plans,
            direct_classes=direct,
            reorientation_classes=reorientation,
            uncovered_classes=uncovered,
            covered_classes=covered,
            fraction=fraction,
            target_fraction=target,
            certified=(fraction >= target),
        )


# Descriptive aliases keep integration code readable without duplicating
# implementation or changing serialization.
BackwardReorientationGraph = TaskGraph
DirectEdge = DirectCoGraspEdge
PlacementEdge = PlacementGraspEdge


__all__ = [
    "BackwardReorientationGraph",
    "CoverageReport",
    "DirectCoGraspEdge",
    "DirectEdge",
    "InitialGraspClass",
    "PlacementEdge",
    "PlacementGraspEdge",
    "TaskGraph",
    "TaskPlan",
    "TaskStep",
]
