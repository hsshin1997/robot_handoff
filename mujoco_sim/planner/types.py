"""Validated data contracts exchanged by handoff-planning stages.

Keeping these records separate from the algorithms makes each stage callable
and testable without importing the 3-D scene, cache implementation, or CLI.
``mujoco_sim.planning`` remains an exact compatibility alias for this planner
implementation's public record types.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..core.se3 import validate_transform


def normalized_placement_robustness(
    support_margin: float,
    edge_clearance: float,
    part_scale: float,
    stage_scale: float,
) -> tuple[float, float, float]:
    """Return scale-invariant support, stage, and bottleneck robustness."""
    values = {
        "support_margin": float(support_margin),
        "edge_clearance": float(edge_clearance),
        "part_scale": float(part_scale),
        "stage_scale": float(stage_scale),
    }
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError("placement robustness inputs must be finite")
    if values["support_margin"] < 0.0 or values["edge_clearance"] < 0.0:
        raise ValueError("placement clearances must be non-negative")
    if values["part_scale"] <= 0.0 or values["stage_scale"] <= 0.0:
        raise ValueError("placement characteristic scales must be positive")
    support = float(np.clip(
        2.0 * values["support_margin"] / values["part_scale"], 0.0, 1.0))
    stage = float(np.clip(
        2.0 * values["edge_clearance"] / values["stage_scale"], 0.0, 1.0))
    return support, stage, min(support, stage)


@dataclass
class DownstreamWitness:
    grasp_name: str
    grasp: np.ndarray
    q_scanner: np.ndarray
    q_preinsert: list[np.ndarray]
    q_insert: list[np.ndarray]
    correction_solutions: list[list[np.ndarray]]
    trajectories: dict[str, list[np.ndarray]]
    quality: float
    sigma_min: float


@dataclass
class ScoreBreakdown:
    manipulability: float
    joint_margin: float
    clearance: float
    reorientation: float
    cycle: float
    total: float


@dataclass
class DirectHandoffPlan:
    X_handoff: np.ndarray
    g_A: np.ndarray
    grasp_name_B: str
    g_B: np.ndarray
    qA_handoff: np.ndarray
    qB_handoff: np.ndarray
    qA_pre: np.ndarray
    qB_pre: np.ndarray
    qA_retreat: np.ndarray
    downstream: DownstreamWitness
    trajectories: dict[str, list[np.ndarray]]
    score: ScoreBreakdown


@dataclass
class RegraspPlan:
    placement_name: str
    X_place: np.ndarray
    g_A_before: np.ndarray
    g_A_after: np.ndarray
    qA_place: np.ndarray
    qA_repick: np.ndarray
    direct: DirectHandoffPlan
    trajectories: dict[str, list[np.ndarray]]


@dataclass(frozen=True)
class StablePlacementWitness:
    """Cached geometric certificate used by the reorientation task graph."""

    name: str
    T_W_P: np.ndarray
    support_margin: float
    support_area: float
    edge_clearance: float
    probability_proxy: float
    minimum_support_margin: float
    part_scale: float
    stage_scale: float
    support_robustness: float
    stage_robustness: float
    robustness: float

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("stable placement name must be non-empty")
        transform = validate_transform(self.T_W_P)
        transform.setflags(write=False)
        object.__setattr__(self, "T_W_P", transform)

        nonnegative = (
            "support_margin", "support_area", "edge_clearance",
            "probability_proxy", "minimum_support_margin",
        )
        positive = ("part_scale", "stage_scale")
        unit_interval = (
            "support_robustness", "stage_robustness", "robustness",
        )
        for field_name in nonnegative + positive + unit_interval:
            value = float(getattr(self, field_name))
            if not np.isfinite(value):
                raise ValueError(f"{field_name} must be finite")
            if field_name in positive and value <= 0.0:
                raise ValueError(f"{field_name} must be positive")
            if field_name not in positive and value < 0.0:
                raise ValueError(f"{field_name} must be non-negative")
            if field_name in unit_interval and value > 1.0:
                raise ValueError(f"{field_name} must not exceed one")
            object.__setattr__(self, field_name, value)
        if self.probability_proxy > 1.0 + 1e-12:
            raise ValueError("probability_proxy must not exceed one")
        expected = normalized_placement_robustness(
            self.support_margin, self.edge_clearance,
            self.part_scale, self.stage_scale)
        actual = (self.support_robustness, self.stage_robustness,
                  self.robustness)
        if not np.allclose(actual, expected, atol=1e-12, rtol=0.0):
            raise ValueError("cached placement robustness is inconsistent")


@dataclass
class PlanningReport:
    direct: DirectHandoffPlan | None = None
    regrasp: RegraspPlan | None = None
    stats: Counter = field(default_factory=Counter)
    candidates: int = 0
    downstream_grasps: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0
    initialization_timings: tuple[dict[str, Any], ...] = ()
    stage_timings: tuple[dict[str, Any], ...] = ()
    bottlenecks: tuple[dict[str, Any], ...] = ()
    mathematical_coverage_certified: bool = False
    physical_certified: bool = False
    # Deprecated compatibility alias for physical_certified.
    certified: bool = False
    limitations: tuple[str, ...] = ()
    coverage: dict = field(default_factory=dict)
    mathematical_corrections: tuple[str, ...] = (
        "G1 queries induced TCP poses, not the part origin",
        "part symmetries left-multiply ^P T_E grasps",
        "reorientation compares part poses rather than TCP and part frames",
    )

    @property
    def feasible(self) -> bool:
        return self.direct is not None or self.regrasp is not None


# Compatibility name retained while callers migrate to the non-private form.
_normalized_placement_robustness = normalized_placement_robustness


__all__ = [
    "DirectHandoffPlan",
    "DownstreamWitness",
    "PlanningReport",
    "RegraspPlan",
    "ScoreBreakdown",
    "StablePlacementWitness",
    "normalized_placement_robustness",
    "_normalized_placement_robustness",
]
