"""Stable cache/JSON codec for grasp and handoff plan records."""
from __future__ import annotations

import numpy as np

from ..modeling.grasps import GraspCandidate
from .types import (DirectHandoffPlan, DownstreamWitness, RegraspPlan,
                    ScoreBreakdown)


def _array(value) -> np.ndarray:
    return np.array(value, dtype=float, copy=True)


def serialize_grasp(candidate: GraspCandidate) -> dict:
    return {
        "T_P_E": candidate.T_P_E,
        "contact_points": candidate.contact_points,
        "contact_normals": candidate.contact_normals,
        "required_opening": candidate.required_opening,
        "approach_direction": candidate.approach_direction,
        "closing_direction": candidate.closing_direction,
        "quality": candidate.quality,
        "antipodal_quality": candidate.antipodal_quality,
        "support_quality": candidate.support_quality,
        "opening_margin": candidate.opening_margin,
        "palm_clearance": candidate.palm_clearance,
    }


def deserialize_grasp(value: dict) -> GraspCandidate:
    arrays = {
        "T_P_E", "contact_points", "contact_normals",
        "approach_direction", "closing_direction",
    }
    return GraspCandidate(**{
        key: _array(item) if key in arrays else item
        for key, item in value.items()
    })


def serialize_downstream(witness: DownstreamWitness) -> dict:
    return {
        "grasp_name": witness.grasp_name,
        "grasp": witness.grasp,
        "q_scanner": witness.q_scanner,
        "q_preinsert": witness.q_preinsert,
        "q_insert": witness.q_insert,
        "correction_solutions": witness.correction_solutions,
        "trajectories": witness.trajectories,
        "quality": witness.quality,
        "sigma_min": witness.sigma_min,
    }


def deserialize_downstream(value: dict) -> DownstreamWitness:
    return DownstreamWitness(
        value["grasp_name"], _array(value["grasp"]),
        _array(value["q_scanner"]),
        [_array(q) for q in value["q_preinsert"]],
        [_array(q) for q in value["q_insert"]],
        [[_array(q) for q in group]
         for group in value["correction_solutions"]],
        {name: [_array(q) for q in path]
         for name, path in value["trajectories"].items()},
        float(value["quality"]), float(value["sigma_min"]),
    )


def serialize_direct(plan: DirectHandoffPlan | None):
    if plan is None:
        return None
    return {
        "X_handoff": plan.X_handoff,
        "g_A": plan.g_A,
        "grasp_name_B": plan.grasp_name_B,
        "g_B": plan.g_B,
        "qA_handoff": plan.qA_handoff,
        "qB_handoff": plan.qB_handoff,
        "qA_pre": plan.qA_pre,
        "qB_pre": plan.qB_pre,
        "qA_retreat": plan.qA_retreat,
        "downstream": serialize_downstream(plan.downstream),
        "trajectories": plan.trajectories,
        "score": vars(plan.score),
    }


def deserialize_direct(value) -> DirectHandoffPlan | None:
    if value is None:
        return None
    arrays = {name: _array(value[name]) for name in (
        "X_handoff", "g_A", "g_B", "qA_handoff", "qB_handoff",
        "qA_pre", "qB_pre", "qA_retreat")}
    score = ScoreBreakdown(**{
        name: float(number) for name, number in value["score"].items()})
    return DirectHandoffPlan(
        arrays["X_handoff"], arrays["g_A"], value["grasp_name_B"],
        arrays["g_B"], arrays["qA_handoff"], arrays["qB_handoff"],
        arrays["qA_pre"], arrays["qB_pre"], arrays["qA_retreat"],
        deserialize_downstream(value["downstream"]),
        {name: [_array(q) for q in path]
         for name, path in value["trajectories"].items()}, score)


def serialize_regrasp(plan: RegraspPlan | None):
    if plan is None:
        return None
    return {
        "placement_name": plan.placement_name,
        "X_place": plan.X_place,
        "g_A_before": plan.g_A_before,
        "g_A_after": plan.g_A_after,
        "qA_place": plan.qA_place,
        "qA_repick": plan.qA_repick,
        "direct": serialize_direct(plan.direct),
        "trajectories": plan.trajectories,
    }


def deserialize_regrasp(value) -> RegraspPlan | None:
    if value is None:
        return None
    return RegraspPlan(
        value["placement_name"], _array(value["X_place"]),
        _array(value["g_A_before"]), _array(value["g_A_after"]),
        _array(value["qA_place"]), _array(value["qA_repick"]),
        deserialize_direct(value["direct"]),
        {name: [_array(q) for q in path]
         for name, path in value["trajectories"].items()},
    )


__all__ = [
    "deserialize_direct", "deserialize_downstream", "deserialize_grasp",
    "deserialize_regrasp", "serialize_direct", "serialize_downstream",
    "serialize_grasp", "serialize_regrasp",
]
