"""Structural invariants for cached and newly generated handoff plans."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from .types import DirectHandoffPlan, RegraspPlan
from ..core.se3 import validate_transform


class PlanValidationError(ValueError):
    pass


def _q(value, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (6,) or not np.all(np.isfinite(array)):
        raise PlanValidationError(f"{name} must be a finite six-joint vector")
    return array


def _path(value, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] != 6:
        raise PlanValidationError(f"trajectory {name!r} must have shape (N, 6)")
    if not np.all(np.isfinite(array)):
        raise PlanValidationError(f"trajectory {name!r} contains non-finite values")
    return array


def _endpoint(path: np.ndarray, expected, name: str, endpoint: str) -> None:
    target = _q(expected, f"{name} {endpoint} witness")
    value = path[0] if endpoint == "start" else path[-1]
    if not np.allclose(value, target, atol=1e-8, rtol=0.0):
        raise PlanValidationError(
            f"trajectory {name!r} {endpoint} does not match its witness")


def validate_direct_plan(
    plan: DirectHandoffPlan,
    *,
    q_start: Mapping[str, Sequence[float]] | None = None,
) -> DirectHandoffPlan:
    """Validate direct-plan transforms, witnesses, paths, and endpoints."""
    if not isinstance(plan.grasp_name_B, str) or not plan.grasp_name_B:
        raise PlanValidationError("receiver grasp name must be non-empty")
    for name, value in (("X_handoff", plan.X_handoff),
                        ("g_A", plan.g_A), ("g_B", plan.g_B),
                        ("downstream.grasp", plan.downstream.grasp)):
        try:
            validate_transform(value)
        except ValueError as error:
            raise PlanValidationError(f"invalid {name}: {error}") from error
    q_values = {
        "qA_handoff": _q(plan.qA_handoff, "qA_handoff"),
        "qB_handoff": _q(plan.qB_handoff, "qB_handoff"),
        "qA_pre": _q(plan.qA_pre, "qA_pre"),
        "qB_pre": _q(plan.qB_pre, "qB_pre"),
        "qA_retreat": _q(plan.qA_retreat, "qA_retreat"),
        "qB_scanner": _q(plan.downstream.q_scanner, "qB_scanner"),
    }
    required = {
        "A_current_to_pre", "A_approach", "B_current_to_pre",
        "B_approach", "A_retreat", "B_to_scanner",
        "A_scanner_clear_to_park",
    }
    missing = sorted(required - set(plan.trajectories))
    if missing:
        raise PlanValidationError(f"direct plan is missing trajectories: {missing}")
    paths = {name: _path(value, name)
             for name, value in plan.trajectories.items()}
    _endpoint(paths["A_current_to_pre"], q_values["qA_pre"],
              "A_current_to_pre", "end")
    _endpoint(paths["A_approach"], q_values["qA_pre"],
              "A_approach", "start")
    _endpoint(paths["A_approach"], q_values["qA_handoff"],
              "A_approach", "end")
    _endpoint(paths["B_current_to_pre"], q_values["qB_pre"],
              "B_current_to_pre", "end")
    _endpoint(paths["B_approach"], q_values["qB_pre"],
              "B_approach", "start")
    _endpoint(paths["B_approach"], q_values["qB_handoff"],
              "B_approach", "end")
    _endpoint(paths["A_retreat"], q_values["qA_handoff"],
              "A_retreat", "start")
    _endpoint(paths["A_retreat"], q_values["qA_retreat"],
              "A_retreat", "end")
    _endpoint(paths["B_to_scanner"], q_values["qB_handoff"],
              "B_to_scanner", "start")
    _endpoint(paths["B_to_scanner"], q_values["qB_scanner"],
              "B_to_scanner", "end")
    _endpoint(paths["A_scanner_clear_to_park"], q_values["qA_retreat"],
              "A_scanner_clear_to_park", "start")
    if q_start is not None:
        _endpoint(paths["A_current_to_pre"], q_start["A"],
                  "A_current_to_pre", "start")
        _endpoint(paths["B_current_to_pre"], q_start["B"],
                  "B_current_to_pre", "start")
        _endpoint(paths["A_scanner_clear_to_park"], q_start["A"],
                  "A_scanner_clear_to_park", "end")

    downstream = plan.downstream
    count = len(downstream.q_insert)
    if not (len(downstream.q_preinsert) == count
            == len(downstream.correction_solutions)):
        raise PlanValidationError("downstream witness target arrays have unequal length")
    if count < 1:
        raise PlanValidationError("downstream witness has no insertion target")
    for index, q_value in enumerate(downstream.q_preinsert):
        _q(q_value, f"q_preinsert[{index}]")
    for index, q_value in enumerate(downstream.q_insert):
        _q(q_value, f"q_insert[{index}]")
    for name, path in downstream.trajectories.items():
        _path(path, f"downstream.{name}")
    scores = vars(plan.score)
    if not scores or not all(np.isfinite(float(value))
                             for value in scores.values()):
        raise PlanValidationError("plan score must contain finite values")
    return plan


def validate_regrasp_plan(
    plan: RegraspPlan,
    *,
    q_start: Mapping[str, Sequence[float]] | None = None,
) -> RegraspPlan:
    """Validate stage place/re-pick paths and the embedded direct plan."""
    if not isinstance(plan.placement_name, str) or not plan.placement_name:
        raise PlanValidationError("placement name must be non-empty")
    for name, value in (("X_place", plan.X_place),
                        ("g_A_before", plan.g_A_before),
                        ("g_A_after", plan.g_A_after)):
        try:
            validate_transform(value)
        except ValueError as error:
            raise PlanValidationError(f"invalid {name}: {error}") from error
    q_place = _q(plan.qA_place, "qA_place")
    q_repick = _q(plan.qA_repick, "qA_repick")
    required = {"A_to_place", "A_place_to_repick"}
    missing = sorted(required - set(plan.trajectories))
    if missing:
        raise PlanValidationError(
            f"reorientation plan is missing trajectories: {missing}")
    place = _path(plan.trajectories["A_to_place"], "A_to_place")
    repick = _path(plan.trajectories["A_place_to_repick"],
                   "A_place_to_repick")
    _endpoint(place, q_place, "A_to_place", "end")
    _endpoint(repick, q_place, "A_place_to_repick", "start")
    _endpoint(repick, q_repick, "A_place_to_repick", "end")
    if q_start is not None:
        _endpoint(place, q_start["A"], "A_to_place", "start")
    validate_direct_plan(plan.direct, q_start=None)
    _endpoint(_path(plan.direct.trajectories["A_current_to_pre"],
                    "A_current_to_pre"), q_repick,
              "A_current_to_pre", "start")
    return plan


__all__ = [
    "PlanValidationError", "validate_direct_plan", "validate_regrasp_plan",
]
