"""Bounded-domain coverage qualification for production handoff policies."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping


PHYSICAL_PREREQUISITE_KEYS = (
    "articulated_gripper_A",
    "articulated_gripper_B",
    "calibrated_gripper_contacts",
    "calibrated_part_contact",
    "pcb_hole_collision_cad",
    "complete_part_collision_cad",
    "calibrated_pin_hole_materials",
    "articulated_gripper_scene_adapter",
    "physical_contact_execution_backend",
)


@dataclass(frozen=True)
class CoverageOutcome:
    class_id: str
    mode: str | None
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.class_id, str) or not self.class_id.strip():
            raise ValueError("coverage class_id must be a non-empty string")
        if self.mode not in (None, "direct", "reorientation"):
            raise ValueError(
                "coverage mode must be direct, reorientation, or None")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("coverage reason must be a non-empty string")

    @property
    def covered(self) -> bool:
        return self.mode in ("direct", "reorientation")


def _calibrated_contact(item: Mapping[str, object]) -> bool:
    profile = next((item.get(name) for name in (
        "contact_material", "contact_parameters", "contact")
        if isinstance(item.get(name), Mapping)), None)
    if not profile or profile.get("calibrated") is not True:
        return False
    friction = profile.get("friction")
    values = friction if isinstance(friction, (list, tuple)) else (friction,)
    try:
        numbers = tuple(float(value) for value in values)
    except (TypeError, ValueError):
        return False
    return bool(numbers and all(math.isfinite(value) and value >= 0.0
                                for value in numbers)
                and numbers[0] > 0.0)


def physical_prerequisites(project) -> dict[str, bool]:
    """Return explicit asset and implementation prerequisites for hardware truth.

    The last two gates intentionally remain false in the current version. They
    prevent a manifest-only change from relabeling ideal-weld/convex-hull replay
    as physical contact certification before the scene and executor adapters
    actually support those assets.
    """
    manifest = project.manifest
    part = project.active_part
    insertion = manifest["insertion"]
    material = insertion.get("contact_materials", {})
    pin_material = material.get("pin", {}) if isinstance(material, Mapping) else {}
    hole_material = material.get("hole", {}) if isinstance(material, Mapping) else {}
    gripper_materials = []
    for robot in ("A", "B"):
        name = manifest["robots"][robot]["gripper"]
        gripper_materials.append(_calibrated_contact(manifest["grippers"][name]))
    return {
        "articulated_gripper_A": bool(project.gripper("A").articulated),
        "articulated_gripper_B": bool(project.gripper("B").articulated),
        "calibrated_gripper_contacts": all(gripper_materials),
        "calibrated_part_contact": _calibrated_contact(part),
        "pcb_hole_collision_cad": bool(insertion.get("collision_cad")),
        # The scene compiler consumes a complete, convex-decomposed part
        # collision model. A pin-only file cannot replace body/palm clearance.
        "complete_part_collision_cad": bool(part.get("collision_cad")),
        "calibrated_pin_hole_materials": bool(
            _calibrated_contact({"contact_material": pin_material})
            and _calibrated_contact({"contact_material": hole_material})),
        "articulated_gripper_scene_adapter": False,
        "physical_contact_execution_backend": False,
    }


def build_coverage_certificate(
    outcomes: Iterable[CoverageOutcome],
    *,
    required_fraction: float = 1.0,
    physical_prerequisites: Mapping[str, bool] | None = None,
    domain_declaration: Mapping[str, object] | None = None,
) -> dict:
    """Issue an exact certificate for an explicitly enumerated start domain."""
    values = tuple(sorted(outcomes, key=lambda item: item.class_id))
    if not values:
        raise ValueError("coverage domain cannot be empty")
    if len({item.class_id for item in values}) != len(values):
        raise ValueError("coverage class IDs must be unique")
    required = float(required_fraction)
    if not 0.0 <= required <= 1.0:
        raise ValueError("required_fraction must lie in [0, 1]")
    direct = [item.class_id for item in values if item.mode == "direct"]
    reorientation = [item.class_id for item in values
                     if item.mode == "reorientation"]
    uncovered = [item.class_id for item in values if not item.covered]
    covered = direct + reorientation
    fraction = len(covered) / len(values)
    supplied = dict(physical_prerequisites or {})
    unknown = sorted(set(supplied) - set(PHYSICAL_PREREQUISITE_KEYS))
    if unknown:
        raise ValueError(f"unknown physical prerequisite keys: {unknown}")
    for name, value in supplied.items():
        if type(value) is not bool:
            raise ValueError(
                f"physical prerequisite {name!r} must be a strict boolean")
    prerequisites = {
        name: supplied.get(name, False)
        for name in PHYSICAL_PREREQUISITE_KEYS
    }
    prerequisite_schema_complete = set(supplied) == set(
        PHYSICAL_PREREQUISITE_KEYS)
    mathematical = fraction >= required
    # Physical certification always means the complete declared domain. A
    # lower mathematical acceptance target must never weaken that label.
    physical = bool(
        fraction == 1.0
        and prerequisite_schema_complete
        and all(prerequisites.values()))
    return {
        "certificate_schema_version": 2,
        "domain_classes": [item.class_id for item in values],
        "direct_classes": direct,
        "reorientation_classes": reorientation,
        "uncovered_classes": uncovered,
        "covered_count": len(covered),
        "domain_count": len(values),
        "required_count": int(math.ceil(required * len(values) - 1e-15)),
        "fraction": fraction,
        "required_fraction": required,
        "mathematical_coverage_certified": mathematical,
        "domain_declaration": dict(domain_declaration or {}),
        "physical_prerequisites": prerequisites,
        "physical_prerequisite_schema_complete": prerequisite_schema_complete,
        "physical_certified": physical,
        "outcomes": {
            item.class_id: {"mode": item.mode, "reason": item.reason}
            for item in values
        },
    }


__all__ = [
    "CoverageOutcome",
    "PHYSICAL_PREREQUISITE_KEYS",
    "build_coverage_certificate",
    "physical_prerequisites",
]
