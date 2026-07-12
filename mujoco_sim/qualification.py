"""Bounded-domain coverage qualification for production handoff policies."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping


@dataclass(frozen=True)
class CoverageOutcome:
    class_id: str
    mode: str | None
    reason: str

    @property
    def covered(self) -> bool:
        return self.mode in ("direct", "reorientation")


def _calibrated_contact(item: Mapping[str, object]) -> bool:
    profile = next((item.get(name) for name in (
        "contact_material", "contact_parameters", "contact")
        if isinstance(item.get(name), Mapping)), None)
    return bool(profile and profile.get("calibrated") is True
                and profile.get("friction") is not None)


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
        "part_pin_collision_cad": bool(
            part.get("pin_collision_cad") or part.get("collision_cad")),
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
    prerequisites = dict(physical_prerequisites or {})
    mathematical = fraction >= required
    physical = mathematical and all(prerequisites.values())
    return {
        "domain_classes": [item.class_id for item in values],
        "direct_classes": direct,
        "reorientation_classes": reorientation,
        "uncovered_classes": uncovered,
        "covered_count": len(covered),
        "required_count": len(values),
        "fraction": fraction,
        "required_fraction": required,
        "mathematical_coverage_certified": mathematical,
        "domain_declaration": dict(domain_declaration or {}),
        "physical_prerequisites": prerequisites,
        "physical_certified": physical,
        "outcomes": {
            item.class_id: {"mode": item.mode, "reason": item.reason}
            for item in values
        },
    }


__all__ = [
    "CoverageOutcome",
    "build_coverage_certificate",
    "physical_prerequisites",
]
