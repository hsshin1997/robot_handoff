"""Shared guards and current-plan loading for physical MuJoCo experiments."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
import sys
from typing import Any

from mujoco_sim.project import DEFAULT_PROJECT, Project


@dataclass(frozen=True)
class PreflightReport:
    experiment: str
    project_path: str
    missing: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return not self.missing


class ExperimentBlocked(RuntimeError):
    """A physical experiment cannot produce defensible results."""


def parse_positive_csv(value: str, *, label: str, allow_zero: bool = True) -> tuple[float, ...]:
    try:
        numbers = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise ValueError(f"{label} must be a comma-separated numeric list") from error
    if not numbers:
        raise ValueError(f"{label} must contain at least one value")
    lower_ok = (lambda item: item >= 0.0) if allow_zero else (lambda item: item > 0.0)
    if not all(lower_ok(item) and item < float("inf") for item in numbers):
        bound = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{label} values must be finite and {bound}")
    return numbers


def _contact_profile(item: dict[str, Any]) -> dict[str, Any] | None:
    for key in ("contact_material", "contact_parameters", "contact"):
        value = item.get(key)
        if isinstance(value, dict):
            return value
    return None


def _calibrated_contact(item: dict[str, Any]) -> bool:
    profile = _contact_profile(item)
    return bool(profile and profile.get("friction") is not None
                and profile.get("calibrated", False) is True)


def gripper_contact_prerequisites(project: Project, robots: tuple[str, ...]) -> list[str]:
    missing = []
    checked_materials: set[str] = set()
    for robot in robots:
        capability = project.gripper(robot)
        if not capability.articulated:
            missing.append(
                f"robot {robot} gripper {capability.name!r} is a static mesh; provide an "
                "articulated MJCF/URDF with finger joints, actuator limits, and collision bodies"
            )
        gripper_name = project.manifest["robots"][robot]["gripper"]
        if gripper_name not in checked_materials:
            gripper = project.manifest["grippers"][gripper_name]
            if not _calibrated_contact(gripper):
                missing.append(
                    f"gripper {gripper_name!r} lacks measured contact_material with "
                    "calibrated: true and friction"
                )
            checked_materials.add(gripper_name)
    part_name = project.manifest["active_task"]["part"]
    if not _calibrated_contact(project.active_part):
        missing.append(
            f"active part {part_name!r} lacks measured contact_material with "
            "calibrated: true and friction"
        )
    return missing


def transport_preflight(project: Project, project_path: str = DEFAULT_PROJECT) -> PreflightReport:
    holder = str(project.manifest["active_task"]["initial_holder"]).upper()
    missing = gripper_contact_prerequisites(project, (holder,))
    return PreflightReport("friction transport speed", project_path, tuple(missing))


def cograsp_preflight(project: Project, project_path: str = DEFAULT_PROJECT) -> PreflightReport:
    missing = gripper_contact_prerequisites(project, ("A", "B"))
    return PreflightReport("contact co-grasp tolerance", project_path, tuple(missing))


def insertion_preflight(project: Project, project_path: str = DEFAULT_PROJECT) -> PreflightReport:
    # Holding stiffness and jaw/part friction influence jamming and self-
    # alignment, so an ideal weld is not acceptable even if pin/hole CAD exists.
    missing = gripper_contact_prerequisites(project, ("B",))
    insertion = project.manifest["insertion"]
    part_name = project.manifest["active_task"]["part"]
    part = project.active_part
    if not insertion.get("collision_cad"):
        missing.append(
            "insertion.collision_cad is absent; provide explicit PCB-hole/chamfer collision CAD"
        )
    if not (part.get("pin_collision_cad") or part.get("collision_cad")):
        missing.append(
            f"active part {part_name!r} has no pin_collision_cad/collision_cad; the visual "
            "mesh convex hull cannot represent individual insertion pins"
        )
    materials = insertion.get("contact_materials")
    if not isinstance(materials, dict):
        missing.append(
            "insertion.contact_materials is absent; provide calibrated pin and hole materials"
        )
    else:
        for name in ("pin", "hole"):
            material = materials.get(name)
            if (not isinstance(material, dict)
                    or material.get("calibrated") is not True
                    or material.get("friction") is None):
                missing.append(
                    f"insertion.contact_materials.{name} must contain calibrated: true and friction"
                )
    return PreflightReport("insertion funnel", project_path, tuple(missing))


def print_blocked(report: PreflightReport, *, stream=None) -> int:
    stream = stream or sys.stderr
    print(f"BLOCKED: {report.experiment} experiment cannot produce physical results.", file=stream)
    print(f"Project: {report.project_path}", file=stream)
    print("Missing physical prerequisites:", file=stream)
    for item in report.missing:
        print(f"  - {item}", file=stream)
    print("No simulation trials were run and no success/slip measurements were reported.", file=stream)
    return 2


def extract_direct_plan_payload(document: Any) -> dict[str, Any]:
    """Extract the current DirectHandoffPlan JSON shape from supported envelopes."""
    if not isinstance(document, dict):
        raise ValueError("plan JSON root must be an object")
    value: Any = document
    if isinstance(value.get("planning"), dict):
        value = value["planning"]
    if isinstance(value, dict) and "direct" in value:
        if isinstance(value["direct"], dict):
            value = value["direct"]
        elif isinstance(value.get("regrasp"), dict):
            value = value["regrasp"].get("direct")
    if isinstance(value, dict) and "plan" in value and value.get("X_handoff") is None:
        value = value["plan"]
    if not isinstance(value, dict) or "X_handoff" not in value or "trajectories" not in value:
        raise ValueError(
            "plan JSON does not contain a current direct plan; pass JSON from "
            "`python -m mujoco_sim.pipeline --json` or omit --plan to derive one"
        )
    return value


def load_or_derive_direct_plan(
    project_path: str,
    plan_path: str | None,
):
    """Return ``(sim, planner, DirectHandoffPlan)`` using current package types.

    Imports are delayed so ``--help`` and blocked preflights do not construct a
    MuJoCo model or trigger an expensive planning search.
    """
    from mujoco_sim.planning import HandoffPlanner
    from mujoco_sim.sim import WorkcellSim

    sim = WorkcellSim(project_path=project_path)
    planner = HandoffPlanner(sim, project_path=project_path)
    if plan_path is not None:
        with open(plan_path, encoding="utf-8") as stream:
            payload = extract_direct_plan_payload(json.load(stream))
        plan = planner._deserialize_direct(payload)
    else:
        report = planner.plan(allow_regrasp=True, return_best=False)
        plan = report.direct if report.direct is not None else (
            report.regrasp.direct if report.regrasp is not None else None
        )
    if plan is None:
        raise ExperimentBlocked("the current project has no feasible direct/regrasp handoff plan")
    return sim, planner, plan


def future_backend_block(experiment: str) -> ExperimentBlocked:
    """Guard the transition after physical assets first become available.

    The existing PipelineExecutor intentionally clamps the part to an ideal
    weld.  Contact experiments must not call it and relabel those kinematic
    results as friction/capture/insertion measurements.
    """
    return ExperimentBlocked(
        f"{experiment} physical prerequisites passed, but the current executor still uses "
        "ideal-weld ownership and virtual capture/insertion predicates. Add the articulated "
        "gripper/contact controller experiment adapter before collecting results."
    )


def load_project(project_path: str) -> Project:
    return Project(os.path.abspath(project_path))


__all__ = [
    "ExperimentBlocked",
    "PreflightReport",
    "cograsp_preflight",
    "extract_direct_plan_payload",
    "future_backend_block",
    "insertion_preflight",
    "load_or_derive_direct_plan",
    "load_project",
    "parse_positive_csv",
    "print_blocked",
    "transport_preflight",
]
