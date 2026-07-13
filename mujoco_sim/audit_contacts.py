"""Audit contacts along the planner-selected support and insertion paths.

This command is intentionally separate from the interactive viewer.  It uses
the same coupled MuJoCo collision state as the planner and reports signed
contact distances (negative means penetration), the phase policy decision,
and the contact point.  It is therefore useful when a transparent collision
mesh and an opaque visual mesh are difficult to distinguish in the viewer.

Examples::

    python -m mujoco_sim.audit_contacts
    python -m mujoco_sim.audit_contacts --reorientation-demo --json

The present demonstration PCB is a solid primitive, not a pin-and-hole CAD
model.  The report labels that condition explicitly; an allowed part/board
contact in placeholder mode is not a physical insertion certificate.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from typing import Callable, Iterable

import numpy as np

from .phase_contacts import REORIENTATION_CONTACTS, insertion_contacts
from .planning import HandoffPlanner
from .project import DEFAULT_PROJECT
from .se3 import inverse
from .sim import MODEL, WorkcellSim


@dataclass(frozen=True)
class ContactSample:
    phase: str
    waypoint: int
    pair: tuple[str, str]
    signed_distance_m: float
    penetration_m: float
    allowed_by_phase_policy: bool
    position_m: tuple[float, float, float]


def _geom_name(sim: WorkcellSim, geom_id: int) -> str:
    return str(sim.model.geom(int(geom_id)).name)


def _contacts(
    sim: WorkcellSim,
    planner: HandoffPlanner,
    *,
    phase: str,
    waypoint: int,
    relevant: Callable[[str, str], bool],
    holders: tuple[str, ...],
    allowed_pairs: tuple[tuple, ...],
) -> list[ContactSample]:
    output: list[ContactSample] = []
    for contact in sim.data.contact[:sim.data.ncon]:
        name1 = _geom_name(sim, contact.geom1)
        name2 = _geom_name(sim, contact.geom2)
        if not relevant(name1, name2):
            continue
        distance = float(contact.dist)
        output.append(ContactSample(
            phase=phase,
            waypoint=int(waypoint),
            pair=(name1, name2),
            signed_distance_m=distance,
            penetration_m=max(0.0, -distance),
            allowed_by_phase_policy=planner.collision._allowed(
                contact,
                allowed_part_holders=holders,
                allowed_geom_pairs=allowed_pairs,
            ),
            position_m=tuple(float(value) for value in contact.pos),
        ))
    return output


def _summarize(samples: Iterable[ContactSample]) -> dict:
    values = list(samples)
    penetrations = [sample.penetration_m for sample in values]
    forbidden = [sample for sample in values
                 if not sample.allowed_by_phase_policy]
    return {
        "contact_samples": len(values),
        "minimum_signed_distance_m": (
            None if not values else min(item.signed_distance_m for item in values)
        ),
        "maximum_penetration_m": max(penetrations, default=0.0),
        "forbidden_contact_samples": len(forbidden),
        "contacts": [asdict(item) for item in values],
    }


def audit_insertion(sim: WorkcellSim, planner: HandoffPlanner, direct) -> dict:
    """Audit all selected insertion descents from scanner-side witnesses."""
    allowed = insertion_contacts(planner.project)
    # Downstream paths are planned and executed only after A reaches its
    # declared insertion-park state. Auditing with the short handoff retreat
    # witness checked a different simultaneous robot configuration.
    qA = np.asarray(planner.q_start["A"], dtype=float)
    samples: list[ContactSample] = []
    targets = []
    for target_name, X_insert in planner.insertion_poses:
        key = f"{target_name}_insert"
        path = direct.downstream.trajectories[key]
        target_samples = []
        for index, qB in enumerate(path):
            X_part = planner.kin.fk("B", qB) @ inverse(direct.g_B)
            gate = planner.collision.check(
                qA, qB, X_part,
                allowed_part_holders=("B",),
                allowed_geom_pairs=allowed,
            )
            current = _contacts(
                sim, planner, phase=f"insertion:{target_name}",
                waypoint=index,
                relevant=lambda left, right: (
                    left.startswith(("pcb_", "insertion_collision"))
                    or right.startswith(("pcb_", "insertion_collision"))
                ),
                holders=("B",), allowed_pairs=allowed,
            )
            target_samples.extend(current)
            if not gate.free:
                # The complete gate can reject a non-PCB collision. Preserve it
                # even though it is outside the filtered contact list.
                targets.append({
                    "name": target_name,
                    "waypoints": len(path),
                    "gate_failure": {
                        "waypoint": index,
                        "reason": gate.reason,
                        "pair": gate.pair,
                        "penetration_m": gate.penetration,
                    },
                })
                break
        else:
            targets.append({
                "name": target_name,
                "waypoints": len(path),
                "gate_failure": None,
            })
        samples.extend(target_samples)

    insertion = planner.project.manifest["insertion"]
    collision_cad = insertion.get("collision_cad")
    exact = bool(collision_cad)
    return {
        "collision_model": {
            "mode": (
                "declared_fixture_cad" if exact
                else "generated_virtual_aperture_placeholder"),
            "collision_cad": collision_cad,
            "fixture_collision_cad_declared": exact,
            # This path audit alone can never certify calibrated contact,
            # seating force, gripper capture, or hardware stopping behavior.
            "physical_certification_from_this_audit": False,
            "note": (
                "Exact fixture CAD was declared; certification still depends on "
                "calibrated pin/hole geometry and contact parameters."
                if exact else
                "The generated PCB/support rings have one bounded virtual "
                "aperture, not measured pin holes. This is a planning aid, "
                "not a physical insertion certificate."
            ),
        },
        "targets": targets,
        **_summarize(samples),
    }


def audit_reorientation(sim: WorkcellSim, planner: HandoffPlanner, plan) -> dict:
    """Audit support and gripper contacts in a selected reorientation plan."""
    samples: list[ContactSample] = []
    phases = (
        ("reorientation:place", plan.trajectories["A_to_place"],
         False, plan.g_A_before),
        ("reorientation:repick", plan.trajectories["A_place_to_repick"],
         True, plan.g_A_after),
    )
    for phase, path, fixed_part, grasp in phases:
        for index, qA in enumerate(path):
            X_part = (plan.X_place if fixed_part else
                      planner.kin.fk("A", qA) @ inverse(grasp))
            planner.collision.check(
                qA, planner.q_start["B"], X_part,
                allowed_part_holders=("A",),
                allowed_geom_pairs=REORIENTATION_CONTACTS,
            )
            samples.extend(_contacts(
                sim, planner, phase=phase, waypoint=index,
                relevant=lambda left, right: (
                    "reorientation_surface" in (left, right)
                ),
                holders=("A",), allowed_pairs=REORIENTATION_CONTACTS,
            ))
    part_contacts = [sample for sample in samples
                     if any(name.startswith("part_collision")
                            for name in sample.pair)]
    gripper_contacts = [sample for sample in samples
                        if any(name.startswith("A_gripper_collision_")
                               for name in sample.pair)]
    return {
        "placement": plan.placement_name,
        "expected_support_contact": True,
        "part_support": _summarize(part_contacts),
        "gripper_support": _summarize(gripper_contacts),
        **_summarize(samples),
    }


def run_audit(
    *,
    project_path: str = DEFAULT_PROJECT,
    model_path: str = MODEL,
    cache_dir: str | None = None,
    reorientation_demo: bool = False,
) -> dict:
    sim = WorkcellSim(model_path=model_path, project_path=project_path)
    planner = HandoffPlanner(sim, project_path=project_path, cache_dir=cache_dir)
    report = planner.plan()
    direct = report.direct or (None if report.regrasp is None else report.regrasp.direct)
    if not report.feasible or direct is None:
        raise RuntimeError("no feasible planner-selected insertion path to audit")
    result = {
        "project": os.path.realpath(project_path),
        "model": os.path.realpath(model_path),
        "planning_branch": "direct" if report.direct is not None else "reorientation",
        "planner_clearance_margin_m": planner.collision.clearance_margin_m,
        "insertion": audit_insertion(sim, planner, direct),
        "reorientation": None,
    }
    if reorientation_demo:
        from .visualize_reorientation_demo import build_demo
        demo_sim, demo_planner, regrasp, _ = build_demo(
            project_path=project_path, model_path=model_path, cache_dir=cache_dir)
        result["reorientation"] = audit_reorientation(
            demo_sim, demo_planner, regrasp)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--cache", default=None)
    parser.add_argument(
        "--reorientation-demo", action="store_true",
        help="also audit the production planner's forced adverse-grasp demo",
    )
    parser.add_argument("--json", action="store_true",
                        help="emit the complete machine-readable report")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_audit(
        project_path=args.project,
        model_path=args.model,
        cache_dir=args.cache,
        reorientation_demo=args.reorientation_demo,
    )
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        insertion = report["insertion"]
        print(f"insertion model: {insertion['collision_model']['mode']}")
        print(f"insertion minimum signed distance: "
              f"{insertion['minimum_signed_distance_m']} m")
        print(f"insertion maximum penetration: "
              f"{insertion['maximum_penetration_m']:.9g} m")
        print(f"insertion forbidden contacts: "
              f"{insertion['forbidden_contact_samples']}")
        print(f"note: {insertion['collision_model']['note']}")
        if report["reorientation"] is not None:
            stage = report["reorientation"]
            print(f"reorientation placement: {stage['placement']}")
            print(f"stage maximum part penetration: "
                  f"{stage['part_support']['maximum_penetration_m']:.9g} m")
            print(f"stage maximum gripper penetration: "
                  f"{stage['gripper_support']['maximum_penetration_m']:.9g} m")
            print(f"stage forbidden contacts: "
                  f"{stage['forbidden_contact_samples']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
