#!/usr/bin/env python3
"""Enumerate the configured grasp-class domain and certify policy coverage.

This can be expensive on the first run; every class policy is content-cached
and the command is resumable. ``100%`` means every enumerated CAD-derived
class is covered. It does not waive missing articulated-gripper, hole-contact,
calibration, or analytic-IK prerequisites.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mujoco_sim.offline_tools.artifacts import (  # noqa: E402
    atomic_write_json, fingerprint_content, fingerprint_file)
from mujoco_sim.planner.planner import HandoffPlanner  # noqa: E402
from mujoco_sim.offline_tools.qualification import (  # noqa: E402
    CoverageOutcome, build_coverage_certificate, physical_prerequisites)
from mujoco_sim.core.se3 import inverse  # noqa: E402
from mujoco_sim.simulation.workcell import MODEL, WorkcellSim  # noqa: E402


def qualify(max_classes: int | None = None, required_fraction: float = 1.0,
            project: str | os.PathLike[str] | None = None,
            model: str | os.PathLike[str] = MODEL,
            cache_dir: str | os.PathLike[str] | None = None) -> dict:
    if max_classes is not None and (
            isinstance(max_classes, bool) or not isinstance(max_classes, int)
            or max_classes <= 0):
        raise ValueError("max_classes must be a positive integer or None")
    project_kwargs = {} if project is None else {"project_path": str(project)}
    sim = WorkcellSim(model_path=str(model), **project_kwargs)
    planner = HandoffPlanner(
        sim, **project_kwargs,
        **({} if cache_dir is None else {"cache_dir": str(cache_dir)}))
    domain_source = planner.project.initial_grasp_domain_source
    candidates = [("nominal", inverse(planner.project.T_tcp_part_start))]
    if domain_source == "known_start_plus_geometry_library":
        capability_A = planner.project.gripper("A")
        capability_B = planner.project.gripper("B")
        comparable = (
            capability_A.model_path == capability_B.model_path
            and capability_A.opening_min == capability_B.opening_min
            and capability_A.opening_max == capability_B.opening_max
            and capability_A.finger_depth == capability_B.finger_depth
            and np.array_equal(capability_A.pad_size, capability_B.pad_size)
        )
        if not comparable:
            raise ValueError(
                "known_start_plus_geometry_library currently requires identical "
                "A/B gripper geometry; an A-specific grasp library is required")
        candidates.extend(planner.g_B_candidates)
    elif domain_source != "known_start":  # Project validates; keep this local guard.
        raise ValueError(f"unsupported initial grasp domain {domain_source!r}")
    unique = []
    for name, grasp in candidates:
        if not any(np.allclose(grasp, old, atol=1e-9) for _, old in unique):
            unique.append((name, grasp))
    full_class_count = len(unique)
    if max_classes is not None:
        unique = unique[:max_classes]
    truncated = len(unique) < full_class_count
    outcomes = []
    for index, (name, grasp) in enumerate(unique, start=1):
        planner.g_A_start = grasp.copy()
        planner.X_start = planner.kin.fk("A", planner.q_start["A"]) @ inverse(grasp)
        direct, _, _ = planner.search_direct(grasp, return_best=False)
        if direct is not None:
            outcome = CoverageOutcome(name, "direct", "verified_direct_policy")
        else:
            reorientation = planner.search_regrasp(Counter())
            outcome = (CoverageOutcome(name, "reorientation",
                                       "verified_backward_reorientation_policy")
                       if reorientation is not None else
                       CoverageOutcome(name, None, "no_policy_within_hop_bound"))
        outcomes.append(outcome)
        print(f"[{index}/{len(unique)}] {name}: {outcome.mode or 'uncovered'}",
              flush=True)
    prerequisites = physical_prerequisites(planner.project)
    certificate = build_coverage_certificate(
        outcomes, required_fraction=required_fraction,
        physical_prerequisites=prerequisites,
        domain_declaration={
            "source": domain_source,
            "project_field": "qualification.initial_grasp_domain",
            "generated_class_count": full_class_count,
            "evaluated_class_count": len(unique),
            "evaluation_complete": not truncated,
            "truncated_prefix_smoke": truncated,
        })
    prefix_passed = certificate["mathematical_coverage_certified"]
    certificate["prefix_smoke_passed"] = bool(prefix_passed)
    certificate["evaluated_domain_complete"] = not truncated
    if truncated:
        certificate["mathematical_coverage_certified"] = False
        certificate["physical_certified"] = False
    certificate["provenance"] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "producer": "scripts/qualify_pipeline.py",
        "project_manifest": planner.project.manifest_path,
        "project_manifest_sha256": fingerprint_file(
            planner.project.manifest_path),
        "solver_sha256": fingerprint_content(planner.project.solver),
        "compiled_model": os.path.realpath(model),
        "compiled_model_sha256": fingerprint_file(model),
        "active_part_sha256": fingerprint_file(
            planner.project.active_part_path),
        "gripper_A_sha256": fingerprint_file(
            planner.project.gripper("A").model_path),
        "gripper_B_sha256": fingerprint_file(
            planner.project.gripper("B").model_path),
    }
    return certificate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=None)
    parser.add_argument("--model", default=MODEL,
                        help="MJCF compiled from --project")
    parser.add_argument("--cache", default=None,
                        help="override content-addressed planner cache directory")
    parser.add_argument("--max-classes", type=int, default=None,
                        help="smoke-test a prefix; omit for the full domain")
    parser.add_argument("--required", type=float, default=1.0)
    parser.add_argument("--output", default=str(
        ROOT / "mujoco_sim" / "cache" / "coverage-certificate.json"))
    args = parser.parse_args()
    report = qualify(
        args.max_classes, args.required, args.project, args.model, args.cache)
    atomic_write_json(args.output, report)
    print(f"coverage: {report['covered_count']}/{report['domain_count']} "
          f"({report['fraction']:.3f})")
    print(f"required covered classes: {report['required_count']}")
    print(f"evaluated full declared domain: "
          f"{report['evaluated_domain_complete']}")
    print(f"mathematical coverage certified: "
          f"{report['mathematical_coverage_certified']}")
    print(f"physical certified: {report['physical_certified']}")
    print(f"report: {args.output}")
    passed = (report["mathematical_coverage_certified"]
              if report["evaluated_domain_complete"]
              else report["prefix_smoke_passed"])
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
