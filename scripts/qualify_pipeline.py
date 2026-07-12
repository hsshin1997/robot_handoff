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
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mujoco_sim.offline import atomic_write_json  # noqa: E402
from mujoco_sim.planning import HandoffPlanner  # noqa: E402
from mujoco_sim.qualification import (CoverageOutcome,
                                      build_coverage_certificate,
                                      physical_prerequisites)  # noqa: E402
from mujoco_sim.se3 import inverse  # noqa: E402
from mujoco_sim.sim import MODEL, WorkcellSim  # noqa: E402


def qualify(max_classes: int | None = None, required_fraction: float = 1.0,
            project: str | os.PathLike[str] | None = None,
            model: str | os.PathLike[str] = MODEL,
            cache_dir: str | os.PathLike[str] | None = None) -> dict:
    project_kwargs = {} if project is None else {"project_path": str(project)}
    sim = WorkcellSim(model_path=str(model), **project_kwargs)
    planner = HandoffPlanner(
        sim, **project_kwargs,
        **({} if cache_dir is None else {"cache_dir": str(cache_dir)}))
    domain_source = planner.project.initial_grasp_domain_source
    candidates = [("nominal", inverse(planner.project.T_tcp_part_start))]
    if domain_source == "known_start_plus_geometry_library":
        candidates.extend(planner.g_B_candidates)
    elif domain_source != "known_start":  # Project validates; keep this local guard.
        raise ValueError(f"unsupported initial grasp domain {domain_source!r}")
    unique = []
    import numpy as np
    for name, grasp in candidates:
        if not any(np.allclose(grasp, old, atol=1e-9) for _, old in unique):
            unique.append((name, grasp))
    if max_classes is not None:
        unique = unique[:max_classes]
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
    return build_coverage_certificate(
        outcomes, required_fraction=required_fraction,
        physical_prerequisites=prerequisites,
        domain_declaration={
            "source": domain_source,
            "project_field": "qualification.initial_grasp_domain",
            "generated_class_count": len(unique),
        })


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
    print(f"coverage: {report['covered_count']}/{report['required_count']} "
          f"({report['fraction']:.3f})")
    print(f"mathematical coverage certified: "
          f"{report['mathematical_coverage_certified']}")
    print(f"physical certified: {report['physical_certified']}")
    print(f"report: {args.output}")
    return 0 if report["mathematical_coverage_certified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
