"""Command-line and programmatic entry point for the complete handoff flow."""
from __future__ import annotations

import argparse
import json

import numpy as np

from .exec import PipelineExecutor
from .planning import HandoffPlanner
from .project import DEFAULT_PROJECT
from .sim import MODEL, WorkcellSim


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "value"):
        return value.value
    if hasattr(value, "__dict__"):
        return {key: _jsonable(item) for key, item in value.__dict__.items()}
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def plan_and_execute(execute: bool = False, allow_regrasp: bool = True,
                     return_best: bool = False,
                     known_start_pose: np.ndarray | None = None,
                     project_path: str = DEFAULT_PROJECT,
                     model_path: str = MODEL,
                     cache_dir: str | None = None):
    """Plan/execute with one explicit project, compiled model, and cache.

    New path arguments are appended after the original API parameters so
    existing positional callers retain their behavior.
    """
    sim = WorkcellSim(model_path=model_path, project_path=project_path)
    planner = HandoffPlanner(
        sim, known_start_pose=known_start_pose,
        project_path=project_path, cache_dir=cache_dir)
    report = planner.plan(allow_regrasp=allow_regrasp, return_best=return_best)
    result = None
    if execute and report.feasible:
        # Planning is kinematic and intentionally mutates the shared MjData.
        # Restore the transaction's known initial state before execution.
        sim.set_arm_qpos("A", planner.q_start["A"])
        sim.set_arm_qpos("B", planner.q_start["B"])
        sim.apply_active_grasp()
        executor = PipelineExecutor(sim, planner)
        result = (executor.execute_direct(report.direct) if report.direct is not None
                  else executor.execute_regrasp(report.regrasp))
    return report, result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true",
                        help="execute the plan in MuJoCo after planning")
    parser.add_argument("--no-regrasp", action="store_true")
    parser.add_argument("--best", action="store_true",
                        help="evaluate all candidates instead of first feasible")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--project", default=DEFAULT_PROJECT,
                        help="project.yaml used by scene state and planner")
    parser.add_argument("--model", default=MODEL,
                        help="MJCF compiled from the selected project")
    parser.add_argument("--cache", default=None,
                        help="override content-addressed planner cache directory")
    return parser


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    report, result = plan_and_execute(
        args.execute, not args.no_regrasp, args.best,
        project_path=args.project, model_path=args.model, cache_dir=args.cache)
    payload = {"planning": _jsonable(report), "execution": _jsonable(result)}
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"feasible: {report.feasible}")
        print(f"downstream grasps: {report.downstream_grasps}")
        print(f"candidates checked: {report.candidates}")
        print(f"gate statistics: {dict(report.stats)}")
        print(f"planning time: {report.elapsed_s:.3f} s")
        print("mathematical known-start coverage certified: "
              f"{report.mathematical_coverage_certified}")
        print(f"physical certified: {report.physical_certified}")
        if report.direct:
            print(f"branch: direct, receiver grasp: {report.direct.grasp_name_B}, "
                  f"score: {report.direct.score.total:.3f}")
        elif report.regrasp:
            print(f"branch: regrasp via {report.regrasp.placement_name}")
        if result:
            print(f"execution: {result.outcome}")
        if report.limitations:
            print("limitations:")
            for limitation in report.limitations:
                print(f"  - {limitation}")


if __name__ == "__main__":
    main()
