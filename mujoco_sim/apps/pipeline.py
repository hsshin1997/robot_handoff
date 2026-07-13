"""Command-line and programmatic entry point for the complete handoff flow."""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from ..execution.executor import PipelineExecutor
from ..planner.planner import HandoffPlanner
from ..modeling.project import DEFAULT_PROJECT
from ..simulation.workcell import MODEL, WorkcellSim


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
                     cache_dir: str | None = None,
                     log_root: str | None = None,
                     strict_debug: bool = False, *,
                     _sim_factory=None, _planner_factory=None,
                     _executor_factory=None):
    """Plan/execute with one explicit project, compiled model, and cache.

    New path arguments are appended after the original API parameters so
    existing positional callers retain their behavior.
    """
    scene_started = time.perf_counter()
    sim_factory = WorkcellSim if _sim_factory is None else _sim_factory
    planner_factory = (HandoffPlanner if _planner_factory is None
                       else _planner_factory)
    executor_factory = (PipelineExecutor if _executor_factory is None
                        else _executor_factory)
    sim = sim_factory(model_path=model_path, project_path=project_path)
    scene_elapsed = time.perf_counter() - scene_started
    planner = planner_factory(
        sim, known_start_pose=known_start_pose,
        project_path=project_path, cache_dir=cache_dir)
    report = planner.plan(allow_regrasp=allow_regrasp, return_best=return_best)
    scene_metric = {
        "path": "setup.scene_and_model",
        "name": "scene_and_model",
        "parent": None,
        "calls": 1,
        "failures": 0,
        "wall_total_s": scene_elapsed,
        "wall_self_s": scene_elapsed,
        "wall_max_s": scene_elapsed,
        "cpu_total_s": 0.0,
    }
    planner_initialization = tuple(getattr(
        planner, "initialization_profile", ()))
    planner_elapsed = float(getattr(
        planner, "initialization_elapsed_s", 0.0))
    planner_metric = {
        "path": "setup.planner",
        "name": "planner",
        "parent": None,
        "calls": 1,
        "failures": 0,
        "wall_total_s": planner_elapsed,
        "wall_self_s": max(
            0.0,
            planner_elapsed - sum(
                item["wall_total_s"] for item in planner_initialization)),
        "wall_max_s": planner_elapsed,
        "cpu_total_s": 0.0,
    }
    report.initialization_timings = (
        scene_metric, planner_metric,
        *tuple(getattr(report, "initialization_timings", ()))
    )
    result = None
    if execute and report.feasible:
        # Planning is kinematic and intentionally mutates the shared MjData.
        # Restore the transaction's known initial state before execution.
        sim.set_arm_qpos("A", planner.q_start["A"])
        sim.set_arm_qpos("B", planner.q_start["B"])
        sim.apply_active_grasp()
        executor = executor_factory(
            sim, planner, log_root=log_root, strict_debug=strict_debug)
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
    parser.add_argument(
        "--profile", action="store_true",
        help="print setup/planning/execution bottleneck details")
    parser.add_argument("--project", default=DEFAULT_PROJECT,
                        help="project.yaml used by scene state and planner")
    parser.add_argument("--model", default=MODEL,
                        help="MJCF compiled from the selected project")
    parser.add_argument("--cache", default=None,
                        help="override content-addressed planner cache directory")
    parser.add_argument(
        "--debug-artifacts", nargs="?", const="logs", default=None,
        metavar="LOG_ROOT",
        help="capture per-stage state.json/contact PNG artifacts (default root: logs)")
    parser.add_argument(
        "--strict-debug", action="store_true",
        help="let debug-recorder failures affect execution (default: isolate them)")
    return parser


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    if args.strict_debug and args.debug_artifacts is None:
        raise SystemExit("--strict-debug requires --debug-artifacts [LOG_ROOT]")
    report, result = plan_and_execute(
        args.execute, not args.no_regrasp, args.best,
        project_path=args.project, model_path=args.model, cache_dir=args.cache,
        log_root=args.debug_artifacts, strict_debug=args.strict_debug)
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
            print("executed modeled robot time: "
                  f"{result.executed_modeled_time_s:.3f} s")
            print("planned modeled serial makespan: "
                  f"{result.planned_modeled_makespan_s:.3f} s")
            if not result.timing_estimate_complete:
                print("timing estimate incomplete; unmodeled operations: "
                      f"{list(result.unmodeled_operations)}")
            print(f"observed computer wall time: {result.wall_elapsed_s:.3f} s")
            print("per-step timing (estimated robot / observed wall):")
            for timing in result.stage_timings:
                print(
                    f"  {timing['label']}: "
                    f"{timing['estimated_robot_duration_s']:.3f} s / "
                    f"{timing['observed_wall_duration_s']:.3f} s")
            if result.debug_run_dir:
                print(f"debug artifacts: {result.debug_run_dir}")
            for error in result.debug_errors:
                print(f"debug warning: {error}")
        if args.profile:
            print("setup timings:")
            for item in sorted(
                    report.initialization_timings,
                    key=lambda value: -value["wall_self_s"]):
                print(
                    f"  {item['path']}: {item['wall_total_s']:.6f} s total, "
                    f"{item['wall_self_s']:.6f} s self")
            print("planning bottlenecks (exclusive wall time):")
            for item in report.bottlenecks:
                print(
                    f"  {item['path']}: {item['wall_self_s']:.6f} s self, "
                    f"{item['calls']} calls")
            if result:
                print("execution bottlenecks (exclusive wall time):")
                for item in sorted(
                        result.profile_spans,
                        key=lambda value: -value["wall_self_s"])[:8]:
                    print(
                        f"  {item['path']}: {item['wall_self_s']:.6f} s self, "
                        f"{item['calls']} calls")
        if report.limitations:
            print("limitations:")
            for limitation in report.limitations:
                print(f"  - {limitation}")


if __name__ == "__main__":
    main()
