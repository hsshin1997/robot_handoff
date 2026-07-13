"""Plan and animate the complete pipeline in MuJoCo's passive viewer.

macOS:
    mjpython -m mujoco_sim.visualize_pipeline

Linux:
    python -m mujoco_sim.visualize_pipeline
"""
from __future__ import annotations

import argparse
import math
import time

import mujoco.viewer

from .exec import PipelineExecutor
from .planning import HandoffPlanner
from .project import DEFAULT_PROJECT
from .sim import MODEL, WorkcellSim


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-regrasp", action="store_true")
    parser.add_argument("--best", action="store_true",
                        help="exhaust the grid instead of taking the first feasible plan")
    parser.add_argument("--hold", type=float, default=3.0,
                        help="seconds to keep the completed result visible; negative waits until closed")
    parser.add_argument(
        "--playback-speed", type=float, default=1.0, metavar="MULTIPLIER",
        help="visualization wall-clock speed multiplier (for example 4)")
    parser.add_argument(
        "--start-delay", type=float, default=1.0, metavar="SECONDS",
        help="pause after opening the viewer before animation starts")
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
    parser.add_argument("--strict-debug", action="store_true",
                        help="fail execution when debug capture fails")
    return parser


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    if args.strict_debug and args.debug_artifacts is None:
        raise SystemExit("--strict-debug requires --debug-artifacts [LOG_ROOT]")
    if not math.isfinite(args.playback_speed) or args.playback_speed <= 0.0:
        raise SystemExit("--playback-speed must be positive and finite")
    if not math.isfinite(args.start_delay) or args.start_delay < 0.0:
        raise SystemExit("--start-delay must be finite and non-negative")

    print("Planning complete handoff pipeline...")
    sim = WorkcellSim(model_path=args.model, project_path=args.project)
    planner = HandoffPlanner(sim, project_path=args.project, cache_dir=args.cache)
    report = planner.plan(allow_regrasp=not args.no_regrasp,
                          return_best=args.best)
    print(f"Planning finished in {report.elapsed_s:.2f} s; "
          f"feasible={report.feasible}; candidates={report.candidates}")
    print(f"Downstream-valid receiver grasps: {report.downstream_grasps}")
    if not report.feasible:
        print(f"No plan found. Gate statistics: {dict(report.stats)}")
        raise SystemExit(2)

    # Planning mutates MjData while checking branches. Restore the known start.
    sim.set_arm_qpos("A", planner.q_start["A"])
    sim.set_arm_qpos("B", planner.q_start["B"])
    sim.apply_active_grasp()

    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        viewer.cam.lookat[:] = [0.425, 0.0, 0.58]
        viewer.cam.distance = 2.25
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -22
        viewer.sync()
        time.sleep(args.start_delay)

        executor = PipelineExecutor(
            sim, planner, viewer=viewer, realtime=True,
            playback_speed=args.playback_speed,
            log_root=args.debug_artifacts, strict_debug=args.strict_debug)
        result = (executor.execute_direct(report.direct) if report.direct is not None
                  else executor.execute_regrasp(report.regrasp))
        print(f"Execution: {result.outcome}")
        print(f"Executed modeled robot time: "
              f"{result.executed_modeled_time_s:.3f} s; "
              f"planned serial makespan: "
              f"{result.planned_modeled_makespan_s:.3f} s; "
              f"observed wall: {result.wall_elapsed_s:.3f} s")
        print("Per-step estimated robot / observed wall durations:")
        for timing in result.stage_timings:
            print(f"  {timing['label']}: "
                  f"{timing['estimated_robot_duration_s']:.3f} s / "
                  f"{timing['observed_wall_duration_s']:.3f} s")
        if result.debug_run_dir:
            print(f"Debug artifacts: {result.debug_run_dir}")
        for error in result.debug_errors:
            print(f"  debug warning: {error}")
        for event in result.events:
            print(
                f"  robot {event.estimated_robot_time_s:7.3f}s  "
                f"wall {event.timestamp_s:7.3f}s  "
                f"{event.state.value}  {event.detail}")

        if args.hold < 0:
            print("Simulation complete. Close the viewer window to exit.")
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.02)
        else:
            deadline = time.monotonic() + args.hold
            while viewer.is_running() and time.monotonic() < deadline:
                viewer.sync()
                time.sleep(0.02)


if __name__ == "__main__":
    main()
