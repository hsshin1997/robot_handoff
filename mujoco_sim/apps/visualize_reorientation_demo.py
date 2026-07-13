"""Forced stage-route demonstration for the connector header.

The part begins in a deliberately rotated orientation in A's TCP. This command
explicitly selects a verified placement/re-pick route so the stage behavior can
be inspected even when the direct-first production planner also has a direct
solution for that initial grasp. It then executes the normal handoff, scanner,
and insertion pipeline.

macOS:
    mjpython -m mujoco_sim.visualize_reorientation_demo --hold -1
"""
from __future__ import annotations

import argparse
from collections import Counter
import math
import time

import mujoco.viewer
import numpy as np

from ..execution.executor import PipelineExecutor
from ..planner.planner import HandoffPlanner
from ..modeling.project import DEFAULT_PROJECT
from ..core.se3 import inverse, make_transform, rpy_matrix
from ..simulation.workcell import MODEL, WorkcellSim


def build_demo(project_path: str = DEFAULT_PROJECT, model_path: str = MODEL,
               cache_dir: str | None = None):
    sim = WorkcellSim(model_path=model_path, project_path=project_path)
    planner = HandoffPlanner(sim, project_path=project_path, cache_dir=cache_dir)
    qA_start = planner.q_start["A"].copy()
    qB_start = planner.q_start["B"].copy()
    good_grasp = planner.g_A_start.copy()

    # Left multiplication is the correct action for g = ^P T_E. This rotated
    # orientation has a verified stage placement/re-pick route in the reference
    # scene. The demo forces that branch for visualization; production remains
    # direct-first.
    bad_grasp = make_transform(rpy_matrix(np.radians([0, 60, 60])), [0, 0, 0]) @ good_grasp
    # Exercise the production backward task graph, including cached stable
    # placements, support robustness, phase-aware clearance, and the terminal
    # insertion-valid direct edge. The demo has no separate hand-built planner.
    X_bad = planner.kin.fk("A", qA_start) @ inverse(bad_grasp)
    planner.g_A_start = bad_grasp.copy()
    planner.X_start = X_bad.copy()
    regrasp = planner.search_regrasp(Counter())
    if regrasp is None:
        raise RuntimeError(
            "no collision-free placement/re-pick with a verified insertion-feasible handoff; "
            "the unsafe reorientation demo will not be displayed"
        )

    # Restore the deliberately bad known initial state after kinematic planning.
    planner.q_start["A"] = qA_start
    sim.set_arm_qpos("A", qA_start)
    sim.set_arm_qpos("B", qB_start)
    sim.release_part()
    sim.set_part_world(X_bad)
    sim.grasp_part("A")
    return sim, planner, regrasp, bad_grasp


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hold", type=float, default=3.0)
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
    print("Building forced adverse-grasp reorientation scenario...")
    try:
        sim, planner, regrasp, bad_grasp = build_demo(
            project_path=args.project, model_path=args.model,
            cache_dir=args.cache)
    except RuntimeError as error:
        raise SystemExit(f"No safe reorientation visualization: {error}") from None
    print(f"Forcing verified stage route via placement "
          f"{regrasp.placement_name}; production planning remains direct-first.")

    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        viewer.cam.lookat[:] = [0.52, 0.12, 0.53]
        viewer.cam.distance = 1.75
        viewer.cam.azimuth = 140
        viewer.cam.elevation = -25
        viewer.sync()
        time.sleep(args.start_delay)
        executor = PipelineExecutor(
            sim, planner, viewer=viewer, realtime=True,
            playback_speed=args.playback_speed,
            log_root=args.debug_artifacts, strict_debug=args.strict_debug)
        executor.owner_grasp = bad_grasp.copy()
        result = executor.execute_regrasp(regrasp)
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
                f"wall {event.timestamp_s:7.3f}s  {event.state.value}")
        if args.hold < 0:
            print("Close the viewer window to exit.")
            while viewer.is_running():
                viewer.sync(); time.sleep(0.02)
        else:
            deadline = time.monotonic() + args.hold
            while viewer.is_running() and time.monotonic() < deadline:
                viewer.sync(); time.sleep(0.02)


if __name__ == "__main__":
    main()
