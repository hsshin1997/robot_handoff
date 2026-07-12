"""Plan and animate the complete pipeline in MuJoCo's passive viewer.

macOS:
    mjpython -m mujoco_sim.visualize_pipeline

Linux:
    python -m mujoco_sim.visualize_pipeline
"""
from __future__ import annotations

import argparse
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
    parser.add_argument("--project", default=DEFAULT_PROJECT,
                        help="project.yaml used by scene state and planner")
    parser.add_argument("--model", default=MODEL,
                        help="MJCF compiled from the selected project")
    parser.add_argument("--cache", default=None,
                        help="override content-addressed planner cache directory")
    return parser


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)

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
        time.sleep(1.0)

        executor = PipelineExecutor(sim, planner, viewer=viewer, realtime=True)
        result = (executor.execute_direct(report.direct) if report.direct is not None
                  else executor.execute_regrasp(report.regrasp))
        print(f"Execution: {result.outcome}")
        for event in result.events:
            print(f"  {event.timestamp_s:7.3f}s  {event.state.value}  {event.detail}")

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
