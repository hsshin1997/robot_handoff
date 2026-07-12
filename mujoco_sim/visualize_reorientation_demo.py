"""Forced reorientation demonstration for the connector header.

The part begins in a deliberately adverse orientation in A's TCP. That grasp fails the configured
co-grasp compatibility gate for every downstream-valid B grasp, so the demo
places it on the flat surface, changes A's grasp, and then executes the normal
handoff/scanner/insertion pipeline.

macOS:
    mjpython -m mujoco_sim.visualize_reorientation_demo --hold -1
"""
from __future__ import annotations

import argparse
from collections import Counter
import time

import mujoco.viewer
import numpy as np

from .exec import PipelineExecutor
from .planning import HandoffPlanner
from .project import DEFAULT_PROJECT
from .se3 import inverse, make_transform, rpy_matrix
from .sim import MODEL, WorkcellSim


def build_demo(project_path: str = DEFAULT_PROJECT, model_path: str = MODEL,
               cache_dir: str | None = None):
    sim = WorkcellSim(model_path=model_path, project_path=project_path)
    planner = HandoffPlanner(sim, project_path=project_path, cache_dir=cache_dir)
    qA_start = planner.q_start["A"].copy()
    qB_start = planner.q_start["B"].copy()
    good_grasp = planner.g_A_start.copy()

    # Left multiplication is the correct action for g = ^P T_E. This adverse
    # orientation is selected from the generic SO(3) grid because its retreat
    # cone conflicts with every current insertion-feasible B grasp.
    bad_grasp = make_transform(rpy_matrix(np.radians([0, 60, 60])), [0, 0, 0]) @ good_grasp
    downstream = planner.filter_downstream()
    if any(planner._gripper_compatibility(bad_grasp, item.grasp)[0]
           for item in downstream):
        raise RuntimeError("the forced grasp unexpectedly passed compatibility")
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
    parser.add_argument("--project", default=DEFAULT_PROJECT,
                        help="project.yaml used by scene state and planner")
    parser.add_argument("--model", default=MODEL,
                        help="MJCF compiled from the selected project")
    parser.add_argument("--cache", default=None,
                        help="override content-addressed planner cache directory")
    return parser


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    print("Building forced adverse-grasp reorientation scenario...")
    sim, planner, regrasp, bad_grasp = build_demo(
        project_path=args.project, model_path=args.model, cache_dir=args.cache)
    print(f"Direct handoff rejected by grasp compatibility; "
          f"using placement {regrasp.placement_name}.")

    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        viewer.cam.lookat[:] = [0.52, 0.12, 0.53]
        viewer.cam.distance = 1.75
        viewer.cam.azimuth = 140
        viewer.cam.elevation = -25
        viewer.sync()
        time.sleep(1.0)
        executor = PipelineExecutor(sim, planner, viewer=viewer, realtime=True)
        executor.owner_grasp = bad_grasp.copy()
        result = executor.execute_regrasp(regrasp)
        print(f"Execution: {result.outcome}")
        for event in result.events:
            print(f"  {event.timestamp_s:7.3f}s  {event.state.value}")
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
