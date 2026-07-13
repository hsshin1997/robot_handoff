"""Build offline G1 reachability maps for both GP7 TCPs."""
import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from mujoco_sim.simulation.kinematics import GP7Kinematics
from mujoco_sim.modeling.project import DEFAULT_PROJECT, Project
from mujoco_sim.planner.reachability import ReachabilityMap
from mujoco_sim.simulation.workcell import MODEL, WorkcellSim


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--samples", type=int, default=None,
                        help="override solver_defaults offline sample count")
    parser.add_argument("--voxel", type=float, default=None,
                        help="override solver_defaults TCP voxel size in metres")
    parser.add_argument("--out", default=os.path.join(ROOT, "mujoco_sim", "cache"))
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    project = Project(args.project)
    samples = (int(project.solver["offline"]["reachability_samples"])
               if args.samples is None else args.samples)
    voxel = (float(project.solver["offline"]["voxel_fraction_of_reach"])
             if args.voxel is None else args.voxel)
    if samples <= 0 or voxel <= 0.0:
        raise ValueError("reachability samples and voxel size must be positive")
    os.makedirs(args.out, exist_ok=True)
    kin = GP7Kinematics(WorkcellSim(
        model_path=args.model, project_path=args.project))
    for robot in ("A", "B"):
        mapping = ReachabilityMap.build(kin, robot, samples, voxel,
                                        seed=17 + ord(robot))
        path = os.path.join(args.out, f"reachability_{robot}.npz")
        mapping.save(path)
        print(f"wrote {path}: {len(mapping.keys)} occupied pose-direction voxels")


if __name__ == "__main__":
    main()
