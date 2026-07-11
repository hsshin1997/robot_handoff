"""Offline plan-cache builder: solve the handoff for a grid of canonical
grasps (every graspable mode x 4 rolls) and store the plans. After this,
online planning is warm-started and typically completes in well under 1 s.

  python scripts/build_cache.py                 # resumable; reruns skip done work
  python scripts/build_cache.py --budget 25     # per-grasp search budget (s)
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np

import kin
from handoff import HandoffPlanner, PlanCache
from rl_env import GraspSampler
from scene import Scene


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=25.0)
    ap.add_argument("--rolls", type=int, default=4)
    ap.add_argument("--time-budget", type=float, default=1e9,
                    help="stop cleanly after this many seconds (resumable)")
    args = ap.parse_args()

    scene = Scene()
    pl = HandoffPlanner(scene)
    sampler = GraspSampler(scene)
    cache = PlanCache()

    rolls = [i * 2 * np.pi / args.rolls for i in range(args.rolls)]
    targets = [(m, r) for m in range(len(sampler.modes)) for r in rolls]

    def covered(T):
        for e in cache.entries:
            E = np.asarray(e["T_fA_part"])
            if (np.linalg.norm(E[:3, 3] - T[:3, 3]) < 1e-6
                    and kin.rot_angle(E[:3, :3], T[:3, :3]) < 1e-6):
                return True
        return False

    t0 = time.time()
    solved = skipped = failed = 0
    for m, r in targets:
        if time.time() - t0 > args.time_budget:
            print("time budget reached — rerun to resume")
            break
        T = sampler.canonical(m, r)
        if covered(T):
            skipped += 1
            continue
        pl.T_fA_part = T
        pl.T_part_fA = kin.inv_T(T)
        rep = pl.search(time_budget=args.budget)
        if rep.feasible:
            cache.add(T, rep.plan)
            solved += 1
            print(f"mode {m} roll {np.degrees(r):.0f}: solved "
                  f"({rep.plan.grasp_name})")
        else:
            failed += 1
            print(f"mode {m} roll {np.degrees(r):.0f}: no direct handoff")
    print(f"\ncache: {len(cache.entries)} plans "
          f"(+{solved} new, {skipped} already cached, {failed} infeasible)")


if __name__ == "__main__":
    main()
