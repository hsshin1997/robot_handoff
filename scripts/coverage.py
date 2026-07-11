"""Coverage / upper-bound measurement: over random initial grasps, what
fraction is solvable by (a) direct handoff search, (b) direct + regrasp?

(b) is the system's true ceiling — the number the RL policy (with oracle
verification and fallbacks) can approach but not exceed.

  python scripts/coverage.py --n 30
  python scripts/coverage.py --n 30 --resume   # append to previous samples

Results accumulate in models/coverage.jsonl (one JSON per sampled grasp),
so long runs can be split across invocations.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np

import kin
from handoff import HandoffPlanner
from regrasp import RegraspPlanner, TABLE_PATH
from rl_env import GraspSampler
from scene import Scene

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "models", "coverage.jsonl")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--time-budget", type=float, default=1e9,
                    help="stop cleanly after this many seconds (resumable)")
    args = ap.parse_args()

    scene = Scene()
    pl = HandoffPlanner(scene)
    rg = RegraspPlanner(scene, pl)
    table = rg.load_table(TABLE_PATH) if os.path.exists(TABLE_PATH) else rg.build_table()
    sampler = GraspSampler(scene)

    rng = np.random.default_rng(args.seed)
    grasps = [sampler.sample(rng) for _ in range(args.n)]   # deterministic set

    done = 0
    if args.resume and os.path.exists(OUT):
        done = sum(1 for _ in open(OUT))
    mode = "a" if args.resume else "w"
    out = open(OUT, mode)

    t0 = time.time()
    for i in range(done, args.n):
        if time.time() - t0 > args.time_budget:
            print(f"time budget reached at sample {i} — rerun with --resume")
            break
        T = grasps[i]
        pl.T_fA_part = T
        pl.T_part_fA = kin.inv_T(T)
        t1 = time.time()
        rep = pl.search(time_budget=25.0)
        direct = bool(rep.feasible)
        regrasp = None
        if not direct:
            rplan = rg.find_regrasp(T, table)
            regrasp = rplan is not None
        rec = {"i": i, "direct": direct, "regrasp": regrasp,
               "dt": round(time.time() - t1, 1)}
        out.write(json.dumps(rec) + "\n")
        out.flush()
        print(rec)
    out.close()

    recs = [json.loads(l) for l in open(OUT)]
    n = len(recs)
    d = sum(r["direct"] for r in recs)
    r_ = sum(1 for r in recs if r["direct"] or r["regrasp"])
    print(f"\n=== coverage over {n} random initial grasps ===")
    print(f"direct handoff only : {d}/{n} = {d/n:.0%}")
    print(f"direct + regrasp    : {r_}/{n} = {r_/n:.0%}   <- system upper bound")
    print(f"unsolvable          : {n-r_}/{n}")


if __name__ == "__main__":
    main()
