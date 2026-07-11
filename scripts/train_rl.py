"""Train the handoff policy with REINFORCE on the one-step env.

  python scripts/train_rl.py --episodes 20000          # train
  python scripts/train_rl.py --eval models/handoff_policy.npz   # evaluate

Deployment contract: the policy PROPOSES handoff parameters; env.step()
verifies them through the exact oracle. On a verified success you get the
full (X_h, g, qA, qB_grasp, qB_insert) tuple in info. On failure, fall
back to handoff.HandoffPlanner.search().
"""
import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np

from rl_env import HandoffEnv
from rl_policy import HandoffPolicy

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
MODEL = os.path.join(ROOT, "models", "handoff_policy.npz")
LOG = os.path.join(ROOT, "models", "train_log.csv")


def evaluate(env: HandoffEnv, policy: HandoffPolicy | None, n: int = 100,
             seed: int = 12345, k: int = 1) -> dict:
    """Policy (or uniform-random baseline if policy is None) on a fixed set
    of held-out initial grasps. k > 1 = best-of-k proposals, which matches
    deployment: every proposal is oracle-verified, so trying k cheap
    proposals before falling back to search is the real usage pattern."""
    rng = np.random.default_rng(seed)
    grasps = [env.sampler.sample(rng) for _ in range(n)]
    arng = np.random.default_rng(seed + 1)
    succ = gates = 0
    for T in grasps:
        obs = env.reset(T)
        best_gate = 0
        for j in range(k):
            if policy is None:
                a = arng.uniform(-1, 1, env.N_CONT)
                gi = int(arng.integers(env.n_grasps))
            else:
                a, gi = policy.act(obs, arng, deterministic=(j == 0))
            r, info = env.step(a, gi)
            best_gate = max(best_gate, info["gate"])
            if best_gate == 3:
                break
        succ += best_gate == 3
        gates += best_gate
    return {"success_rate": succ / n, "mean_gate": gates / n}


def deploy(env: HandoffEnv, args) -> None:
    """The full deployment chain on one random initial grasp:
    1. policy proposes k candidates, each oracle-verified   (~0.5 s)
    2. fallback: brute-force search                          (~10-40 s)
    3. fallback: regrasp — place on nest, re-pick, handoff   (~1-30 s)
    """
    from regrasp import RegraspPlanner

    rng = np.random.default_rng(args.seed + 999)
    T = env.sampler.sample(rng)
    obs = env.reset(T)
    print("initial grasp (T_flangeA_part):")
    print(np.round(T, 4))

    if os.path.exists(MODEL):
        pol = HandoffPolicy.load(MODEL, env.obs_dim, env.N_CONT, env.n_grasps)
        arng = np.random.default_rng(0)
        for j in range(args.k):
            a, gi = pol.act(obs, arng, deterministic=(j == 0))
            r, info = env.step(a, gi)
            if info["gate"] == 3:
                print(f"\n[1] policy proposal {j+1}/{args.k}: VERIFIED SUCCESS")
                print(f"    grasp {info['grasp']}, X_h pos "
                      f"{np.round(info['X_h'][:3,3],3).tolist()}")
                return
        print(f"\n[1] policy: {args.k} proposals, none feasible -> search")
    else:
        print("\n[1] no trained model found -> search")

    rep = env.pl.search()
    if rep.feasible:
        print(f"[2] search: FEASIBLE (grasp {rep.plan.grasp_name})")
        return
    print(f"[2] search: infeasible (dominant: {rep.dominant_failure()}) -> regrasp")

    rg = RegraspPlanner(env.s, env.pl)
    rplan = rg.find_regrasp(T)
    if rplan is not None:
        print(f"[3] regrasp: SUCCESS — place {rplan.placement}, re-pick, "
              f"then handoff with {rplan.handoff.grasp_name}")
    else:
        print("[3] regrasp: no plan — part rejected")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval", metavar="MODEL", help="evaluate a saved model and exit")
    ap.add_argument("--eval-n", type=int, default=100)
    ap.add_argument("--resume", action="store_true",
                    help="continue from the saved model, append to the log")
    ap.add_argument("--no-final-eval", action="store_true")
    ap.add_argument("--deploy", action="store_true",
                    help="full chain on one random grasp: policy -> search -> regrasp")
    ap.add_argument("--k", type=int, default=8, help="proposals per grasp in --deploy")
    args = ap.parse_args()

    env = HandoffEnv(seed=args.seed)
    print(f"G* = {sorted(env.g_star)}   graspable modes = {len(env.sampler.modes)}")

    if args.deploy:
        deploy(env, args)
        return

    if args.eval:
        pol = HandoffPolicy.load(args.eval, env.obs_dim, env.N_CONT, env.n_grasps)
        print("random k=1 :", evaluate(env, None, args.eval_n))
        print("policy k=1 :", evaluate(env, pol, args.eval_n))
        print("random k=8 :", evaluate(env, None, args.eval_n, k=8))
        print("policy k=8 :", evaluate(env, pol, args.eval_n, k=8))
        return

    os.makedirs(os.path.dirname(MODEL), exist_ok=True)
    if args.resume and os.path.exists(MODEL):
        pol = HandoffPolicy.load(MODEL, env.obs_dim, env.N_CONT, env.n_grasps)
        print(f"resumed from {MODEL} (baseline {pol.baseline:.3f})")
    else:
        pol = HandoffPolicy(env.obs_dim, env.N_CONT, env.n_grasps, seed=args.seed)
    rng = np.random.default_rng(args.seed + 100 + (os.getpid() if args.resume else 0))

    log = open(LOG, "a" if args.resume else "w", newline="")
    writer = csv.writer(log)
    if not args.resume:
        writer.writerow(["episode", "reward_mean", "success_rate", "elapsed_s"])
    t0 = time.time()
    batch, ep = [], 0
    while ep < args.episodes:
        obs = env.reset()
        a, gi = pol.act(obs, rng)
        r, _ = env.step(a, gi)
        batch.append((obs, a, gi, r))
        ep += 1
        if len(batch) >= args.batch:
            m = pol.update(batch)
            batch = []
            writer.writerow([ep, f"{m['reward_mean']:.4f}",
                             f"{m['success_rate']:.4f}", f"{time.time()-t0:.0f}"])
            log.flush()
            if (ep // args.batch) % 10 == 0:
                print(f"ep {ep:6d}  reward {m['reward_mean']:.3f}  "
                      f"success {m['success_rate']:.2f}  "
                      f"({ep/(time.time()-t0):.1f} ep/s)")
                pol.save(MODEL)
    pol.save(MODEL)
    print(f"model saved to {MODEL}")
    if not args.no_final_eval:
        print("\nfinal evaluation on held-out grasps:")
        print("random baseline:", evaluate(env, None, args.eval_n))
        print("policy         :", evaluate(env, pol, args.eval_n))


if __name__ == "__main__":
    main()
