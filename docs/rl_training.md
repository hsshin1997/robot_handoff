# RL Handoff Policy — Technical Details & Training Guide

> **Legacy PyBullet experiment.** The current MuJoCo safety architecture limits
> learning to proposal/ranking behind deterministic feasibility gates; see the
> learning section in [mujoco_handoff_pipeline.md](mujoco_handoff_pipeline.md).

How the learned handoff proposer works, why it is built this way, how to
train it, and what changes when the part, gripper, or cell changes.

---

## 1. Architecture: policy proposes, oracle verifies

```
                 ┌─────────────────────────────────────────────┐
                 │                DEPLOYMENT                    │
                 │                                              │
 initial grasp  ─┤→ policy → k proposals ─→ oracle verify ──ok─→│→ execute
 ĝ_A (from pick) │              │                 │             │  (X_h, g, qA,
                 │              │               fail            │   qB_grasp,
                 │              │                 ↓             │   qB_insert)
                 │              └──────→ brute-force search ────│
                 └─────────────────────────────────────────────┘
```

The policy is **never trusted**. Every proposal runs through the exact same
feasibility oracle (`handoff.py` gates → `kin.py` IK + collision) that the
brute-force search uses. A proposal that passes returns a fully verified
joint-space plan; if all `k` proposals fail, the system falls back to grid
search. The learned component can therefore only *speed things up* — it can
never emit an unverified or colliding plan. This matters: pure end-to-end RL
policies give you no such guarantee.

Why one-step RL (contextual bandit) and not sequential control? Because the
scoped problem has no dynamics: insertion success is defined kinematically
(B reaches the insert pose collision-free within limits). The decision is a
single parameter choice per handoff — *where* to present the part and *which*
grasp B takes. Learning joint-level motions would burn millions of samples to
rediscover what IK already computes exactly.

## 2. The MDP (one step)

### Observation (12-D)

| dims | content | encoding |
|---|---|---|
| 3 | initial-grasp translation error | `(t − [0.200, 0, 0]) × 20` (≈ [−1, 1]) |
| 6 | initial-grasp rotation | first two columns of R (6-D rotation rep — continuous, no quaternion double-cover issues) |
| 3 | part half-extents | `× 20` |

The observation encodes *how A happens to be holding the part* (`T_flangeA_part`,
which varies pick to pick) and the part's size. Everything else (robot bases,
fixtures, insert pose) is constant per cell and lives in the config — the
policy absorbs it implicitly during training.

### Action (5 continuous + 1 categorical)

| component | range | meaning |
|---|---|---|
| x, y, z | [−1,1] → search box in `cell.yaml handoff_search` | part position at transfer |
| yaw | [−1,1] → ±90° | part yaw at transfer |
| roll | [−1,1] → ±180° | part roll about its x axis |
| grasp index | categorical over `grasp_set_G` | which grasp B uses |

`X_h = Trans(x,y,z) · Rz(yaw) · Rx(roll)` — same parameterization the grid
search sweeps, so policy and search explore the same space.

### Reward (shaped by oracle gates)

| event | reward |
|---|---|
| chosen grasp ∉ G* (can never insert) | 0.0 |
| gate 1 passes (A can present at X_h) | 0.3 |
| gate 2 passes (B takes at co-grasp instant) | 0.6 |
| gate 3 passes (B carries to insert, waypoints clear) | 1.0 + 0.2·margin |

The margin bonus is `(joint-limit margin + clearance/5 cm)/2`, pushing solutions
away from limits and obstacles, not just barely-feasible ones. Shaping is what
makes sparse success learnable: the policy first learns to put the part where
A can reach (gate 1), then where both arms work (gate 2), then full success.

G* (grasps that can reach the insert pose) is precomputed once per env —
gate 3's grasp filter depends only on `g`, never on `X_h` or `ĝ_A`
(the §3.2 factorization from the pipeline doc).

## 3. Initial-grasp sampler (`rl_env.GraspSampler`)

Simulates bin-pick variability from the part CAD alone:

1. Read the part mesh bbox (`part_mesh` in config).
2. Enumerate physical grasp modes: (approach axis ±x/±y/±z) × (closing axis)
   pairs where the part's extent along the closing axis fits the finger gap
   (24 mm) — 12 modes for the current relay.
3. Sample: mode, uniform roll about the approach axis, plus pick-error jitter
   (σ ≈ 1.3 mm translation, 2.3° rotation).
4. Emit `T_flangeA_part` with the part COG at the TCP (200 mm from flange).

Every training episode draws a fresh grasp, so the policy learns a *strategy
conditioned on how the part was picked* — not one fixed answer.

## 4. Policy & learning algorithm (`rl_policy.py`)

Pure numpy (no torch — keeps the project dependency list at
pybullet/numpy/pyyaml):

- **Network**: obs(12) → 64 tanh → 64 tanh → two heads:
  - Gaussian head: tanh mean (5), state-independent learned log-std
    (clipped to [−2, 0.5] so exploration never collapses early);
  - Categorical head: logits over the grasp set.
- **Algorithm**: REINFORCE with an EMA baseline,
  `∇J = E[(R − b) ∇log π(a|s)]`, Adam (lr 3·10⁻³), batch = 48 episodes.
  With one-step episodes there is no credit-assignment problem, so plain
  policy gradient is adequate; PPO/SAC would add machinery without changing
  what's learnable here.
- Gradients are hand-derived (tanh-squashed Gaussian + softmax categorical,
  shared trunk) — see `HandoffPolicy.update`.

## 5. Training

```bash
# from the repo root, venv active
python scripts/train_rl.py --episodes 200000          # fresh run (overnight)
python scripts/train_rl.py --episodes 50000 --resume  # continue any time
python scripts/train_rl.py --eval models/handoff_policy.npz --eval-n 100
```

- Checkpoints: `models/handoff_policy.npz` (every ~480 episodes and at exit).
- Log: `models/train_log.csv` (`episode, reward_mean, success_rate, elapsed_s`)
  — plot reward_mean to watch convergence.
- Throughput: ~60–80 episodes/s (each episode = one oracle evaluation;
  failures at early gates are cheapest).
- `--resume` continues from the checkpoint — this is the "continuous
  learning" loop: run it whenever the cell is idle, forever.
- Evaluation reports single-shot and **best-of-k** success. Best-of-k is the
  deployment metric: proposals are oracle-verified, so trying 8 cheap
  proposals (~0.5 s total) before falling back to search is the real usage.

Smoke-run reference (11k episodes, ~13 min CPU): random k=8 = 0% success,
policy k=1 = 0% but mean-gate 0.68, policy k=8 = **12%** verified success.
The curve was still rising; treat 11k episodes as 5% of a real run.

Two ceilings to keep in mind when reading success rates:

1. Not every sampled initial grasp admits *any* direct handoff — some need
   the regrasp branch (§8 below). The exhaustive-search success rate on the
   same grasp distribution is the true ceiling; the policy can approach it,
   not exceed it.
2. The insert pose and grasp set are placeholders; a richer `grasp_set_G`
   raises the ceiling for both search and policy.

## 6. File map

| file | role |
|---|---|
| `src/rl_env.py` | grasp sampler + one-step env (wraps the oracle) |
| `src/rl_policy.py` | numpy MLP policy + REINFORCE/Adam |
| `scripts/train_rl.py` | training loop, eval, checkpointing, CSV log |
| `tests/test_rl.py` | 8 tests: sampler validity, env determinism, reward correctness, learning sanity, save/load |
| `src/handoff.py` | the oracle gates + brute-force search (fallback) |
| `src/kin.py` | the single IK + collision path everything uses |

## 7. What happens when things change (scaling)

The key split: the **oracle adapts instantly** (it is model-based — it reads
whatever URDF/config you give it), while the **policy must be retrained**
(its weights encode the old geometry implicitly). Nothing needs re-coding in
either case, with the exceptions noted.

| change | oracle / env | policy | code changes |
|---|---|---|---|
| **New part** (mesh + insert pose in config) | automatic — sampler re-derives grasp modes from the CAD bbox; G* re-filtered | retrain (or fine-tune with `--resume`) | none, if the part is boxy and axis-aligned grasps suffice. Exotic shapes need entries added to `grasp_set_G` |
| **New gripper** (fingers/body dims, TCP offset) | automatic once URDF + config updated | retrain | update `gp7.urdf` gripper link, `TCP_OFFSET` in `rl_env.py`, and the 0.200 translations in `cell.yaml` |
| **Fixtures move** (PCB, bin, nest, insert pose) | automatic | retrain (usually quick fine-tune) | none |
| **Different robots / bases** | automatic (URDF + base transforms) | retrain | none |
| **Many parts, one policy** | supported by design | train ONE policy over randomized parts | extend the sampler to randomize part dims per episode (domain randomization) — the observation already carries part half-extents precisely so one policy can serve a part family |

The honest limits of "universal": the current grasp model is bbox-based
(parallel-jaw grasps on axis-aligned face pairs). That covers relays,
headers, and most boxy SMT/THT components. Parts needing curved-surface or
off-axis grasps require enriching `grasp_set_G` / the sampler — the oracle
and training loop are unchanged. And a policy trained on one part transfers
zero-shot only as far as its training distribution covered that geometry;
domain randomization over part dims is how you buy generalization.

## 8. Reorientation / regrasp branch (`src/regrasp.py`)

Implemented (pipeline doc §6, one placement hop). When no direct handoff
exists for how A holds the part: place it on the nest, re-pick with a
handoff-viable grasp, then run the cached direct handoff.

- **Stable placements**: the 6 axis-aligned face-down orientations of the
  boxy part on the nest plate, × 8 yaws.
- **Viability table** (`models/regrasp_table.json`): for each canonical
  re-pick grasp, whether a direct handoff exists — with the full cached plan.
  Precomputed once (`RegraspPlanner.build_table()`, resumable), because
  handoff viability of the new grasp is independent of the placement — the
  same factorization trick as G*.
- **Online search** (`find_regrasp`): find (placement, yaw, viable grasp g')
  where A can place from the current grasp AND re-pick with g' (descending
  approach, IK + collision through the same kin.py path). Typically < 1 s
  because the expensive part is all in the table.

Deployment chain (both entry points):

```
scripts/run.py                 : search -> regrasp fallback -> reject
scripts/train_rl.py --deploy   : policy (k proposals) -> search -> regrasp -> reject
```

`--gui` replays the whole regrasp sequence: place → retreat → re-pick →
present → B grasps → insert.

### Measured coverage (16 random initial grasps, `scripts/coverage.py`)

| method | solved |
|---|---|
| direct handoff only | 69% |
| direct + regrasp | **88% — current system upper bound** |

The remaining ~12%: initial grasps from which A cannot even place the part
on the nest without collision. Paths to push toward 100%: complete the
viability table (2 of 12 modes unmeasured), add roll-180 canonical grasps,
more placement yaws, a pre-place retreat/approach search, or a second
placement hop. Re-run `python scripts/coverage.py --n 50` after any such
change to re-measure.
