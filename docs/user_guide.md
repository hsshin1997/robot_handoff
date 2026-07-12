# Handoff Simulation — User Guide

> **Legacy PyBullet guide.** This document describes the earlier prototype and
> is not the current MuJoCo workflow. Use
> [mujoco_handoff_pipeline.md](mujoco_handoff_pipeline.md) and
> [mujoco_setup.md](mujoco_setup.md) for the implemented system.

Everything needed to configure, run, test, and extend the dual-GP7 part
handoff simulation. Companion docs: `handoff_pipeline_detailed.md` (the math),
`rl_training.md` (the optional learned proposer).

---

## 1. What the system does

Given how Robot A is holding a part, it answers: **can the part be handed to
Robot B such that B can insert it on the PCB — and with which poses?**

```
INPUT                       PIPELINE                                OUTPUT
T_flangeA_part   ->   plan cache -> budgeted search -> regrasp  ->  executable plan
(how A holds          (verify old   (grid over        (place on     or explicit
 the part)             plans ~0.1s)  pose x grasp)     nest,        rejection
                                                       re-pick)
```

Every stage runs through one shared oracle (IK + joint limits + collision in
`src/kin.py`); nothing is ever returned unverified.

## 2. Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # pybullet, numpy, pyyaml
python scripts/run.py                  # smoke check: loads cell, plans, prints
```

Runs headless (`p.DIRECT`) everywhere including macOS; `--gui` opens the
viewer. Linux with an NVIDIA driver gives the smoothest GUI.

## 3. Frames and units — read this first

- **World frame** = the workcell STL frame in meters (mm scaled by 0.001).
  Robot A's mount flange is the world origin, z up. Cell floor at z = −0.610.
- **Robot bases**: A at (0,0,0) yaw 0 (faces +x), B at (0.850,0,0) yaw 180°
  (faces −x, toward A).
- **Flange** = the `tool0` link. **TCP** = center between the fingertips,
  **200 mm along tool0 +x** (see the gripper in `assets/gp7/gp7.urdf`).
- **Part frame** = bounding-box center of the part mesh (the loader recenters
  automatically).
- All transforms in config are **4×4 row-major homogeneous matrices, meters**.
- Grasp conventions:
  - `T_flangeA_part` — part pose in A's flange frame ("how A holds it").
  - `grasp_set_G` entries — B's **flange pose in the part frame**
    (`T_part_flangeB`); flange x = approach direction, TCP lands on the part
    origin (translation = −0.200 × approach axis).

## 4. Configuring the cell (`config/cell.yaml`)

### 4.1 Robots

```yaml
robotA_base / robotB_base   # 4x4 world <- base; from cell calibration
home_qA / home_qB           # 6 joint angles (rad); must be mutually
                            # collision-free — verify with tests after changing
```

### 4.2 The part

```yaml
part_mesh: "parts/<name>/<file>.stl"   # BINARY STL, meters
part_half_extents: [x, y, z]           # box fallback if part_mesh is absent
```

pybullet's STL loader cannot read ASCII STL (it segfaults). Convert once if
needed — see `parts/conn_header/` for an example of an original + converted
pair. Collision uses the convex hull of the mesh.

### 4.3 The initial grasp (the per-cycle input)

```yaml
T_flangeA_part:              # part in A's flange frame
  - [1.0, 0.0, 0.0, 0.200]   # identity rotation, part COG at the TCP
  - [0.0, 1.0, 0.0, 0.0]
  - [0.0, 0.0, 1.0, 0.0]
  - [0.0, 0.0, 0.0, 1.0]
```

In production this comes from the picker/measurement per cycle; in the sim
you set it here (or pass it programmatically, §7). A commented "regrasp demo"
variant sits next to it in the file — a grasp that provably has no direct
handoff, useful for watching the reorientation recovery in `--gui`.

### 4.4 Fixtures

```yaml
T_world_pcb + pcb_half_extents   # PCB box; a stand to the floor is auto-added
T_world_insert                   # PART pose at insertion (world). B's flange
                                 # target is derived as T_world_insert @ g.
                                 # Keep it consistent with the PCB top +
                                 # part half height + small hover.
bin:  center_xy / interior / wall_height / wall_thickness / interior_floor_z
nest: center_xy / size / top_z   # the reorientation plate (regrasp branch)
```

Placement rule of thumb: keep bin/nest clear of the scanner head overhang
(y > 0.44, z 0.76–1.04 in this cell) or top-down approaches jam the wrist.

### 4.5 Grasp set for B

Eight generated entries (side/top/bottom × 2 rolls). Add entries to raise the
feasibility ceiling; each is just a `T_part_flangeB` matrix + name. The
search filters them to G* (insert-feasible) automatically at runtime.

### 4.6 Search parameters

```yaml
handoff_search:
  x / y / z: [min, max, step]     # part-position grid between the arms
  orientations_rpy_deg: [...]     # part orientations tried at handoff
  joint_limit_margin: 0.09        # rad, distance kept from joint limits
  ik_restarts: 8                  # IK seed diversity per target
  n_path_waypoints: 4             # collision checks along handoff -> insert
```

The grid must cover the dual-reach region: A's flange sits 0.2 m behind the
part along its approach (too close to A's base folds the arm), and for top
grasps B's flange is 0.2 m above the part (so part z + 0.2 must stay inside
B's 0.927 m reach).

## 5. Inputs and outputs, precisely

### Inputs

| when | what | where |
|---|---|---|
| per cycle | `T_flangeA_part` | config, or `planner.T_fA_part` in code |
| per product | part mesh, `grasp_set_G`, `T_world_insert` | config |
| per cell | base transforms, fixtures, home poses | config |

### Outputs

`python scripts/run.py` prints one of:

1. **Direct handoff plan**
   - `X_h` — 4×4 world part pose at the transfer instant
   - grasp name (which G entry B uses)
   - `qA` (A presents), `qB_grasp` (B takes), `qB_insert` (B at insert),
     plus checked intermediate waypoints
   - margin score and estimated physical execution time (s)
2. **Regrasp plan** — placement name + `X_place` (part pose on the nest),
   `qA_place`, `qA_pick`, the new grasp, then the full handoff plan as above
3. **Rejection** — no plan exists; with `--thorough` you also get per-gate
   failure counts telling you *which* constraint dominated (e.g.
   `gate1_A_presents` → widen the grid or A can't present; `gate2_B_takes` →
   co-grasp geometry; empty G* → no grasp can insert).

All joint vectors are 6 values, radians, ordered S L U R B T.

## 6. Running

```bash
python scripts/run.py                # fast pipeline: cache -> 1.5s search -> regrasp
python scripts/run.py --gui          # same + animated replay (loops)
python scripts/run.py --thorough     # exhaustive search (definitive feasibility)
python scripts/run.py --best        # exhaustive, maximize safety margins
python scripts/run.py --fastest      # exhaustive, minimize physical execution time
python scripts/run.py --no-regrasp   # disable the reorientation fallback
python scripts/run.py --no-search    # just load the scene (inspection)
python scripts/render_check.py OUT   # headless camera snapshots (pip install pillow)
```

Typical timings (this cell): cache hit ≈ 0.4 s, fast pipeline worst case
≈ 2.5 s, exhaustive sweep 10–60 s. Estimated *physical* handoff execution:
≈ 1 s direct, ≈ 2.3 s with reorientation (kinematic estimate at 70% of GP7
rated axis speeds — relative metric for comparing plans, not a controller
trajectory time).

## 7. Multiple runs / batch use (programmatic API)

```python
import sys; sys.path.insert(0, "src")
import numpy as np, kin
from scene import Scene
from handoff import HandoffPlanner, PlanCache
from regrasp import RegraspPlanner

scene = Scene()                        # load ONCE, reuse across queries
pl = HandoffPlanner(scene)
rg = RegraspPlanner(scene, pl)

for T in my_grasp_list:                # each T = 4x4 T_flangeA_part
    pl.T_fA_part = T
    pl.T_part_fA = kin.inv_T(T)
    kind, plan, timings = pl.plan_fast(regrasp_planner=rg)
    # kind: 'cache' | 'search' | 'regrasp' | None (reject)
```

Ready-made batch scripts:

```bash
python scripts/coverage.py --n 50            # feasibility rate over random grasps
python scripts/coverage.py --n 50 --resume   # continue an interrupted study
python scripts/build_cache.py                # pre-solve canonical grasps (~15 min)
```

Both are resumable (results accumulate in `models/coverage.jsonl` /
`models/plan_cache.json`). Measured coverage with the relay: 71% direct,
**100% with regrasp** over 24 random pick-jittered grasps.

## 8. The caches (`models/`)

| file | contents | rebuild when |
|---|---|---|
| `regrasp_table.json` | per canonical re-pick grasp: is a direct handoff possible + the full plan | gripper, grasp set, insert pose, bases, or part *shape class* changes |
| `plan_cache.json` | every solved handoff: initial grasp → (X_h, grasp); warm-starts future queries | same triggers |
| `coverage.jsonl` | batch study results | whenever you want a fresh study |
| `handoff_policy.npz`, `train_log.csv` | RL artifacts (optional path) | see rl_training.md |

Rebuild = delete the file; tables regenerate automatically (regrasp table on
first use, ~5–10 min; plan cache grows per solved query or via
`build_cache.py`). Caches are *verified on use* (a stale cached plan fails
verification and falls through), so a stale cache costs speed, not
correctness — except the regrasp table's plans, which are trusted as-is:
delete it on any geometry change.

## 9. Testing

```bash
python tests/test_kin.py       # 11: FK/IK round-trips, limits, collision +/-
python tests/test_handoff.py   # 5:  oracle gates, G*, known-good/known-bad, search
python tests/test_regrasp.py   # 4:  placements, viability table, full regrasp plan
python tests/test_rl.py        # 8:  RL env/policy (optional path)
```

No pytest needed (plain `python`), pytest-compatible if you have it. The
suites pin their own nominal grasp, so they pass regardless of what
`T_flangeA_part` you're currently experimenting with in the config.
Run all four after ANY config/URDF change — they catch geometry regressions
(they've caught real ones: a nest under the scanner, finger-finger
interference at co-grasp).

## 10. Changing hardware

| change | steps |
|---|---|
| new part | drop binary-STL (meters) in `parts/`, update `part_mesh`, `T_world_insert`; delete caches; run tests |
| new gripper | edit the gripper link + `tcp` joint in `gp7.urdf`; update the 0.200 offsets in config and `TCP_OFFSET` in `src/rl_env.py`; delete caches; run tests |
| moved fixtures | update config poses; delete caches; possibly retune `handoff_search` grid; run tests |
| different robots | new URDF + base transforms; check `reach_min/max` in `src/handoff.py`; delete caches; run tests |

## 11. Troubleshooting

- **"no feasible handoff" everywhere** → run `--thorough` and read the
  failure stats. `gate1_A_presents` dominating = search grid doesn't fit A's
  comfortable workspace; `gate2_B_takes` = co-grasp geometry (finger
  interference, B reach); empty `G*` = no grasp can reach the insert pose.
- **Search suddenly slow** → usually a grasp for which many poses pass gate 1
  and grind gate-2 IK; use `plan_fast` (budgeted) or pass
  `search(time_budget=...)`.
- **Part floats oddly in renders** → part mesh is ASCII STL or not in meters.
- **GUI laggy on macOS** → point `workcell_visual` at the decimated
  `workcell_viz.stl`; it's a pybullet viewer limitation, not the planner.
- **Changed something and results look wrong** → delete `models/*.json`,
  rerun tests, then re-plan.
