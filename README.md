# handoff-sim

> **Current implementation:** the production-oriented path is the MuJoCo
> pipeline documented in [docs/mujoco_handoff_pipeline.md](docs/mujoco_handoff_pipeline.md)
> and launched via `python -m mujoco_sim.pipeline`. The PyBullet material below
> is retained as a legacy prototype and does not describe the current planner,
> configuration contract, collision policy, or certification status.

Part handoff simulation between two Yaskawa GP7 arms (PyBullet, no ROS).

## Setup

Requires Python 3.11+ (macOS Apple Silicon works; pybullet builds from source on first install).

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Two independent ways to solve a handoff — both use the same oracle
(`src/kin.py` IK + collision, `src/handoff.py` gates), so results are
directly comparable:

### 1. Search (no learning — the baseline method)

Brute-force grid over (handoff pose x grasp), first-feasible or best-scored.
Deterministic, no training, ~10 s per query:

```
python scripts/run.py          # search + print the (X_h, g, qA, qB_grasp, qB_insert) tuple
python scripts/run.py --best   # scan all candidates, keep the highest margin score
python scripts/run.py --gui    # search, then replay: A presents -> B grasps -> B inserts
python scripts/run.py --no-search  # just load the scene (visual inspection)
```

### 2. Learned proposer (RL, optional)

A policy trained to propose handoff parameters from the initial grasp;
every proposal is verified by the same oracle, with search as fallback.
See `docs/rl_training.md`:

```
python scripts/train_rl.py --episodes 200000            # train
python scripts/train_rl.py --eval models/handoff_policy.npz  # evaluate
```

### Utilities

```
python scripts/render_check.py # save camera snapshots (needs: pip install pillow)
python tests/test_kin.py       # kinematics/collision suite
python tests/test_handoff.py   # oracle + search suite
python tests/test_rl.py        # RL env/policy suite
```

Deactivate with `deactivate`.

### Linux

Same code, no changes. Copy the whole folder (assets included), then:

```
sudo apt install -y python3-venv        # if not already present
cd handoff
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt         # x86_64: prebuilt wheel, no compile
python scripts/run.py --gui
```

Notes:

- For a smooth GUI on an NVIDIA card, make sure the proprietary driver is
  active (`nvidia-smi` works). The stock open-source stack is fine too, just
  slower.
- The GUI needs a display (X11/Wayland). Headless servers can only use
  `p.DIRECT` + `render_check.py`.

### GUI performance

Everything visual is decimated: workcell 724k -> 35k tris (`workcell_viz.stl`),
GP7 visual meshes capped at 4k tris/link, gripper 3k (~24k/robot). Whole scene
is ~85k triangles. `scene.py` also disables shadows, preview panes, and mouse
picking, pauses rendering during load, and uses a fixed 1280x800 window.
Original robot meshes are recoverable from the ROS-Industrial motoman repo.

If the GUI still lags after all this, it's pybullet's macOS viewer itself
(single-threaded OpenGL, poor Apple Silicon support) — run on Linux for a
smooth GUI, or use headless snapshots (`render_check.py`).

## Frames

World frame = workcell STL frame in meters (mm scaled by 0.001):

- Robot A (left/picker) mount flange = world origin, yaw 0 — faces +X at home.
- Robot B (back/inserter) at (0.850, 0, 0), yaw 180° — faces -X, toward A.
- Both pedestal tops at z = 0; cell floor at z = -0.610.

Base poses live in `config/cell.yaml` as 4x4 transforms.

## Workcell collision

The GUI/renders use `workcell_viz.stl`, a decimated 35k-triangle copy of the
raw 724k-triangle STL (kept as `workcell.stl`, unused at runtime). Both are
visual-only. Collision geometry is the simplified
box set in `assets/workcell/collision_boxes.yaml` (2 pedestals + 28 boxes for
the extrusion frame, scanner heads, and plates), auto-extracted from the STL's
connected components.

## Layout

```
config/cell.yaml     cell geometry + (later) grasps, insert pose
assets/gp7/          GP7 URDF + meshes (+ gripper, tool0/tcp frames)
assets/workcell/     raw STL (visual) + collision_boxes.yaml
src/scene.py         world loading, robots, part attach/detach
scripts/run.py       launcher
scripts/render_check.py  headless camera renders
```
