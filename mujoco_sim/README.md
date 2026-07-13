# MuJoCo source map

Edit implementation code inside the responsibility-based packages:

| Package | Responsibility |
|---|---|
| `core/` | SE(3), uncertainty, stable paths, profiling |
| `modeling/` | Project schema, CAD, grippers, meshes, grasps, placements |
| `simulation/` | MuJoCo state, GP7 kinematics, collision/contact policies |
| `planner/` | Planner facade, plan contracts, motion, task graph, stages |
| `execution/` | State machine, trajectory timing, operation scheduling |
| `offline_tools/` | Cache artifacts, precomputation, qualification |
| `diagnostics/` | Contact audits and per-stage debug artifacts |
| `apps/` | Pipeline, viewer, and visualization implementations |
| `experiments/` | Focused handoff experiments |

The short Python files immediately under `mujoco_sim/` are compatibility
aliases and stable `python -m` launchers. They contain no algorithm logic and
should generally not be edited. Configuration YAML, `models/`, and `cache/`
remain at the package root so the reorganization does not change existing
projects, compiled MJCF paths, or artifact fingerprints.

See [the architecture guide](../docs/mujoco_architecture.md) for dependency
rules, profiling, timing, and regression-test tiers.
