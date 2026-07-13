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
| `config/` | Active configuration, templates, and deprecated tombstones |

Only five short `python -m` launchers remain directly under `mujoco_sim/`;
they contain no algorithm logic. Active YAML lives under `config/`, while
`models/` and `cache/` retain their stable locations. See
[the configuration map](config/README.md) before adding a new setting.

See [the architecture guide](../docs/mujoco_architecture.md) for dependency
rules, profiling, timing, and regression-test tiers.
