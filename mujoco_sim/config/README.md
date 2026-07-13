# MuJoCo configuration map

| Path | Owner | Purpose |
|---|---|---|
| `project.yaml` | Cell integrator | Robots, CAD, frames, task regions, insertion target |
| `solver_defaults.yaml` | System developer | Numerical, safety, search, and execution defaults |
| `internal/scene_fallback.yaml` | System developer | Temporary photo-matched fixture primitives used when CAD is absent |
| `templates/gripper_asset.template.yaml` | Integrator template | Contract for a supplied articulated gripper model |
| `deprecated/` | Migration only | Files that are not read by the current pipeline |

Only `project.yaml` is the normal per-cell user input. Keep part-specific facts
out of `solver_defaults.yaml`; keep numerical search tuning out of
`project.yaml`. Relative CAD references in a repository-owned project are
resolved from the repository root. External project manifests continue to
resolve their relative assets beside the external manifest first.
