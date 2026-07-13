# MuJoCo handoff code architecture and performance workflow

This document is the code map for debugging and optimization. The mathematical
method remains in
[handoff_pipeline_detailed.md](handoff_pipeline_detailed.md); the operator
commands remain in [mujoco_user_guide.md](mujoco_user_guide.md).

## 1. Package layout and dependency direction

Implementation code is grouped by responsibility. Files directly under
`mujoco_sim/` are intentionally tiny compatibility aliases or established
`python -m` launchers; new algorithm code belongs in one of these packages:

```text
mujoco_sim/
├── core/           SE(3), uncertainty, stable paths, profiling
├── modeling/       project manifest, CAD, mesh, gripper, grasps, placements
├── simulation/     MuJoCo state, GP7 kinematics, collision/contact policy
├── planner/        planner facade, contracts, motion, task graph, stages
│   └── stages/     direct, downstream, and reorientation searches
├── execution/      executor state machine, schedule, and timing model
├── offline_tools/  artifact cache, precomputation, and qualification
├── diagnostics/    debug artifacts and contact-path audit
├── apps/           pipeline/viewer/visualization implementations
├── experiments/    qualification experiments
├── models/         compiled MJCF and generated CAD
└── *.yaml          stable user/system configuration locations
```

The dependency direction is kept acyclic:

```text
core -> offline artifact primitives -> modeling -> simulation
                                            |          |
                                            +----+-----+
                                                 v
                                          planner/planner.py
                                                 |
                                                 v
                                      execution/executor.py
                                                 |
                                                 v
                                      apps and diagnostics

project/CAD/scene
       |
       v
HandoffPlanner (`planner/planner.py`; legacy alias `planning.py`)
       |
       +-- plan records and validation (`planner/types.py`, `validation.py`)
       +-- phase contact policy (`simulation/contact_policies.py`)
       +-- deterministic IK + collision/motion runtime
       +-- downstream certification (`planner/stages/downstream.py`)
       +-- direct search ordering (`planner/stages/direct.py`)
       +-- reorientation task graph (`planner/stages/reorientation.py`)
       |
       v
PipelineExecutor (`execution/executor.py`; legacy alias `exec.py`)
       +-- geometric trajectory timing (`execution/timing.py`)
       +-- resource/dependency schedule (`execution/schedule.py`)
       +-- transactional state events
       +-- continuous collision monitoring
       +-- debug artifact recorder
```

Legacy imports such as `mujoco_sim.planning`, `mujoco_sim.sim`, and
`mujoco_sim.se3` resolve to the exact canonical module objects, preserving
class identity and existing monkeypatch/integration hooks. The five documented
root launchers delegate to `apps/` or `diagnostics/` while retaining their
existing command lines. New code should import canonical package modules.

## 2. Planning stages

1. Project/scene initialization loads prepared SI-unit CAD, the grasp library,
   reachability maps, kinematic limits, and the collision runtime.
2. Downstream certification starts from each Robot B surface grasp and verifies
   scanner IK, every pre-insertion/insertion IK, correction-envelope
   controllability, and collision-free paths with Robot A parked.
3. Direct search enumerates the user handoff region. A fast branch-continuous
   pass runs first; the complete deterministic multi-start pass remains the
   fallback. Candidate gates check grasp-patch compatibility, reachability, IK,
   joint/manipulability margin, simultaneous co-grasp collision/clearance, and
   every approach/retreat/scanner/park path.
4. Reorientation search runs only after direct failure in production. It works
   backward from insertion-valid direct grasps, connects stable support poses
   to place/re-pick motions, and solves the bounded task graph.
5. Plan validation checks every cached/new transform, six-joint witness,
   required trajectory name, and trajectory endpoint before execution.

All IK and collision calls mutate the shared MuJoCo `MjData`. A planner and
executor must therefore share exactly one `WorkcellSim`; independent parallel
candidate evaluation requires independent simulation states or an explicitly
synchronized state pool.

## 3. Execution stages

Transactional state changes and performance stages are deliberately separate.
`ExecutionEvent` answers “who owns the part and which safety state completed?”
The hierarchical profile answers “where did computer wall time go?” The robot
operation graph answers “what is on the modeled cycle-time critical path?”

The current certified sequence is serial:

```text
A transit/approach -> B transit/approach -> capture/transfer
-> A retreat -> B scanner -> A park -> B pre-insert -> insertion
```

The reorientation branch prepends A place, release, re-pick, and capture.
Gripper and scanner operations without calibrated duration are explicitly
listed as unmodeled; they are never silently treated as a complete hardware
cycle estimate.

## 4. Bottleneck profiling

Run:

```bash
python -m mujoco_sim.pipeline --execute --profile
```

The report separates:

- scene/model and planner setup;
- inclusive and exclusive planning-stage wall time;
- cache access, IK, direct-edge checking, and RRT fallback;
- explicit execution-stage simulation/collision time;
- viewer synchronization, playback pacing, and diagnostic I/O; and
- modeled robot-operation makespan, resource busy time, and critical path.

Optimize by **exclusive** wall time, not the inclusive parent alone. On a warm
query, setup may dominate planning. During headless replay, dense MuJoCo
collision monitoring may dominate computer wall time even though it does not
increase modeled robot time.

## 5. Timing model and parallel-arm work

`JointVelocityTimingModel` integrates normalized joint travel along the
geometric path with the cubic-smoothstep peak-velocity factor. Collision
waypoints are safety samples, not separate commands; collinearly densifying a
path therefore leaves its modeled duration unchanged.

The model is not yet hardware minimum-time retiming. It omits acceleration,
jerk, controller blending, payload derating, settling, and measured device/PLC
latency. Add those to a new versioned timing model rather than changing the
meaning of the current version.

The operation scheduler supports dependencies and exclusive resources. Current
plans are scheduled serially because their collision certificates assume the
other arm at a fixed witness. Parallel A/B operations are fail-closed: both
operations must reference the same coordinated dual-arm collision certificate.
The safe optimization sequence is:

1. choose candidate stages that can overlap;
2. jointly parameterize both arm paths in time;
3. collision-check the full `(q_A(t), q_B(t), X_P(t))` trajectory, including
   stopping-distance margin;
4. attach the resulting certificate to both operations; and
5. compare scheduled makespan, not summed arm work.

For least-time motion, replace per-edge smoothstep with a controller-matched
velocity/acceleration/jerk retimer and then optimize the coordinated critical
path. Never remove collision samples to make the visualization or reported
cycle faster.

## 6. Test tiers

The repository keeps direct executable tests; `pytest` is optional.

```bash
python scripts/run_mujoco_tests.py --tier t1   # pure deterministic unit tests
python scripts/run_mujoco_tests.py --tier t2   # headless scene integration
python scripts/run_mujoco_tests.py --tier t3   # direct + stage end-to-end gate
python scripts/run_mujoco_tests.py --tier all
```

- T1 covers math, geometry, cache primitives, stage ordering, fail-closed
  qualification, plan validation, profiling, timing, and scheduling.
- T2 covers the compiled scene, gripper bindings, collision semantics, planner
  component integration, CLI paths, executor aborts, and debug artifacts.
- T3 requires both the direct and forced-stage reference policies to plan and
  execute to `COMPLETE`; infeasibility is a test failure.

Full cold-cache domain qualification and hardware validation remain release
activities. Hardware validation must add controller timestamps, force/contact
limits, gripper/scanner/PLC latency, and stopping-distance trials.

## 7. Where to make common improvements

| Goal | Primary module | Required regression |
|---|---|---|
| Faster IK | `simulation/kinematics.py` / planning runtime | target-keyed determinism and FK residual |
| Better single-arm path | `planner/motion.py` | exact phase collision replay and endpoints |
| Faster direct search | `planner/stages/direct.py` | warm/exhaustive completeness and candidate order |
| Better insertion grasp | `planner/stages/downstream.py` | all targets, correction envelope, parked-A state |
| Better stage strategy | `planner/stages/reorientation.py` | stable support, place/re-pick, terminal direct edge |
| Parallel arms | `execution/schedule.py` plus coordinated planner | joint time-indexed collision certificate |
| Minimum-time trajectory | `execution/timing.py` | speed/accel/jerk bounds and density invariance |
| Runtime visualization | `execution/executor.py` viewer/pacing spans | identical operation graph and safety outcome |
