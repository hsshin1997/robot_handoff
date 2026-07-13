# MuJoCo Contact-Dynamics Layer — Implementation Plan

> **Superseded historical plan.** The implemented architecture and its current
> physical-certification limits are documented in
> [mujoco_handoff_pipeline.md](mujoco_handoff_pipeline.md) and
> [handoff_pipeline_detailed.md](handoff_pipeline_detailed.md). In particular,
> the present static gripper and virtual-aperture PCB do not provide the contact-dynamics
> validation envisioned below.

Goal: add a contact-rich dynamic simulation that VALIDATES and EXECUTES the
plans produced by the existing kinematic planner — not a rewrite of it.

```
  PyBullet layer (exists, unchanged)          MuJoCo layer (new)
  ─────────────────────────────────           ─────────────────────────────
  finds (X_h, g, qA, qB, segments)   ──────▶  executes segments with real
  geometric feasibility, regrasp,             contact: grasp stability,
  caches, coverage                            co-grasp transfer, insertion
```

Division of labor: geometry questions ("does a feasible handoff exist?")
stay in PyBullet where they're already solved and fast. Dynamics questions
("does the part slip at speed? does the transfer survive 1 mm misalignment?
does the pin find the hole?") go to MuJoCo.

## What the dynamic sim must answer (scope contract)

1. **Grasp transport** — part held only by friction between actuated
   fingers: does it stay put through the planned segments at target speeds?
   Output: max safe speed fraction / required grip force per plan.
2. **Co-grasp transfer** — B closes while A holds; A opens and retreats.
   Inject gripper-to-gripper misalignment ε and measure success rate vs ε.
   This measures the doc's §2/§5 capture region empirically and produces the
   real tolerance budget for robot-to-robot calibration.
3. **Insertion** — pins into PCB holes with chamfers, under position error,
   with a compliant controller + search policy (spiral/dither). Output:
   success rate vs initial error → the accuracy the upstream handoff must
   deliver.
4. *(optional, phase 5)* Train a residual force-guided insertion policy
   (this is where RL genuinely belongs — unknown contact dynamics).

Known limitation, stated upfront: real PCB clearances (50–100 µm) are at the
edge of rigid-body contact simulation. Results validate *strategies and
relative tolerances*, not absolute forces. Mitigation: progressive clearance
tightening (start 0.5 mm, tighten to 0.1 mm), elliptic friction cones, small
timesteps (≤ 0.5 ms), `noslip` iterations, and domain randomization on
friction/clearance before trusting any conclusion.

## Phase 1 — Models (MJCF)

New directory `mujoco_sim/` (the PyBullet package is untouched):

```
mujoco_sim/
  models/
    gp7.xml            # arm converted from the URDF (MuJoCo reads URDF, but
                       #   hand-tuned MJCF: joint damping/armature, actuator
                       #   gains ~ GP7 rated speeds/torques)
    gripper.xml        # ACTUATED parallel jaw: 2 prismatic fingers,
                       #   force-limited position actuators, high-friction
                       #   pads (condim 4-6), same 24 mm stroke / 200 mm TCP
    part_header.xml    # header: body as boxes, PINS AS CAPSULES (primitive
                       #   contacts, never fine meshes at this scale)
    pcb.xml            # board + hole array: each hole = 4 chamfered boxes
                       #   forming a funnel + through pocket; clearance
                       #   parameterized for progressive tightening
    cell.xml           # static geoms ported from collision_boxes.yaml + nest
    scene.xml          # includes: 2x gp7+gripper at the calibrated bases
  sim.py               # model loading, stepping, state/contact readout
  exec.py              # segment executor: HandoffPlan/RegraspPlan ->
                       #   time-parameterized joint trajectories (trapezoid,
                       #   GP7 speed caps) -> position servo tracking;
                       #   grip/release events; slip + wrench monitoring
  experiments/
    transport_speed.py     # question 1
    cograsp_tolerance.py   # question 2 (Monte Carlo over ε)
    insertion_funnel.py    # question 3 (success vs initial error)
  tests/test_mujoco_exec.py
```

Model-building order and acceptance checks:
1. One arm + gripper, gravity on: holds home pose, tracks a joint step
   without oscillation (tune gains). Fingers close on a fixed block and hold
   it against gravity by friction alone.
2. Part model: dropped on the nest from 5 mm, settles in < 0.5 s without
   jitter or penetration (contact params healthy at part scale).
3. Both arms in the cell; forward-play a cached plan's segments open-loop;
   no unexpected contacts (cross-check vs the kinematic checker — the two
   sims must agree on clearance, this validates the port).
4. PCB hole: scripted straight-down insertion from perfect alignment
   succeeds at 0.5 mm clearance; measure the tolerance funnel.

## Phase 2 — Execution layer (`exec.py`)

- Loader for plans: read a solved `HandoffPlan`/`RegraspPlan` (planner
  output or the JSON caches — the `segments` dict is the interface; nothing
  in the planner changes).
- Time parameterization: trapezoidal profiles per segment at a configurable
  fraction of GP7 rated speeds (replaces the kinematic `phase_time` estimate
  with actual tracked motion).
- Event sequence: A grip (already holding) → run A_approach/B_approach →
  B grip (force-limited close) → dwell → A open → A_retreat → transit →
  insert descent [→ compliant search if contact].
- Monitors: part pose in gripper frame (slip), finger forces, part-world
  wrench, contact lists per step. Success predicates per experiment.

## Phase 3 — Validation experiments

Each experiment consumes cached plans (so PyBullet and MuJoCo run the SAME
handoffs) and writes JSONL results like `coverage.py` does:

1. `transport_speed.py`: sweep speed fraction 0.2→1.0; report slip onset.
   → sets the real speed derating (currently a guessed 0.7).
2. `cograsp_tolerance.py`: inject ε ~ N(0, σ) in gripper-to-gripper offset
   at the transfer; N≈200 trials; success vs |ε| curve.
   → the empirical capture region; feeds the calibration budget (doc: keep
   robot-to-robot under ~0.5 mm — now measurable instead of assumed).
3. `insertion_funnel.py`: initial lateral/angular error sweep × clearance
   levels × with/without spiral search; success-rate heatmap.
   → the accuracy contract the handoff+scan pipeline must meet.

## Phase 4 — Acceptance gate for hardware

A plan is "hardware-ready" when: it exists (PyBullet), AND transports
without slip at the chosen speed (exp 1), AND transfers at 3σ of the
measured cell calibration error (exp 2), AND inserts from the expected
post-scan accuracy (exp 3). That chain — plan → dynamic validation →
tolerance budgets — is the actual deliverable of this layer.

## Phase 5 (optional) — insertion RL

Residual policy (impedance controller + learned Δ-action), trained in
MuJoCo with domain randomization (friction, clearance, initial error),
observation = wrench + pose estimate. Deploy as the force-guided search in
§5 of the pipeline doc. Only start this if the scripted spiral search from
experiment 3 proves insufficient.

## Dependencies & effort

- `pip install mujoco` (DeepMind's bindings; excellent native Apple Silicon
  support — better than PyBullet's on macOS). Optional `mujoco.viewer` for
  interactive inspection. No other new deps.
- Rough effort: Phase 1 ≈ 2–4 days (the PCB hole + contact tuning is most of
  it), Phase 2 ≈ 2 days, Phase 3 ≈ 2–3 days, Phase 5 open-ended.

## What stays in PyBullet (explicitly)

Planning, search, regrasp logic, caches, coverage studies, all existing
tests. If MuJoCo experiments reveal a systematic problem (e.g. a grasp mode
that always slips), the fix flows back as a config change (grasp set,
speeds, margins) — the planner code itself shouldn't need to change.
