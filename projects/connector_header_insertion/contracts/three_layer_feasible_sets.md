# Three-layer insertion feasible-set contract

This contract replaces a dense world-pose lookup table with bounded sets,
proof obligations, and reusable witnesses. All transforms use the repository
convention: `T_X_Y` maps coordinates from frame `Y` into frame `X`, and all
stored physical values use SI units.

## 1. Robot-independent task set

For contact mode `m`, define the authored continuous parameter domain

```text
Theta_m = U_m x V_m x S1
theta   = (u, v, roll)
```

`u` and `v` locate the grasp centre on an authored pair of opposing connector
housing patches. `roll` rotates the gripper by the right-hand rule around the
mode's nominal closing axis. Each mode stores a constructive map
`Phi_m(theta)` that returns `T_P_E` and a nominal aperture. Its positive-roll
quadrature is derived as `closing_axis_P x roll_zero_approach_axis_P`, so roll
sign is independent of the handedness used for the patch-coordinate axes.

For insertion progress `s in [0, 1]`, with pre-insert distance `d`,
connector-frame insertion direction `a_P`, and `a_B = R_B_P a_P`, the nominal
part path is

```text
R_B_P(s) = R_B_P_insert
t_B_P(s) = t_B_P_insert - (1 - s) d a_B
```

Let `c_i(theta, s) <= 0` denote every robot-independent condition: housing
contact, aperture and stroke, full pad capture, gripper/part separation away
from intended contacts, gripper/PCB/fixture clearance, wrench capacity, and
the declared uncertainty envelope. The ideal task-feasible set inside the
authored contact-mode domain is

```text
F_task = {theta in union(Theta_m) |
          for every s in [0,1], every c_i(theta,s) <= 0}
```

The implementation covers each `Theta_m` with bounded cells `C_j` and uses
three-valued logic:

- `SAFE`: every declared constraint is proved for every `theta` in `C_j` and
  every `s` in `[0,1]`;
- `REJECTED`: a necessary condition is proved impossible for every `theta` in
  `C_j`; and
- `UNRESOLVED`: neither whole-cell statement has been proved.

This creates an inner and outer approximation inside the authored modes:

```text
F_task_inner = union(SAFE cells)
F_task_outer = union(SAFE and UNRESOLVED cells)
F_task_inner subseteq F_task subseteq F_task_outer
```

A sampled grasp, sampled collision, or sampled no-collision observation is a
witness about one point only. It never promotes a complete cell to `SAFE` or
`REJECTED`. Every cell also stores a constructive `center_pose`; this is a
query/visualization seed and is not a certificate for the cell.

Optional whole-cell certificates are accepted only through the layer-1
certificate interface. A certificate must be explicitly SHA-pinned, bind the
base artifact semantic hash and connector/project identity, and prove the full
required constraint list. No certificates are supplied in the current config.

Current artifact: `generated/sets/insertion_task_set.json`.

- 2,304 total cells;
- 0 `SAFE`;
- 1,344 analytically `REJECTED` because a necessary pad-footprint or aperture
  condition is impossible throughout the cell;
- 960 `UNRESOLVED` outer cells; and
- 14 positive finite-PCB vertex-penetration witnesses among 24 bounded sampled
  witness evaluations.

## 2. Target- and robot-conditioned set

For a measured seated target `T_W_P_insert`, robot `r`, and task parameter
`theta`, define the desired contact-frame path

```text
T_W_E(theta, s) = T_W_P(s) Phi_m(theta)
```

The ideal robot-conditioned set contains task-feasible parameters for which a
continuous joint path exists on one IK branch:

```text
F_robot(r, T_W_P) = {
  theta in F_task |
  there exists continuous q(s) such that
  FK_r(q(s)) = T_W_E(theta,s),
  joint limits, singularity margins, and all swept collisions hold
}
```

The numerical checker evaluates constructive cell-center poses from selected
layer-1 cells at nine path samples. It retains a branch only when numerical IK
is available at every sample and the configured joint-step, joint-limit,
singularity, and FK-error margins pass. The example selection is stratified
across contact modes and `(u,v,roll)` indices rather than taking an arbitrary
ID prefix. Its maximin metric treats `u` and `v` linearly and roll on `S1`,
using the shorter distance across the periodic seam.

These are path witnesses, not `F_robot`: numerical IK is incomplete, the
checker does not prove between-sample continuity, check the workcell swept
collision, or cover every point in a task cell. Numerical failure is therefore
reported as `NO_WITNESS...`, never as proof that the cell is robot-infeasible.

Optional robot-cell certificates use a separate strict import interface. They
must be SHA-pinned and bind the exact task artifact, target, robot, world/TCP
calibrations, and all continuous collision/path hard gates. No such
certificates are supplied for the current result.

The target is supplied either as `T_W_P_insert` directly or as

```text
T_W_P_insert = T_W_B T_B_P_insert
```

so the set is recomputed for the measured board pose instead of indexing a
world-pose database.

Current artifact: `generated/sets/robot_insertion_set.json`.

- 24 of 960 unresolved-cell centers evaluated;
- 12 provisional same-branch center-path witnesses;
- 12 numerical no-witness results; and
- 0 certified receiver cells.

## 3. Handoff preimage

Let `R` be the individually certified receiver cells from layer 2 and `G0` the
declared current-grasp domain. The direct preimage is

```text
Pre_direct(R) = {
  g in G0 | there exists a synchronized, collision-free donor/receiver
  trajectory from g to some receiver cell in R
}
```

For stable placement nodes `p` and directed, validated place/pick transitions,
the transfer preimage is

```text
Pre_transfer(R) = {
  g in G0 | g ->place p ->pick g' ->handoff R is a validated directed path
}
```

The layer-3 partition is fail-closed:

- `DIRECT`: positive synchronized dual-arm trajectory evidence reaches an
  individually certified receiver cell;
- `TRANSFER`: positive place, pick, and final handoff trajectory evidence
  reaches an individually certified receiver cell;
- `UNCOVERED`: a domain-bound exhaustive certificate proves that neither route
  exists; and
- `UNKNOWN`: inputs or proof are incomplete. Failed or absent search remains
  `UNKNOWN`.

Every positive trajectory and every mandatory hard-check record must have an
explicit matching SHA-256. Every hard check binds both the trajectory evidence
ID and its exact content hash. The complete current-grasp declaration and each
class (ID, exact representative list, and domain) also receive canonical
hashes. Direct and first-place edges require a separately pinned whole-class
coverage record bound to those hashes; an exact trajectory for one
representative alone does not prove membership of the complete class. Empty
preimage evidence uses the same domain/class bindings. All evidence also binds
the task ID, receiver-pose ID, exact receiver-artifact hash,
robot/model/project identities, and world/tool calibrations. Joint paths must
be finite, numeric, rectangular, dimensionally consistent, and time ordered.
Bare self-declared check booleans are not accepted.

These SHA-bound imports are explicit trust roots: the software verifies their
identity, context, declared proof gates, and internal structure, but an
operator or independent certifier remains responsible for the scientific
validity of the external proof itself.

Current artifact: `generated/sets/handoff_preimage_set.json`.

- 0 `DIRECT`;
- 0 `TRANSFER`;
- 0 `UNCOVERED`;
- 1 `UNKNOWN` current-grasp class.

## Visualization contract

The interactive viewer uses area-weighted samples from the actual PCB,
connector, gripper-body, and two finger STL files. Geometry sampling is for
display only and performs no feasibility check. CAD points are depth-sorted;
pose glyphs are deliberately overlaid as annotations.

Every layer-1 cell is shown by its constructive center pose. Rejected cell
centers are hidden by default. The selected cell center displays the detailed
three-component gripper and can be moved along the nominal 40 mm insertion
path. Any sampled phase-1 witness attached to that cell is shown separately in
the details panel.

Colors and glyph shapes distinguish:

- task `UNRESOLVED` and not robot-evaluated;
- a provisional numerical robot path witness;
- robot-evaluated but no numerical witness;
- task-safe only;
- robot-certified safe; and
- whole-cell task rejection.

Layer 3 is shown only as a global current/donor-grasp preimage summary. Its
donor classes are not incorrectly joined to receiver pose glyphs. Before any
overlay is rendered, the viewer verifies the layer-1 semantic digest, displayed
CAD/source-file hashes, socket transform, layer-2 source binding and internal
cell partition, and the layer-3 receiver hash and class partition. The embedded
payload records all three generated artifact hashes for traceability.

At present there are no certified-safe glyphs. The viewer must not visually
or textually upgrade an unresolved cell or provisional path witness.

## Remaining proof obligations

Before any positive set can drive hardware autonomously:

1. calibrate the flange-to-contact-frame transform and bound its uncertainty;
2. replace provisional jaw stroke, force, and contact-patch assumptions with
   measured or manufacturer-authoritative values;
3. prove pad capture, part/gripper separation, and exact finite-workcell swept
   collision over whole cells and continuous paths;
4. add insertion wrench, friction, connector compliance, manufacturing
   tolerance, and pin/hole lead-in models;
5. enumerate or conservatively cover relevant IK branches; and
6. generate provenance-bound synchronized direct-handoff or directed
   place/pick trajectory evidence.
