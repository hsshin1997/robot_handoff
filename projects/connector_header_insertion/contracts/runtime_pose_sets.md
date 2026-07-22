# Runtime insertion-pose set contract

The insertion solution is a continuous, usually disconnected subset of robot
configuration space. It is not a finite table of world poses and it should not
be treated as one smooth cone.

This project separates three different objects that are easy to conflate:

1. a connector-relative grasp family;
2. a world insertion target supplied at runtime; and
3. robot configurations and paths that realize one member of that family.

## Connector-relative family

Let `P` be the native connector frame and `E` the gripper contact/TCP frame.
An admissible parallel-jaw grasp is represented by a low-dimensional family

```text
g_l(u, v, rho) = T_P_E
```

where `l` selects an allowed opposing pair of housing patches, `(u, v)` locates
the contacts on those patches, and `rho` rolls the gripper about the closing
line. Aperture, pad support, friction, palm depth, pin exclusion, and task
clearance restrict that parameter domain.

`phase1_pose_library.json` is a deterministic finite sampling of this family.
Its rows are reusable seeds and geometric evidence. A missing row is not proof
that the nearby continuous pose is impossible.

## World target composition

The preferred user input is a measured board pose `T_W_B`. The authored PCB
socket contract supplies the seated connector pose `T_B_P_insert`, so

```text
T_W_P_insert = T_W_B @ T_B_P_insert
```

A user may instead provide `T_W_P_insert` directly. For every connector-relative
grasp seed:

```text
T_W_E_insert    = T_W_P_insert    @ T_P_E
T_W_E_preinsert = T_W_P_preinsert @ T_P_E
```

`T_W_P_preinsert` is translated opposite the insertion axis by the declared
approach distance. The object/gripper relationship remains fixed during the
nominal straight insertion.

Small vision or fixture corrections are a bounded continuous variable, not
extra table rows. This phase permits insertion-frame translation and yaw about
`+Z_I` only. With `delta = (dx, dy, dz, yaw)`:

```text
Delta_I(delta) = Trans_I(dx, dy, dz) @ RotZ_I(yaw)
```

Yaw is applied about the insertion-frame origin first, followed by translation
expressed in the nominal `I` axes. The target family becomes

```text
T_W_E(delta) = T_W_I @ Delta_I(delta) @ T_I_P @ T_P_E
```

The bounds on `delta` must be checked as a sweep or robust envelope before a
pose is certified.

## Robot-specific feasibility

For a selected robot `R`, a composed TCP target is reachable only if at least
one joint configuration satisfies

```text
FK_R(q) = T_W_E
```

with joint-limit and singularity margins. An insertion pose is task-feasible
only when the same IK branch can be followed from approach through seating and
retreat while the real robot, gripper, connector, PCB, fixtures, and other arm
remain collision-free. Contact-force and uncertainty requirements are
additional gates.

Therefore the query outputs use the following claim levels:

- `preinsert_diagnostic`: approach arithmetic only; `T_W_E_insert` is null
  because the grasp is already known to fail a seated geometry gate;
- `composed_target`: transform arithmetic only;
- `ik_reachable`: FK-verified numerical IK for the configured TCP model;
- `path_validated`: continuous joint-space path and collision checks passed;
- `insertion_qualified`: path, finite-board geometry, mechanics, and declared
  uncertainty envelope passed.

Only the last two are executable task claims. In particular, an IK result made
with the repository's current placeholder/static gripper TCP is useful for
software testing but does not certify the supplied long-finger assembly.

## Correct role of a lookup database

A sparse database remains useful when it stores reusable evidence:

- content-addressed connector-relative grasp IDs or certified parameter cells;
- IK branches and reachability seeds for a named robot/calibration fingerprint;
- complete handoff and insertion trajectories;
- clearance, joint-limit, singularity, and force margins; and
- the exact domain or uncertainty neighborhood that was validated.

A nearby entry is only a proposal unless its whole neighborhood was certified.
Every runtime candidate still passes exact IK, collision, and path validation
for the current measured world state. New exact successes may be cached, which
makes the database sparse and demand-driven rather than an attempted grid over
all of `SE(3)`.

## Scalable planning sequence

```text
parameterized housing contact patches
    -> deterministic grasp seeds / adaptive parameter cells
    -> compose with measured board socket pose
    -> reachability rejection and IK branch search
    -> continuous collision, correction, and insertion-path validation
    -> sparse certified handoff policy edges
    -> transfer-stage place/reobserve/repick graph if direct handoff fails
```

For a flat transfer stage, placement does not determine planar `(x, y, yaw)`.
Use a locating nest or measure the part again after release before querying the
next grasp/reorientation edge.

Normal queries default to purpose `insertion` and require both pre-insert and
seated compatibility. Pre-insert-only rows are available only through explicit
purpose `preinsert_diagnostic`; their nominal seated transform may be retained
as a named witness but is never published as an executable target.
