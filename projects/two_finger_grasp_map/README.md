# Two-finger set-valued grasp map

This is the clean, part-only starting point for insertion grasp generation. It
does not use the previous task/robot/handoff layers and it does not enumerate a
world-pose lookup table.

Given a part mesh, its seated pose `T_W_P_insert`, and the part-frame insertion
axis, the generator returns a finite union of continuous two-finger grasp
families:

```text
Gamma(T_W_P_insert) = union_k {
    T_W_E(u, v, roll) = T_W_P_insert T_P_E,k(u, v, roll)
    | (u, v) in D_k, roll in R_k
}
```

Each family `k` comes from two opposed, lateral surface regions on the
outward-wound part mesh. `D_k` is a union of convex contact-center
polygons—not a list of poses. An arbitrary point inside a polygon and an
arbitrary allowed roll can be evaluated at runtime to obtain the part-relative
and world-frame gripper poses.

The part-relative family is independent of the world translation and
orientation. The seated pose simply maps it into the requested world frame by
left multiplication with `T_W_P_insert`.

## Required inputs

An arbitrary 4x4 part pose does **not** encode a direction of motion. The
minimal insertion task therefore contains:

- the STL path and its explicit scale to metres;
- a closed, manifold, consistently wound part mesh (this version assumes but
  does not certify that topology);
- `T_W_P_insert`, mapping the part frame into the world at seating;
- `insertion_axis_P`, the direction in which the part moves during insertion;
- the two-finger opening range and friction assumption; and
- simple surface/roll tolerances used to construct the candidate map.

For the connector example, `insertion_axis_P = [0, -1, 0]`. The included
nominal `T_W_P_insert` maps that direction to world down `[0, 0, -1]`.

## Run

From the repository root:

```bash
.venv/bin/python scripts/generate_two_finger_grasp_map.py \
  --config projects/two_finger_grasp_map/config/connector_header.yaml
```

The command writes:

```text
projects/two_finger_grasp_map/generated/connector_header_grasp_map.json
```

The JSON stores the paired surface planes, continuous family domains, affine
aperture functions, and roll intervals. It does not store a sampled pose table.

## What “exposed” means in this version

This first version uses a deliberately local, object-only definition:

1. candidate facets belong to the outward-wound mesh boundary;
2. their normals face laterally relative to the insertion axis; and
3. paired projected contact regions are antipodal under the declared friction
   cone and fit the gripper opening range.

This does **not** prove that a ray, finger pad, or complete finger can reach the
facet without another part feature blocking it. The result is therefore an
object-only candidate map, not an insertion-safe map. A PCB, socket, fixture,
finger/body geometry, and insertion path are required to decide whether the
complete gripper can occupy or sweep through a family. Robot IK and handoff are
intentionally outside this module.

Within that stated ideal point-contact model, the polygon domains are
continuous and constructive: choose a family, choose any `(u, v)` in one of
its polygons, choose any roll in its interval, and call `family.evaluate(...)`
to obtain the two contacts, aperture, `T_P_E`, and `T_W_E`.

The STL does not identify plastic housing versus fragile metal pins. Therefore
the unmasked connector result can include mathematically valid contact
families on surfaces that should be forbidden in practice. A later certified
map must add an authored graspable-surface mask as well as the PCB and complete
gripper geometry.

## Current connector result

With the checked-in configuration, the generator produces 44 continuous
families made from 5,471 convex domain components. Required ideal contact
separations range from approximately 2.540 mm to 20.955 mm, and the configured
part insertion axis maps to world `[0, 0, -1]`.

These are local-surface candidates under the declared mesh tolerances. The
counts are not claims of 44 collision-free or robot-reachable grasps.

## Verify

```bash
.venv/bin/python tests/test_two_finger_grasp_map.py
```

The focused suite checks an analytic box, disconnected surface regions,
continuous membership/evaluation, opening and edge constraints, world-frame
composition, serialization, invalid inputs, and the connector mesh.

## Frame convention

- `T_X_Y` maps coordinates from `Y` into `X`.
- `+Y_E` is the jaw-closing direction from contact 0 to contact 1.
- `+Z_E` points from the gripper palm toward the contact midpoint.
- Roll is about `+Y_E`; roll zero aligns `+Z_E` with the insertion direction
  projected perpendicular to the closing axis.
