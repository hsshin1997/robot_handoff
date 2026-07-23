# Surface-relative and CAD-generated parallel-jaw grasps

This repository now exposes two related operations:

1. map a surface-relative grasp
   `(u, v, d, psi, alpha, beta, w)` to an abstract gripper-pose transform; and
2. generate repeatable antipodal parallel-jaw candidates directly from STL,
   OBJ, STEP, or STP CAD.

They share the repository transform convention:

```text
T_X_Y maps coordinates expressed in Y into frame X.
translations: metres
angles: radians
```

## 1. Map `(u, v, d, psi, alpha, beta, w)` to SE(3)

Use `mujoco_sim.modeling.surface_grasp.surface_grasp_to_se3`.
The caller supplies a surface-frame evaluator because raw triangle meshes do
not have one canonical global UV chart.

The evaluated surface frame has:

```text
+X_S = local u tangent
+Y_S = local v tangent
+Z_S = outward surface normal
```

The parameterized gripper-pose frame `G` uses the repository gripper-axis
convention:

```text
+X_G = pad-width direction
+Y_G = jaw opening/closing direction
+Z_G = approach direction from palm towards the surface
```

Example for the plane `z = 0`:

```python
import numpy as np

from mujoco_sim.core.se3 import make_transform
from mujoco_sim.modeling.surface_grasp import surface_grasp_to_se3


def plane_frame_at(u_m: float, v_m: float) -> np.ndarray:
    return make_transform(np.eye(3), [u_m, v_m, 0.0])


T_W_G, opening_m = surface_grasp_to_se3(
    (
        0.20,   # u [m]
        -0.10,  # v [m]
        0.08,   # d: surface-anchor standoff along -Z_G [m]
        0.40,   # psi: rotation about outward surface normal [rad]
        -0.20,  # alpha: intrinsic tilt about +Y_G closing axis [rad]
        0.10,   # beta: intrinsic tilt about +X_G pad-width axis [rad]
        0.025,  # w: required jaw opening [m]
    ),
    plane_frame_at,
)
```

The mapping is

```text
R_X_G = R_X_S Rz(psi) A Ry(alpha) Rx(beta)
p_X_G = q_X - d R_X_G e_z

    [0  1  0]
A = [1  0  0]
    [0  0 -1]
```

Therefore the surface anchor always satisfies

```text
q_X = p_X_G + d * approach_X
```

`w` is returned separately because a jaw opening is an internal gripper
coordinate, not part of SE(3).

For a known surface frame without a UV evaluator, call the lower-level
`surface_grasp_to_transform(T_X_S_uv, ...)`.

`G` is the abstract pose frame selected by this surface parameterization. It
is not the same frame as the CAD generator's contact-midpoint frame `E`
described below. If `G` is intended to be a physical TCP, its origin and axes
must match the calibrated controller TCP convention.

## 2. Generate candidates directly from CAD

The generic command is:

```bash
.venv/bin/python scripts/generate_parallel_jaw_grasps.py \
  parts/connector_header/connector_header_part.STL \
  --units mm \
  --min-opening-m 0.002 \
  --max-opening-m 0.024 \
  --pad-width-m 0.008 \
  --pad-height-m 0.010 \
  --finger-depth-m 0.040 \
  --friction-coefficient 0.5 \
  --surface-samples 3200 \
  --closing-directions-per-surface 5 \
  --approaches-per-pair 24 \
  --max-candidates 0 \
  --output build/parallel_jaw_grasps/connector_header.json
```

`--max-candidates 0` retains every deduplicated, accepted candidate at the
declared finite resolution. A positive value deliberately produces a bounded,
ranked output subset; it is applied after geometric candidate construction and
therefore does not bound generation runtime.

The gripper dimensions are always specified in metres. CAD source units must
be explicit because STL has no unit metadata:

```text
--units m|mm|cm|in
```

or:

```text
--scale-to-m <factor>
```

STL and OBJ work directly. STEP/STP is tessellated without later triangle
reduction and requires `FreeCADCmd`/`freecadcmd`:

```bash
... --freecad /absolute/path/to/FreeCADCmd
```

Before ray sampling, the command performs an exact-coordinate topology audit.
It rejects degenerate faces, boundary edges, nonmanifold edges, inconsistent
paired-edge orientation, zero-volume components, and mixed winding-volume
signs across disconnected components because inward-normal ray semantics are
unreliable or ambiguous for those meshes. Mixed signs can describe a valid
nested cavity, but the CLI does not infer material containment, so it fails
closed. The audit never welds, repairs, or reorients the supplied geometry.

For diagnostic-only generation on such a mesh, the caller must opt in:

```bash
... --allow-unreliable-mesh
```

That changes the output claim level to
`unreliable_mesh_sampled_candidate`; it does not make the mesh valid.

### What the generator checks

For each area-stratified low-discrepancy surface sample, the generator:

1. requires a closed, consistently wound exact-coordinate mesh topology;
2. samples inward closing rays inside the source contact's Coulomb friction
   cone and keeps each first opposing-surface hit;
3. checks that the two contact normals satisfy the Coulomb friction cones;
4. checks the required opening against the gripper range;
5. samples approach roll about the closing axis;
6. rejects candidates that exceed the idealized fingertip-to-palm depth;
7. scores local pad support, opening margin, and palm clearance; and
8. deduplicates nearby contact poses while preserving spatial coverage when a
   result cap is requested.

Every candidate contains:

```text
T_P_E
two contact points and normals in CAD frame P
closing and approach directions
required opening w
quality components
```

`E` is an ideal contact frame at the contact midpoint. It is not automatically
the robot TCP or the origin of the physical gripper CAD. If `T_G_E` is the
calibrated fixed transform from the ideal contact frame into physical gripper
frame `G`, use:

```text
T_P_G = T_P_E @ inverse(T_G_E)
```

## 3. Visualize selected candidates

Render a dependency-free PNG gallery from the generated JSON:

```bash
.venv/bin/python scripts/render_parallel_jaw_grasp_candidates.py \
  build/parallel_jaw_grasps/connector_header.json \
  --generated-root build/parallel_jaw_grasps/cad \
  --selection pose-diverse \
  --count 4 \
  --output build/parallel_jaw_grasps/connector_header.preview.png
```

The command also writes
`build/parallel_jaw_grasps/connector_header.preview.json`. That companion
artifact records the exact displayed candidate IDs and transforms, selection
policy, prepared-CAD fingerprint, view definitions, and visualization claim
boundary.

The default `pose-diverse` mode starts with generator source rank 0, then
chooses candidates by deterministic max-min display distance over:

```text
contact-midpoint position
unoriented jaw-closing axis
oriented approach axis
required opening
```

This is a display-coverage heuristic, not a new grasp-quality or feasibility
ranking. Use generator order instead with:

```bash
... --selection ranked --count 4
```

To inspect exact candidates, repeat `--candidate-id` in the desired order:

```bash
... \
  --candidate-id grasp_0123456789abcdef \
  --candidate-id grasp_fedcba9876543210
```

The visualizer verifies the JSON pose invariants before drawing:

```text
origin of E = midpoint(c0, c1)
+Y_E = normalized(c1 - c0) = serialized closing direction
+Z_E = serialized palm-to-contact approach direction
||c1 - c0|| = required opening
```

Each candidate is displayed in three candidate-aligned orthographic views:

```text
XY: view along -Z_E
XZ: view along +Y_E
YZ: view along -X_E
```

The XZ panel therefore uses a cross for `+Y_E` (into the page); the XY and
YZ panels use dots for their positive out-of-page axes.

The RGB axes are:

```text
red   +X_E  pad-width direction
green +Y_E  contact 0 to contact 1
blue  +Z_E  palm-to-contact approach
```

White points are the exact contacts, yellow arrows are their outward surface
normals, and orange arrows show symmetric jaw closing. Cyan rectangles and
lines are an ideal schematic only:

- zero-thickness pad rectangles use the configured pad width and height;
- jaw centerlines run from each contact toward `-Z_E`; and
- the dashed crossbar marks the configured fingertip-to-palm depth.

No finger thickness, finite palm solid, physical TCP, or calibrated gripper
CAD is inferred. The preview therefore does not certify full part/gripper
collision, approach-sweep clearance, robot reachability, or task feasibility.

The preview locates the exact prepared CAD by its recorded artifact
fingerprint under `--generated-root`. If the original CAD was relocated, pass
`--cad`; the replacement file is accepted only when its SHA-256 matches the
generated JSON. A positive `--max-render-triangles` may be used for a
display-only area-stratified triangle subset on very large meshes; the default
of `0` projects every prepared CAD triangle.

## Meaning of “all feasible”

A CAD surface and rigid gripper pose form a continuous set, so a finite JSON
file cannot enumerate literally every feasible SE(3) pose. The generic script
returns all accepted candidates at the declared surface, friction-cone
closing-direction, and roll sampling resolution when `--max-candidates 0` is
used.

The output is deliberately labeled:

```text
claim_level = resolution_qualified_object_geometry_candidate
continuous_exhaustive = false
```

The two APIs also use different surface semantics:

- the seven-parameter helper anchors an approach pose on one surface, so its
  zero-angle approach is opposite that surface normal; and
- antipodal parallel-jaw generation pairs two opposing contact surfaces, so
  the jaw-closing line is approximately normal to each contact while the
  approach direction is tangent to those contact surfaces.

Consequently the CAD generator does not claim that its contact candidates are
direct samples of the helper's `(u, v, d, psi, alpha, beta, w)` chart. It
returns contact-frame families that still require physical TCP registration
and any desired insertion-depth selection.

It does not yet certify:

- poses between the declared samples;
- closing directions between the finite friction-cone samples;
- exact finger, palm, or gripper-body collision against the part;
- the collision-free approach sweep;
- fixtures, other objects, or environment clearance;
- robot IK, motion planning, and joint limits; or
- task wrench, compliance, dynamics, and calibration uncertainty.

Those checks need actual gripper collision geometry, a scene, and the task
requirements. The JSON records this boundary so sampled candidates cannot be
mistaken for physically certified grasps.

Results are repeatable for fixed inputs, settings, NumPy version, and numerical
backend. Cross-platform bitwise identity is not certified because symmetric
CAD can have non-unique principal directions in the covariance
eigendecomposition used to seed gripper roll.

For opposed planar facets where a continuous set representation is preferable,
the existing `scripts/generate_two_finger_grasp_map.py` command produces finite
unions of continuous local `(u, v, roll)` families. Its `roll` is about the
jaw-closing axis and corresponds to `alpha` above; it is not `psi` about a
surface normal.

## Verification

```bash
.venv/bin/python tests/test_surface_grasp.py
.venv/bin/python tests/test_generate_parallel_jaw_grasps.py
.venv/bin/python tests/test_geometry_grasps.py
.venv/bin/python tests/test_mujoco_cad_preprocess.py
.venv/bin/python tests/test_mujoco_part_mesh.py
```
