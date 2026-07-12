# MuJoCo handoff pipeline

This document describes the current executable implementation. The mathematical
derivation is in [handoff_pipeline_detailed.md](handoff_pipeline_detailed.md).

The pipeline is geometry-driven and downstream-constrained. A user supplies the
cell assets, calibrated poses, task regions, and the part pin/PCB hole feature
frames. The solver derives grasp contacts, jaw aperture, stable placements,
handoff candidates, reorientation choices, IK branches, and motion paths. There
are no per-part approach-axis, roll, center-grasp, or finger-opening tables.

The current connector-header reference project can plan and replay both direct
handoff and forced reorientation. Those are verified simulation results, not a
physical qualification claim. The supplied gripper is still one static STL and
the PCB is a solid board with no hole collision model, so
`physical_certification` is currently `false`.

## 1. Configuration contract

### User-owned input: `project.yaml`

[project.yaml](../mujoco_sim/project.yaml) is the only project file that a cell
integrator or part author should edit. It contains physical facts and task
semantics:

- robot URDF/MJCF models, calibrated world-base transforms, and initial joints;
- gripper model and mount-to-TCP transform;
- workstation visual CAD and collision CAD;
- part CAD, mass, and the semantic `part_to_pin` transform;
- the known startup `TCP-to-part` pose;
- the finite source used for offline initial-grasp coverage qualification;
- handoff, scanner, reorientation-stage, and insertion regions; and
- PCB world pose and one or more `PCB-to-hole` feature transforms.

STL has no unit metadata. Every STL/OBJ/STEP entry must declare units or an
explicit scale. MuJoCo uses metres and radians.

Robot and gripper CAD alone are insufficient for articulation. A robot needs
joint axes and limits in URDF/MJCF. A moving parallel-jaw gripper needs separate
palm/finger bodies, prismatic joints, joint limits, collision bodies, and named
pad/contact geometry in URDF/MJCF. These are reusable asset properties, not
part tuning.

The current executable scene/kinematics adapter supports the two calibrated
GP7s in this cell and requires both entries to use the GP7 model. The generic
geometry, task-frame, cache, collision-policy, and task-graph layers are
robot-independent, but using a different robot requires a matching scene and
kinematics adapter. Likewise, gripper joint/aperture discovery is implemented
and tested, but the present GP7 scene compiler still instantiates the supplied
static gripper surface assembly; an articulated gripper must be integrated into
that adapter before physical close/contact execution is available.

### System-owned input: `solver_defaults.yaml`

[solver_defaults.yaml](../mujoco_sim/solver_defaults.yaml) holds numerical and
safety policy: sampling budgets, IK tolerances, joint margins, clearance,
uncertainty, adaptive edge resolution, bounded RRT budgets, insertion correction
envelope, and execution interlocks. These defaults are system/versioned policy;
they should not be copied into every part project.

The current defaults include at most 128 geometry grasps, a 0.035 rad maximum
joint increment and 0.5 mm positive buffer for swept collision validation, a
6.5 mm handoff clearance including calibration uncertainty, and bounded
RRT-Connect with a 1 s/3000-node budget.

### Deprecated compatibility files

`pipeline_config.yaml` and `grasp_config.yaml` are deprecated as user inputs.
The current planner compiles its internal compatibility structure from
`project.yaml` and `solver_defaults.yaml` and never reads either deprecated
file. They remain only as migration tombstones. Do not add new part rules to
them.

`scene_config.yaml` is also not a part-planning interface. It currently retains
the internal photo-matched primitive fixture fallback used when table/bin CAD is
not supplied. Replace that fallback with workstation collision CAD in
`project.yaml`; do not treat it as another user tuning layer.

## 2. CAD ingestion and physical-model truth

`build_mujoco_scene.py` runs the CAD preparation path automatically and writes
content-addressed metadata under `mujoco_sim/models/generated_cad/`.
`prepare_project_cad.py` remains useful as a standalone audit/prewarm command.

- Binary and ASCII STL are normalized to deterministic binary STL.
- Every input visual triangle is preserved. No triangle is decimated, welded,
  simplified, or reordered.
- Large STL files are split only because one MuJoCo STL asset must contain
  fewer than 200,000 faces. Splitting does not change the triangles.
- OBJ visual input is copied byte-for-byte.
- STEP/STP is tessellated by `FreeCADCmd`/`freecadcmd` with explicit linear and
  angular deflection, then every resulting triangle is preserved through the
  same STL path.

“Exact visual CAD” means exact preservation of the supplied polygon mesh, or of
the explicitly parameterized STEP tessellation. It does not mean that a
tessellated STEP file remains an analytic B-rep.

MuJoCo itself cannot load STEP. The scene builder calls the same preprocessing
code and references the resulting STL/OBJ chunks in the generated MJCF. For a
STEP-backed project asset, `FreeCADCmd`/`freecadcmd` must therefore be on
`PATH`, named by `FREECADCMD`, or used once through
`prepare_project_cad.py --freecad /path/to/FreeCADCmd` to populate the content
cache. Do not point a hand-written
MuJoCo `<mesh>` directly at STEP.

Visual and collision fidelity are separate. MuJoCo uses the convex hull of a
mesh geom for collision; a concave rendered triangle surface is not an exact
concave collision shape. Use primitives or separately exported convex pieces
for collision. Export every articulated moving body separately and preserve a
common assembly frame.

`workstation.collision_cad` may be the current surveyed collision-box YAML or
STL/OBJ/STEP collision CAD with declared units/scale and an optional world
pose. Mesh sources still obey the convex-hull rule above, so concave equipment
should be supplied as separately exported convex entries in
`additional_collision_cad`.

The current workcell visual uses the complete source STL split into chunks.
The current gripper visual also uses the complete source STL. Its eight
connected components are all loaded as fixed collision geoms for each robot.
This is materially safer than the old palm proxy, but each component is still a
convex-hull collision geom and none can move relative to the palm.

### Static versus articulated gripper

The current `gp7_parallel` asset is declared
`parallel_jaw_static_fallback`. The manufacturer opening range and pad sizes in
`project.yaml` are used to reject impossible geometry grasps and compute the
required aperture. They do not create finger joints. Therefore:

- the viewer cannot show the fingers opening or closing;
- aperture/capture is a virtual predicate;
- part ownership transfers between ideal MuJoCo welds; and
- physical grasp/capture certification is impossible with this asset.

When an articulated MJCF/URDF is supplied, the gripper inspector obtains the
aperture range from its slide/prismatic joint limits and maps each candidate's
contact separation to finger joint positions. Named pad geometry should then
replace the static-holder contact allowance.

## 3. Frame contract and feature-derived insertion

All poses use `^X T_Y`: a transform mapping coordinates in frame `Y` into
frame `X`. A grasp is

$$
g = {}^P T_E,
\qquad
{}^W T_E = {}^W T_P\,g.
$$

With a known startup part pose, A's actual rigid grasp is recovered from FK:

$$
g_A = ({}^W T_P^{start})^{-1}\,{}^W T_{E_A}(q_A^{start}).
$$

The project manifest stores the equivalent startup `^E T_P`; the simulator
uses its inverse consistently at the boundary.

Insertion is not configured as a hand-tuned world part pose. For PCB frame
`C`, hole frame `H`, and part pin frame `F`, feature equality gives

$$
{}^W T_P^{ins},{}^P T_F
= {}^W T_C,{}^C T_H,
$$

and therefore

$$
\boxed{{}^W T_P^{ins}
= {}^W T_C,{}^C T_H\,({}^P T_F)^{-1}}.
$$

The hole's `+Z` axis is defined to point into the hole. If
$a_H={}^W R_H e_z$, the pre-insertion translation is

$$
t_P^{pre}=t_P^{ins}-d_{app}a_H.
$$

This is intentionally relative to the physical hole axis, not world `+Z` and
not an assumed part axis. Correction-envelope translations are expressed in
the hole frame and yaw is a left-applied world rotation about $a_H$.

## 4. Geometry-derived grasps

`geometry_grasps.py` parses the part STL in its native CAD frame and generates
parallel-jaw contact pairs as follows:

1. deterministically sample the surface, stratified by triangle area;
2. cast an inward ray from each sample to the first opposing surface;
3. require both contact normals to lie in the friction cones;
4. require contact separation to lie inside the gripper aperture range;
5. derive gripper roll from mesh covariance rather than fixed part axes;
6. check pad support and fingertip-to-palm depth; and
7. rank and SE(3)-deduplicate while retaining spatial coverage along elongated
   parts.

Every candidate stores `^P T_E`, both contact points and normals, required
opening, closing and approach directions, and quality terms. Its origin is the
contact midpoint, `+E_Y` is the closing line, and `+E_Z` is the approach
direction. This removes the former rule that forced both robots to grasp the
part center.

During co-grasp filtering, A's occupied contact patch and all eight fixed
gripper collision components exclude overlapping B grasps. A candidate is not
accepted merely because its two contact points are good: exact robot/gripper/
part/cell collision and complete approach/retreat paths remain hard gates.

For a part symmetry `S` expressed in the part frame, the grasp orbit uses the
left action `S g` (or `S^{-1}g` if `S` was defined in the inverse relabeling
direction). Right multiplication rotates about the TCP and is incorrect for
`g = ^P T_E`.

## 5. Planning pipeline

### 5.1 Offline/cached downstream factorization

For every receiver grasp, B is checked at the scanner and every feature-derived
pre-insertion/insertion target. The witness includes:

- FK-verified multi-seed numerical IK;
- joint-limit, singularity, and wrist-dither margins;
- collision-checked scanner-to-PCB and insertion paths;
- all 16 axis-relative correction-envelope vertices;
- branch-continuous IK at each correction vertex; and
- worst downstream manipulability and minimum singular value.

The current connector project generates 128 receiver candidates; 23 survive
the complete downstream filter. This `23/128` is a measured result for the
current CAD, frames, and solver version, not a universal acceptance rate.

### 5.2 Direct handoff

For A's measured current grasp and each downstream-valid B grasp, the planner
samples only inside the declared handoff region. It applies, cheap to expensive:

1. reachability lookup at the induced TCP poses;
2. exact FK-verified IK;
3. joint-limit and singularity gates;
4. distinct occupied contact patches and full component collision;
5. pre-handoff approaches, co-grasp, A retreat, and B-to-scanner paths; and
6. clearance including the calibration 3-sigma margin.

Every non-authorized swept-path pair has a positive 0.5 mm broadphase buffer;
the simultaneous handoff has the stricter 5 mm clearance plus 1.5 mm
calibration 3-sigma requirement. During place/re-pick, part/support contact and
positive finger/support proximity are phase-authorized, but finger-table
penetration remains zero-tolerance.

The normal low-latency mode returns the first completely valid plan.
`--best` exhausts the bounded grid and uses normalized manipulability, joint
margin, clearance, part-orientation reorientation, and cycle-travel terms.
Safety conditions are never score terms.

The G1 query is the induced TCP pose

$$
{}^{R_0}T_E=({}^W T_{R_0})^{-1}\,X_h\,g_R,
$$

not the part origin. The reorientation score compares part orientations
$R_h$ and $R_{ins}$; it does not compare TCP orientation $R_hR_g$ with a part
orientation.

### 5.3 Motion planning

Each requested segment first receives adaptive joint-space edge validation.
The number of checks scales with motion length so a long edge cannot silently
receive the same fixed sample count as a short edge. If the direct edge fails,
the planner runs deterministic, bounded bidirectional RRT-Connect using the
complete MuJoCo state and the appropriate held-part or fixed-part transform.
Successful sparse paths are densely revalidated for execution.

### 5.4 Stable placement and reorientation

Stable placements are generated from CAD rather than bounding-box face rules:

- connected components and closedness are evaluated;
- closed components provide uniform-density volume COM estimates, with an
  explicit bounding-box-center fallback for unreliable/open CAD;
- coplanar support facets and their convex support polygons are found;
- projected COM must lie inside with positive support margin; and
- the complete part footprint must fit inside the declared rectangular stage
  at a sampled yaw.

The system policy rejects supports below `0.005` of the part bounding-box
diagonal. Placement-edge robustness is not a constant: it is the bottleneck of
the support margin normalized by part scale and the stage-edge clearance
normalized by stage scale. The task graph uses that value for robustness
tie-breaking after cycle cost and hop count.

The discrete planner searches backward from B grasps already proved insertion
feasible. Direct co-grasp edges connect candidate A grasps to those B grasps;
placement-grasp edges connect feasible A grasps to stable placements. The
`TaskGraph` has a hard direct-first policy. Only when the initial class has no
direct edge does it search bounded reorientation paths, minimizing cycle cost,
then hops, maximizing bottleneck robustness, and finally using deterministic ID
ordering.

The returned sequence is explicit:

```text
current A grasp -> place -> re-pick A grasp -> verified direct A/B edge
                 -> insertion-feasible B grasp
```

Thus the re-picked A grasp must actually connect to an insertion-valid B grasp;
it is not selected from a visual orientation heuristic.

## 6. Collision semantics and execution safety

Collision is a hard gate during planning and execution. Expected manipulation
contact is phase-specific and bounded:

- a static holder may contact the part only through
  `<robot>_gripper_collision_*`, never through link 6 or the wrist;
- holder penetration is capped at 0.75 mm in the current fallback policy;
- placement temporarily permits only the part/reorientation-surface pair; and
- geometric insertion temporarily permits only the part/PCB pair.

These allowances do not disable other contacts. Gripper-to-gripper,
gripper-to-wrist, wrist-to-part, robot-to-cell, and unexpected fixture contacts
remain visible to the checker.

Execution is transactional: A owns the part, B approaches under the co-grasp
policy, capture is checked, ownership transfers atomically, A retreats, B moves
to the scanner, downstream targets are recomputed from the measured B grasp,
and B approaches insertion. A continuous monitor checks every replay waypoint
and aborts immediately on an unexpected collision. With the present static
gripper, closing/capture and force guard are idealized predicates rather than
finger-contact dynamics.

## 7. Content-addressed offline computation and measured latency

Artifacts are canonical JSON stored by SHA-256 key. Keys include an artifact
version, source CAD/scene fingerprints, relevant project/task transforms,
solver parameters, and upstream artifact identities. A CAD, calibration,
feature-frame, or solver change therefore produces a new key instead of
silently reusing stale feasibility.

The production pass materializes the known-start direct-first policy:

| Artifact | Reused result |
|---|---|
| Grasp cache | Geometry contact pairs and required aperture |
| Downstream task-policy cache | Scanner/insertion IK, correction, collision, and paths |
| Direct task-policy cache | Known-start direct handoff witness and trajectories |
| Stable-pose cache | COM/support/footprint-valid stage instances |
| Reorientation task-policy cache | Backward placement/re-pick/direct policy, when direct is unavailable |

`qualify_pipeline.py` enumerates the manifest-declared admissible grasp domain;
in doing so it materializes direct or reorientation policies for every class.

Reference snapshot measured on 2026-07-12 on the Mac Studio for project
fingerprint `aaff9b2dfcdbb71721f6fe8776d8bf0fbdceb892ab55ac403f04cb47acfef9f0`
and solver fingerprint
`d652ff9f31a7181d1dbdb6ba37bd2c201d8a76a3afddbb1dc9d656accd451139`
(downstream v4, direct v7, stable pose v4, reorientation v7):

| Operation | Current measurement |
|---|---:|
| Downstream-valid receiver grasps | 23 of 128 |
| Downstream filter, cold | 149.9 s |
| Direct search, cold policy entry | 2.93 s |
| Direct policy cache hit | 26.9 ms |
| Reorientation policy, cold adverse grasp | 4.39 s |
| Reorientation policy cache hit | 4.58 ms |
| Stable-placement cache hit | 4.59 ms |

These are engineering measurements, not deadlines or worst-case bounds. They
exclude physical robot motion, and end-to-end process CT also includes scene/
planner construction, communication, sensing, and execution. The production
strategy is to precompute after every content change and make the per-cycle
decision a cache lookup whenever the observed initial grasp belongs to the
declared offline domain.

Optional learned ordering can reduce cold-search work further; it cannot make a
candidate safe or valid. See §10.

## 8. Exact commands

From the repository root:

```bash
source .venv/bin/activate
```

Prepare/audit all CAD referenced by the project:

```bash
python scripts/prepare_project_cad.py --project mujoco_sim/project.yaml
```

For STEP/STP, install FreeCAD and either put `FreeCADCmd`/`freecadcmd` on
`PATH` or provide it explicitly:

```bash
python scripts/prepare_project_cad.py --project mujoco_sim/project.yaml --freecad /absolute/path/to/FreeCADCmd
```

Build the scene from the project assets:

```bash
python scripts/build_mujoco_scene.py
# Alternate manifest/output:
python scripts/build_mujoco_scene.py --project path/project.yaml --output path/scene.xml
```

Populate the content-addressed production caches:

```bash
python scripts/build_reachability.py --project mujoco_sim/project.yaml \
  --model mujoco_sim/models/scene.xml --out mujoco_sim/cache
python scripts/precompute_pipeline.py --project mujoco_sim/project.yaml \
  --model mujoco_sim/models/scene.xml --production
```

Run the focused implementation tests (the environment uses direct executable
test files; `pytest` is not required):

```bash
for test in tests/test_mujoco*.py tests/test_geometry_grasps.py tests/test_motion_planning.py; do
  python "$test" || exit 1
done
```

Plan/execute headlessly:

```bash
python -m mujoco_sim.pipeline --execute --json
# Use the same manifest/model/cache triplet for an alternate compiled scene:
python -m mujoco_sim.pipeline --project path/project.yaml --model path/scene.xml --cache path/cache --execute
```

Visualize the verified direct pipeline on macOS:

```bash
mjpython -m mujoco_sim.visualize_pipeline --hold -1
```

Visualize the verified forced-reorientation pipeline on macOS:

```bash
mjpython -m mujoco_sim.visualize_reorientation_demo --hold -1
```

Those visualization modules use `launch_passive()`, so macOS requires
`mjpython`. Use ordinary `python` on Linux. The reorientation demo refuses to
open if it cannot find collision-checked place/re-pick paths connected to an
insertion-feasible direct handoff.

## 9. Coverage and certification

For a declared admissible initial-grasp-class domain $\mathcal D$, the task
graph reports disjoint `direct`, `reorientation`, and `uncovered` sets and

$$
\mathrm{coverage}
= \frac{|\mathcal D_{direct}|+|\mathcal D_{reorientation}|}{|\mathcal D|}.
$$

A coverage certificate is issued only when this fraction meets or exceeds the
requested target (normally `1.0`). “100%” means 100% of the explicitly declared finite
domain under the fingerprinted CAD, calibration, and solver policy. It does not
mean every continuous pose, arbitrary part, calibration error, or unmodeled
obstacle.

The normal runtime and current manifest both declare the supplied known-start
singleton, matching this task's known-pose assumption. A project whose picker
can deliver any geometry-library grasp may instead declare
`known_start_plus_geometry_library`; `qualify_pipeline.py` then enumerates that
much broader domain offline.

Task-graph coverage and physical certification are separate. The current
reference project cannot be physically certified because:

1. the gripper is a static STL with no articulated fingers or measured pad
   contact dynamics;
2. the PCB is a solid collision board and `insertion.collision_cad` does not
   provide actual hole/chamfer geometry;
3. the part has no separate pin collision model or calibrated pin/hole/contact
   materials; and
4. execution still uses ideal weld ownership and virtual capture/insertion
   predicates rather than an articulated contact controller.

Consequently even a feasible direct/reorientation replay and 100% singleton
coverage must report `physical_certified: false` (and the precompute summary's
`physical_certification.certified` is also false).

Additional production work includes measured friction/COM, calibrated
uncertainty, real aperture/force feedback, scanner noise, actual fixture and
pin/hole collision CAD, and hardware validation of stopping distances.

## 10. Learning: useful, but outside the safety boundary

Learning is appropriate for proposal ordering, not validity:

- rank geometry grasps likely to survive downstream gates;
- rank handoff region samples by expected feasibility or cycle time;
- rank stable placements/re-picks; and
- predict which cached policy neighborhood is worth checking first.

Start with supervised learning from deterministic planner logs. Useful labels
are gate outcome, solve time, bottleneck clearance, path length, and insertion
success. Evaluate top-k feasible recall and wall-clock reduction on held-out
parts/cell perturbations. Fall back to deterministic ordering under uncertainty
or distribution shift.

The current `SafetyGatedRanker` is deliberately conservative: it never returns
a proposal marked invalid by deterministic gates. A future pre-gate learned
ranker may reorder candidates, but every selected candidate must still pass
geometry, IK, collision, uncertainty, motion, and execution-monitor checks.

RL is unnecessary for the present static-cell planning problem and adds a much
harder validation burden. It may later optimize high-level scheduling or
closed-loop contact behavior with appropriate hardware safeguards, but an RL
score, policy, or value estimate is never a collision/safety certificate.
