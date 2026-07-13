# MuJoCo simulator user guide

This is the end-to-end guide for configuring, building, planning, executing,
visualizing, and checking the current MuJoCo handoff simulator. The detailed
method is documented in
[mujoco_handoff_pipeline.md](mujoco_handoff_pipeline.md), and the full offline
build procedure is in
[mujoco_offline_policies.md](mujoco_offline_policies.md). The code-stage map,
profiling workflow, and safe parallel-motion extension points are in
[mujoco_architecture.md](mujoco_architecture.md).

## Current scope

The current executable adapter models two calibrated Yaskawa GP7 arms. The
geometry, collision-policy, task-frame, cache, and task-graph layers are
reusable, but a different articulated robot still needs a matching
scene/kinematics adapter; a CAD path alone does not define its joint chain.

The reference gripper is one static STL assembly. Its complete supplied surface
is rendered and its connected components are collision checked, but its fingers
cannot move. Opening feasibility and ownership transfer are virtual predicates
and ideal welds. The current PCB is a segmented placeholder with one bounded
virtual aperture rather than measured holes. These limitations prevent
physical certification even when a complete
simulation plan succeeds. The source-model requirements and the remaining
adapter work for a moving gripper are specified in
[mujoco_gripper_integration.md](mujoco_gripper_integration.md).

## 1. Install the environment

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

MuJoCo uses metres, kilograms, seconds, and radians internally. Pose entries in
`project.yaml` accept metres plus either `rpy_deg`, `rotation_matrix`, or a 4x4
`matrix` where supported by the project loader.

## 2. Configure one project

Edit [../mujoco_sim/project.yaml](../mujoco_sim/project.yaml). This is the
user-owned project interface. Do not add per-part rules to
`pipeline_config.yaml` or `grasp_config.yaml`; they are deprecated placeholders.
`solver_defaults.yaml` is system-owned numerical and safety policy.

At minimum, configure:

- each robot model, calibrated `world_base`, initial joints, and gripper name;
- the gripper model, mounting transform, and manufacturer capability envelope;
- workstation visual CAD and collision CAD;
- the active part CAD, declared CAD units, and mass;
- the known startup `initial_tcp_to_part` pose;
- handoff, scanner, reorientation-support, and insertion regions;
- the exact final world pose of the native part frame; and
- the insertion/correction frame whose `+Z` points into the hole.

### Articulated gripper assets

The repository now contains a validated articulated-gripper descriptor contract
and [descriptor template](../mujoco_sim/gripper_asset.template.yaml). It checks
source MJCF/URDF mount and TCP frames, prismatic/slide limits, moving-finger pad
geometry, visual/collision names, aperture mapping, namespaces, and the
post-compilation scene binding.

That contract is not yet a generic scene importer. The current GP7 scene
compiler still takes the reference surface CAD and creates fixed geoms. A new
articulated model therefore needs model-specific code that imports its complete
kinematic subtree, assets, actuators/tendons/equalities, materials, and meshes,
attaches it at the flange, and binds its names after compilation. Changing the
manifest path to an MJCF, URDF, or descriptor alone does not make the fingers
move. Follow [mujoco_gripper_integration.md](mujoco_gripper_integration.md) when
the full gripper model is available.

### Frame convention

Every transform is written `^X T_Y`: it maps coordinates expressed in frame
`Y` into frame `X`.

- `active_task.initial_tcp_to_part` is `^E T_P`, the part pose in the current
  holder TCP frame.
- `insertion.targets[].world_part_pose` is the exact final `^W T_P`. The native
  part frame is the frame in which the supplied part CAD is expressed.
- `insertion.targets[].world_insertion_frame` is `^W T_I`. Its X/Y axes define
  lateral correction directions and its `+Z` axis defines motion into the
  hole.

An explicit insertion target looks like:

```yaml
regions:
  insertion:
    type: box
    center_m: [0.425, -0.455, 0.380]
    size_m: [0.360, 0.240, 0.120]

insertion:
  # Fixture pose used when PCB collision CAD is supplied. It does not modify
  # the exact part target below.
  pcb_world_pose:
    position_m: [0.425, -0.455, 0.347]
    rpy_deg: [180.0, 0.0, 0.0]

  targets:
    - name: pcb_slot_0
      world_part_pose:       # exact ^W T_P at full insertion
        # This right-angle header's short PCB tails point native -Z; native -Y
        # is the long exposed end. Native z=0 is the housing seating plane.
        position_m: [0.4123, -0.455, 0.347]
        rotation_matrix:
          - [1.0, 0.0, 0.0]
          - [0.0, 1.0, 0.0]
          - [0.0, 0.0, 1.0]
      world_insertion_frame: # X/Y correction; +Z points into the hole
        position_m: [0.425, -0.455, 0.347]
        rotation_matrix:
          - [1.0, 0.0, 0.0]
          - [0.0, -1.0, 0.0]
          - [0.0, 0.0, -1.0]
```

The planner preserves `world_part_pose` exactly. It constructs pre-insertion by
translating that part pose opposite `+Z_I` by
`solver_defaults.yaml:insertion.approach_distance_m`. Both final and
pre-insertion part origins must lie inside `regions.insertion`; malformed poses,
duplicate target names, and out-of-region targets fail when the project loads.

The generated reference fixture uses a bounded rectangular virtual aperture in
its PCB and support collision rings, so the short native `-Z` tails can reach
the seated pose without disabling collision against the surrounding fixture.
This aperture is suitable for planning/visualization only. A real insertion
project must supply PCB/hole and supporting-fixture collision CAD, including
under-board pin clearance, and retain the calibrated seated `world_part_pose`.

The older feature-frame schema remains available: `parts.<part>.part_to_pin`,
`insertion.pcb_world_pose`, and `insertion.holes[].pcb_to_hole` derive
`^W T_P` from pin/hole equality. Do not mix `insertion.targets` and
`insertion.holes` in one project.

### CAD and collision rules

STL and OBJ are supported directly. STEP/STP is tessellated through
`FreeCADCmd`/`freecadcmd`; see
[mujoco_offline_policies.md](mujoco_offline_policies.md#1-prepare-cad-and-compile-the-scene).
STL has no units, so every STL entry needs `*_units` or `*_scale_to_m`.

Visual fidelity and collision fidelity are separate:

- the scene preparation path preserves every supplied visual triangle and only
  chunks large STL assets to satisfy MuJoCo's per-mesh face limit;
- contact-grasp generation samples and raycasts the actual part triangles, not
  a convex hull; and
- MuJoCo mesh collision uses a convex hull for each mesh geom. Concave
  workstation, gripper, pin, or hole contact therefore requires separately
  exported convex pieces or suitable primitive collision geometry.

A rendered gap or overlap cannot by itself identify the collision
representation. Use the contact audit in section 7.

For a physically meaningful part collision model, keep the native visual CAD
for surface-grasp generation and declare a separate complete collision
decomposition:

```yaml
parts:
  connector_header:
    cad: parts/conn_header/conn_header_bin.stl
    cad_units: m
    collision_cad: parts/conn_header/connector_collision.step
    collision_cad_units: mm
    collision_cad_static_assembly: true
```

The complete collision model must cover both the body and every insertion pin;
a pin-only mesh would omit palm/body clearance. Disconnected components are
compiled as separate `part_collision*` geoms, so each exported convex piece
retains its own MuJoCo hull. The visual triangles remain the authoritative
surface for grasp generation.

When measured PCB/hole collision CAD is available, declare it separately from
the exact part target:

```yaml
insertion:
  collision_cad: assets/fixture/pcb_holes.step
  collision_cad_units: mm
  collision_cad_world_pose:  # ^W T_C for the supplied fixture CAD frame
    position_m: [0.425, -0.455, 0.347]
    rpy_deg: [180.0, 0.0, 0.0]
  targets:
    # world_part_pose and world_insertion_frame as shown above
```

`collision_cad_world_pose` is required in explicit-target mode unless the
legacy `pcb_world_pose` is retained as the same fixture transform. The scene
compiler names these geoms `insertion_collision*` and removes the solid
`pcb_board`/`pcb_fixture_base` placeholders while keeping the other generated
tables, bins, and reorientation surface. Declared fixture CAD receives no
penetration allowance: its semantic part/fixture pair may enter the broadphase
margin or touch, but any negative signed distance fails. Export concave
holes/chamfers as a deliberate convex decomposition; one concave mesh would
still collide as its convex hull.

### Optional XYZ and ASCII-PCD grasp proposal templates

`proposal_templates` can provide prior TCP locations or poses from a point-cloud
or template process. They are optional hints, not a substitute for part CAD.
Declare them either once at the project root or under the active part, but not
in both locations. The current handoff planner consumes only
`frame: part` plus `role: grasp_tcp`.

For a three-column XYZ position template, create a UTF-8 text file such as
`parts/conn_header/grasp_positions.xyz`:

```text
# ^P t_E position hints in millimetres
12.30 -4.20 0.80
18.10 -4.10 0.75
```

Declare it in `project.yaml`:

```yaml
proposal_templates:
  - name: connector_tcp_positions
    path: parts/conn_header/grasp_positions.xyz
    format: xyz
    frame: part
    role: grasp_tcp
    xyz_units: mm
```

Each row is a three-DoF position hint. It does not prescribe orientation; the
planner ranks nearby CAD-derived contact grasps while retaining all unmatched
CAD-valid candidates as fallback.

For a six-column pose template, each row is
`x y z roll pitch yaw`:

```text
# millimetres, then degrees
12.30 -4.20 0.80 180.0 0.0 90.0
18.10 -4.10 0.75 180.0 0.0 90.0
```

```yaml
proposal_templates:
  - name: connector_tcp_poses
    path: parts/conn_header/grasp_poses.xyz
    format: xyz
    frame: part
    role: grasp_tcp
    xyz_units: mm
    rpy_units: deg
```

RPY uses intrinsic XYZ, equivalently the active rotation
`Rz(yaw) @ Ry(pitch) @ Rx(roll)`. `rpy_units` must be `deg` or `rad`; rows must
have exactly three or six finite numeric values. Spaces or commas are accepted,
and `#` begins an inline comment.

ASCII PCD is also supported. Fields are located by name, so their order is not
significant. This example provides XYZ plus a preferred TCP `+Z` approach hint:

```text
VERSION .7
FIELDS x y z normal_x normal_y normal_z
SIZE 4 4 4 4 4 4
TYPE F F F F F F
COUNT 1 1 1 1 1 1
WIDTH 2
HEIGHT 1
POINTS 2
DATA ascii
12.30 -4.20 0.80 0.0 0.0 1.0
18.10 -4.10 0.75 1.0 0.0 0.0
```

```yaml
proposal_templates:
  - name: connector_tcp_cloud
    path: parts/conn_header/grasp_cloud.pcd
    format: ascii_pcd
    frame: part
    role: grasp_tcp
    xyz_units: mm
```

`normal_x normal_y normal_z` must be present together and are normalized before
matching. XYZ plus normals remains a three-DoF position proposal with a normal
hint; it is not interpreted as XYZ+RPY. Extra named PCD fields are ignored.
Binary/compressed PCD, partial normals, inconsistent `WIDTH*HEIGHT/POINTS`, and
more than the configured `template_max_proposals` are rejected.

In every format, the planner first generates antipodal contacts on the actual
part triangles. Templates then prioritize nearby candidates subject to the
system position, rotation, and normal association limits. They cannot create a
raw grasp, remove the complete CAD-derived fallback, bypass IK/collision/motion
checks, or become MuJoCo collision geometry. World-frame `part_pose` templates
exist in the generic parser data model but are intentionally rejected by the
current grasp consumer; insertion still uses the explicit target schema above.

### Reorientation support

`regions.reorientation` is a planar support frame and usable rectangle:

```yaml
reorientation:
  type: support_rectangle
  world_pose:
    position_m: [0.735, 0.315, 0.332]
    rpy_deg: [0.0, 0.0, 0.0]
  size_m: [0.240, 0.200]
```

Stable placements are derived from the part's actual triangle mesh. A placement
is retained only when its COM projection is inside the support polygon with the
configured margin and the entire footprint fits the stage. The place, release,
re-pick, and lift paths are then checked against the complete scene.

## 3. Build the scene

```bash
source .venv/bin/activate
python scripts/build_mujoco_scene.py \
  --project mujoco_sim/project.yaml \
  --output mujoco_sim/models/scene.xml
```

The build creates deterministic prepared CAD beside the MJCF under
`mujoco_sim/models/generated_cad/`. Rebuild after changing CAD, units, robot
bases, gripper/TCP transforms, initial joints, fixture geometry, or task-facing
scene inputs.

Check that the static scene loads:

```bash
mjpython -m mujoco_sim.viewer \
  --project mujoco_sim/project.yaml \
  --model mujoco_sim/models/scene.xml
```

This managed viewer displays the model but does not animate a pipeline.

## 4. Build production tables and policies

For the complete offline procedure and cache semantics, follow
[mujoco_offline_policies.md](mujoco_offline_policies.md). The normal production
commands are:

```bash
python scripts/build_reachability.py \
  --project mujoco_sim/project.yaml \
  --model mujoco_sim/models/scene.xml \
  --out mujoco_sim/cache

python scripts/precompute_pipeline.py \
  --project mujoco_sim/project.yaml \
  --project-root . \
  --model mujoco_sim/models/scene.xml \
  --cache-dir mujoco_sim/cache \
  --production
```

Use a distinct cache directory for every project. The planner can run without
precomputation, but the first request then performs the missing work on the
critical path.

## 5. Plan and execute headlessly

Plan only:

```bash
python -m mujoco_sim.pipeline \
  --project mujoco_sim/project.yaml \
  --model mujoco_sim/models/scene.xml \
  --cache mujoco_sim/cache
```

Plan and replay the deterministic MuJoCo executor without a GUI:

```bash
python -m mujoco_sim.pipeline \
  --project mujoco_sim/project.yaml \
  --model mujoco_sim/models/scene.xml \
  --cache mujoco_sim/cache \
  --execute
```

Useful options:

- `--json` emits the planning report and execution events for automation;
- `--no-regrasp` allows direct handoff only;
- `--best` evaluates the bounded candidate grid instead of returning the first
  complete feasible plan; and
- `--profile` prints setup, planning, execution, and critical-path bottlenecks;
  and
- `--debug-artifacts [LOG_ROOT]` captures one diagnostic bundle per executed
  stage; it has no effect on a plan-only run.

The normal output identifies feasibility, downstream-valid grasps, candidates,
gate statistics, timing, direct versus reorientation branch, execution outcome,
mathematical known-start coverage, physical certification, and explicit model
limitations. A successful replay should end with `pipeline_complete`.

### Cycle-time reporting

Execute headlessly to report both clocks without viewer pacing:

```bash
python -m mujoco_sim.pipeline --execute --no-regrasp --profile
```

The output distinguishes executed modeled robot time, planned modeled serial
makespan, observed computer wall time, and one duration for every completed
pipeline state. It also lists setup/planning/execution bottlenecks. The timing
model integrates geometric joint-path travel using the configured GP7 velocity
limits, phase speed fractions, and cubic-smoothstep peak factor. Collision
waypoints are safety samples rather than separate commands, so collinearly
densifying a path does not change modeled duration.

This is a joint-velocity simulation model, not a controller-matched minimum-time
trajectory or hardware measurement. Gripper/scanner operations without a
calibrated duration are explicitly reported as unmodeled and make
`timing_estimate_complete=false`. The estimate also omits PLC/network
handshakes, acceleration/jerk/blending, payload derating, and settling. For a
production cycle-time certificate, record controller timestamps at the same
state boundaries and add measured I/O/device delays. `observed computer wall
time` is diagnostic only and changes with CPU/GPU load, debug capture, and
`--playback-speed`.

`--best` can be substantially slower and has a separate content-addressed
policy key. Safety gates are identical; scoring never turns a failed collision,
IK, correction, or motion check into a valid plan.

## 6. Visualize the complete simulation

On macOS, passive MuJoCo applications must be launched with `mjpython`:

```bash
mjpython -m mujoco_sim.visualize_pipeline \
  --project mujoco_sim/project.yaml \
  --model mujoco_sim/models/scene.xml \
  --cache mujoco_sim/cache \
  --playback-speed 4 \
  --start-delay 5 \
  --hold -1
```

On Linux, use ordinary `python` for the same command. A negative `--hold` keeps
the completed state visible until the window is closed; a nonnegative value is
the number of seconds to hold it. `--playback-speed` changes visualization wall
time only: the same stored trajectories, 1 ms simulation steps, and continuous
collision monitor remain active. Values from `2` to `4` are useful for review;
`8` is convenient for quickly checking an entire cycle.

The forced connector-header reorientation example is:

```bash
mjpython -m mujoco_sim.visualize_reorientation_demo \
  --project mujoco_sim/project.yaml \
  --model mujoco_sim/models/scene.xml \
  --cache mujoco_sim/cache \
  --playback-speed 4 \
  --start-delay 5 \
  --hold -1
```

This demo intentionally forces a verified stage placement/re-pick route for
visualization. Production planning remains direct-first and may also have a
direct route for this rotated initial grasp.

### Record a simulation video on macOS

1. Launch either command with `--playback-speed 2` or `4` and
   `--start-delay 5`.
2. Press `Shift`+`Command`+`5` and choose **Record Selected Portion**.
3. Select the MuJoCo window, start recording, and then focus that window.
4. Stop from the macOS menu bar. macOS saves a `.mov` file by default.

For an MP4 copy, if FFmpeg is installed:

```bash
ffmpeg -i handoff.mov -c:v libx264 -pix_fmt yuv420p handoff.mp4
```

## Execution-stage debug artifacts

Debug capture is opt-in. A headless execution with the default `logs` root is:

```bash
python -m mujoco_sim.pipeline --execute --debug-artifacts
```

Pass an explicit root when several cells or test runs share a workspace:

```bash
python -m mujoco_sim.pipeline \
  --project mujoco_sim/project.yaml \
  --model mujoco_sim/models/scene.xml \
  --cache mujoco_sim/cache \
  --execute \
  --debug-artifacts diagnostics/handoff
```

The same `--debug-artifacts [LOG_ROOT]` and `--strict-debug` options exist on
`mujoco_sim.visualize_pipeline` and
`mujoco_sim.visualize_reorientation_demo`. On macOS, continue to launch these
passive-viewer modules with `mjpython`:

```bash
mjpython -m mujoco_sim.visualize_pipeline \
  --debug-artifacts logs \
  --hold -1
```

Each executor creates one unique UTC run name. Every recorded event
gets its own sanitized stage directory:

```text
logs/
  <YYYYMMDDTHHMMSS.ffffffZ>/
    <step>/
      contacts.png
      state.json
```

For example, direct execution normally produces stages such as `owned_by_A`,
`A_at_handoff`, `B_at_prehandoff`, `owned_by_B`, `A_clear`,
`scanned_virtual_exact`, `at_preinsert_00`,
`inserted_virtual_geometry_00`, and `complete`. Reorientation adds its place and
re-pick events. If a UTC name or step repeats, a deterministic `__02` suffix is
added instead of overwriting an earlier artifact.

`state.json` contains the stage event, plan and execution metadata, A/B joint
vectors, `^W T_E` for both TCPs, `^W T_P`, the active collision policy, and
every current contact's geom names/IDs, signed distance, penetration, world
point/frame, contact-frame wrench, world force/torque, and allowed-policy
decision. `contact_summary` records the count, allowed/unexpected counts, and
minimum signed distance. The `render` object states whether the PNG is a real
MuJoCo render.

`contacts.png` is rendered offscreen with MuJoCo contact points and forces
enabled. Offscreen OpenGL is best-effort: it can be unavailable in macOS
non-viewer processes, CI, or headless Linux without EGL. A render failure does
not discard the numerical evidence. The recorder writes a deterministic CPU
top/side projection of both robot chains, collision-geom centers, TCPs, the
part, fixtures, and allowed/forbidden contact points, and records:

```json
{
  "mujoco_rendered": false,
  "contact_visualization": false,
  "fallback_image": true,
  "fallback_kind": "cpu_top_side_contact_projection",
  "error": "<OpenGL error>"
}
```

The projection remains a useful contact schematic but is not an exact CAD
silhouette. For the full rendered scene, verify
`state.json:render.mujoco_rendered` is `true`.
Using `mjpython` is mandatory for the macOS passive viewer and may also provide
the required main-thread graphics context for a debug-enabled run. On headless
Linux, configure `MUJOCO_GL=egl` and a working EGL driver. The numerical JSON
and fallback PNG remain available when rendering cannot start.

By default, recorder initialization, serialization, and file-write errors are
isolated from execution and returned through `debug_errors`; the CLI prints
them as debug warnings. Add `--strict-debug` only in a diagnostic/CI workflow
when such recorder errors should affect the execution result:

```bash
python -m mujoco_sim.pipeline \
  --execute \
  --debug-artifacts logs \
  --strict-debug
```

`--strict-debug` without `--debug-artifacts` is rejected. An offscreen render
failure that successfully produces the documented fallback is not a recorder
failure, so strict mode does not require `mujoco_rendered: true`.

When `--debug-artifacts` is omitted—the production default—the executor creates
no recorder or log directory and performs no state serialization or offscreen
rendering. Planning and execution CT therefore retain the normal production
critical path. Enabling capture adds per-stage I/O/rendering and should not be
used for customer CT measurement.

## 7. Audit stage and PCB contacts

Run the contact audit against the normal plan:

```bash
python -m mujoco_sim.audit_contacts \
  --project mujoco_sim/project.yaml \
  --model mujoco_sim/models/scene.xml \
  --cache mujoco_sim/cache
```

For the forced reorientation branch:

```bash
python -m mujoco_sim.audit_contacts \
  --project mujoco_sim/project.yaml \
  --model mujoco_sim/models/scene.xml \
  --cache mujoco_sim/cache \
  --reorientation-demo
```

Use `--json` for a machine-readable report. Interpret signed contact distance
as follows:

- negative: geometric penetration;
- zero: touching; and
- positive: proximity exposed by MuJoCo's configured contact margin.

Part-to-stage contact at a stable placement is required and authorized only in
the named place/re-pick phase, with a bounded tolerance. Positive
gripper-to-stage proximity can be visible because the swept-path checker uses a
0.5 mm safety buffer, but gripper-to-stage penetration is forbidden. All
robot-to-fixture, wrist-to-part, gripper-to-gripper, and phase-unexpected
contacts remain hard failures.

The current reference insertion is different: its PCB is a solid primitive and
has no hole collision shape. The executor labels this
`placeholder_solid_board_10um_tolerance`; it checks the target pose, approach
path, IK, correction envelope, and surrounding collisions, but authorizes only
the part/board placeholder pair with at most 10 micrometres penetration during
insertion. Visible part/board overlap is therefore a declared model limitation,
not a physically certified insertion. Replace it with fixture/hole collision
CAD, a complete convex-decomposed part collision model containing the pins,
and calibrated contact materials before making a physical claim.

The execution monitor checks every replay waypoint and aborts on unexpected
penetration. In the GUI, MuJoCo's contact-point/contact-force visualization can
help localize a pair, but the signed audit and execution result are
authoritative.

For the current reference build, the verified forced-stage audit reports the
insertion model as `generated_virtual_aperture_placeholder`, the selected
stable placement, signed insertion and support distances, gripper/stage
penetration, and forbidden-contact counts. Values are run-specific; zero
forbidden contacts is required, but the placeholder result is not a physical
insertion certificate.

These numbers describe this exact compiled model and selected policy; rerun the
audit after any CAD, pose, solver, or policy change. `--json` includes every
sampled contact point and can be large; use the default text output for a quick
summary.

## 8. Qualify the configured domain

Smoke-check one class:

```bash
python scripts/qualify_pipeline.py \
  --project mujoco_sim/project.yaml \
  --model mujoco_sim/models/scene.xml \
  --cache mujoco_sim/cache \
  --max-classes 1 \
  --output mujoco_sim/cache/coverage-smoke.json
```

Qualify every declared class:

```bash
python scripts/qualify_pipeline.py \
  --project mujoco_sim/project.yaml \
  --model mujoco_sim/models/scene.xml \
  --cache mujoco_sim/cache \
  --required 1.0 \
  --output mujoco_sim/cache/coverage-certificate.json
```

Do not present a prefix smoke result as full-domain coverage. Also keep
`mathematical_coverage_certified` separate from `physical_certified`. The
former is scoped exactly to `qualification.initial_grasp_domain`; the latter
requires all physical-model prerequisites listed in the certificate.

## 9. Use an alternate project

Always pass the project/model/cache triplet together:

```bash
python scripts/build_mujoco_scene.py \
  --project path/to/project.yaml \
  --output path/to/build/scene.xml

python scripts/precompute_pipeline.py \
  --project path/to/project.yaml \
  --project-root path/to \
  --model path/to/build/scene.xml \
  --cache-dir path/to/cache \
  --production

python -m mujoco_sim.pipeline \
  --project path/to/project.yaml \
  --model path/to/build/scene.xml \
  --cache path/to/cache \
  --execute
```

For external manifests, prefer absolute asset paths or keep relative assets
under the manifest/project root. The compiled MJCF references prepared CAD
beside itself, so do not move `scene.xml` without its `generated_cad/`
directory.

## 10. Troubleshooting

### `RuntimeError: Caught an unknown exception!` on macOS

For `launch_passive` visualization, use `mjpython`, and ensure the environment
was created from a macOS framework Python:

```bash
python3 -c "import sysconfig; print(sysconfig.get_config_var('PYTHONFRAMEWORK') or 'NOT a framework build')"
```

If it prints `NOT a framework build`, install a framework Python from
python.org or Homebrew, recreate `.venv`, reinstall requirements, and retry.
Headless planning and tests do not need a framework build or display. The
static viewer also uses `launch_passive`; on macOS launch it with
`mjpython -m mujoco_sim.viewer`.

### The scene is correct but the animated plan is not

Confirm every command uses the same `--project`, `--model`, and `--cache`.
Rebuild the MJCF after changing the manifest. An alternate model with the
default project, or fixed-name reachability maps copied from another project,
is an invalid pairing even if it loads.

### No plan is found

Run with `--json` and inspect gate statistics. Then check, in this order:

1. CAD units and model scale;
2. calibrated world-base and mount-to-TCP transforms;
3. the known `^E T_P` start pose;
4. exact `^W T_P` insertion pose and `+Z_I` direction;
5. whether all target/pre-target origins lie in their declared regions;
6. collision CAD decomposition and unexpected initial penetration; and
7. whether direct-only failure succeeds when reorientation is enabled.

Do not start by loosening clearance, IK, or uncertainty gates. Those are safety
policy and a failed physical/frame assumption should remain visible.

### Grasp points appear to be on a convex hull

The candidate contacts are generated on actual triangles and opposing-surface
ray intersections. The convex-hull limitation belongs to MuJoCo mesh collision,
not grasp generation. Inspecting only the transparent collision geom can make
these look equivalent. Use the stored candidate contact points/normals and the
visual part mesh to audit grasps; supply convex-decomposed collision pieces for
concave contact geometry.

### STEP cannot be loaded

Install FreeCAD and pass `--freecad /path/to/FreeCADCmd` to
`prepare_project_cad.py`, or set `FREECADCMD`. Build the MJCF again after the
prepared tessellation exists. A hand-written MuJoCo `<mesh>` cannot point
directly to STEP/STP.

### A cache result looks stale

The JSON artifacts are content-addressed and verify their key and payload on
read. A relevant project/CAD/solver change should create a new digest rather
than return the old one. Verify that the intended cache directory was passed,
then compare `project-metadata.json`. Reachability `.npz` files are the
exception: rebuild or isolate them manually per project. See
[Fingerprints and automatic invalidation](mujoco_offline_policies.md#fingerprints-and-automatic-invalidation).

### Headless Linux rendering

Planning/execution needs no viewer. For offscreen rendering on a headless Linux
host, configure MuJoCo EGL (`MUJOCO_GL=egl`) and a working EGL driver. Ordinary
headless pipeline commands require neither EGL nor a display.

## Verification and release checklist

After changing planning, collision, execution, CAD preparation, or project
schema code, run:

```bash
python -m compileall -q mujoco_sim scripts
python scripts/run_mujoco_tests.py --tier t1
python scripts/run_mujoco_tests.py --tier t2
python scripts/run_mujoco_tests.py --tier t3
python -m mujoco_sim.pipeline --execute --profile
python -m mujoco_sim.audit_contacts --json
python scripts/qualify_pipeline.py --max-classes 1
```

`--max-classes 1` is a smoke run, not a full-domain certificate. Before
deployment, rebuild the full offline policies, run qualification without
truncation, review every physical prerequisite, and validate collision stopping
distances and contact parameters against the real cell.
