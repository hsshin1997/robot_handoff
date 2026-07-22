# General workcell URDF generator

`scripts/setup_workcell_urdf.py` builds one connected workcell URDF from a
YAML manifest. The generator is vendor-neutral: the manifest may contain any
number of robot URDFs, static or articulated grippers, static cell geometry,
named frames, and fixed or robot-mounted cameras.

The list lengths define the workcell size. There are no separate
`number_of_robots` or `number_of_cameras` fields to keep synchronized.

Use this generator when constructing a reusable model. The older
`scripts/build_workcell_urdf.py` remains the exact, GP7-specific builder for
the current handoff cell.

## Quick start

The checked-in example recreates the main structure of the current two-GP7
cell while exercising the general schema:

```bash
.venv/bin/python scripts/setup_workcell_urdf.py config/workcell_generator.yaml
```

Validate all source files, transforms, references, and camera values without
writing output:

```bash
.venv/bin/python scripts/setup_workcell_urdf.py \
  config/workcell_generator.yaml --validate-only
```

The output paths are set in the manifest. Command-line overrides are useful
for a temporary build or CI check; run `--help` for the complete list.

Do not hand-edit generated artifacts. Change the source URDF, source geometry,
or manifest and run the setup command again.

## Generated artifacts

One build emits three kinds of artifact:

- a single-rooted URDF containing the workcell's geometry, joints, and named
  frames;
- one standard ROS-style `camera_info` YAML per calibrated camera stream; and
- a generation report with imported-model inventories, camera operating
  envelopes, derived field-of-view values, and warnings.

URDF is the canonical frame and geometry model. Camera intrinsics, distortion,
working distance, measurable depth range, and focus range are companion
metadata because URDF has no portable standard fields for them.

Use a dedicated `camera_info_dir` for each generated workcell. The report's
`outputs.camera_info` list is the authoritative inventory; generation does
not delete obsolete files left in that directory by an older manifest.

## Manifest outline

`config/workcell_generator.yaml` is the executable reference manifest. Its
top-level structure is:

```yaml
schema_version: 1
name: example_workcell
root_link: world

output:
  urdf: build/example_workcell.urdf
  camera_info_dir: build/camera_info
  report: build/example_workcell.report.yaml
  mesh_uri_mode: relative
  extension_policy: drop
  mujoco_fusestatic: false

package_roots: {}
frames: []
static_bodies: []
robots: []
grippers: []
cameras: []
attached_frames: []
```

Names must be unique within their section and should contain only letters,
digits, and underscores, starting with a letter or underscore. All distances
and translations are in metres unless a field explicitly declares another
unit.

### Transforms

Every transform follows one rule:

```text
parent_T_child maps coordinates expressed in child into parent
```

Use a translation plus exactly one orientation representation:

```yaml
parent_T_child:
  position_m: [0.85, 0.0, 0.0]
  rpy_deg: [0.0, 0.0, 180.0]
```

`rotation_matrix` and `quaternion_xyzw` are alternatives to `rpy_deg`. A
complete homogeneous matrix is also accepted:

```yaml
parent_T_child:
  matrix:
    - [r00, r01, r02, tx]
    - [r10, r11, r12, ty]
    - [r20, r21, r22, tz]
    - [0.0, 0.0, 0.0, 1.0]
```

The generator checks that every transform is finite and is a proper rigid
transform. It rejects reflections, malformed matrices, and materially
non-orthonormal rotations.

### Parent references

A component can attach to a known link directly or use a typed reference.
Typed references make intent clear and continue to work after imported names
are namespaced:

```yaml
parent: {frame: cell}
parent: {body: workcell_frame}
parent: {robot: robot_a, link: tool0}
parent: {gripper: robot_a_gripper, link: tcp}
parent: {camera: wrist_camera, link: optical}
```

Use a robot link such as `tool0` for a hand-eye transform when the transform
is relative to the robot flange. A TCP is tool-dependent and may move when the
gripper changes.

## Frames and static cell geometry

`frames` add fixed, geometry-free links. They are useful for the cell datum,
survey monuments, calibration targets, task regions, and other semantic
coordinates:

```yaml
frames:
  - name: cell
    parent: world
    parent_T_child:
      position_m: [0.0, 0.0, 0.0]
      rpy_deg: [0.0, 0.0, 0.0]
```

`static_bodies` add fixed links with visuals, collisions, and optional
inertial data. Geometry may be a mesh, box, cylinder, or sphere. Mesh units
must be explicit so millimetre CAD cannot silently be interpreted as metres:

```yaml
static_bodies:
  - name: workcell_frame
    parent: {frame: cell}
    parent_T_body:
      position_m: [0.0, 0.0, 0.0]
      rpy_deg: [0.0, 0.0, 0.0]
    visuals:
      - geometry:
          mesh:
            path: assets/workcell/workcell.stl
            units: mm
        material:
          name: frame_gray
          rgba: [0.45, 0.48, 0.52, 1.0]
    collisions:
      - geometry:
          box_set:
            path: assets/workcell/collision_boxes.yaml
            units: from_file
            groups: [pedestals, boxes]
```

A `box_set` expands a surveyed or simplified list of boxes into collision
elements. Visual CAD and collision geometry are deliberately independent;
high-resolution render meshes generally make poor collision models.

## Robots from any vendor

Each robot is imported from a normal URDF. The generator does not assume a
Yaskawa GP7, six joints, particular link names, or even only revolute joints.
For example, the two entries in one manifest can point to a Nachi URDF and a
Universal Robots URDF:

```yaml
robots:
  - name: nachi_left
    path: vendor/nachi/robot.urdf
    parent: {frame: cell}
    parent_T_root:
      position_m: [0.0, 0.0, 0.0]
      rpy_deg: [0.0, 0.0, 0.0]
    root_link: base_link
    flange_link: tool0

  - name: ur_right
    path: vendor/ur/robot.urdf
    parent: {frame: cell}
    parent_T_root:
      position_m: [0.85, 0.0, 0.0]
      rpy_deg: [0.0, 0.0, 180.0]
    root_link: base_link
    flange_link: tool0
```

`root_link` is an assertion against the source model, so selecting the wrong
URDF fails early. `flange_link` records the intended tool attachment point in
the report and helps reviewers catch a link-name mismatch.

Every imported link, joint, material, and transmission is namespaced with the
instance name. Internal references, including mimic-joint and transmission
references, are rewritten consistently. Thus two source URDFs may both use
names such as `base_link` and `joint_1` without colliding in the workcell
model.

If a robot source URDF already embeds a tool, remove that branch before
attaching a separately configured gripper:

```yaml
    prune_subtrees_at_joints: [tool0-gripper]
```

The named source joint and all links below it are omitted. This is important
for the checked-in GP7 model, which already contains a gripper branch.

Relative mesh paths resolve from the source URDF. `file://` paths are
supported. For `package://vendor_description/...` paths, add the package's
filesystem root:

```yaml
package_roots:
  vendor_description: ../vendor/vendor_description
```

The output `mesh_uri_mode` controls whether generated mesh paths are relative,
absolute, or preserved as `package://` URIs where possible.

For direct MuJoCo URDF loading, set `mujoco_fusestatic: false` as in the
checked-in manifest. This emits MuJoCo's compiler extension so fixed semantic
links such as camera optical frames and TCPs remain addressable by name.
Without it, MuJoCo's URDF importer may fuse fixed links into their parents.

## Grippers and other end effectors

Grippers attach to any imported robot link. Two modes cover common model
sources.

An articulated gripper is another URDF import. Its joints, mimic relationships,
materials, and transmissions use the same namespace-safe importer as a robot.
Its root is connected to the selected flange with a fixed mounting transform.
Name the source link that represents the gripper TCP:

```yaml
grippers:
  - name: two_finger_tool
    type: urdf
    path: vendor/gripper/two_finger.urdf
    parent: {robot: ur_right, link: tool0}
    parent_T_root:
      position_m: [0.0, 0.0, 0.0]
      rpy_deg: [0.0, 0.0, 0.0]
    root_link: gripper_base
    tcp_link: tcp
```

A static gripper is a fixed mesh or primitive assembly with explicit visual,
collision, inertial, and TCP data. A typical static entry has this shape:

```yaml
grippers:
  - name: left_gripper
    type: mesh
    parent: {robot: nachi_left, link: tool0}
    parent_T_mount:
      position_m: [0.0, 0.0, 0.0]
      rpy_deg: [0.0, 90.0, 0.0]
    visuals:
      - geometry:
          mesh:
            path: assets/gripper/gripper.stl
            units: mm
    collisions:
      - geometry:
          box: {size_m: [0.08, 0.10, 0.18]}
    tcp:
      name: tcp
      mount_T_tcp:
        position_m: [0.0, 0.0, 0.23]
        rpy_deg: [0.0, 0.0, 0.0]
```

Provide an intentional collision approximation; the generator does not treat
a visual mesh as collision geometry automatically. For an articulated model,
identify an existing source TCP link or add a fixed TCP transform according to
the model type.

The generator constructs the kinematic model only. Gripper actuator commands,
controller configuration, allowed-collision rules, payload limits, and grasp
state remain application configuration.

## Cameras

A camera contributes a physical `camera_link` and a ROS-convention optical
frame. The optical axes are +X image-right, +Y image-down, and +Z forward.

Attach an eye-to-hand camera to a cell frame, or an eye-in-hand camera to a
robot or gripper link. Supply exactly one of `parent_T_camera_link` and
`parent_T_camera_optical`; the direct optical transform is often the least
ambiguous representation of a hand-eye calibration result:

```yaml
cameras:
  - name: overhead
    enabled: true
    mode: eye_to_hand
    pose_status: calibrated
    parent: {frame: cell}
    parent_T_camera_optical:
      matrix:
        - [r00, r01, r02, tx]
        - [r10, r11, r12, ty]
        - [r20, r21, r22, tz]
        - [0.0, 0.0, 0.0, 1.0]

    intrinsics:
      status: calibrated
      image_width: 1920
      image_height: 1200
      fx: 1450.0
      fy: 1448.0
      cx: 959.5
      cy: 599.5
      skew: 0.0
      distortion_model: plumb_bob
      distortion_coefficients: [k1, k2, p1, p2, k3]

    operating_envelope:
      working_distance_m: {min: 0.35, nominal: 0.55, max: 0.80}
      view_depth_m: {near: 0.20, far: 1.20}
      depth_measurement_range_m: {min: 0.25, max: 1.00}
      focus:
        depth_of_field_m: {near: 0.40, far: 0.72}
```

`pose_status` distinguishes a `nominal` layout pose from a `measured` or
`calibrated` pose. A nominal transform is useful for layout and visualization,
but must not be represented downstream as measured calibration. Disable a
placeholder camera if even a nominal transform is not meaningful.

The optional intrinsic `status` is tracked separately because a camera can
have calibrated intrinsics while its workcell pose is still only nominal.

### Intrinsic calibration

The normal pinhole inputs are image width and height plus `fx`, `fy`, `cx`,
`cy`, and optional `skew`, all in pixels. Also supply the distortion model and
coefficients produced by calibration. Supported coefficient counts are:

- `plumb_bob`: 5 (`k1`, `k2`, `p1`, `p2`, `k3`);
- `rational_polynomial`: 8; and
- `equidistant`: 4.

You may paste conventional ROS calibration matrices instead of splitting K
into scalar values. `camera_matrix`, `distortion_coefficients`,
`rectification_matrix`, and `projection_matrix` accept either nested matrices
or standard `{rows, cols, data}` mappings; `K`, `D`, `R`, and `P` are aliases.
Do not provide both K and the scalar focal/principal-point fields.

Optional rectification and projection matrices may be supplied when the
calibration process produced them. The generated camera-info file contains
the conventional `image_width`, `image_height`, `camera_name`, `K`, `D`, `R`,
and `P` values. Intrinsics are always tied to the stated image resolution;
rescaling or cropping an image requires corresponding intrinsics.

For a pinhole model, the report derives horizontal and vertical angular FOV
from the intrinsics, including an off-centre principal point. At distance
`z`, the nominal image footprint is also reported from `z * width / fx` and
`z * height / fy`. Raw fisheye FOV is not inferred from pinhole formulas.

For a device with separate color, depth, or infrared calibration, replace the
single `intrinsics` mapping with a `streams` list. Each stream has a unique
`name`, a `type` (`color`, `mono`, `depth`, or `infrared`), and a
`calibration` mapping using the same fields. The generator writes one
camera-info file per stream while keeping one physical/optical camera frame.

### Working distance and the three depth ranges

These values answer different questions and should not be collapsed into one
"depth of FOV" field:

- `working_distance_m` is the preferred mounting/use interval, with a nominal
  distance;
- `view_depth_m` is the near/far 3-D viewing or rendering frustum;
- `depth_measurement_range_m` is where a depth camera claims usable depth
  measurements; and
- `focus.depth_of_field_m` is the optical interval expected to be acceptably
  in focus.

For an RGB camera without a depth sensor, omit
`depth_measurement_range_m`. For a fixed-focus camera, provide depth of field
when it is known from the lens and aperture setup. The generator validates
range ordering but does not invent missing vendor specifications.

### Frames that depend on imported components

Use `attached_frames` for semantic or calibration frames whose parent is not
available until a robot, gripper, or camera has been created. For example:

```yaml
attached_frames:
  - name: wrist_calibration_target
    parent: {gripper: left_gripper, link: tcp}
    parent_T_child:
      position_m: [0.0, 0.0, 0.05]
      rpy_deg: [0.0, 0.0, 0.0]
```

These are ordinary fixed URDF links. The separate section only controls build
order so typed parent references can be resolved.

## Import validation and extension policy

Before writing anything, the generator validates the complete frame graph,
component names, source URDF roots, joint references, mimic references,
transmission references, transforms, mesh paths, intrinsics, and range
ordering. Each generated file is written atomically after validation.

Imported source files must be expanded URDF XML. Raw `.xacro` files and
unresolved `${...}` or `$(...)` expressions are rejected; run the vendor's
Xacro process first and reference its resulting URDF.

Standard URDF links, joints, materials, and transmissions are imported.
Vendor-specific top-level extensions such as Gazebo plugins or `ros2_control`
blocks cannot always be safely namespaced:

- `extension_policy: drop` omits unsupported extensions and records warnings
  in the report; and
- `extension_policy: reject` fails the build instead.

Use `reject` in strict CI when every source element must be accounted for.

## What the generated URDF does not replace

The unified model is useful for TF, calibration, visualization, collision
geometry, and as an input to systems that accept a single URDF. It does not by
itself provide:

- robot or gripper controllers and initial joint positions;
- an SRDF or application-specific allowed-collision matrix;
- calibrated dynamics, actuator limits beyond the source URDF, or payload
  certification;
- camera drivers, image topics, exposure settings, trigger timing, or temporal
  calibration; or
- automatic conversion of the current planner from separate bodies to one
  unified collision body.

Keep those concerns in the runtime configuration that owns them. Treat the
manifest and measured calibration records as source data and regenerate the
URDF whenever the physical layout or installed model changes.
