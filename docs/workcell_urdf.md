# Full workcell URDF

`assets/workcell/handoff_workcell.urdf` is the generated, single-rooted model
for calibration, TF publication, visualization, and geometry inspection. It is
not a replacement for the current MuJoCo scene compiler or the legacy
multi-body PyBullet runtime. Treat it as kinematics/collision-only: it is not a
dynamically calibrated model.

## Contents

The `world -> cell` tree contains:

- the exact workcell visual CAD, scaled from millimetres to metres;
- 30 stable workcell collision boxes and a finite floor approximation;
- the current 33-box provisional table/bin/reorientation/PCB fixture model;
- two independently prefixed GP7 trees (`A_*` and `B_*`), each with six
  revolute joints, arm visual/collision meshes, gripper CAD, `tool0`, and TCP;
- semantic frames for the tables, bins, reorientation surface, handoff and
  scanner regions, PCB, part target, and insertion direction; and
- any calibrated camera, hand-eye target, or additional frames enabled in
  `config/workcell_calibration.yaml`.

The task-dependent connector is intentionally separate: it is a free object,
not part of the fixed workcell tree. Its CAD and initial grasp remain in the
project manifest and simulation state.

The authoritative robot mounts are copied from
`mujoco_sim/config/project.yaml`: A is at the cell origin and B is at
`[0.850, 0, 0]` with 180 degrees of yaw. The generated model uses the current
gripper CAD TCP of 0.23292807 m, resolving the old GP7-only URDF's 0.200 m
placeholder.

Rebuild after changing any source asset or transform:

```bash
.venv/bin/python scripts/build_workcell_urdf.py
.venv/bin/python tests/test_workcell_urdf.py
```

Do not hand-edit the generated URDF. URDF also cannot store the robots'
initial joint state; that remains in `mujoco_sim/config/project.yaml`.

Mesh URIs are relative to the generated file, so keep the default URDF with
the repository's `assets/` layout. This repository is not currently a ROS
description package; if it is packaged for ROS/RViz, install the assets in that
package and generate `package://` mesh URIs as part of that packaging step.

## Adding a camera or hand-eye result

All calibration matrices use this explicit convention:

```text
parent_T_child maps coordinates from child into parent
```

Translation is in metres. The camera matrix is named
`parent_T_camera_optical` because an unlabeled “hand-eye matrix” is ambiguous:
calibration packages may return either camera-to-gripper or
gripper-to-camera. Invert the result once at the configuration boundary if it
is supplied in the opposite direction.

For an eye-to-hand camera, use `parent: cell`. For a camera mounted on robot A,
use `parent: A_tool0`; for robot B, use `parent: B_tool0`. Use `tool0`, not TCP,
because TCP changes with tooling.

Example eye-to-hand entry:

```yaml
cameras:
  - name: overhead
    enabled: true
    calibrated: true
    mode: eye_to_hand
    parent: cell
    parent_T_camera_optical:
      matrix:
        - [r00, r01, r02, tx]
        - [r10, r11, r12, ty]
        - [r20, r21, r22, tz]
        - [0.0, 0.0, 0.0, 1.0]
```

Example eye-in-hand entry:

```yaml
cameras:
  - name: robot_a_wrist
    enabled: true
    calibrated: true
    mode: eye_in_hand
    parent: A_tool0
    parent_T_camera_optical:
      matrix:
        - [r00, r01, r02, tx]
        - [r10, r11, r12, ty]
        - [r20, r21, r22, tz]
        - [0.0, 0.0, 0.0, 1.0]
```

The builder validates the matrix as SE(3), derives the mechanical
`camera_*_link`, and emits a ROS optical frame with +Z forward, +X image-right,
and +Y image-down. Small rotation drift from six-decimal serialization is
projected to the nearest proper rotation; materially invalid matrices are
rejected. A camera is omitted unless both `enabled` and `calibrated` are true,
so an identity placeholder cannot silently enter TF.

Camera intrinsics, distortion, image size, serial number, calibration
covariance, and timestamp do not belong in URDF. Store intrinsics in a ROS
`camera_info` YAML and keep calibration metadata beside the measured matrix.

## Loading in PyBullet

The complete workcell is one PyBullet multibody. Robot-to-cell and
robot-to-robot contacts are therefore self-collisions and require these flags:

```python
flags = (
    p.URDF_USE_SELF_COLLISION
    | p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT
)
body = p.loadURDF(
    "assets/workcell/handoff_workcell.urdf",
    useFixedBase=True,
    flags=flags,
)
```

Do not use `URDF_MERGE_FIXED_LINKS`; it can collapse the environment branch
and suppress internal robot/environment contacts. The current planner assumes
separate body IDs for both robots and the environment, so integrating this
unified URDF into that runtime requires collision-query changes. It is safe to
use immediately as the canonical calibration and frame artifact.

## Fidelity limits

- Fixture primitives are reconstructed from photographs and the requested
  0.93 m table height; they are not surveyed production geometry.
- GP7 masses and inertias come from the existing source URDF and are simplified.
- Frame-only fixed links intentionally have no invented inertials. PyBullet
  warns and assigns defaults to them, so do not use this unified body for
  payload, torque, or other dynamics calculations.
- The gripper CAD is fixed-open. Its collision geometry and inertia remain
  provisional until articulated, measured gripper data is supplied.
- The finite floor box approximates MuJoCo's infinite plane.
- URDF does not encode allowed-collision policies; use SRDF or an application
  collision matrix when this model is integrated with a motion planner.
