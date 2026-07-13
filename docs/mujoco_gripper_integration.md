# Articulated gripper integration contract

The future full gripper can replace the static reference assembly, but it is
not a CAD-path-only change. The current scene compiler turns one STL into fixed
visual/collision geoms; it does not import an MJCF or URDF kinematic subtree.
The validated contract in `mujoco_sim/gripper.py` defines the exact boundary
that an articulated scene adapter must satisfy.

## What to provide

Prefer MJCF. URDF is accepted by the contract validator, but MJCF can represent
MuJoCo contact parameters, equality constraints, and actuators directly.

The source model must contain:

1. one named mount body/link whose frame is attached to the GP7 flange;
2. separate palm and finger bodies/links (not one fused CAD mesh);
3. named slide/prismatic finger joints with physical limits;
4. named, separate visual and collision geometry;
5. at least two named pad collision geoms on the actual moving fingers;
6. a named TCP site (MJCF) or TCP link (URDF);
7. realistic mass/inertia, joint damping, contact friction, and actuator data.

Keep every mesh in its native link frame and declare its units. Concave visual
CAD is fine for rendering. Collision must use primitives or a deliberate convex
decomposition; MuJoCo collides with a mesh's convex hull, not its rendered
triangle surface.

## Descriptor

Copy `mujoco_sim/config/templates/gripper_asset.template.yaml` next to the gripper model, then
replace every example name with the exact source-model name. The validator
checks all of the following before the asset can be adapted:

- mount and TCP frames exist;
- joints are slide/prismatic and have finite limits;
- descriptor limits do not contradict model limits;
- pad names refer to actual collision geometry;
- each commanded finger joint has a declared pad in its moving subtree;
- every collision/visual name exists under the mount subtree;
- flange/mount/TCP transforms are valid SE(3) transforms;
- compiled-scene names are deterministically namespaced per robot.

Validate a descriptor from Python:

```bash
python - <<'PY'
from mujoco_sim.modeling.gripper import load_gripper_asset_contract

asset = load_gripper_asset_contract("path/to/gripper_asset.yaml")
print("model:", asset.model_path)
print("aperture [m]:", asset.actuation.aperture_range)
print("flange_to_tcp:\n", asset.T_F_E)
PY
```

No part-specific jaw-opening value is needed. The grasp candidate supplies the
actual contact separation; `ParallelJawActuation.joint_positions()` maps that
separation into the source joint limits.

## Scene-adapter obligations

The remaining model-specific adapter must:

1. import/copy the source gripper subtree and every referenced mesh, material,
   default class, actuator, tendon, and equality constraint;
2. attach its mount frame to each robot flange using `T_F_M`;
3. prefix all imported names using `scene_name_template`;
4. expose the TCP as a MuJoCo site at `T_M_E`;
5. keep named visual geoms non-colliding and named collision geoms colliding;
6. install joint actuators/mimic coupling and command the resolved aperture;
7. replace broad static `*_gripper_collision_*` holder allowances with the
   descriptor's exact pad geom names;
8. bind and fail closed with `bind_gripper_scene()` after compilation.

The post-compilation binding check is already implemented and tested against a
small articulated MJCF. It catches partial imports, missing joints, missing
pads, and namespace mismatches before planning starts.

## Different grippers on A and B

The contract and scene namespace support a distinct descriptor for each robot.
The current `scripts/build_mujoco_scene.py`, however, explicitly requires A and
B to share one robot URDF and one static gripper asset. Therefore different
grippers are not yet a supported scene configuration. The articulated adapter
must compile each robot's selected descriptor independently; removing the
existing equality check alone would be insufficient because the current asset
and geom names are also shared.

## Acceptance checks after the model arrives

- Both minimum and maximum apertures compile and move the intended fingers.
- Requested aperture equals measured pad-to-pad separation across the range.
- TCP pose agrees with the calibrated physical tool frame.
- Palm/finger self-collision behaves as intended; adjacent exclusions are
  explicit, not broad masks.
- Pad/part contact occurs on the supplied pad surfaces, not on a hull proxy.
- Closing force, capture, slip, release, and insertion-force guards pass.
- A and B can be instantiated independently without shared-name collisions.
