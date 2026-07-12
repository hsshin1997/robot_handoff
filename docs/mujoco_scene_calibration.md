# MuJoCo scene calibration

## Authoritative inputs

Physical and task inputs are centralized in `mujoco_sim/project.yaml`:

- robot URDF/MJCF and `world_base` pose;
- gripper model and mount-to-TCP transform;
- exact workstation visual CAD and collision representation;
- active part CAD, mass, and semantic part-to-pin frame;
- known startup `^E T_P` grasp;
- handoff, reorientation, scanner, and insertion regions;
- PCB world frame and PCB-to-hole frames.

`scene_config.yaml` now contains only the current lab's internal photo-matched
table/bin primitive fallback. It is enabled by
`workstation.generated_fixture_primitives: true`; turn that off when measured
fixture CAD is supplied through `additional_collision_cad`.

## CAD fidelity

The 723,724-triangle workcell and 3,000-triangle gripper visuals are retained
without downsampling. They are split only to satisfy MuJoCo's per-STL face
limit. Exact visuals are not automatically exact concave collision meshes:
MuJoCo uses a mesh convex hull for collision. Supply primitives, convex
decomposition, or separate articulated bodies for qualified collision/contact
work.

The current gripper STL has eight disconnected but unlabeled fixed components.
Those components now participate in collision checking, but they cannot open
or close. The gripper utility can derive aperture from slide/prismatic limits
in an articulated MJCF/URDF; the GP7 scene adapter must also instantiate those
bodies, joints, actuators, and pad contact names before execution can use it.

## Survey before hardware qualification

Measure and enter, in one world frame:

1. both robot base poses;
2. flange-to-gripper and gripper-to-TCP transforms;
3. reorientation support frame and usable boundary;
4. PCB frame and each functional hole frame;
5. scanner region/presentation constraints;
6. robot-to-robot calibration covariance.

Then rebuild and requalify:

```bash
python scripts/prepare_project_cad.py
python scripts/build_mujoco_scene.py
python scripts/precompute_pipeline.py --production
python scripts/qualify_pipeline.py
```

Any CAD, calibration, feature-frame, or solver-policy change invalidates the
affected content-addressed artifacts automatically.
