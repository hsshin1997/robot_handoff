# MuJoCo setup and launch

This page is the short setup reference. The complete operational workflow is in
[mujoco_user_guide.md](mujoco_user_guide.md), and reproducible production-cache
construction is in [mujoco_offline_policies.md](mujoco_offline_policies.md).

## Install and build

```bash
source .venv/bin/activate
pip install -r requirements.txt

# Optional explicit CAD audit/prewarm. The scene builder also runs this.
# STEP/STP needs FreeCADCmd/freecadcmd (or FREECADCMD).
python scripts/prepare_project_cad.py

# Generate scene.xml from project.yaml.
python scripts/build_mujoco_scene.py

# Build low-latency grasp, stable-pose, downstream, and task-policy caches.
python scripts/build_reachability.py --project mujoco_sim/project.yaml \
  --model mujoco_sim/models/scene.xml --out mujoco_sim/cache
python scripts/precompute_pipeline.py --model mujoco_sim/models/scene.xml --production
```

For a physical/contact audit before qualification, run
`python -m mujoco_sim.audit_contacts`; see
[Audit stage and PCB contacts](mujoco_user_guide.md#7-audit-stage-and-pcb-contacts).

`project.yaml` is the user-owned interface: robots, bases, gripper, workcell,
part, known startup grasp, regions, exact world-part insertion targets, and
insertion/correction frames. Optional XYZ/ASCII-PCD proposal templates may
prioritize actual-CAD grasp candidates. STL units must be declared because STL
does not store units. `solver_defaults.yaml` is a system-owned numerical/safety
policy. The old `grasp_config.yaml` and `pipeline_config.yaml` are deprecated
placeholders and are not read by the default planner.

The current executable robot adapter is dual GP7. A different robot or an
articulated gripper needs a corresponding scene/kinematics adapter; replacing a
manifest CAD path alone does not provide kinematics or actuated finger contact.
For grippers, use the validated contract and model-specific integration
checklist in
[mujoco_gripper_integration.md](mujoco_gripper_integration.md).

## Plan, execute, and visualize

Headless plan and deterministic MuJoCo execution:

```bash
python -m mujoco_sim.pipeline
python -m mujoco_sim.pipeline --execute
```

Opt-in per-stage contacts, transforms, joint state, and PNG diagnostics:

```bash
python -m mujoco_sim.pipeline --execute --debug-artifacts
```

Diagnostics are disabled by default and add no capture work to production CT.
See [Execution-stage debug artifacts](mujoco_user_guide.md#execution-stage-debug-artifacts).

For an alternate project, keep the manifest and compiled MJCF paired:

```bash
python scripts/build_mujoco_scene.py --project path/project.yaml --output path/scene.xml
python scripts/precompute_pipeline.py --project path/project.yaml --model path/scene.xml --cache-dir path/cache --production
python scripts/qualify_pipeline.py --project path/project.yaml --model path/scene.xml --cache path/cache --output path/coverage.json
python -m mujoco_sim.pipeline --project path/project.yaml --model path/scene.xml --cache path/cache --execute
```

Interactive full-pipeline animation on macOS:

```bash
mjpython -m mujoco_sim.visualize_pipeline --hold -1
```

Verified reorientation example:

```bash
mjpython -m mujoco_sim.visualize_reorientation_demo --hold -1
```

Linux can normally use `python` instead of `mjpython`. On macOS, the standalone
model viewer uses the same passive viewer launcher:

```bash
mjpython -m mujoco_sim.viewer
```

The static viewer and pipeline visualizers use the non-blocking passive viewer.
The pipeline visualizers are preferred
because they animate the checked trajectories, ownership transfer, scanner,
and insertion stages rather than only displaying a static MJCF.

## Qualification and tests

```bash
# Small smoke certificate; omit --max-classes for the complete declared domain.
python scripts/qualify_pipeline.py --max-classes 1

python scripts/run_mujoco_tests.py --tier t1
python scripts/run_mujoco_tests.py --tier t2
python scripts/run_mujoco_tests.py --tier t3
```

The coverage certificate separates mathematical policy coverage from physical
certification. With the current single static gripper STL and virtual-aperture
PCB placeholder, absent pin/contact calibration, and ideal-weld executor, policy
coverage may be 100%, but physical certification is correctly false.

## macOS viewer troubleshooting

`RuntimeError: Caught an unknown exception!` from `mjpython` usually means the
environment does not use a macOS framework build of Python. Check:

```bash
python3 -c "import sysconfig; print(sysconfig.get_config_var('PYTHONFRAMEWORK') or 'NOT a framework build')"
```

Use a framework Python from python.org or Homebrew, recreate `.venv`, and use
`mjpython` for passive GUI modules. Headless planning/tests do not need a
framework build or a display. On headless Linux, use EGL for offscreen
rendering (`MUJOCO_GL=egl`).
