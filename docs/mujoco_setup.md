# MuJoCo setup and launch

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

`project.yaml` is the user-owned interface: robots, bases, gripper, workcell,
part, known startup grasp, regions, and pin/hole frames. STL units must be
declared because STL does not store units. `solver_defaults.yaml` is a
system-owned numerical/safety policy. The old `grasp_config.yaml` and
`pipeline_config.yaml` are deprecated placeholders and are not read by the
default planner.

The current executable robot adapter is dual GP7. A different robot or an
articulated gripper needs a corresponding scene/kinematics adapter; replacing a
manifest CAD path alone does not provide kinematics or actuated finger contact.

## Plan, execute, and visualize

Headless plan and deterministic MuJoCo execution:

```bash
python -m mujoco_sim.pipeline
python -m mujoco_sim.pipeline --execute
```

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

Linux can normally use `python` instead of `mjpython`. The standalone model
viewer remains:

```bash
python -m mujoco_sim.viewer
```

The pipeline visualizers use the non-blocking passive viewer and are preferred
because they animate the checked trajectories, ownership transfer, scanner,
and insertion stages rather than only displaying a static MJCF.

## Qualification and tests

```bash
# Small smoke certificate; omit --max-classes for the complete declared domain.
python scripts/qualify_pipeline.py --max-classes 1

for test in tests/test_mujoco*.py tests/test_geometry_grasps.py tests/test_motion_planning.py; do
  python "$test" || exit 1
done
```

The coverage certificate separates mathematical policy coverage from physical
certification. With the current single static gripper STL and solid PCB
placeholder, absent pin/contact calibration, and ideal-weld executor, policy
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
