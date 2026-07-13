#!/usr/bin/env python3
"""Regression gates for the modular source and configuration layout."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _subprocess_environment() -> dict[str, str]:
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(ROOT) if not existing else os.pathsep.join((str(ROOT), existing)))
    return environment


def test_root_python_surface_contains_only_launchers():
    package = ROOT / "mujoco_sim"
    expected = {
        "__init__.py",
        "audit_contacts.py",
        "pipeline.py",
        "viewer.py",
        "visualize_pipeline.py",
        "visualize_reorientation_demo.py",
    }
    assert {path.name for path in package.glob("*.py")} == expected
    assert not list(package.glob("*.yaml"))


def test_configuration_defaults_and_asset_resolution_survive_moves():
    from mujoco_sim.core import paths
    from mujoco_sim.modeling.project import (DEFAULT_PROJECT, DEFAULT_SOLVER,
                                             Project)
    from mujoco_sim.modeling.part_mesh import DEFAULT_GENERATED_CAD
    from mujoco_sim.simulation.workcell import MODEL

    assert paths.PACKAGE_ROOT == ROOT / "mujoco_sim"
    assert paths.CONFIG_ROOT == paths.PACKAGE_ROOT / "config"
    assert Path(DEFAULT_PROJECT) == paths.DEFAULT_PROJECT_PATH
    assert Path(DEFAULT_SOLVER) == paths.DEFAULT_SOLVER_PATH
    assert Path(MODEL) == paths.DEFAULT_MODEL_PATH
    assert Path(DEFAULT_GENERATED_CAD) == paths.GENERATED_CAD_ROOT
    for required in (
        paths.DEFAULT_PROJECT_PATH,
        paths.DEFAULT_SOLVER_PATH,
        paths.DEFAULT_SCENE_CONFIG_PATH,
        paths.DEFAULT_GRIPPER_TEMPLATE_PATH,
        paths.DEFAULT_PIPELINE_CONFIG_PATH,
        paths.DEFAULT_MODEL_PATH,
        paths.FREECAD_CONVERTER_PATH,
    ):
        assert required.is_absolute() and required.is_file(), required

    assert (paths.DEPRECATED_CONFIG_ROOT / "grasp_config.yaml").is_file()
    project = Project()
    references = [
        *(robot["model"] for robot in project.manifest["robots"].values()),
        *(gripper["model"] for gripper in
          project.manifest["grippers"].values()),
        project.manifest["workstation"]["visual_cad"],
        project.manifest["workstation"]["collision_cad"],
        *(part["cad"] for part in project.manifest["parts"].values()),
    ]
    assert all(Path(project.resolve_asset(reference)).is_file()
               for reference in references)


def test_canonical_modules_import_in_fresh_process_without_cycles():
    modules = (
        "mujoco_sim.core.se3",
        "mujoco_sim.offline_tools.artifacts",
        "mujoco_sim.modeling.cad_preprocess",
        "mujoco_sim.modeling.project",
        "mujoco_sim.simulation.workcell",
        "mujoco_sim.planner.motion",
        "mujoco_sim.simulation.collision",
        "mujoco_sim.planner.planner",
        "mujoco_sim.execution.executor",
        "mujoco_sim.diagnostics.contact_audit",
        "mujoco_sim.apps.pipeline",
    )
    script = "import importlib; " + "; ".join(
        f"importlib.import_module({name!r})" for name in reversed(modules))
    with tempfile.TemporaryDirectory() as directory:
        completed = subprocess.run(
            [sys.executable, "-c", script], cwd=directory,
            env=_subprocess_environment(), capture_output=True, text=True,
            check=False)
    assert completed.returncode == 0, completed.stderr


def test_stable_entrypoints_work_outside_repository_directory():
    entrypoints = (
        "mujoco_sim.pipeline",
        "mujoco_sim.viewer",
        "mujoco_sim.visualize_pipeline",
        "mujoco_sim.visualize_reorientation_demo",
        "mujoco_sim.audit_contacts",
    )
    with tempfile.TemporaryDirectory() as directory:
        for module in entrypoints:
            completed = subprocess.run(
                [sys.executable, "-m", module, "--help"], cwd=directory,
                env=_subprocess_environment(), capture_output=True, text=True,
                check=False)
            assert completed.returncode == 0, (module, completed.stderr)
            assert "usage:" in completed.stdout.lower(), module


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS  {test.__name__}")
    print(f"\n{len(tests)}/{len(tests)} passed")
