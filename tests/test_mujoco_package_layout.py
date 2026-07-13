#!/usr/bin/env python3
"""Regression gates for the modular package layout and legacy API surface."""
from __future__ import annotations

import importlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


MODULE_ALIASES = {
    "mujoco_sim.paths": "mujoco_sim.core.paths",
    "mujoco_sim.se3": "mujoco_sim.core.se3",
    "mujoco_sim.uncertainty": "mujoco_sim.core.uncertainty",
    "mujoco_sim.profiling": "mujoco_sim.core.profiling",
    "mujoco_sim.cad_preprocess": "mujoco_sim.modeling.cad_preprocess",
    "mujoco_sim.part_mesh": "mujoco_sim.modeling.part_mesh",
    "mujoco_sim.geometry_grasps": "mujoco_sim.modeling.grasps",
    "mujoco_sim.placements": "mujoco_sim.modeling.placements",
    "mujoco_sim.pose_templates": "mujoco_sim.modeling.pose_templates",
    "mujoco_sim.gripper": "mujoco_sim.modeling.gripper",
    "mujoco_sim.project": "mujoco_sim.modeling.project",
    "mujoco_sim.sim": "mujoco_sim.simulation.workcell",
    "mujoco_sim.kinematics": "mujoco_sim.simulation.kinematics",
    "mujoco_sim.collision": "mujoco_sim.simulation.collision",
    "mujoco_sim.phase_contacts": "mujoco_sim.simulation.contact_policies",
    "mujoco_sim.motion_planning": "mujoco_sim.planner.motion",
    "mujoco_sim.reachability": "mujoco_sim.planner.reachability",
    "mujoco_sim.task_graph": "mujoco_sim.planner.task_graph",
    "mujoco_sim.learning": "mujoco_sim.planner.learning",
    "mujoco_sim.planning_types": "mujoco_sim.planner.types",
    "mujoco_sim.plan_codec": "mujoco_sim.planner.codec",
    "mujoco_sim.plan_validation": "mujoco_sim.planner.validation",
    "mujoco_sim.planning": "mujoco_sim.planner.planner",
    "mujoco_sim.execution_types": "mujoco_sim.execution.types",
    "mujoco_sim.execution_schedule": "mujoco_sim.execution.schedule",
    "mujoco_sim.trajectory_timing": "mujoco_sim.execution.timing",
    "mujoco_sim.exec": "mujoco_sim.execution.executor",
    "mujoco_sim.offline": "mujoco_sim.offline_tools.artifacts",
    "mujoco_sim.precompute": "mujoco_sim.offline_tools.precompute",
    "mujoco_sim.qualification": "mujoco_sim.offline_tools.qualification",
    "mujoco_sim.debug_artifacts": "mujoco_sim.diagnostics.artifacts",
}


def _subprocess_environment() -> dict[str, str]:
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(ROOT) if not existing else os.pathsep.join((str(ROOT), existing)))
    return environment


def test_legacy_modules_are_exact_canonical_aliases():
    for legacy_name, canonical_name in MODULE_ALIASES.items():
        legacy = importlib.import_module(legacy_name)
        canonical = importlib.import_module(canonical_name)
        assert legacy is canonical, (legacy_name, canonical_name)

    old_stages = importlib.import_module("mujoco_sim.planner_stages")
    new_stages = importlib.import_module("mujoco_sim.planner.stages")
    assert old_stages.DirectHandoffSearch is new_stages.DirectHandoffSearch
    assert old_stages.ReorientationSearch is new_stages.ReorientationSearch


def test_default_paths_remain_at_package_root_after_moves():
    from mujoco_sim.core import paths
    from mujoco_sim.modeling.project import DEFAULT_PROJECT, DEFAULT_SOLVER
    from mujoco_sim.modeling.part_mesh import DEFAULT_GENERATED_CAD
    from mujoco_sim.simulation.workcell import MODEL

    assert paths.PACKAGE_ROOT == ROOT / "mujoco_sim"
    assert Path(DEFAULT_PROJECT) == paths.DEFAULT_PROJECT_PATH
    assert Path(DEFAULT_SOLVER) == paths.DEFAULT_SOLVER_PATH
    assert Path(MODEL) == paths.DEFAULT_MODEL_PATH
    assert Path(DEFAULT_GENERATED_CAD) == paths.GENERATED_CAD_ROOT
    for required in (
        paths.DEFAULT_PROJECT_PATH,
        paths.DEFAULT_SOLVER_PATH,
        paths.DEFAULT_SCENE_CONFIG_PATH,
        paths.DEFAULT_MODEL_PATH,
        paths.FREECAD_CONVERTER_PATH,
    ):
        assert required.is_absolute() and required.is_file(), required


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


def test_legacy_entrypoints_work_outside_repository_directory():
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
