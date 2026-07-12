"""Checks for the minimal project manifest and feature-frame compiler."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from mujoco_sim.project import Project  # noqa: E402
from mujoco_sim.se3 import inverse  # noqa: E402


def test_manifest_assets_and_regions_are_valid():
    project = Project()
    assert project.initial_grasp_domain_source == "known_start"
    assert os.path.isfile(project.active_part_path)
    assert os.path.isfile(project.gripper("A").model_path)
    assert project.region("handoff").contains([0.4625, 0.0, 0.64])
    support = project.support_region()
    assert support.contains_local_xy([[0.0, 0.0], [0.10, 0.08]])
    assert not support.contains_local_xy([[0.13, 0.0]])


def test_pin_hole_feature_equation_is_exact():
    project = Project()
    hole_by_name = {
        target.name: target for target in project.insertion_targets()
    }
    for name, T_W_P in project.insertion_part_poses():
        target = hole_by_name[name]
        assert np.allclose(T_W_P, target.T_W_P_insert, atol=1e-12)
        # ^W T_P ^P T_pin == ^W T_H
        assert np.allclose(T_W_P @ project.T_part_pin, target.T_W_H, atol=1e-12)


def test_preinsert_offset_uses_hole_axis_not_world_axis():
    project = Project()
    target = project.insertion_targets(0.037)[0]
    delta = target.T_W_P_preinsert[:3, 3] - target.T_W_P_insert[:3, 3]
    assert np.allclose(delta, -0.037 * target.insertion_axis_world, atol=1e-12)
    assert np.isclose(np.linalg.norm(delta), 0.037)
    region = project.region("insertion")
    assert region.contains(target.T_W_P_insert[:3, 3])
    assert region.contains(target.T_W_P_preinsert[:3, 3])


def test_start_grasp_convention_round_trip():
    project = Project()
    assert np.allclose(inverse(project.T_tcp_part_start) @ project.T_tcp_part_start,
                       np.eye(4), atol=1e-14)


def test_external_project_resolves_assets_beside_its_manifest():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for name in ("robot.urdf", "gripper.stl", "cell.stl", "collision.yaml",
                     "part.stl"):
            (root / name).write_bytes(b"placeholder")
        manifest = {
            "schema_version": 1,
            "robots": {
                "A": {"model": "robot.urdf", "gripper": "g"},
                "B": {"model": "robot.urdf", "gripper": "g"},
            },
            "grippers": {"g": {"model": "gripper.stl"}},
            "workstation": {
                "visual_cad": "cell.stl",
                "collision_cad": "collision.yaml",
            },
            "parts": {"p": {"cad": "part.stl"}},
            "active_task": {"part": "p"},
            "qualification": {
                "initial_grasp_domain": {"source": "known_start"},
            },
        }
        path = root / "project.yaml"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        project = Project(str(path))
        assert project.active_part_path == str((root / "part.stl").resolve())
        assert project.resolve_asset("cell.stl") == str((root / "cell.stl").resolve())


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS  {test.__name__}")
    print(f"\n{len(tests)}/{len(tests)} passed")
