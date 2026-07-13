"""Checks for explicit and legacy insertion-task project schemas."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from mujoco_sim.modeling.part_mesh import load_project_part_mesh  # noqa: E402
from mujoco_sim.modeling.project import DEFAULT_PROJECT, Project  # noqa: E402
from mujoco_sim.core.se3 import inverse, make_transform  # noqa: E402


def _modified_default(mutator):
    with open(DEFAULT_PROJECT, encoding="utf-8") as stream:
        manifest = yaml.safe_load(stream)
    mutator(manifest)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "project.yaml"
        path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
        return Project(str(path))


def _assert_project_error(mutator, expected: str):
    try:
        _modified_default(mutator)
    except ValueError as error:
        assert expected in str(error), str(error)
    else:
        raise AssertionError(f"expected project validation error containing {expected!r}")


def test_manifest_assets_and_regions_are_valid():
    project = Project()
    assert project.initial_grasp_domain_source == "known_start"
    assert os.path.isfile(project.active_part_path)
    assert os.path.isfile(project.gripper("A").model_path)
    assert project.region("handoff").contains([0.4625, 0.0, 0.64])
    support = project.support_region()
    assert support.contains_local_xy([[0.0, 0.0], [0.10, 0.08]])
    assert not support.contains_local_xy([[0.13, 0.0]])


def test_explicit_world_part_pose_and_insertion_axes_are_exact():
    project = Project()
    target = project.insertion_targets()[0]
    declared = project.manifest["insertion"]["targets"][0]
    expected_part = make_transform(
        declared["world_part_pose"]["rotation_matrix"],
        declared["world_part_pose"]["position_m"],
    )
    expected_frame = make_transform(
        declared["world_insertion_frame"]["rotation_matrix"],
        declared["world_insertion_frame"]["position_m"],
    )
    assert np.array_equal(target.T_W_P_insert, expected_part)
    assert np.array_equal(target.T_W_I, expected_frame)
    assert np.array_equal(target.T_W_H, expected_frame)  # legacy API alias
    assert np.array_equal(target.insertion_axis_world, expected_frame[:3, 2])
    assert np.array_equal(target.T_W_I[:3, 0], [1.0, 0.0, 0.0])
    assert np.array_equal(target.T_W_I[:3, 1], [0.0, -1.0, 0.0])
    assert np.array_equal(project.insertion_part_poses()[0][1], expected_part)


def test_connector_target_inserts_short_pin_end():
    """The right-angle header's native -Z short tails must enter the PCB."""
    project = Project()
    target = project.insertion_targets()[0]
    mesh = load_project_part_mesh(project).mesh

    vertices = mesh.triangles.reshape(-1, 3)
    short_tip_vertices = vertices[np.isclose(vertices[:, 2], mesh.bounds_min[2])]
    pin_x = 0.5 * (
        np.min(short_tip_vertices[:, 0]) + np.max(short_tip_vertices[:, 0]))
    short_tip_P = np.array([pin_x, 0.0, mesh.bounds_min[2], 1.0])
    seat_P = np.array([pin_x, 0.0, 0.0, 1.0])
    short_tip_W = target.T_W_P_insert @ short_tip_P
    seat_W = target.T_W_P_insert @ seat_P

    assert np.allclose(
        target.T_W_P_insert[:3, :3] @ np.array([0.0, 0.0, -1.0]),
        target.insertion_axis_world,
    )
    assert np.allclose(seat_W[:3], target.T_W_I[:3, 3], atol=1e-8)
    assert np.isclose(
        np.dot(short_tip_W[:3] - seat_W[:3], target.insertion_axis_world),
        0.0032,
        atol=1e-8,
    )


def test_legacy_pin_hole_feature_equation_is_preserved():
    def legacy(manifest):
        manifest["parts"]["connector_header"]["part_to_pin"] = {
            "position_m": [0.0127075, -0.010796, 0.0],
            "rotation_matrix": [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
            ],
        }
        insertion = manifest["insertion"]
        insertion.pop("targets")
        insertion["holes"] = [{
            "name": "legacy_slot",
            "pcb_to_hole": {
                "position_m": [0.0, 0.0, 0.0],
                "rpy_deg": [0.0, 0.0, 0.0],
            },
        }]

    project = _modified_default(legacy)
    target = project.insertion_targets()[0]
    assert target.name == "legacy_slot"
    assert np.allclose(
        target.T_W_P_insert @ project.T_part_pin,
        target.T_W_I,
        atol=1e-12,
    )


def test_preinsert_offset_uses_hole_axis_not_world_axis():
    project = Project()
    target = project.insertion_targets(0.037)[0]
    delta = target.T_W_P_preinsert[:3, 3] - target.T_W_P_insert[:3, 3]
    assert np.allclose(delta, -0.037 * target.insertion_axis_world, atol=1e-12)
    assert np.isclose(np.linalg.norm(delta), 0.037)
    region = project.region("insertion")
    assert region.contains(target.T_W_P_insert[:3, 3])
    assert region.contains(target.T_W_P_preinsert[:3, 3])


def test_explicit_and_legacy_modes_cannot_be_mixed():
    def ambiguous(manifest):
        manifest["insertion"]["holes"] = [{
            "name": "also_legacy",
            "pcb_to_hole": {
                "position_m": [0.0, 0.0, 0.0],
                "rpy_deg": [0.0, 0.0, 0.0],
            },
        }]

    _assert_project_error(ambiguous, "ambiguous")


def test_explicit_target_requires_both_part_and_insertion_frames():
    def missing_frame(manifest):
        del manifest["insertion"]["targets"][0]["world_insertion_frame"]

    _assert_project_error(missing_frame, "requires world_insertion_frame")


def test_invalid_explicit_insertion_frame_is_rejected():
    def reflected_frame(manifest):
        manifest["insertion"]["targets"][0]["world_insertion_frame"] = {
            "position_m": [0.425, -0.455, 0.347],
            "rotation_matrix": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, -1.0],
            ],
        }

    _assert_project_error(reflected_frame, "not a valid SE(3) pose")


def test_explicit_fixture_collision_cad_requires_its_own_world_pose():
    def missing_pose(manifest):
        manifest["insertion"].pop("pcb_world_pose", None)
        manifest["insertion"]["collision_cad"] = (
            "parts/conn_header/conn_header_bin.stl")

    _assert_project_error(missing_pose, "collision_cad_world_pose")

    def declared_pose(manifest):
        manifest["insertion"]["collision_cad"] = (
            "parts/conn_header/conn_header_bin.stl")
        manifest["insertion"]["collision_cad_world_pose"] = {
            "position_m": [0.425, -0.455, 0.347],
            "rpy_deg": [0.0, 0.0, 0.0],
        }

    project = _modified_default(declared_pose)
    assert project.manifest["insertion"]["collision_cad_world_pose"][
        "position_m"] == [0.425, -0.455, 0.347]


def test_insert_and_preinsert_must_both_lie_in_insertion_region():
    def insert_outside(manifest):
        manifest["insertion"]["targets"][0]["world_part_pose"]["position_m"] = [
            1.0, 1.0, 1.0,
        ]

    _assert_project_error(insert_outside, "lies outside regions.insertion")

    def preinsert_outside(manifest):
        # Insert is just inside the upper Z face. The configured insertion +Z
        # points down, so the default 40 mm pre-insert offset exits upward.
        manifest["insertion"]["targets"][0]["world_part_pose"]["position_m"] = [
            0.425, -0.455, 0.439,
        ]

    _assert_project_error(preinsert_outside, "pre-insertion target")


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
            "regions": {
                "insertion": {
                    "type": "box",
                    "center_m": [0.0, 0.0, 0.0],
                    "size_m": [1.0, 1.0, 1.0],
                },
            },
            "insertion": {
                "targets": [{
                    "name": "target",
                    "world_part_pose": {
                        "position_m": [0.0, 0.0, 0.0],
                        "rpy_deg": [0.0, 0.0, 0.0],
                    },
                    "world_insertion_frame": {
                        "position_m": [0.0, 0.0, 0.0],
                        "rpy_deg": [0.0, 0.0, 0.0],
                    },
                }],
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
