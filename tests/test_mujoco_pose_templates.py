"""Acceptance tests for conservative XYZ/ASCII-PCD pose proposals."""
from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import mujoco_sim.planning as planning_module
from mujoco_sim.geometry_grasps import GraspCandidate
from mujoco_sim.planning import HandoffPlanner
from mujoco_sim.pose_templates import (
    PROPOSAL_ONLY_USAGE,
    RPY_CONVENTION,
    TemplateFrame,
    TemplateRole,
    load_declared_pose_templates,
    load_pose_template,
    rank_contact_validated_grasps,
)
from mujoco_sim.se3 import rpy_matrix


def _assert_raises(error_type, callback, text: str):
    try:
        callback()
    except error_type as error:
        assert text in str(error), str(error)
    else:
        raise AssertionError(f"expected {error_type.__name__}")


def _candidate(position, rotation=None) -> GraspCandidate:
    transform = np.eye(4)
    transform[:3, :3] = np.eye(3) if rotation is None else rotation
    transform[:3, 3] = position
    closing = transform[:3, 1]
    approach = transform[:3, 2]
    opening = 0.01
    points = np.vstack((
        transform[:3, 3] - 0.5 * opening * closing,
        transform[:3, 3] + 0.5 * opening * closing,
    ))
    normals = np.vstack((-closing, closing))
    return GraspCandidate(
        T_P_E=transform,
        contact_points=points,
        contact_normals=normals,
        required_opening=opening,
        approach_direction=approach,
        closing_direction=closing,
        quality=0.8,
        antipodal_quality=1.0,
        support_quality=0.8,
        opening_margin=0.5,
        palm_clearance=0.01,
    )


def test_xyz_rows_are_explicit_three_or_six_dof_with_intrinsic_xyz_rpy():
    content = """# part-frame grasp TCP hints, millimetres and degrees
100 200 300
100, 200, 300, 90, 0, 90  # XYZ + roll pitch yaw
"""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "poses.xyz"
        path.write_text(content, encoding="utf-8")
        template = load_pose_template(
            path, frame="part", role="grasp_tcp",
            xyz_units="mm", rpy_units="deg")

    assert template.frame == TemplateFrame.PART
    assert template.role == TemplateRole.GRASP_TCP
    assert template.usage == PROPOSAL_ONLY_USAGE
    assert template.rpy_convention == RPY_CONVENTION
    assert [item.dof for item in template.proposals] == [3, 6]
    assert template.proposals[0].T_frame_target is None
    assert np.allclose(template.proposals[0].position_m, [0.1, 0.2, 0.3])
    explicit = template.proposals[1].T_frame_target
    assert np.allclose(explicit[:3, 3], [0.1, 0.2, 0.3])
    assert np.allclose(
        explicit[:3, :3], rpy_matrix(np.radians([90.0, 0.0, 90.0])))
    assert not template.proposals[0].position_m.flags.writeable


def test_ascii_pcd_uses_named_xyz_and_normal_fields_not_six_dof_inference():
    pcd = """VERSION .7
FIELDS normal_z x normal_x y z normal_y intensity
SIZE 4 4 4 4 4 4 4
TYPE F F F F F F F
COUNT 1 1 1 1 1 1 1
WIDTH 2
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS 2
DATA ascii
1 100 0 200 300 0 5
0 400 1 500 600 0 7
"""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "cloud.pcd"
        path.write_text(pcd, encoding="utf-8")
        template = load_pose_template(
            path, frame="part", role="grasp_tcp", xyz_units="mm")

    assert template.source_format == "ascii_pcd"
    first, second = template.proposals
    # Six named XYZ+normal fields remain position proposals. They are not
    # reinterpreted positionally as XYZ+RPY.
    assert first.dof == second.dof == 3
    assert first.rpy_rad is None and first.T_frame_target is None
    assert np.allclose(first.position_m, [0.1, 0.2, 0.3])
    assert np.allclose(first.normal_hint, [0.0, 0.0, 1.0])
    assert np.allclose(second.position_m, [0.4, 0.5, 0.6])
    assert np.allclose(second.normal_hint, [1.0, 0.0, 0.0])


def test_parser_rejects_ambiguous_rows_binary_pcd_and_partial_normals():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        ambiguous = root / "bad.xyz"
        ambiguous.write_text("1 2 3 4\n", encoding="utf-8")
        _assert_raises(
            ValueError,
            lambda: load_pose_template(
                ambiguous, frame="part", role="grasp_tcp"),
            "exactly 3 (XYZ) or 6 (XYZ+RPY)",
        )

        binary = root / "binary.pcd"
        binary.write_text("""FIELDS x y z
SIZE 4 4 4
TYPE F F F
WIDTH 1
HEIGHT 1
POINTS 1
DATA binary
""", encoding="utf-8")
        _assert_raises(
            ValueError,
            lambda: load_pose_template(
                binary, frame="part", role="grasp_tcp"),
            "only DATA ascii",
        )

        partial = root / "partial.pcd"
        partial.write_text("""FIELDS x y z normal_x normal_y
SIZE 4 4 4 4 4
TYPE F F F F F
COUNT 1 1 1 1 1
WIDTH 1
HEIGHT 1
POINTS 1
DATA ascii
0 0 0 1 0
""", encoding="utf-8")
        _assert_raises(
            ValueError,
            lambda: load_pose_template(
                partial, frame="part", role="grasp_tcp"),
            "normals must include",
        )


def test_generic_semantics_parse_but_grasp_consumer_rejects_unsupported_role_frame():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "part_poses.xyz"
        path.write_text("0 0 0 0 0 0\n", encoding="utf-8")
        template = load_pose_template(
            path, frame="world", role="part_pose")

    assert template.frame == TemplateFrame.WORLD
    assert template.role == TemplateRole.PART_POSE
    candidates = {"cad": _candidate([0.0, 0.0, 0.0])}
    _assert_raises(
        ValueError,
        lambda: rank_contact_validated_grasps(
            candidates, [template], part_scale=1.0),
        "supports only frame='part', role='grasp_tcp'",
    )


def test_template_ranking_only_permutes_cad_contact_candidates():
    rotation_90 = rpy_matrix([0.0, 0.0, np.pi / 2.0])
    candidates = {
        "wrong_rotation": _candidate([0.0, 0.0, 0.0], rotation_90),
        "validated_near": _candidate([0.01, 0.0, 0.0]),
        "validated_far": _candidate([0.8, 0.0, 0.0]),
    }
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "explicit.xyz"
        path.write_text("0 0 0 0 0 0\n", encoding="utf-8")
        template = load_pose_template(
            path, frame="part", role="grasp_tcp", rpy_units="rad")
        ordered, matches = rank_contact_validated_grasps(
            candidates, [template], part_scale=1.0,
            position_tolerance_fraction=0.2,
            rotation_tolerance_deg=10.0,
            max_matches_per_proposal=2,
        )

    assert ordered[0] == "validated_near"
    assert set(ordered) == set(candidates)
    assert len(ordered) == len(candidates)
    assert [item.grasp_name for item in matches] == ["validated_near"]
    # No raw template transform exists in the output; every result still maps
    # to one of the original CAD/contact-certified candidate objects.
    assert all(name in candidates for name in ordered)


def test_planner_integration_prioritizes_templates_without_injecting_grasps():
    generated = [
        _candidate([0.0, 0.0, 0.0]),
        _candidate([0.5, 0.0, 0.0]),
    ]
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        template_path = root / "preferred.xyz"
        template_path.write_text("500 0 0 0 0 0\n", encoding="utf-8")
        asset = Path(__file__).resolve()

        class FakeProject:
            manifest = {
                "active_task": {"part": "synthetic"},
                "proposal_templates": [{
                    "path": str(template_path),
                    "frame": "part",
                    "role": "grasp_tcp",
                    "xyz_units": "mm",
                    "rpy_units": "deg",
                }],
            }
            active_part = {}
            active_part_path = str(asset)
            solver = {"geometry": {
                "surface_samples": 8,
                "approaches_per_pair": 1,
                "max_grasp_candidates": 8,
                "template_position_tolerance_mesh_fraction": 0.1,
                "template_rotation_tolerance_deg": 10.0,
                "template_normal_tolerance_deg": 20.0,
                "template_matches_per_proposal": 1,
                "template_max_proposals": 10,
            }}

            @staticmethod
            def resolve_asset(path):
                return str(Path(path).resolve())

            @staticmethod
            def gripper(robot):
                return SimpleNamespace(
                    name="synthetic_gripper",
                    opening_min=0.002,
                    opening_max=0.02,
                    pad_size=np.array([0.005, 0.02]),
                    finger_depth=0.03,
                    model_path=str(asset),
                )

        planner = HandoffPlanner.__new__(HandoffPlanner)
        planner.project = FakeProject()
        planner.part_geometry = SimpleNamespace(
            artifact_fingerprint="prepared-synthetic")
        planner.part_mesh = SimpleNamespace(extent=np.ones(3))
        planner.cache_dir = str(root / "cache")

        original = planning_module.generate_antipodal_grasps
        planning_module.generate_antipodal_grasps = lambda *args, **kwargs: generated
        try:
            ordered = planner._receiver_grasps()
        finally:
            planning_module.generate_antipodal_grasps = original

    assert [name for name, _ in ordered] == ["geom_001", "geom_000"]
    assert len(planner.grasp_template_matches) == 1
    assert set(name for name, _ in ordered) == set(planner.grasp_candidates)
    for name, transform in ordered:
        assert np.array_equal(transform, planner.grasp_candidates[name].T_P_E)


def test_declaration_loader_requires_explicit_frame_and_role():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "poses.xyz"
        path.write_text("0 0 0\n", encoding="utf-8")
        _assert_raises(
            ValueError,
            lambda: load_declared_pose_templates({"path": str(path)}),
            "missing required fields",
        )


if __name__ == "__main__":
    tests = [
        value for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
        print(f"PASS  {test.__name__}")
    print(f"\n{len(tests)}/{len(tests)} passed")
