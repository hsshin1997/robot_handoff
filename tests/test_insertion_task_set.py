"""Tests for the robot-independent continuous insertion task-set layer."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mujoco_sim.modeling.insertion_task_set import (  # noqa: E402
    CELL_REJECTED,
    CELL_SAFE,
    CELL_UNRESOLVED,
    ContactMode,
    FinitePCBFootprint,
    GripperComponentVertices,
    SampledGripperGeometry,
    apply_whole_cell_task_certificates,
    artifact_sha256,
    build_parameter_cells,
    build_task_set_document,
    certificate_binding_sha256,
    finite_pcb_vertex_witness,
    query_constructive_task_pose,
    wrap_periodic_angle,
)


CONNECTOR_SHA = "a" * 64


def _mode() -> ContactMode:
    return ContactMode(
        mode_id="synthetic_opposed_y",
        description="Synthetic opposing Y faces.",
        closing_axis_P=np.array([0.0, 1.0, 0.0]),
        position_u_axis_P=np.array([1.0, 0.0, 0.0]),
        position_v_axis_P=np.array([0.0, 0.0, 1.0]),
        contact_midplane_coordinate_P_m=0.2,
        roll_zero_approach_axis_P=np.array([1.0, 0.0, 0.0]),
        aperture_model={"type": "constant", "value_m": 0.4},
        u_bounds_P_m=(0.0, 1.0),
        v_bounds_P_m=(0.0, 1.0),
        possible_aperture_range_m=(0.2, 0.4),
        u_cells=4,
        v_cells=4,
        roll_cells=8,
    )


def _seed(
    *,
    seed_id: str = "g_synthetic",
    centre=(0.5, 0.2, 0.5),
    quality: float = 0.8,
) -> dict:
    transform = np.eye(4)
    transform[:3, 3] = centre
    return {
        "id": seed_id,
        "library_index": 0,
        "status": "phase1_preinsert_only_candidate",
        "family": "close_y_approach_+z",
        "preinsert_compatible": True,
        "seated_compatible": False,
        "T_P_E": transform.tolist(),
        "contact_points_P_m": [
            [centre[0], 0.0, centre[2]],
            [centre[0], 0.4, centre[2]],
        ],
        "closing_direction_P": [0.0, 1.0, 0.0],
        "approach_direction_P": [0.0, 0.0, 1.0],
        "required_aperture_m": 0.4,
        "quality": quality,
        "seated_pcb_clearance_m": -0.01,
        "preinsert_pcb_clearance_m": 0.03,
    }


def _rect_top_triangles(x0, x1, y0, y1, z=0.0):
    return np.array([
        [[x0, y0, z], [x1, y0, z], [x1, y1, z]],
        [[x0, y0, z], [x1, y1, z], [x0, y1, z]],
    ], dtype=float)


def _board_with_square_hole() -> np.ndarray:
    top = np.concatenate([
        _rect_top_triangles(0.0, 0.4, 0.0, 1.0),
        _rect_top_triangles(0.6, 1.0, 0.0, 1.0),
        _rect_top_triangles(0.4, 0.6, 0.0, 0.4),
        _rect_top_triangles(0.4, 0.6, 0.6, 1.0),
    ])
    bottom = _rect_top_triangles(0.0, 1.0, 0.0, 1.0, z=-0.1)
    return np.concatenate((top, bottom))


def test_periodic_roll_and_seed_parameterization_are_frame_explicit():
    mode = _mode()
    parameters = mode.parameterize_seed(
        _seed(), minimum_closing_alignment=0.99)
    assert parameters is not None
    u, v, roll = parameters
    assert np.isclose(u, 0.5)
    assert np.isclose(v, 0.5)
    # Positive roll about +Y_P points from +X_P toward -Z_P, so +Z_P is -pi/2.
    assert np.isclose(roll, -np.pi / 2.0)
    assert np.isclose(wrap_periodic_angle(3.0 * np.pi), -np.pi)


def test_constructive_map_is_proper_se3_and_right_hand_roll_is_independent_of_v():
    mode = _mode()
    zero, aperture = mode.construct_pose(0.3, 0.7, 0.0)
    positive, _ = mode.construct_pose(0.3, 0.7, np.pi / 2.0)
    assert np.allclose(zero[:3, 3], [0.3, 0.2, 0.7])
    assert np.allclose(zero[:3, 2], [1.0, 0.0, 0.0])
    # Right-hand +roll about +Y maps +X toward -Z. The position-v axis remains
    # +Z, demonstrating that angular handedness is not inferred from it.
    assert np.allclose(positive[:3, 2], [0.0, 0.0, -1.0], atol=1e-12)
    assert np.allclose(mode.positive_roll_quadrature_axis_P, [0.0, 0.0, -1.0])
    assert np.isclose(np.linalg.det(positive[:3, :3]), 1.0)
    assert np.allclose(positive[:3, :3].T @ positive[:3, :3], np.eye(3))
    assert np.isclose(aperture, 0.4)


def test_cells_are_continuous_ranges_and_samples_never_promote_safe():
    cells, representatives = build_parameter_cells(
        [_mode()],
        [_seed()],
        pad_size_m=[0.6, 0.8],
        usable_opening_range_m=[0.1, 0.5],
        minimum_closing_alignment=0.99,
    )
    assert len(cells) == 4 * 4 * 8
    assert not any(cell["classification"] == CELL_SAFE for cell in cells)
    assert any(cell["classification"] == CELL_REJECTED for cell in cells)
    assert any(cell["classification"] == CELL_UNRESOLVED for cell in cells)
    represented = [cell for cell in cells if cell["representative"] is not None]
    assert len(represented) == 1
    cell = represented[0]
    assert cell["id"] in representatives
    assert cell["classification"] == CELL_UNRESOLVED
    assert cell["representative"]["seed_grasp_id"] == "g_synthetic"
    assert cell["witnesses"][0]["claim"] == "SAMPLED_PREINSERT_ONLY_WITNESS"
    assert "never a whole-cell claim" in cell["witnesses"][0]["scope"]
    for candidate in cells:
        center = candidate["center_pose"]
        theta = center["theta"]
        for name, value in theta.items():
            low, high = candidate["bounds"][name]
            assert low < value < high
        expected, expected_aperture = _mode().construct_pose(
            theta["u_P_m"], theta["v_P_m"], theta["roll_rad"])
        assert np.allclose(center["T_P_E"], expected, atol=2e-12)
        assert np.isclose(center["required_aperture_m"], expected_aperture)
        assert center["source"] == "contact_mode_constructive_map"


def test_finite_pcb_footprint_uses_top_triangles_and_preserves_hole():
    footprint = FinitePCBFootprint(
        _board_with_square_hole(),
        top_surface_tolerance_m=1e-8,
        spatial_bin_size_m=0.2,
    )
    points = np.array([
        [0.15, 0.5, -0.05],  # board material, away from triangulation edge
        [0.5, 0.5, -0.05],  # through hole
        [1.2, 0.5, -0.05],  # outside finite outline
        [0.2, 0.5, 0.05],   # above the board
    ])
    result = footprint.contains_nominal_solid(
        points, interior_tolerance_m=1e-5)
    assert result.tolist() == [True, False, False, False]


def test_finite_pcb_collision_is_a_positive_witness_not_cell_proof():
    footprint = FinitePCBFootprint(
        _board_with_square_hole(),
        top_surface_tolerance_m=1e-8,
        spatial_bin_size_m=0.2,
    )
    component = GripperComponentVertices(
        name="one_vertex_component",
        # Avoid a top-triangle tessellation edge so this is strictly inside
        # the nominal solid under the configured interior tolerance.
        vertices_C_m=np.array([[0.15, 0.5, -0.05]]),
        T_G_C_reference=np.eye(4),
        aperture_multiplier=0.0,
        source_unique_vertex_count=1,
    )
    gripper = SampledGripperGeometry(
        T_G_E=np.eye(4),
        reference_aperture_m=0.3,
        opening_axis_G=np.array([0.0, 1.0, 0.0]),
        components=(component,),
    )
    witness = finite_pcb_vertex_witness(
        footprint=footprint,
        gripper=gripper,
        T_P_E=np.eye(4),
        required_aperture_m=0.3,
        T_B_P_insert=np.eye(4),
        insertion_axis_P=np.array([0.0, 0.0, -1.0]),
        preinsert_distance_m=0.2,
        path_samples=5,
        interior_tolerance_m=1e-5,
    )
    assert witness["claim"] == "FINITE_PCB_INTERPENETRATION_WITNESS"
    assert 0.0 < witness["path_progress"] <= 1.0
    assert "one sampled gripper vertex" in witness["scope"]


def test_task_set_document_exposes_inner_outer_sets_and_stable_digest():
    library = {
        "schema_version": 1,
        "project_id": "synthetic_library",
        "asset_stats": {"part": {"sha256": CONNECTOR_SHA}},
        "task_geometry": {
            "insertion_axis_P": [0.0, 0.0, -1.0],
            "preinsert_distance_m": 0.2,
        },
        "candidates": [_seed()],
    }
    socket = {
        "schema_version": 1,
        "assets": {"connector": {"sha256": CONNECTOR_SHA}},
        "T_B_P_insert": np.eye(4).tolist(),
    }
    document = build_task_set_document(
        project_id="synthetic_task_set",
        modes=[_mode()],
        pose_library=library,
        socket_contract=socket,
        pad_size_m=[0.6, 0.8],
        usable_opening_range_m=[0.1, 0.5],
        minimum_closing_alignment=0.99,
        input_provenance={"fixture": "synthetic"},
    )
    assert document["artifact_type"] == "robot_independent_insertion_task_set"
    assert document["counts"]["safe"] == 0
    assert document["safe_inner_cell_ids"] == []
    assert len(document["unresolved_cell_ids"]) > 0
    assert document["certification_boundary"][
        "certified_safe_set_available"] is False
    query = query_constructive_task_pose(
        document,
        contact_mode="synthetic_opposed_y",
        u_P_m=0.5,
        v_P_m=0.5,
        roll_rad=2.0 * np.pi + np.pi / 2.0,
    )
    expected, _ = _mode().construct_pose(0.5, 0.5, np.pi / 2.0)
    assert np.allclose(query["T_P_E"], expected)
    assert query["cell_id"] in {
        cell["id"] for cell in document["cells"]}
    assert query["cell_classification"] in {
        CELL_REJECTED, CELL_UNRESOLVED}
    try:
        query_constructive_task_pose(
            document,
            contact_mode="synthetic_opposed_y",
            u_P_m=1.1,
            v_P_m=0.5,
            roll_rad=0.0,
        )
    except ValueError as error:
        assert "outside the authored" in str(error)
    else:
        raise AssertionError("query outside authored domain must fail")
    inconsistent = json.loads(json.dumps(document))
    inconsistent["parameterization"]["contact_modes"][0][
        "constructive_map"]["positive_roll_quadrature_axis_P"] = [0.0, 0.0, 1.0]
    try:
        query_constructive_task_pose(
            inconsistent,
            contact_mode="synthetic_opposed_y",
            u_P_m=0.5,
            v_P_m=0.5,
            roll_rad=0.0,
        )
    except ValueError as error:
        assert "quadrature is inconsistent" in str(error)
    else:
        raise AssertionError("inconsistent roll-handedness metadata must fail")
    digest = artifact_sha256(document)
    document["semantic_sha256"] = digest
    assert artifact_sha256(document) == digest
    # It is a plain self-contained JSON schema, not a NumPy-only object graph.
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "artifact.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] == 1


def test_whole_cell_certificate_import_is_exact_bound_and_fail_closed():
    library = {
        "schema_version": 1,
        "project_id": "synthetic_library",
        "asset_stats": {"part": {"sha256": CONNECTOR_SHA}},
        "task_geometry": {
            "insertion_axis_P": [0.0, 0.0, -1.0],
            "preinsert_distance_m": 0.2,
        },
        "candidates": [_seed()],
    }
    socket = {
        "schema_version": 1,
        "assets": {
            "connector": {"sha256": CONNECTOR_SHA},
            "pcb": {"sha256": "c" * 64},
        },
        "T_B_P_insert": np.eye(4).tolist(),
    }

    def fresh_document():
        return build_task_set_document(
            project_id="synthetic_task_set",
            modes=[_mode()],
            pose_library=library,
            socket_contract=socket,
            pad_size_m=[0.6, 0.8],
            usable_opening_range_m=[0.1, 0.5],
            minimum_closing_alignment=0.99,
            input_provenance={
                "task_set_config": {
                    "path": "synthetic.yaml", "sha256": "d" * 64,
                },
                "pose_library": {"sha256": "e" * 64},
            },
        )

    required = ["exact_continuous_collision", "uncertainty_envelope"]
    base = fresh_document()
    unresolved_id = base["unresolved_cell_ids"][0]
    certificate = {
        "schema_version": 1,
        "artifact_type": (
            "robot_independent_insertion_task_whole_cell_certificate"),
        "certificate_id": "synthetic_exact_proof_v1",
        "bindings": {
            "base_artifact_certificate_binding_sha256": (
                certificate_binding_sha256(base)),
            "project_id": base["project_id"],
            "connector_sha256": CONNECTOR_SHA,
        },
        "proved_constraints": [*required, "extra_valid_proof"],
        "cell_claims": [{
            "cell_id": unresolved_id,
            "classification": "SAFE",
        }],
    }
    imported = {
        "path": "synthetic_certificate.json",
        "expected_sha256": "f" * 64,
        "actual_sha256": "f" * 64,
        "document": certificate,
    }
    result = apply_whole_cell_task_certificates(
        base, [imported], required_proved_constraints=required)
    assert result["counts"]["safe"] == 1
    assert result["safe_inner_cell_ids"] == [unresolved_id]
    certified = next(cell for cell in result["cells"]
                     if cell["id"] == unresolved_id)
    assert certified["classification"] == CELL_SAFE
    assert certified["whole_cell_task_certificate"]["sha256"] == "f" * 64

    mismatched_hash = dict(imported, actual_sha256="0" * 64)
    try:
        apply_whole_cell_task_certificates(
            fresh_document(), [mismatched_hash],
            required_proved_constraints=required,
        )
    except ValueError as error:
        assert "file SHA mismatch" in str(error)
    else:
        raise AssertionError("mismatched certificate file hash must fail")

    missing_proof = json.loads(json.dumps(certificate))
    missing_proof["proved_constraints"] = [required[0]]
    try:
        apply_whole_cell_task_certificates(
            fresh_document(), [dict(imported, document=missing_proof)],
            required_proved_constraints=required,
        )
    except ValueError as error:
        assert "missing required proofs" in str(error)
    else:
        raise AssertionError("incomplete proof list must fail")

    wrong_binding = json.loads(json.dumps(certificate))
    wrong_binding["bindings"]["base_artifact_certificate_binding_sha256"] = (
        "0" * 64)
    try:
        apply_whole_cell_task_certificates(
            fresh_document(), [dict(imported, document=wrong_binding)],
            required_proved_constraints=required,
        )
    except ValueError as error:
        assert "bindings do not exactly match" in str(error)
    else:
        raise AssertionError("unbound certificate must fail")


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
    print(f"passed {len(tests)} insertion task-set tests")
