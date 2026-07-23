"""Focused tests for sampled parallel-jaw grasp candidate previews."""
from __future__ import annotations

import copy
import contextlib
import hashlib
import io
import json
from pathlib import Path
import struct
import sys
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mujoco_sim.modeling.cad_preprocess import write_binary_stl  # noqa: E402
from mujoco_sim.modeling.grasps import ParallelJawGripper  # noqa: E402
from mujoco_sim.modeling.parallel_jaw_grasp_visualization import (  # noqa: E402
    ARTIFACT_TYPE,
    _fill_triangle_with_depth,
    ideal_parallel_jaw_glyph_E,
    render_parallel_jaw_candidate_image,
    select_candidate_records,
)
from scripts.generate_parallel_jaw_grasps import generate_document  # noqa: E402
from scripts.render_parallel_jaw_grasp_candidates import main  # noqa: E402


def _candidate(
    test_label: str,
    index: int,
    *,
    origin,
    closing,
    approach,
    opening: float = 0.010,
) -> dict:
    y_axis = np.asarray(closing, dtype=float)
    y_axis /= np.linalg.norm(y_axis)
    z_axis = np.asarray(approach, dtype=float)
    z_axis /= np.linalg.norm(z_axis)
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    z_axis = np.cross(x_axis, y_axis)
    rotation = np.column_stack((x_axis, y_axis, z_axis))
    translation = np.asarray(origin, dtype=float)
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    contacts = np.stack((
        translation - 0.5 * opening * y_axis,
        translation + 0.5 * opening * y_axis,
    ))
    normals = np.stack((-y_axis, y_axis))
    support_quality = 0.5
    opening_margin = 0.5
    palm_clearance = 0.020
    friction_cosine = 1.0 / np.sqrt(1.0 + 0.5**2)
    normalized_antipodal = (1.0 - friction_cosine) / (
        1.0 - friction_cosine)
    quality = (
        0.45 * normalized_antipodal
        + 0.25 * support_quality
        + 0.15 * opening_margin
        + 0.15 * (palm_clearance / 0.040)
    )
    record = {
        "test_label": test_label,
        "index": index,
        "T_P_E": transform.tolist(),
        "contact_points_P_m": contacts.tolist(),
        "contact_normals_P": normals.tolist(),
        "closing_direction_P": y_axis.tolist(),
        "approach_direction_P": z_axis.tolist(),
        "required_opening_m": opening,
        "quality": quality,
        "antipodal_quality": 1.0,
        "support_quality": support_quality,
        "opening_margin": opening_margin,
        "idealized_palm_clearance_m": palm_clearance,
    }
    identity = {
        "T_P_E": np.round(transform, 10).tolist(),
        "contacts_P_m": np.round(contacts, 10).tolist(),
        "required_opening_m": round(float(opening), 10),
    }
    payload = json.dumps(
        identity,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    record["id"] = "grasp_" + hashlib.sha256(payload).hexdigest()[:16]
    return record


def _selection_document() -> dict:
    candidates = [
        _candidate(
            "g0", 0,
            origin=[0.0, 0.0, 0.0],
            closing=[0.0, 1.0, 0.0],
            approach=[0.0, 0.0, 1.0],
        ),
        _candidate(
            "closing_sign_duplicate", 1,
            origin=[0.0, 0.0, 0.0],
            closing=[0.0, -1.0, 0.0],
            approach=[0.0, 0.0, 1.0],
        ),
        _candidate(
            "far", 2,
            origin=[0.045, 0.0, 0.0],
            closing=[0.0, 1.0, 0.0],
            approach=[0.0, 0.0, 1.0],
        ),
        _candidate(
            "opposite_approach", 3,
            origin=[0.0, 0.0, 0.0],
            closing=[0.0, 1.0, 0.0],
            approach=[0.0, 0.0, -1.0],
        ),
    ]
    return {
        "schema_version": 1,
        "artifact_type": "sampled_parallel_jaw_grasp_candidates",
        "claim_level": "resolution_qualified_object_geometry_candidate",
        "continuous_exhaustive": False,
        "candidate_cap_applied": False,
        "all_deduplicated_accepted_candidates_returned": True,
        "cad": {
            "path": "/not/needed/for/selection.stl",
            "sha256": "0" * 64,
            "artifact_fingerprint": "synthetic",
            "format": "stl",
            "triangle_count": 12,
            "bounds_min_P_m": [-0.05, -0.01, -0.01],
            "bounds_max_P_m": [0.05, 0.01, 0.01],
            "extent_P_m": [0.10, 0.02, 0.02],
            "scale_to_m": [1.0, 1.0, 1.0],
            "topology_audit": {
                "normal_ray_assumptions_accepted": True,
                "closed_consistently_wound_two_manifold": True,
                "degenerate_face_count": 0,
                "boundary_edge_count": 0,
                "nonmanifold_edge_count": 0,
                "inconsistent_paired_edge_orientation_count": 0,
                "zero_or_unresolved_volume_component_count": 0,
                "mixed_component_winding_signs": False,
            },
        },
        "sampling": {
            "surface_samples": 64,
            "closing_directions_per_surface": 1,
            "approaches_per_contact_pair": 4,
            "max_candidates": None,
        },
        "gripper_model": {
            "type": "ideal_symmetric_parallel_jaw",
            "opening_range_m": [0.004, 0.020],
            "pad_size_m": [0.008, 0.010],
            "finger_tip_to_palm_depth_m": 0.040,
            "friction_coefficient": 0.5,
        },
        "feasibility_contract": {"not_checked": []},
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


def _assert_raises(error_type, text: str, function) -> None:
    try:
        function()
    except error_type as error:
        assert text in str(error)
    else:
        raise AssertionError(f"expected {error_type.__name__}: {text}")


def _assert_cli_error(function) -> None:
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            function()
    except SystemExit as error:
        assert error.code == 2
    else:
        raise AssertionError("expected argparse SystemExit(2)")


def _box_triangles() -> np.ndarray:
    low = np.array([-0.02, -0.01, -0.005])
    high = np.array([0.02, 0.01, 0.005])
    vertices = np.array([
        [low[0], low[1], low[2]],
        [high[0], low[1], low[2]],
        [high[0], high[1], low[2]],
        [low[0], high[1], low[2]],
        [low[0], low[1], high[2]],
        [high[0], low[1], high[2]],
        [high[0], high[1], high[2]],
        [low[0], high[1], high[2]],
    ])
    faces = np.array([
        [0, 2, 1], [0, 3, 2],
        [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4],
        [3, 7, 6], [3, 6, 2],
        [0, 4, 7], [0, 7, 3],
        [1, 2, 6], [1, 6, 5],
    ])
    return vertices[faces]


def _write_box(path: Path) -> None:
    records = []
    for triangle in _box_triangles():
        normal = np.cross(
            triangle[1] - triangle[0],
            triangle[2] - triangle[0],
        )
        normal /= np.linalg.norm(normal)
        records.append(struct.pack(
            "<12fH",
            *normal.astype(np.float32),
            *triangle.astype(np.float32).ravel(),
            0,
        ))
    write_binary_stl(path, records)


def test_pose_diverse_selection_is_deterministic_and_frame_aware():
    document = _selection_document()
    first = select_candidate_records(
        document, count=4, selection_mode="pose_diverse")
    second = select_candidate_records(
        document, count=4, selection_mode="pose-diverse")
    assert first == second
    labels = [candidate["test_label"] for candidate in first["candidates"]]
    assert labels[0] == "g0"
    assert labels[-1] == "closing_sign_duplicate"
    assert labels.index("far") < labels.index("closing_sign_duplicate")
    assert labels.index("opposite_approach") < labels.index(
        "closing_sign_duplicate")
    metric = first["selection"]["distance_metric"]
    assert metric["closing_axis_sign_invariant"] is True
    assert metric["opposite_approach_distinct"] is True

    far_id = document["candidates"][2]["id"]
    base_id = document["candidates"][0]["id"]
    explicit = select_candidate_records(
        document,
        candidate_ids=(far_id, base_id),
    )
    assert explicit["selection"]["mode"] == "explicit_ids"
    assert [candidate["test_label"] for candidate in explicit["candidates"]] == [
        "far", "g0"]


def test_frame_inconsistent_contacts_fail_closed():
    document = _selection_document()
    broken = copy.deepcopy(document)
    half_y = float(np.sqrt(0.005**2 - 0.001**2))
    broken["candidates"][0]["contact_points_P_m"][0] = [
        -0.001, -half_y, 0.0]
    broken["candidates"][0]["contact_points_P_m"][1] = [
        +0.001, +half_y, 0.0]
    _assert_raises(
        ValueError,
        "contacts are inconsistent with the serialized E frame",
        lambda: select_candidate_records(broken, count=1),
    )


def test_source_schema_identity_and_claim_fields_fail_closed():
    source = _selection_document()
    cases = []

    continuous = copy.deepcopy(source)
    continuous["continuous_exhaustive"] = True
    cases.append((continuous, "continuous_exhaustive=false"))

    missing_sampling = copy.deepcopy(source)
    del missing_sampling["sampling"]
    cases.append((missing_sampling, "sampling must be an object"))

    reordered = copy.deepcopy(source)
    reordered["candidates"][0]["index"] = 3
    cases.append((reordered, "must equal its generator output rank"))

    false_identity = copy.deepcopy(source)
    false_identity["candidates"][0]["id"] = "grasp_0000000000000000"
    cases.append((false_identity, "does not match its pose/contact identity"))

    bad_normals = copy.deepcopy(source)
    bad_normals["candidates"][0]["contact_normals_P"][0] = [0.0, 1.0, 0.0]
    cases.append((bad_normals, "antipodal quality disagrees"))

    malformed_limitations = copy.deepcopy(source)
    malformed_limitations["feasibility_contract"]["not_checked"] = "none"
    cases.append((malformed_limitations, "must contain non-empty strings"))

    contradictory_topology = copy.deepcopy(source)
    contradictory_topology["cad"]["topology_audit"][
        "boundary_edge_count"] = 2
    cases.append((contradictory_topology, "claim_level contradicts"))

    for document, expected in cases:
        _assert_raises(
            ValueError,
            expected,
            lambda value=document: select_candidate_records(value, count=1),
        )
    _assert_raises(
        TypeError,
        "must be a boolean",
        lambda: select_candidate_records(
            source,
            count=1,
            allow_unreliable_input="false",  # type: ignore[arg-type]
        ),
    )
    unreliable_bad_cone = copy.deepcopy(source)
    unreliable_bad_cone["claim_level"] = "unreliable_mesh_sampled_candidate"
    unreliable_bad_cone["cad"]["topology_audit"][
        "normal_ray_assumptions_accepted"] = False
    unreliable_bad_cone["cad"]["topology_audit"][
        "closed_consistently_wound_two_manifold"] = False
    unreliable_bad_cone["cad"]["topology_audit"]["boundary_edge_count"] = 2
    unreliable_bad_cone["candidates"][0]["contact_normals_P"] = [
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ]
    unreliable_bad_cone["candidates"][0]["antipodal_quality"] = 0.0
    unreliable_bad_cone["candidates"][0]["quality"] = 0.275
    _assert_raises(
        ValueError,
        "violate the declared Coulomb friction cone",
        lambda: select_candidate_records(
            unreliable_bad_cone,
            count=1,
            allow_unreliable_input=True,
        ),
    )


def test_nontrivial_rotation_uses_the_correct_part_to_E_transpose():
    document = _selection_document()
    candidate = _candidate(
        "oblique",
        0,
        origin=[0.013, -0.017, 0.021],
        closing=[1.0, 2.0, 0.5],
        approach=[-0.2, -0.1, 0.8],
    )
    document["candidates"] = [candidate]
    document["candidate_count"] = 1
    selected = select_candidate_records(document, count=1)
    assert selected["candidates"][0]["test_label"] == "oblique"
    transform = np.asarray(candidate["T_P_E"], dtype=float)
    contacts = np.asarray(candidate["contact_points_P_m"], dtype=float)
    contacts_E = (
        contacts - transform[:3, 3]
    ) @ transform[:3, :3]
    assert np.allclose(
        contacts_E,
        [[0.0, -0.005, 0.0], [0.0, 0.005, 0.0]],
        atol=1e-12,
        rtol=0.0,
    )


def test_ideal_glyph_has_exact_contact_pad_and_palm_coordinates():
    glyph = ideal_parallel_jaw_glyph_E(
        0.020,
        [0.008, 0.010],
        0.040,
    )
    assert glyph["physical_finger_or_palm_solids_defined"] is False
    assert np.allclose(
        glyph["contacts_E_m"],
        [[0.0, -0.010, 0.0], [0.0, 0.010, 0.0]],
    )
    pads = np.asarray(glyph["pad_rectangles_E_m"])
    assert pads.shape == (2, 4, 3)
    assert np.allclose(np.unique(pads[0, :, 0]), [-0.004, 0.004])
    assert np.allclose(np.unique(pads[0, :, 1]), [-0.010])
    assert np.allclose(np.unique(pads[0, :, 2]), [-0.005, 0.005])
    assert np.allclose(
        glyph["palm_depth_line_E_m"],
        [[0.0, -0.010, -0.040], [0.0, 0.010, -0.040]],
    )


def test_triangle_rasterizer_uses_per_pixel_depth_not_draw_order():
    triangle = np.array([[2.0, 2.0], [17.0, 2.0], [2.0, 17.0]])
    near = np.array([1.0, 1.0, 1.0])
    far = np.array([0.0, 0.0, 0.0])
    for order in ("near_first", "far_first"):
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        depth = np.full((20, 20), -np.inf)
        if order == "near_first":
            _fill_triangle_with_depth(
                image, depth, triangle, near, [255, 0, 0])
            _fill_triangle_with_depth(
                image, depth, triangle, far, [0, 0, 255])
        else:
            _fill_triangle_with_depth(
                image, depth, triangle, far, [0, 0, 255])
            _fill_triangle_with_depth(
                image, depth, triangle, near, [255, 0, 0])
        assert image[5, 5].tolist() == [255, 0, 0]
        assert depth[5, 5] == 1.0


def test_generated_box_renders_deterministic_png_and_cli_metadata():
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        cad = temporary / "box.stl"
        generated_root = temporary / "prepared"
        candidate_json = temporary / "box-grasps.json"
        first_png = temporary / "first.png"
        second_png = temporary / "second.png"
        _write_box(cad)
        document = generate_document(
            cad,
            units="m",
            scale_to_m=None,
            gripper=ParallelJawGripper(
                min_opening=0.008,
                max_opening=0.025,
                pad_size=(0.008, 0.008),
                pad_depth=0.050,
                friction_coefficient=0.5,
            ),
            surface_samples=80,
            closing_directions_per_surface=1,
            approaches_per_pair=4,
            max_candidates=5,
            generated_root=generated_root,
        )
        candidate_json.write_text(
            json.dumps(document, allow_nan=False, indent=2) + "\n",
            encoding="utf-8",
        )
        first = render_parallel_jaw_candidate_image(
            candidate_json,
            first_png,
            count=2,
            generated_root=generated_root,
            width=1200,
            row_height=260,
        )
        second = render_parallel_jaw_candidate_image(
            candidate_json,
            second_png,
            count=2,
            generated_root=generated_root,
            width=1200,
            row_height=260,
        )
        assert first_png.read_bytes() == second_png.read_bytes()
        first_without_path = copy.deepcopy(first)
        second_without_path = copy.deepcopy(second)
        first_without_path["output_image"].pop("path")
        second_without_path["output_image"].pop("path")
        assert first_without_path == second_without_path
        payload = first_png.read_bytes()
        assert payload[:8] == b"\x89PNG\r\n\x1a\n"
        width, height = struct.unpack(">II", payload[16:24])
        assert (width, height) == (1200, 760)
        assert len(payload) > 5_000
        assert first["artifact_type"] == ARTIFACT_TYPE
        assert first["view"]["frame"] == "each candidate ideal contact frame E"
        assert [
            (
                view["name"],
                view["horizontal_axis"],
                view["vertical_axis"],
                view["view_direction"],
                view["out_of_page_axis"],
            )
            for view in first["view"]["projections"]
        ] == [
            ("XY", "+X_E", "+Y_E", "-Z_E", "+Z_E"),
            ("XZ", "+X_E", "+Z_E", "+Y_E", "-Y_E"),
            ("YZ", "+Y_E", "+Z_E", "-X_E", "+X_E"),
        ]
        assert first["cad"]["complete_triangle_projection"] is True
        assert len(first["displayed_candidates"]) == 2
        assert first["certification"] == {
            "physical_gripper_geometry_shown": False,
            "physical_tcp_shown": False,
            "full_part_gripper_collision_checked": False,
            "approach_sweep_checked": False,
            "environment_collision_checked": False,
            "robot_reachability_checked": False,
            "task_feasibility_checked": False,
        }
        assert first["source_candidates"]["sha256"] == hashlib.sha256(
            candidate_json.read_bytes()).hexdigest()

        explicit_png = temporary / "explicit.png"
        explicit_metadata = temporary / "explicit.json"
        selected_id = document["candidates"][2]["id"]
        exit_code = main([
            str(candidate_json),
            "--candidate-id", selected_id,
            "--generated-root", str(generated_root),
            "--output", str(explicit_png),
            "--metadata-output", str(explicit_metadata),
            "--width", "1200",
            "--row-height", "260",
        ])
        assert exit_code == 0
        cli_metadata = json.loads(
            explicit_metadata.read_text(encoding="utf-8"))
        assert cli_metadata["selection"]["mode"] == "explicit_ids"
        assert cli_metadata["selection"]["displayed"] == [{
            "id": selected_id,
            "index": 2,
            "source_rank": 2,
        }]
        assert explicit_png.is_file()

        invalid_fingerprint = copy.deepcopy(document)
        invalid_fingerprint["cad"]["artifact_fingerprint"] = "../../escape"
        invalid_fingerprint_json = temporary / "invalid-fingerprint.json"
        invalid_fingerprint_json.write_text(
            json.dumps(invalid_fingerprint, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        _assert_raises(
            ValueError,
            "artifact_fingerprint must be a SHA-256 hex digest",
            lambda: render_parallel_jaw_candidate_image(
                invalid_fingerprint_json,
                temporary / "invalid-fingerprint.png",
                count=1,
                generated_root=generated_root,
                width=1200,
                row_height=260,
            ),
        )
        wrong_cad = temporary / "wrong.stl"
        wrong_cad.write_bytes(b"not the expected CAD")
        _assert_raises(
            ValueError,
            "CAD override does not match cad.sha256",
            lambda: render_parallel_jaw_candidate_image(
                candidate_json,
                temporary / "wrong-cad.png",
                count=1,
                generated_root=generated_root,
                cad_path=wrong_cad,
                width=1200,
                row_height=260,
            ),
        )

        diagnostic_json = temporary / "diagnostic.json"
        diagnostic_png = temporary / "diagnostic.png"
        diagnostic = copy.deepcopy(document)
        diagnostic["claim_level"] = "unreliable_mesh_sampled_candidate"
        diagnostic["cad"]["topology_audit"]["boundary_edge_count"] = 2
        diagnostic["cad"]["topology_audit"][
            "normal_ray_assumptions_accepted"] = False
        diagnostic["cad"]["topology_audit"][
            "closed_consistently_wound_two_manifold"] = False
        diagnostic_json.write_text(
            json.dumps(diagnostic, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        diagnostic_metadata = render_parallel_jaw_candidate_image(
            diagnostic_json,
            diagnostic_png,
            count=1,
            generated_root=generated_root,
            width=1200,
            row_height=260,
            max_render_triangles=4,
            allow_unreliable_input=True,
        )
        warnings = diagnostic_metadata["view"]["display_warnings"]
        assert "DISPLAYING 1 OF 5" in warnings
        assert "UNRELIABLE MESH INPUT" in warnings
        assert "SOURCE CAP CONFIGURED" in warnings
        assert "CAD DISPLAY SUBSET" in warnings
        assert diagnostic_metadata["claim_level"] == "visualization_only"
        assert diagnostic_metadata["source_candidates"]["claim_level"] == (
            "unreliable_mesh_sampled_candidate")
        assert diagnostic_metadata["cad"]["complete_triangle_projection"] is False
        assert (
            diagnostic_metadata["output_image"]["sha256"]
            == hashlib.sha256(diagnostic_png.read_bytes()).hexdigest()
        )

        collision_png = temporary / "collision.png"
        _assert_cli_error(lambda: main([
            str(candidate_json),
            "--generated-root", str(generated_root),
            "--output", str(collision_png),
            "--metadata-output", str(collision_png),
        ]))
        _assert_cli_error(lambda: main([
            str(candidate_json),
            "--generated-root", str(generated_root),
            "--output", str(collision_png),
            "--metadata-output", str(candidate_json),
        ]))
        _assert_cli_error(lambda: main([
            str(candidate_json),
            "--generated-root", str(generated_root),
            "--output", str(collision_png),
            "--metadata-output", str(cad),
        ]))
        fresh_root = temporary / "fresh-prepared"
        future_chunk = (
            fresh_root
            / document["cad"]["artifact_fingerprint"]
            / "visual"
            / "visual-0000.stl"
        )
        _assert_cli_error(lambda: main([
            str(candidate_json),
            "--generated-root", str(fresh_root),
            "--output", str(temporary / "fresh-preview.png"),
            "--metadata-output", str(future_chunk),
            "--count", "1",
            "--width", "1200",
            "--row-height", "260",
        ]))
        assert future_chunk.is_file()
        assert future_chunk.stat().st_size > 84
        assert not future_chunk.read_bytes().startswith(b"{")


if __name__ == "__main__":
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(f"passed {len(tests)} parallel-jaw visualization tests")
