"""Focused tests for representative views of the continuous grasp map."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import zlib

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mujoco_sim.modeling.two_finger_grasp_map import SCOPE  # noqa: E402
from mujoco_sim.modeling.two_finger_grasp_visualization import (  # noqa: E402
    ARTIFACT_TYPE,
    render_top_down_candidate_image,
    select_orientation_diverse_candidates,
    select_representative_candidates,
)


MAP_PATH = (
    ROOT
    / "projects"
    / "two_finger_grasp_map"
    / "generated"
    / "connector_header_grasp_map.json"
)


def _document() -> dict:
    return json.loads(MAP_PATH.read_text(encoding="utf-8"))


def _assert_raises(error_type, text: str, function) -> None:
    try:
        function()
    except error_type as error:
        assert text in str(error)
    else:
        raise AssertionError(f"expected {error_type.__name__}: {text}")


def _inside_convex_polygon(
    point: np.ndarray,
    vertices: np.ndarray,
    *,
    tolerance: float = 1e-10,
) -> bool:
    edges = np.roll(vertices, -1, axis=0) - vertices
    relative = point - vertices
    cross = edges[:, 0] * relative[:, 1] - edges[:, 1] * relative[:, 0]
    return bool(
        np.all(cross >= -tolerance) or np.all(cross <= tolerance)
    )


def _decode_rgb_png(path: Path) -> np.ndarray:
    payload = path.read_bytes()
    assert payload[:8] == b"\x89PNG\r\n\x1a\n"
    offset = 8
    width = height = None
    idat = bytearray()
    while offset < len(payload):
        length = struct.unpack(">I", payload[offset:offset + 4])[0]
        kind = payload[offset + 4:offset + 8]
        data = payload[offset + 8:offset + 8 + length]
        offset += 12 + length
        if kind == b"IHDR":
            width, height, depth, color_type, compression, filtering, interlace = (
                struct.unpack(">IIBBBBB", data)
            )
            assert (depth, color_type, compression, filtering, interlace) == (
                8, 2, 0, 0, 0)
        elif kind == b"IDAT":
            idat.extend(data)
        elif kind == b"IEND":
            break
    assert width is not None and height is not None
    rows = zlib.decompress(bytes(idat))
    stride = 1 + 3 * width
    assert len(rows) == height * stride
    assert all(rows[row * stride] == 0 for row in range(height))
    image = np.empty((height, width, 3), dtype=np.uint8)
    for row in range(height):
        start = row * stride + 1
        image[row] = np.frombuffer(
            rows[start:start + 3 * width], dtype=np.uint8
        ).reshape(width, 3)
    return image


def test_connector_selection_is_deterministic_and_stays_in_continuous_domain():
    document = _document()
    first = select_representative_candidates(document, count=6)
    second = select_representative_candidates(document, count=6)
    assert first == second

    selection = first["selection"]
    assert selection["family_id"] == "family_0037"
    assert selection["long_parameter"] == "u_m"
    assert selection["cross_parameter"] == "v_m"
    assert selection["sample_fractions"] == [
        0.1,
        0.26,
        0.42000000000000004,
        0.58,
        0.74,
        0.9,
    ]
    expected_u = np.array([
        -0.01267967953681946,
        -0.007607807626724243,
        -0.0025359357166290265,
        0.0025359361934661866,
        0.007607808103561403,
        0.01267968001365662,
    ])
    candidates = first["candidates"]
    assert [candidate["candidate_id"] for candidate in candidates] == [
        "C1", "C2", "C3", "C4", "C5", "C6"]
    assert np.allclose(
        [candidate["parameters"]["u_m"] for candidate in candidates],
        expected_u,
        atol=1e-14,
        rtol=0.0,
    )
    assert np.allclose(
        [candidate["parameters"]["v_m"] for candidate in candidates],
        0.0010723088830709457,
        atol=1e-14,
        rtol=0.0,
    )

    family = next(
        family
        for family in document["families"]
        if family["family_id"] == selection["family_id"]
    )
    domains = [
        np.asarray(domain["vertices_uv_m"], dtype=float)
        for domain in family["parameterization"]["domains"]
    ]
    T_W_P = np.asarray(document["inputs"]["T_W_P_insert"], dtype=float)
    for candidate in candidates:
        parameter = np.array([
            candidate["parameters"]["u_m"],
            candidate["parameters"]["v_m"],
        ])
        assert any(_inside_convex_polygon(parameter, domain)
                   for domain in domains)
        assert candidate["family_id"] == "family_0037"
        assert candidate["scope"] == SCOPE
        assert candidate["insertion_safe"] is False
        assert np.isclose(
            candidate["aperture_m"],
            0.010794999599456786,
            atol=1e-14,
            rtol=0.0,
        )
        assert np.allclose(
            candidate["T_W_E"],
            T_W_P @ np.asarray(candidate["T_P_E"], dtype=float),
            atol=1e-12,
            rtol=0.0,
        )

    assert np.allclose(
        candidates[0]["contacts_P_m"],
        [
            [0.03068878748416901, 0.004402787685394287, 0.004636642932891846],
            [0.03068878748416901, 0.004402787685394287, 0.015431642532348633],
        ],
        atol=1e-14,
        rtol=0.0,
    )


def test_render_writes_valid_png_and_fail_closed_metadata():
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "representative.png"
        metadata = render_top_down_candidate_image(
            MAP_PATH,
            output,
            count=3,
            width=800,
            height=600,
        )
        image = _decode_rgb_png(output)
        assert image.shape == (600, 800, 3)
        assert output.stat().st_size > 1_000
        assert len(np.unique(image.reshape(-1, 3), axis=0)) > 10

    assert metadata["artifact_type"] == ARTIFACT_TYPE
    assert metadata["scope"] == SCOPE
    assert metadata["insertion_safe"] is False
    assert metadata["selection"]["family_id"] == "family_0037"
    assert metadata["selection"]["sample_count"] == 3
    assert len(metadata["candidates"]) == 3
    assert metadata["view"]["image_size_px"] == [800, 600]
    assert metadata["view"]["view_direction"] == "-Z_W"
    assert metadata["view"]["approach_glyph"] == "into_page_cross"
    assert metadata["render_semantics"] == {
        "part_geometry": "actual STL orthographic projection",
        "gripper_geometry": "schematic ideal parallel-jaw glyph",
        "samples_are_visualization_only": True,
        "samples_define_map": False,
    }
    limitations = " ".join(metadata["limitations"])
    assert "gripper collision is not checked" in limitations
    assert "PCB and insertion-path clearance are not checked" in limitations
    assert "insertion safety" in limitations
    assert metadata["source_map"]["sha256"] == hashlib.sha256(
        MAP_PATH.read_bytes()
    ).hexdigest()


def test_connector_orientation_plot_contains_horizontal_and_vertical_groups():
    document = _document()
    selected = select_orientation_diverse_candidates(
        document,
        count_per_orientation=2,
    )
    groups = selected["selection"]["orientation_groups"]
    assert selected["selection"]["mode"] == "orientation_diverse"
    assert selected["selection"]["detected_orientation_group_count"] == 2
    assert selected["selection"]["orientation_group_count"] == 2
    assert selected["selection"]["omitted_orientation_group_count"] == 0
    assert sum(group["clustered_family_count"] for group in groups) == 44
    assert [
        (group["plot_prefix"], group["top_down_orientation"], group["family_id"])
        for group in groups
    ] == [
        ("H", "horizontal", "family_0037"),
        ("V", "vertical", "family_0019"),
    ]
    assert np.allclose(groups[0]["closing_axis_P"], [0.0, 0.0, 1.0])
    assert np.allclose(groups[0]["closing_axis_W"], [-1.0, 0.0, 0.0])
    assert np.allclose(groups[1]["closing_axis_P"], [1.0, 0.0, 0.0])
    assert np.allclose(groups[1]["closing_axis_W"], [0.0, -1.0, 0.0])
    candidates = selected["candidates"]
    assert [candidate["candidate_id"] for candidate in candidates] == [
        "H1", "H2", "V1", "V2"]
    for candidate in candidates:
        contacts = np.asarray(candidate["contacts_W_m"])
        difference = contacts[1, :2] - contacts[0, :2]
        if candidate["top_down_orientation"] == "horizontal":
            assert abs(difference[0]) > 0.010
            assert abs(difference[1]) < 1e-12
        else:
            assert abs(difference[0]) < 1e-12
            assert abs(difference[1]) > 0.020
        assert candidate["external_visibility_certified"] is False
        assert candidate["insertion_safe"] is False

    opposite_axis = copy.deepcopy(document)
    opposite_axis["families"][0]["closing_axis_P"] = [-1.0, 0.0, 0.0]
    sign_invariant = select_orientation_diverse_candidates(
        opposite_axis,
        count_per_orientation=1,
    )
    assert sign_invariant["selection"]["orientation_group_count"] == 2
    assert sum(
        group["clustered_family_count"]
        for group in sign_invariant["selection"]["orientation_groups"]
    ) == 44

    truncated = select_orientation_diverse_candidates(
        document,
        count_per_orientation=1,
        maximum_orientation_groups=1,
    )
    assert truncated["selection"]["detected_orientation_group_count"] == 2
    assert truncated["selection"]["orientation_group_count"] == 1
    assert truncated["selection"]["omitted_orientation_group_count"] == 1

    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "orientations.png"
        metadata = render_top_down_candidate_image(
            MAP_PATH,
            output,
            selection_mode="orientations",
            count_per_orientation=2,
            width=1100,
            height=800,
        )
        image = _decode_rgb_png(output)
        assert image.shape == (800, 1100, 3)
    rendered_groups = metadata["selection"]["orientation_groups"]
    assert metadata["selection"]["configured_opening_range_m"] == [0.002, 0.024]
    assert np.isclose(
        rendered_groups[0]["full_part_support_span_m"],
        0.017894301414489745,
    )
    assert rendered_groups[0][
        "full_part_support_span_within_opening_range"] is True
    assert np.isclose(
        rendered_groups[1]["full_part_support_span_m"],
        0.03566160023212433,
    )
    assert rendered_groups[1][
        "full_part_support_span_within_opening_range"] is False
    assert "distinct closing-axis groups" in metadata["limitations"][0]
    with tempfile.TemporaryDirectory() as directory:
        _assert_raises(
            ValueError,
            "height >= 888",
            lambda: render_top_down_candidate_image(
                MAP_PATH,
                Path(directory) / "too-short.png",
                selection_mode="orientations",
                count_per_orientation=5,
                width=1100,
                height=800,
            ),
        )


def test_invalid_selection_and_render_inputs_fail_closed():
    document = _document()
    _assert_raises(
        ValueError,
        "integer in [1, 10]",
        lambda: select_representative_candidates(document, count=0),
    )
    _assert_raises(
        ValueError,
        "integer in [1, 10]",
        lambda: select_representative_candidates(document, count=True),
    )
    _assert_raises(
        ValueError,
        "roll_rad must be finite",
        lambda: select_representative_candidates(
            document, count=1, roll_rad=float("nan")),
    )
    invalid_scope = copy.deepcopy(document)
    invalid_scope["scope"] = "FULLY_CERTIFIED"
    invalid_scope["insertion_safe"] = True
    _assert_raises(
        ValueError,
        "insertion_safe=false",
        lambda: select_representative_candidates(invalid_scope, count=1),
    )

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        _assert_raises(
            ValueError,
            "must name a .png",
            lambda: render_top_down_candidate_image(
                MAP_PATH, root / "candidate.jpg", count=1),
        )
        _assert_raises(
            ValueError,
            "selection_mode must be",
            lambda: render_top_down_candidate_image(
                MAP_PATH,
                root / "candidate.png",
                selection_mode="unknown",
                count=1,
            ),
        )
        stale = copy.deepcopy(document)
        stale["provenance"]["part"]["sha256"] = "0" * 64
        stale_path = root / "stale-map.json"
        stale_path.write_text(json.dumps(stale), encoding="utf-8")
        output = root / "must-not-exist.png"
        _assert_raises(
            ValueError,
            "does not match grasp-map provenance",
            lambda: render_top_down_candidate_image(
                stale_path, output, count=1, width=800, height=600),
        )
        assert not output.exists()


def test_world_pose_controls_approach_glyph_and_unreadable_closing_axis_fails():
    document = _document()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        in_plane = copy.deepcopy(document)
        in_plane["inputs"]["T_W_P_insert"] = [
            [0.0, 0.0, 1.0, 0.425],
            [0.0, -1.0, 0.0, -0.455],
            [1.0, 0.0, 0.0, 0.350],
            [0.0, 0.0, 0.0, 1.0],
        ]
        in_plane_path = root / "in-plane-map.json"
        in_plane_path.write_text(json.dumps(in_plane), encoding="utf-8")
        metadata = render_top_down_candidate_image(
            in_plane_path,
            root / "in-plane.png",
            count=1,
            width=800,
            height=600,
        )
        assert metadata["view"]["approach_glyph"] == "world_xy_arrow"
        assert np.allclose(
            np.asarray(metadata["candidates"][0]["T_W_E"])[:3, 2],
            [0.0, 1.0, 0.0],
            atol=1e-12,
            rtol=0.0,
        )

        closing_into_view = copy.deepcopy(document)
        closing_into_view["inputs"]["T_W_P_insert"] = [
            [1.0, 0.0, 0.0, 0.425],
            [0.0, 1.0, 0.0, -0.455],
            [0.0, 0.0, 1.0, 0.350],
            [0.0, 0.0, 0.0, 1.0],
        ]
        closing_path = root / "closing-into-view-map.json"
        closing_path.write_text(json.dumps(closing_into_view), encoding="utf-8")
        _assert_raises(
            ValueError,
            "cannot be represented honestly in world XY",
            lambda: render_top_down_candidate_image(
                closing_path,
                root / "closing-into-view.png",
                count=1,
                width=800,
                height=600,
            ),
        )


if __name__ == "__main__":
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(f"passed {len(tests)} two-finger grasp visualization tests")
