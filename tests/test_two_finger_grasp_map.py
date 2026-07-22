"""Focused tests for the continuous object-only two-finger grasp map."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mujoco_sim.core.se3 import compose, transform_from_rpy  # noqa: E402
from mujoco_sim.modeling.grasps import TriangleMesh  # noqa: E402
from mujoco_sim.modeling.two_finger_grasp_map import (  # noqa: E402
    SCOPE,
    generate_two_finger_grasp_map,
    load_scaled_binary_stl,
)


def _box_triangles(minimum, maximum):
    low = np.asarray(minimum, dtype=float)
    high = np.asarray(maximum, dtype=float)
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
        [0, 2, 1], [0, 3, 2],       # -Z
        [4, 5, 6], [4, 6, 7],       # +Z
        [0, 1, 5], [0, 5, 4],       # -Y
        [3, 7, 6], [3, 6, 2],       # +Y
        [0, 4, 7], [0, 7, 3],       # -X
        [1, 2, 6], [1, 6, 5],       # +X
    ])
    return vertices[faces]


def _box_mesh(*, source="analytic_box"):
    return TriangleMesh.from_triangles(
        _box_triangles([0.0, -0.03, -0.05], [0.04, 0.03, 0.05]),
        source=source,
    )


def _box_map(T_W_P_insert=None, **overrides):
    arguments = dict(
        mesh=_box_mesh(),
        T_W_P_insert=np.eye(4) if T_W_P_insert is None else T_W_P_insert,
        insertion_axis_P=[0.0, 0.0, -1.0],
        opening_range_m=[0.03, 0.05],
        friction_coefficient=0.5,
        maximum_surface_tilt_from_lateral_rad=np.deg2rad(2.0),
        maximum_antipodal_normal_error_rad=np.deg2rad(2.0),
        roll_bounds_rad=[-0.4, 0.6],
    )
    arguments.update(overrides)
    return generate_two_finger_grasp_map(**arguments)


def _assert_raises(error_type, text, function):
    try:
        function()
    except error_type as error:
        assert text in str(error)
    else:
        raise AssertionError(f"expected {error_type.__name__}: {text}")


def test_analytic_box_is_one_continuous_family_not_sampled_poses():
    grasp_map = _box_map()
    assert SCOPE == "OBJECT_ONLY_LOCAL_SURFACE_CANDIDATE"
    assert grasp_map.scope == SCOPE
    assert grasp_map.insertion_safe is False
    assert len(grasp_map.families) == 1

    family = grasp_map.families[0]
    assert family.contains(0.0, 0.0, 0.0)
    assert not family.contains(0.0, 0.0, 0.61)
    assert not family.contains(0.2, 0.0, 0.0)
    assert len(family.domains) == 2  # exact union of the two triangle charts
    assert np.isclose(sum(domain.area_m2 for domain in family.domains), 0.006)
    assert np.allclose(family.closing_axis_P, [1.0, 0.0, 0.0])
    assert np.allclose(family.aperture_coefficients_m, [0.04, 0.0, 0.0])

    evaluation = family.evaluate(0.0, 0.0, 0.0)
    assert evaluation.scope == SCOPE
    assert evaluation.insertion_safe is False
    assert np.isclose(evaluation.aperture_m, 0.04)
    assert np.allclose(evaluation.contacts_P_m, [[0.0, 0.0, 0.0],
                                                 [0.04, 0.0, 0.0]])
    assert np.allclose(evaluation.T_P_E[:3, 3], [0.02, 0.0, 0.0])
    assert np.allclose(evaluation.T_P_E[:3, 1], family.closing_axis_P)
    assert np.allclose(evaluation.T_P_E[:3, 2], [0.0, 0.0, -1.0])
    assert np.isclose(np.linalg.det(evaluation.T_P_E[:3, :3]), 1.0)


def test_disconnected_coplanar_patches_remain_an_exact_union():
    first = _box_triangles([0.0, -0.05, -0.02], [0.04, -0.03, 0.02])
    second = _box_triangles([0.0, 0.03, -0.02], [0.04, 0.05, 0.02])
    mesh = TriangleMesh.from_triangles(
        np.concatenate((first, second)), source="two_disconnected_boxes")
    grasp_map = generate_two_finger_grasp_map(
        mesh,
        T_W_P_insert=np.eye(4),
        insertion_axis_P=[0.0, 0.0, -1.0],
        opening_range_m=[0.039, 0.041],
        friction_coefficient=0.3,
        lateral_normal_threshold=0.01,
        maximum_antipodal_normal_error_rad=np.deg2rad(1.0),
        roll_bounds_rad=[-0.1, 0.1],
    )
    assert len(grasp_map.families) == 1
    family = grasp_map.families[0]
    # u is -Y for this deterministic chart.  Both components exist; their gap
    # is not convexified into a fictitious contact region.
    assert family.contains(0.04, 0.0, 0.0)
    assert family.contains(-0.04, 0.0, 0.0)
    assert not family.contains(0.0, 0.0, 0.0)
    assert np.isclose(sum(domain.area_m2 for domain in family.domains), 0.0016)


def test_opening_bounds_and_edge_margin_clip_the_continuous_domain():
    assert len(_box_map(opening_range_m=[0.041, 0.05]).families) == 0
    untrimmed = _box_map(contact_edge_margin_m=0.0).families[0]
    trimmed = _box_map(contact_edge_margin_m=0.002).families[0]
    untrimmed_area = sum(domain.area_m2 for domain in untrimmed.domains)
    trimmed_area = sum(domain.area_m2 for domain in trimmed.domains)
    assert 0.0 < trimmed_area < untrimmed_area
    assert trimmed.contact_edge_margin_m == 0.002


def test_world_pose_changes_only_world_outputs():
    T_W_P = transform_from_rpy([0.31, -0.22, 0.47], [0.2, -0.3, 0.7])
    identity_map = _box_map()
    transformed_map = _box_map(T_W_P_insert=T_W_P)
    first = identity_map.families[0].evaluate(0.007, -0.011, 0.2)
    second = transformed_map.families[0].evaluate(0.007, -0.011, 0.2)
    assert np.allclose(first.contacts_P_m, second.contacts_P_m)
    assert np.allclose(first.T_P_E, second.T_P_E)
    assert np.allclose(second.T_W_E, compose(T_W_P, second.T_P_E))
    expected_contacts = first.contacts_P_m @ T_W_P[:3, :3].T + T_W_P[:3, 3]
    assert np.allclose(second.contacts_W_m, expected_contacts)


def test_serialization_states_set_semantics_and_certification_boundary():
    document = _box_map().to_dict()
    json.dumps(document, sort_keys=True, allow_nan=False)
    assert document["artifact_type"] == "two_finger_continuous_grasp_map"
    assert document["scope"] == SCOPE
    assert document["insertion_safe"] is False
    assert document["set_representation"]["type"] == (
        "finite_union_of_continuous_families")
    assert document["set_representation"]["family_count"] == 1
    assert np.allclose(document["inputs"]["insertion_axis_W"], [0.0, 0.0, -1.0])
    assert document["inputs"]["minimum_surface_area_m2"] == 0.0
    assert document["inputs"]["plane_tolerance_m"] > 0.0
    assert document["inputs"]["normal_tolerance"] >= 0.0
    family = document["families"][0]
    assert family["type"] == "continuous_ideal_point_contact_parallel_jaw_family"
    assert family["aperture_map"]["type"] == "affine"
    assert family["parameterization"]["parameters"] == [
        "u_m", "v_m", "roll_rad"]
    assert all(domain["type"] == "convex_polygon"
               for domain in family["parameterization"]["domains"])
    assert any("PCB" in limitation for limitation in document["limitations"])


def test_invalid_inputs_fail_closed():
    invalid_transform = np.eye(4)
    invalid_transform[0, 0] = 2.0
    _assert_raises(ValueError, "rotation", lambda: _box_map(
        T_W_P_insert=invalid_transform))
    _assert_raises(ValueError, "nonzero", lambda: _box_map(
        insertion_axis_P=[0.0, 0.0, 0.0]))
    _assert_raises(ValueError, "opening_range_m", lambda: _box_map(
        opening_range_m=[0.05, 0.03]))
    _assert_raises(ValueError, "non-negative", lambda: _box_map(
        friction_coefficient=-0.1))
    _assert_raises(ValueError, "not both", lambda: _box_map(
        lateral_normal_threshold=0.1))
    _assert_raises(ValueError, "non-negative", lambda: _box_map(
        contact_edge_margin_m=-0.001))
    _assert_raises(ValueError, "outside family", lambda: (
        _box_map().families[0].evaluate(1.0, 1.0, 0.0)))
    _assert_raises(ValueError, "positive", lambda: load_scaled_binary_stl(
        "not-read-when-scale-is-invalid.stl", scale_to_m=0.0))


def test_connector_header_smoke_produces_useful_continuous_families():
    mesh = load_scaled_binary_stl(
        ROOT / "parts/connector_header/connector_header_part.STL",
        scale_to_m=0.001,
    )
    grasp_map = generate_two_finger_grasp_map(
        mesh,
        T_W_P_insert=np.eye(4),
        insertion_axis_P=[0.0, -1.0, 0.0],
        opening_range_m=[0.002, 0.024],
        friction_coefficient=0.5,
        maximum_surface_tilt_from_lateral_rad=np.deg2rad(25.0),
        maximum_antipodal_normal_error_rad=np.deg2rad(5.0),
        minimum_surface_area_m2=1e-6,
        contact_edge_margin_m=0.00015,
        roll_bounds_rad=[-np.pi, np.pi],
    )
    assert grasp_map.families
    assert any(
        np.allclose(family.closing_axis_P, [0.0, 0.0, 1.0], atol=1e-6)
        for family in grasp_map.families
    )
    for family in grasp_map.families:
        for domain in family.domains:
            centroid = np.mean(domain.vertices_uv_m, axis=0)
            evaluation = family.evaluate(float(centroid[0]), float(centroid[1]), 0.0)
            assert 0.002 - 1e-9 <= evaluation.aperture_m <= 0.024 + 1e-9
            assert np.allclose(
                evaluation.T_W_E,
                compose(grasp_map.T_W_P_insert, evaluation.T_P_E),
            )


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
    print(f"passed {len(tests)} continuous two-finger grasp-map tests")
