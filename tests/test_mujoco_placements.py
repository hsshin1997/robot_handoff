"""Deterministic geometry tests for stable placements and stage regions."""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from mujoco_sim.geometry_grasps import TriangleMesh
from mujoco_sim.placements import (
    RectangularStage,
    estimate_center_of_mass,
    generate_stable_placements,
    instantiate_on_rectangular_stage,
    signed_support_margin,
)


def _mesh(triangles: np.ndarray) -> TriangleMesh:
    triangles = np.asarray(triangles, dtype=float)
    cross = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    double_area = np.linalg.norm(cross, axis=1)
    normals = cross / double_area[:, None]
    vertices = triangles.reshape(-1, 3)
    return TriangleMesh(
        triangles=triangles,
        normals=normals,
        areas=0.5 * double_area,
        bounds_min=np.min(vertices, axis=0),
        bounds_max=np.max(vertices, axis=0),
    )


def _box_triangles(size, rotation=None, translation=None):
    half = 0.5 * np.asarray(size, dtype=float)
    vertices = np.array([
        [-half[0], -half[1], -half[2]],
        [+half[0], -half[1], -half[2]],
        [+half[0], +half[1], -half[2]],
        [-half[0], +half[1], -half[2]],
        [-half[0], -half[1], +half[2]],
        [+half[0], -half[1], +half[2]],
        [+half[0], +half[1], +half[2]],
        [-half[0], +half[1], +half[2]],
    ])
    faces = np.array([
        [0, 2, 1], [0, 3, 2],       # -Z
        [4, 5, 6], [4, 6, 7],       # +Z
        [0, 1, 5], [0, 5, 4],       # -Y
        [3, 7, 6], [3, 6, 2],       # +Y
        [0, 4, 7], [0, 7, 3],       # -X
        [1, 2, 6], [1, 6, 5],       # +X
    ])
    if rotation is not None:
        vertices = vertices @ np.asarray(rotation, dtype=float).T
    if translation is not None:
        vertices = vertices + np.asarray(translation, dtype=float)
    return vertices[faces]


def _rotation():
    axis = np.array([1.0, -2.0, 0.7])
    axis /= np.linalg.norm(axis)
    angle = np.radians(37.0)
    skew = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def _triangular_prism(translation=None) -> TriangleMesh:
    # An asymmetric triangular cross-section extruded along Y.  The eight
    # triangles form five outward-facing planar patches.
    vertices = np.array([
        [-1.0, -0.3, 0.0],
        [+1.0, -0.3, 0.0],
        [-0.4, -0.3, 1.0],
        [-1.0, +0.3, 0.0],
        [+1.0, +0.3, 0.0],
        [-0.4, +0.3, 1.0],
    ])
    if translation is not None:
        vertices += np.asarray(translation, dtype=float)
    faces = np.array([
        [0, 1, 2],                  # -Y triangular end
        [3, 5, 4],                  # +Y triangular end
        [0, 3, 4], [0, 4, 1],       # bottom
        [1, 4, 5], [1, 5, 2],       # long sloped side
        [2, 5, 3], [2, 3, 0],       # short sloped side
    ])
    return _mesh(vertices[faces])


def test_closed_mesh_volume_centroid_and_open_mesh_bbox_fallback():
    translation = np.array([1.0e6, -2.0e6, 3.0e6])
    prism = _triangular_prism(translation)
    expected = translation + np.array([-0.4 / 3.0, 0.0, 1.0 / 3.0])
    assert np.allclose(estimate_center_of_mass(prism), expected, atol=1e-9)

    triangle = np.array([[[2.0, -1.0, 4.0],
                          [5.0, -1.0, 4.0],
                          [2.0, 3.0, 4.0]]])
    open_mesh = _mesh(triangle)
    expected_bbox_center = 0.5 * (open_mesh.bounds_min + open_mesh.bounds_max)
    assert np.allclose(estimate_center_of_mass(open_mesh), expected_bbox_center)


def test_rotated_box_facets_group_and_produce_six_stable_supports():
    size = np.array([2.0, 1.0, 0.5])
    rotation = _rotation()
    translation = np.array([0.7, -0.2, 1.3])
    mesh = _mesh(_box_triangles(size, rotation, translation))
    placements = generate_stable_placements(mesh)

    # Twelve triangles must group into the six physical box faces.
    assert len(placements) == 6
    assert all(len(item.triangle_indices) == 2 for item in placements)
    assert np.isclose(sum(item.probability_proxy for item in placements), 1.0)
    center = translation
    assert np.allclose(estimate_center_of_mass(mesh), center, atol=1e-10)

    vertices = mesh.triangles.reshape(-1, 3)
    rotations = []
    for placement in placements:
        R_N_P = placement.T_N_P[:3, :3]
        rotations.append(R_N_P)
        assert np.allclose(R_N_P @ placement.support_normal_P, [0.0, 0.0, -1.0])
        vertices_N = vertices @ R_N_P.T + placement.T_N_P[:3, 3]
        assert np.min(vertices_N[:, 2]) >= -1e-9
        assert np.isclose(np.min(vertices_N[:, 2]), 0.0, atol=1e-9)
        assert np.isclose(
            signed_support_margin(
                placement.center_of_mass_N[:2], placement.support_polygon_N
            ),
            placement.support_margin,
        )
        assert placement.support_margin > 0.0
        assert placement.support_area > 0.0

    # Rotation deduplication is part of the public contract.
    for index, first in enumerate(rotations):
        for second in rotations[index + 1:]:
            cosine = np.clip((np.trace(first @ second.T) - 1.0) / 2.0, -1.0, 1.0)
            assert np.arccos(cosine) > np.radians(1.0) - 1e-10

    # Frozen dataclasses alone do not freeze ndarray storage; verify that the
    # implementation makes transform and polygon buffers read-only as well.
    try:
        placements[0].T_N_P[0, 0] = 7.0
    except ValueError:
        pass
    else:
        raise AssertionError("StablePlacement transform is mutable")
    try:
        placements[0].support_polygon_N[0, 0] = 7.0
    except ValueError:
        pass
    else:
        raise AssertionError("StablePlacement support polygon is mutable")


def test_asymmetric_prism_has_geometry_derived_sloped_placements():
    mesh = _triangular_prism()
    first = generate_stable_placements(mesh)
    second = generate_stable_placements(mesh)

    assert len(first) == 5
    assert len(second) == len(first)
    expected_center = np.array([-0.4 / 3.0, 0.0, 1.0 / 3.0])
    assert np.allclose(estimate_center_of_mass(mesh), expected_center)
    assert any(
        np.count_nonzero(np.abs(item.support_normal_P) > 1e-8) == 2
        for item in first
    )
    for a, b in zip(first, second):
        assert np.array_equal(a.T_N_P, b.T_N_P)
        assert np.array_equal(a.support_polygon_N, b.support_polygon_N)
        assert a.probability_proxy == b.probability_proxy


def test_disconnected_solids_correct_winding_and_merge_coplanar_contacts():
    left = _box_triangles([1.0, 1.0, 1.0], translation=[-1.0, 0.0, 0.0])
    right = _box_triangles([1.0, 1.0, 1.0], translation=[1.0, 0.0, 0.0])
    right = right[:, [0, 2, 1]]  # Independent CAD component has reversed winding.
    # A small damaged/open internal component must not invalidate the reliable
    # volume centroids of the two dominant closed solids.
    fragment = np.array([[[-0.05, -0.05, 0.0],
                          [+0.05, -0.05, 0.0],
                          [0.0, +0.05, 0.0]]])
    mesh = _mesh(np.concatenate((left, right, fragment)))

    assert np.allclose(estimate_center_of_mass(mesh), [0.0, 0.0, 0.0])
    placements = generate_stable_placements(mesh)
    assert len(placements) == 6
    bottom = next(item for item in placements
                  if np.allclose(item.support_normal_P, [0.0, 0.0, -1.0]))
    # Four coplanar triangles from two disconnected components jointly form
    # the support; neither individual square contains the combined COM.
    assert len(bottom.triangle_indices) == 4
    assert np.min(bottom.support_polygon_N[:, 0]) < -1.4
    assert np.max(bottom.support_polygon_N[:, 0]) > 1.4
    assert bottom.support_margin > 0.49


def test_supplied_com_and_minimum_margin_reject_unstable_faces():
    mesh = _mesh(_box_triangles([2.0, 1.0, 0.5]))
    # An attached payload moves the total COM beyond the box in +X.  Four
    # placements project that COM outside their support polygons.  The two X
    # faces remain stable because X becomes the vertical direction there.
    placements = generate_stable_placements(
        mesh,
        center_of_mass_P=[1.2, 0.0, 0.0],
        minimum_support_margin=0.1,
    )
    assert len(placements) == 2
    assert all(abs(item.support_normal_P[0]) > 1.0 - 1e-12
               for item in placements)
    assert all(item.support_margin >= 0.1 for item in placements)


def test_signed_support_margin_has_inside_boundary_outside_sign():
    square = np.array([[-1.0, -1.0], [1.0, -1.0],
                       [1.0, 1.0], [-1.0, 1.0]])
    assert np.isclose(signed_support_margin([0.0, 0.0], square), 1.0)
    assert np.isclose(signed_support_margin([1.0, 0.0], square), 0.0)
    assert np.isclose(signed_support_margin([1.2, 0.0], square), -0.2)
    assert np.isclose(signed_support_margin([0.0, 0.0], square[::-1]), 1.0)


def test_rectangular_stage_filters_yaws_using_complete_part_footprint():
    mesh = _mesh(_box_triangles([2.0, 1.0, 0.5]))
    placements = generate_stable_placements(mesh)
    angle = np.radians(23.0)
    T_W_S = np.eye(4)
    T_W_S[:3, :3] = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    T_W_S[:3, 3] = [0.4, -0.1, 0.93]
    stage = RectangularStage(T_W_S, size_xy=(2.2, 1.2), edge_margin=0.05)
    instances = instantiate_on_rectangular_stage(
        mesh,
        placements,
        stage,
        yaw_samples_deg=[0.0, 90.0, 360.0],  # 360 must deduplicate with 0.
    )

    assert len(instances) == 8
    half_usable = np.array([1.05, 0.55])
    inverse_stage = np.linalg.inv(T_W_S)
    vertices = mesh.triangles.reshape(-1, 3)
    for instance in instances:
        assert np.all(np.abs(instance.footprint_polygon_S) <= half_usable + 1e-10)
        assert np.all(instance.translation_bounds_S[:, 0]
                      <= instance.translation_bounds_S[:, 1])
        vertices_W = (
            vertices @ instance.T_W_P[:3, :3].T
            + instance.T_W_P[:3, 3]
        )
        vertices_S = (
            vertices_W @ inverse_stage[:3, :3].T
            + inverse_stage[:3, 3]
        )
        assert np.min(vertices_S[:, 2]) >= -1e-9
        assert np.isclose(np.min(vertices_S[:, 2]), 0.0, atol=1e-9)
        assert instance.edge_clearance >= 0.0

    too_small = RectangularStage(np.eye(4), size_xy=(0.4, 0.4))
    assert not instantiate_on_rectangular_stage(
        mesh, placements, too_small, yaw_samples_deg=[0.0, 90.0]
    )


def test_real_connector_multicomponent_cad_has_stable_supports():
    path = os.path.join(ROOT, "parts", "conn_header", "conn_header_bin.stl")
    mesh = TriangleMesh.from_binary_stl(path)
    bbox_center = 0.5 * (mesh.bounds_min + mesh.bounds_max)
    center = estimate_center_of_mass(mesh)
    # This STL contains 21 usable closed solids plus two damaged/open pieces.
    # Per-component volume aggregation must be used instead of falling back to
    # the whole-file bbox center, which lies outside its principal support.
    assert not np.allclose(center, bbox_center, atol=1e-5)
    placements = generate_stable_placements(mesh)
    assert len(placements) >= 3
    assert np.isclose(sum(item.probability_proxy for item in placements), 1.0)
    principal = max(placements, key=lambda item: item.support_area)
    assert np.allclose(principal.support_normal_P, [0.0, 0.0, 1.0])
    assert principal.support_margin > 0.0005
    vertices_N = (
        mesh.vertices @ principal.T_N_P[:3, :3].T
        + principal.T_N_P[:3, 3]
    )
    assert np.min(vertices_N[:, 2]) >= -1e-10


if __name__ == "__main__":
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
        print(f"PASS  {test.__name__}")
    print(f"\n{len(tests)}/{len(tests)} passed")
