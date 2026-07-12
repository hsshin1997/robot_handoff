"""Focused tests for geometry-driven parallel-jaw grasp generation."""
from __future__ import annotations

import os
import struct
import sys
import tempfile

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from mujoco_sim.geometry_grasps import (
    ParallelJawGripper,
    generate_antipodal_grasps,
    load_binary_stl,
    sample_surface_patches,
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
    triangles = vertices[faces]
    if rotation is not None:
        triangles = triangles @ np.asarray(rotation, dtype=float).T
    if translation is not None:
        triangles = triangles + np.asarray(translation, dtype=float)
    return triangles


def _write_binary_stl(path, triangles):
    triangles = np.asarray(triangles, dtype=float)
    with open(path, "wb") as stream:
        stream.write(b"geometry-grasp-test".ljust(80, b"\0"))
        stream.write(struct.pack("<I", len(triangles)))
        for triangle in triangles:
            normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
            normal /= np.linalg.norm(normal)
            stream.write(struct.pack("<12fH", *normal.astype(np.float32),
                                     *triangle.astype(np.float32).ravel(), 0))


def _load_temporary_box(size, rotation=None, translation=None):
    directory = tempfile.TemporaryDirectory()
    path = os.path.join(directory.name, "box.stl")
    triangles = _box_triangles(size, rotation, translation)
    _write_binary_stl(path, triangles)
    return directory, load_binary_stl(path), triangles


def _rotation():
    # A deliberately non-axis-aligned frame to guard against fixed part-axis
    # grasp rules.
    axis = np.array([1.0, -2.0, 0.7])
    axis /= np.linalg.norm(axis)
    angle = np.radians(37.0)
    skew = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def test_binary_stl_parser_preserves_native_frame_and_geometry():
    size = np.array([0.12, 0.022, 0.014])
    translation = np.array([0.41, -0.27, 0.083])
    owner, mesh, triangles = _load_temporary_box(size, translation=translation)
    try:
        assert mesh.triangles.shape == (12, 3, 3)
        assert np.allclose(mesh.triangles, triangles, atol=2e-8)
        assert np.allclose(mesh.bounds_min, translation - size / 2.0, atol=2e-8)
        assert np.allclose(mesh.bounds_max, translation + size / 2.0, atol=2e-8)
        expected_area = 2.0 * (size[0] * size[1]
                               + size[0] * size[2]
                               + size[1] * size[2])
        assert np.isclose(mesh.surface_area, expected_area, rtol=2e-6)
        assert np.allclose(np.linalg.norm(mesh.normals, axis=1), 1.0)
        assert np.all(mesh.areas > 0.0)
    finally:
        owner.cleanup()


def test_surface_patch_sampling_is_deterministic_and_on_surface():
    owner, mesh, _ = _load_temporary_box([0.08, 0.03, 0.02],
                                         rotation=_rotation(),
                                         translation=[0.2, 0.1, -0.3])
    try:
        first = sample_surface_patches(mesh, 73)
        second = sample_surface_patches(mesh, 73)
        assert len(first) == 73
        for a, b in zip(first, second):
            assert np.array_equal(a.point, b.point)
            assert np.array_equal(a.normal, b.normal)
            triangle = mesh.triangles[a.triangle_index]
            assert abs(float((a.point - triangle[0]) @ a.normal)) < 1e-9
            assert np.all(a.point >= mesh.bounds_min - 1e-9)
            assert np.all(a.point <= mesh.bounds_max + 1e-9)
        assert np.isclose(sum(p.represented_area for p in first),
                          mesh.surface_area)
    finally:
        owner.cleanup()


def test_antipodal_grasps_have_consistent_contacts_and_part_frame_transform():
    rotation = _rotation()
    owner, mesh, _ = _load_temporary_box(
        [0.12, 0.022, 0.014], rotation=rotation,
        translation=[0.31, -0.12, 0.07])
    gripper = ParallelJawGripper(
        min_opening=0.006,
        max_opening=0.030,
        pad_size=(0.012, 0.010),
        pad_depth=0.15,
        friction_coefficient=0.5,
    )
    try:
        candidates = generate_antipodal_grasps(
            mesh, gripper, surface_samples=320,
            approaches_per_pair=4, max_candidates=64)
        assert candidates
        assert all(candidates[index].quality >= candidates[index + 1].quality
                   for index in range(len(candidates) - 1))
        friction_cosine = 1.0 / np.sqrt(1.0 + gripper.friction_coefficient**2)
        for candidate in candidates:
            rotation_P_E = candidate.T_P_E[:3, :3]
            separation = candidate.contact_points[1] - candidate.contact_points[0]
            assert np.allclose(candidate.midpoint,
                               np.mean(candidate.contact_points, axis=0))
            assert np.allclose(rotation_P_E[:, 1], candidate.closing_direction)
            assert np.allclose(rotation_P_E[:, 2], candidate.approach_direction)
            assert np.allclose(rotation_P_E.T @ rotation_P_E, np.eye(3), atol=1e-8)
            assert np.isclose(np.linalg.det(rotation_P_E), 1.0)
            assert np.isclose(np.linalg.norm(separation), candidate.required_opening)
            assert np.allclose(separation / np.linalg.norm(separation),
                               candidate.closing_direction)
            assert candidate.contact_normals[0] @ (-candidate.closing_direction) \
                >= friction_cosine - 1e-8
            assert candidate.contact_normals[1] @ candidate.closing_direction \
                >= friction_cosine - 1e-8
            assert gripper.min_opening <= candidate.required_opening <= gripper.max_opening
    finally:
        owner.cleanup()


def test_elongated_part_produces_spatially_distinct_noncentral_grasps():
    rotation = _rotation()
    translation = np.array([-0.2, 0.3, 0.5])
    owner, mesh, _ = _load_temporary_box(
        [0.16, 0.024, 0.014], rotation=rotation, translation=translation)
    gripper = ParallelJawGripper(0.008, 0.032, (0.010, 0.010), 0.20)
    try:
        candidates = generate_antipodal_grasps(
            mesh, gripper, surface_samples=480,
            approaches_per_pair=4, max_candidates=32)
        assert len(candidates) >= 8
        # Project grasp locations onto the actual (rotated) long dimension.
        coordinates = np.array([
            (candidate.midpoint - translation) @ rotation[:, 0]
            for candidate in candidates
        ])
        assert np.ptp(coordinates) > 0.10
        assert len(np.unique(np.round(coordinates, 3))) >= 6
        # Center-only rules would make every projection zero.
        assert np.any(np.abs(coordinates) > 0.05)
    finally:
        owner.cleanup()


def test_opening_range_is_a_hard_gripper_capability_gate():
    owner, mesh, _ = _load_temporary_box([0.10, 0.022, 0.014])
    try:
        too_narrow = ParallelJawGripper(0.002, 0.010, (0.01, 0.01), 0.12)
        too_wide = ParallelJawGripper(0.030, 0.050, (0.01, 0.01), 0.12)
        assert not generate_antipodal_grasps(mesh, too_narrow, surface_samples=128)
        assert not generate_antipodal_grasps(mesh, too_wide, surface_samples=128)
    finally:
        owner.cleanup()


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS  {test.__name__}")
    print(f"\n{len(tests)}/{len(tests)} passed")
