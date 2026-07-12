"""Project-aware planning mesh ingestion across CAD formats and units."""
from __future__ import annotations

import os
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from mujoco_sim.cad_preprocess import prepare_cad, write_binary_stl  # noqa: E402
from mujoco_sim.part_mesh import (load_obj_triangles,
                                  load_prepared_triangle_mesh,
                                  load_project_part_mesh)  # noqa: E402


def _record(triangle: np.ndarray) -> bytes:
    triangle = np.asarray(triangle, dtype=float)
    normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
    normal /= np.linalg.norm(normal)
    return struct.pack(
        "<12fH", *normal.astype(np.float32),
        *triangle.astype(np.float32).ravel(), 0)


def _tetrahedron(offset=(10.0, 20.0, 30.0)) -> np.ndarray:
    offset = np.asarray(offset, dtype=float)
    vertices = offset + np.array([
        [0.0, 0.0, 0.0],
        [1000.0, 0.0, 0.0],
        [0.0, 1000.0, 0.0],
        [0.0, 0.0, 1000.0],
    ])
    return vertices[np.array([
        [0, 2, 1],
        [0, 1, 3],
        [0, 3, 2],
        [1, 2, 3],
    ])]


def test_prepared_stl_chunks_are_combined_in_order_and_scaled_once():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "part.stl"
        triangles = _tetrahedron()
        write_binary_stl(source, [_record(triangle) for triangle in triangles])
        preparation = prepare_cad(
            source, root / "generated",
            scale_to_m=(0.001, 0.002, 0.003),
            role="part-visual", max_faces=2)
        assert len(preparation.metadata["visual"]["chunks"]) == 2

        mesh = load_prepared_triangle_mesh(preparation)
        expected = triangles * np.array([0.001, 0.002, 0.003])[None, None, :]
        assert mesh.triangles.shape == (4, 3, 3)
        assert np.allclose(mesh.triangles, expected, atol=1e-7)
        assert np.allclose(mesh.bounds_min, expected.reshape(-1, 3).min(axis=0))
        assert np.allclose(mesh.bounds_max, expected.reshape(-1, 3).max(axis=0))
        # Normals and areas must correspond to the anisotropically scaled SI
        # triangles rather than to independently loaded source chunks.
        cross = np.cross(expected[:, 1] - expected[:, 0],
                         expected[:, 2] - expected[:, 0])
        assert np.allclose(mesh.areas, 0.5 * np.linalg.norm(cross, axis=1))
        assert np.allclose(np.linalg.norm(mesh.normals, axis=1), 1.0)


def test_ascii_stl_uses_project_declared_units_and_prepared_binary_geometry():
    ascii_stl = """solid offset_triangle
facet normal 0 0 1
  outer loop
    vertex 100 200 300
    vertex 1100 200 300
    vertex 100 1200 300
  endloop
endfacet
endsolid offset_triangle
"""
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "part_ascii.stl"
        source.write_text(ascii_stl, encoding="utf-8")
        project = SimpleNamespace(
            active_part_path=str(source),
            active_part={"cad": str(source), "cad_units": "mm"},
        )
        result = load_project_part_mesh(project, root / "generated")
        mesh = result.mesh
        assert result.preparation.metadata["visual"]["input_stl_encoding"] == "ascii-stl"
        assert result.preparation.metadata["visual"]["format"] == "binary-stl"
        assert np.allclose(mesh.triangles[0], [
            [0.1, 0.2, 0.3],
            [1.1, 0.2, 0.3],
            [0.1, 1.2, 0.3],
        ])
        assert np.isclose(mesh.areas[0], 0.5)
        assert result.artifact_fingerprint == result.preparation.metadata[
            "artifact_fingerprint"]


def test_project_declared_xyz_cad_scale_is_applied_without_recentering():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "scaled_part.stl"
        triangle = np.array([
            [10.0, 20.0, 30.0],
            [20.0, 20.0, 30.0],
            [10.0, 30.0, 30.0],
        ])
        write_binary_stl(source, [_record(triangle)])
        project = SimpleNamespace(
            active_part_path=str(source),
            active_part={
                "cad": str(source),
                "cad_scale_to_m": [0.1, 0.01, 0.001],
            },
        )
        mesh = load_project_part_mesh(project, root / "generated").mesh
        expected = triangle * np.array([0.1, 0.01, 0.001])
        assert np.allclose(mesh.triangles[0], expected)
        assert np.allclose(mesh.bounds_min, [1.0, 0.2, 0.03])
        assert np.allclose(mesh.bounds_max, [2.0, 0.3, 0.03])


def test_obj_polygon_triangulation_is_deterministic_and_preserves_vertices():
    obj = """# one quad in an offset native CAD frame, millimetres
v 100 200 300
v 1100 200 300
v 1100 1200 300
v 100 1200 300
vt 0 0
vt 1 0
vt 1 1
vt 0 1
vn 0 0 1
f -4/1/1 -3/2/1 -2/3/1 -1/4/1
"""
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "part.obj"
        source.write_text(obj, encoding="utf-8")
        first = load_obj_triangles(source)
        second = load_obj_triangles(source)
        assert np.array_equal(first, second)
        assert first.shape == (2, 3, 3)

        preparation = prepare_cad(
            source, root / "generated", units="mm", role="part-visual")
        mesh = load_prepared_triangle_mesh(preparation)
        expected_vertices = np.array([
            [0.1, 0.2, 0.3],
            [1.1, 0.2, 0.3],
            [1.1, 1.2, 0.3],
            [0.1, 1.2, 0.3],
        ])
        actual_vertices = np.unique(mesh.triangles.reshape(-1, 3), axis=0)
        assert np.allclose(actual_vertices,
                           np.unique(expected_vertices, axis=0))
        assert np.isclose(mesh.surface_area, 1.0)
        assert np.allclose(mesh.bounds_min, [0.1, 0.2, 0.3])
        assert np.allclose(mesh.bounds_max, [1.1, 1.2, 0.3])


def test_obj_concave_face_uses_nonoverlapping_ear_clipping():
    obj = """v 0 0 0
v 2 0 0
v 2 2 0
v 1 1 0
v 0 2 0
f 1 2 3 4 5
"""
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / "concave.obj"
        source.write_text(obj, encoding="utf-8")
        triangles = load_obj_triangles(source)
        assert triangles.shape == (3, 3, 3)
        areas = 0.5 * np.linalg.norm(np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0]), axis=1)
        # Shoelace area of the concave polygon is exactly 3.
        assert np.isclose(np.sum(areas), 3.0)


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS  {test.__name__}")
    print(f"\n{len(tests)}/{len(tests)} passed")
