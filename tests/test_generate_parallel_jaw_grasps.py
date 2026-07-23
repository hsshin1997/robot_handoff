"""Tests for the direct CAD-to-parallel-jaw grasp CLI."""
from __future__ import annotations

import json
from pathlib import Path
import struct
import sys
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mujoco_sim.core.se3 import validate_transform  # noqa: E402
from mujoco_sim.modeling.cad_preprocess import write_binary_stl  # noqa: E402
from mujoco_sim.modeling.grasps import (  # noqa: E402
    ParallelJawGripper,
    TriangleMesh,
)
from scripts.generate_parallel_jaw_grasps import (  # noqa: E402
    ARTIFACT_TYPE,
    CLAIM_LEVEL,
    UNRELIABLE_MESH_CLAIM_LEVEL,
    _mesh_topology_audit,
    generate_document,
    main,
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
    for triangle in _box_triangles([-0.02, -0.01, -0.005],
                                   [0.02, 0.01, 0.005]):
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        normal /= np.linalg.norm(normal)
        records.append(struct.pack(
            "<12fH",
            *normal.astype(np.float32),
            *triangle.astype(np.float32).ravel(),
            0,
        ))
    write_binary_stl(path, records)


def _write_open_triangle(path: Path) -> None:
    triangle = np.array([
        [0.0, 0.0, 0.0],
        [0.02, 0.0, 0.0],
        [0.0, 0.02, 0.0],
    ])
    normal = np.array([0.0, 0.0, 1.0])
    record = struct.pack(
        "<12fH",
        *normal.astype(np.float32),
        *triangle.astype(np.float32).ravel(),
        0,
    )
    write_binary_stl(path, [record])


def _write_box_obj(path: Path) -> None:
    path.write_text(
        "\n".join([
            "v -0.02 -0.01 -0.005",
            "v  0.02 -0.01 -0.005",
            "v  0.02  0.01 -0.005",
            "v -0.02  0.01 -0.005",
            "v -0.02 -0.01  0.005",
            "v  0.02 -0.01  0.005",
            "v  0.02  0.01  0.005",
            "v -0.02  0.01  0.005",
            "f 1 4 3 2",
            "f 5 6 7 8",
            "f 1 2 6 5",
            "f 4 8 7 3",
            "f 1 5 8 4",
            "f 2 3 7 6",
            "",
        ]),
        encoding="utf-8",
    )


def _gripper() -> ParallelJawGripper:
    return ParallelJawGripper(
        min_opening=0.008,
        max_opening=0.025,
        pad_size=(0.008, 0.008),
        pad_depth=0.05,
        friction_coefficient=0.5,
    )


def test_generate_document_returns_all_deduplicated_candidates_repeatably():
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        cad = temporary / "box.stl"
        generated = temporary / "prepared"
        _write_box(cad)
        first = generate_document(
            cad,
            units="m",
            scale_to_m=None,
            gripper=_gripper(),
            surface_samples=240,
            approaches_per_pair=8,
            max_candidates=None,
            generated_root=generated,
        )
        second = generate_document(
            cad,
            units="m",
            scale_to_m=None,
            gripper=_gripper(),
            surface_samples=240,
            approaches_per_pair=8,
            max_candidates=None,
            generated_root=generated,
        )
        assert first == second
        assert first["artifact_type"] == ARTIFACT_TYPE
        assert first["claim_level"] == CLAIM_LEVEL
        assert first["continuous_exhaustive"] is False
        assert first["all_deduplicated_accepted_candidates_returned"] is True
        assert first["candidate_cap_applied"] is False
        assert first["cad"]["triangle_count"] == 12
        topology = first["cad"]["topology_audit"]
        assert topology["closed_consistently_wound_two_manifold"] is True
        assert topology["boundary_edge_count"] == 0
        assert first["candidate_count"] > 0
        assert first["candidate_count"] == len(first["candidates"])
        assert first["sampling"]["deduplication"]["position_tolerance_m"] > 0.0
        assert first["sampling"]["closing_directions_per_surface"] == 5
        assert len({item["id"] for item in first["candidates"]}) == len(
            first["candidates"]
        )
        for item in first["candidates"]:
            transform = validate_transform(np.asarray(item["T_P_E"]))
            assert np.allclose(transform[:3, 1], item["closing_direction_P"])
            assert np.allclose(transform[:3, 2], item["approach_direction_P"])
            assert 0.008 <= item["required_opening_m"] <= 0.025


def test_cli_writes_a_capped_machine_readable_subset():
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        cad = temporary / "box.stl"
        output = temporary / "grasps.json"
        generated = temporary / "prepared"
        _write_box(cad)
        exit_code = main([
            str(cad),
            "--units", "m",
            "--min-opening-m", "0.008",
            "--max-opening-m", "0.025",
            "--pad-width-m", "0.008",
            "--pad-height-m", "0.008",
            "--finger-depth-m", "0.05",
            "--surface-samples", "160",
            "--approaches-per-pair", "8",
            "--max-candidates", "5",
            "--generated-root", str(generated),
            "--output", str(output),
        ])
        assert exit_code == 0
        document = json.loads(output.read_text(encoding="utf-8"))
        assert document["all_deduplicated_accepted_candidates_returned"] is False
        assert document["candidate_cap_applied"] is True
        assert 0 < document["candidate_count"] <= 5
        assert document["sampling"]["max_candidates"] == 5
        assert any(
            "continuous poses" in limitation
            for limitation in document["feasibility_contract"]["not_checked"]
        )


def test_open_mesh_fails_closed_unless_explicitly_downgraded():
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        cad = temporary / "open.stl"
        _write_open_triangle(cad)
        arguments = {
            "units": "m",
            "scale_to_m": None,
            "gripper": _gripper(),
            "surface_samples": 12,
            "approaches_per_pair": 4,
            "max_candidates": None,
            "generated_root": temporary / "prepared",
        }
        try:
            generate_document(cad, **arguments)
        except ValueError as error:
            assert "closed, consistently wound two-manifold" in str(error)
            assert "boundary_edges=3" in str(error)
        else:
            raise AssertionError("open mesh should fail without explicit opt-in")

        document = generate_document(
            cad,
            **arguments,
            allow_unreliable_mesh=True,
        )
        assert document["claim_level"] == UNRELIABLE_MESH_CLAIM_LEVEL
        assert document["candidate_count"] == 0
        topology = document["cad"]["topology_audit"]
        assert topology["normal_ray_assumptions_accepted"] is False
        assert topology["boundary_edge_count"] == 3
        assert any(
            "unreliable mesh" in limitation
            for limitation in document["feasibility_contract"]["not_checked"]
        )


def test_obj_cad_uses_the_same_si_mesh_and_topology_pipeline():
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        cad = temporary / "box.obj"
        _write_box_obj(cad)
        document = generate_document(
            cad,
            units="m",
            scale_to_m=None,
            gripper=_gripper(),
            surface_samples=80,
            closing_directions_per_surface=1,
            approaches_per_pair=4,
            max_candidates=8,
            generated_root=temporary / "prepared",
        )
        assert document["cad"]["format"] == "obj"
        assert document["cad"]["topology_audit"][
            "closed_consistently_wound_two_manifold"
        ] is True
        assert 0 < document["candidate_count"] <= 8


def test_mixed_winding_across_disconnected_components_is_downgraded():
    outward = _box_triangles([-0.02, -0.01, -0.005],
                             [0.02, 0.01, 0.005])
    inward = _box_triangles([0.08, -0.01, -0.005],
                            [0.12, 0.01, 0.005])[:, [0, 2, 1]]
    mesh = TriangleMesh.from_triangles(np.concatenate((outward, inward)))
    topology = _mesh_topology_audit(mesh)
    assert topology["component_count"] == 2
    assert topology["boundary_edge_count"] == 0
    assert topology["mixed_component_winding_signs"] is True
    assert topology["normal_ray_assumptions_accepted"] is False


if __name__ == "__main__":
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(f"passed {len(tests)} direct CAD grasp-generation tests")
