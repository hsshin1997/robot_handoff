"""Acceptance tests for exact, general MuJoCo CAD preprocessing."""
from __future__ import annotations

import json
from pathlib import Path
import struct
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mujoco_sim.cad_preprocess import (  # noqa: E402
    CANONICAL_STL_HEADER,
    COLLISION_WARNING,
    CADPreprocessError,
    FreeCADUnavailableError,
    binary_stl_bytes,
    connected_components,
    detect_freecad,
    prepare_cad,
    read_stl,
    scale_to_metres,
    verify_preparation,
)
from mujoco_sim.offline import fingerprint_content, fingerprint_file  # noqa: E402
from scripts.prepare_project_cad import prepare_project  # noqa: E402


def _record(normal, first, second, third, attribute=0) -> bytes:
    return struct.pack("<12fH", *(tuple(normal) + tuple(first) + tuple(second) + tuple(third)),
                       attribute)


RECORDS = (
    _record((0, 0, 1), (0, 0, 0), (1, 0, 0), (0, 1, 0), 7),
    _record((0, 0, 1), (1, 0, 0), (1, 1, 0), (0, 1, 0), 11),
    _record((0, 0, 1), (10, 0, 0), (11, 0, 0), (10, 1, 0), 13),
    _record((1, 0, 0), (20, 0, 0), (20, 1, 0), (20, 0, 1), 17),
    _record((0, 1, 0), (30, 0, 0), (30, 0, 1), (31, 0, 0), 19),
)


def _write_binary(path: Path, records=RECORDS, header=b"arbitrary source header") -> None:
    path.write_bytes(header[:80].ljust(80, b"X") + struct.pack("<I", len(records)) + b"".join(records))


def _write_ascii(path: Path) -> None:
    path.write_text(
        """solid triangle
 facet normal 0 0 1
  outer loop
   vertex 0 0 0
   vertex 1 0 0
   vertex 0 1 0
  endloop
 endfacet
endsolid triangle
""",
        encoding="utf-8",
    )


def test_binary_and_ascii_stl_normalize_deterministically():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        binary = root / "binary.stl"
        ascii_path = root / "ascii.stl"
        _write_binary(binary, RECORDS[:1], header=b"solid misleading binary header")
        _write_ascii(ascii_path)

        binary_data = read_stl(binary)
        ascii_data = read_stl(ascii_path)
        assert binary_data.source_format == "binary-stl"
        assert ascii_data.source_format == "ascii-stl"
        # ASCII has no attribute-byte payload; geometry and normals are equal.
        assert binary_data.records[0][0:48] == ascii_data.records[0][0:48]
        assert binary_stl_bytes(binary_data.records).startswith(CANONICAL_STL_HEADER)
        assert binary_stl_bytes(binary_data.records) == binary_stl_bytes(binary_data.records)

        first = prepare_cad(binary, root / "generated", units="m", max_faces=2)
        repeated = prepare_cad(binary, root / "generated", units="m", max_faces=2)
        assert first.artifact_dir == repeated.artifact_dir
        assert first.metadata == repeated.metadata
        assert fingerprint_file(first.metadata_path) == fingerprint_file(repeated.metadata_path)


def test_exact_chunking_preserves_every_binary_record_and_explicit_scale():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "assembly.stl"
        _write_binary(source)
        prepared = prepare_cad(source, root / "generated", units="mm", max_faces=2)
        chunks = prepared.metadata["visual"]["chunks"]
        assert [chunk["face_count"] for chunk in chunks] == [2, 2, 1]
        assert all(chunk["face_count"] < 200_000 for chunk in chunks)
        recovered = []
        for chunk in chunks:
            path = prepared.artifact_dir / chunk["path"]
            assert path.read_bytes()[:80] == CANONICAL_STL_HEADER
            recovered.extend(read_stl(path).records)
        assert tuple(recovered) == RECORDS
        assert prepared.metadata["source"]["scale_to_m"] == [0.001, 0.001, 0.001]
        assert prepared.metadata["visual"]["downsampled"] is False
        assert prepared.metadata["visual"]["face_count"] == len(RECORDS)
        assert prepared.metadata["collision"]["representation"] == "not-generated"
        assert COLLISION_WARNING in prepared.metadata["warnings"]


def test_static_components_are_exact_visual_groups_not_collision_hulls():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "static.stl"
        _write_binary(source, RECORDS[:3])
        prepared = prepare_cad(
            source, root / "generated", units="m", max_faces=2, static_assembly=True,
        )
        static = prepared.metadata["static_assembly"]
        assert connected_components(RECORDS[:3]) == ((0, 1), (2,))
        assert static["component_count"] == 2
        assert [component["face_count"] for component in static["components"]] == [2, 1]
        assert static["collision_decomposition"] is False
        assert "not necessarily convex" in static["warning"]
        expected_components = (RECORDS[:2], RECORDS[2:3])
        for component, expected in zip(static["components"], expected_components):
            recovered = []
            for chunk in component["chunks"]:
                recovered.extend(read_stl(prepared.artifact_dir / chunk["path"]).records)
            assert tuple(recovered) == expected

        # Metadata is the final atomic publication marker and verifies every output.
        verified = verify_preparation(prepared.metadata_path)
        assert verified.metadata == prepared.metadata
        payload = dict(prepared.metadata)
        claimed = payload.pop("metadata_content_sha256")
        assert fingerprint_content(payload) == claimed
        assert not list(prepared.artifact_dir.rglob("*.tmp"))


def test_units_are_never_inferred_and_obj_is_an_exact_registered_copy():
    try:
        scale_to_metres()
    except ValueError as error:
        assert "ambiguous" in str(error)
    else:
        raise AssertionError("unitless STL/OBJ scale was accepted")

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "visual.obj"
        content = b"# material-preserving source\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"
        source.write_bytes(content)
        prepared = prepare_cad(source, root / "generated", scale_to_m=0.01)
        output = prepared.artifact_dir / prepared.metadata["visual"]["chunks"][0]["path"]
        assert output.read_bytes() == content
        assert prepared.metadata["visual"]["source_encoding_preserved"] is True
        assert prepared.metadata["source"]["scale_to_m"] == [0.01, 0.01, 0.01]


def test_step_failure_without_freecad_is_clear_and_actionable():
    try:
        detect_freecad("/definitely/not/a/FreeCADCmd")
    except FreeCADUnavailableError as error:
        message = str(error)
        assert "FreeCADCmd/freecadcmd" in message
        assert "FREECADCMD" in message
        assert "export" in message
    else:
        raise AssertionError("a nonexistent explicit FreeCAD executable was accepted")


def test_project_cli_core_prepares_all_referenced_cad_and_atomic_index():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        workcell = root / "workcell.stl"
        gripper = root / "gripper.obj"
        part = root / "part.stl"
        _write_binary(workcell, RECORDS[:3])
        gripper.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
        _write_ascii(part)
        manifest = {
            "schema_version": 1,
            "workstation": {
                "visual_cad": "workcell.stl", "visual_cad_units": "mm",
                "additional_collision_cad": [],
            },
            "grippers": {
                "jaw": {
                    "model": "gripper.obj", "model_units": "mm", "kind": "fixed",
                }
            },
            "parts": {"part": {"cad": "part.stl", "cad_units": "m"}},
            "robots": {},
        }
        project = root / "project.yaml"
        project.write_text(json.dumps(manifest), encoding="utf-8")
        generated = root / "generated"
        index = prepare_project(project, generated, project_root=root, max_faces=2)

        assert [entry["name"] for entry in index["entries"]] == [
            "grippers.jaw.model", "parts.part.cad", "workstation.visual_cad",
        ]
        assert index["exact_visual_preservation"] is True
        assert index["visual_downsampling"] is False
        assert index["collision_warning"] == COLLISION_WARNING
        stored = json.loads((generated / "index.json").read_text(encoding="utf-8"))
        assert stored == index
        payload = dict(index)
        claimed = payload.pop("index_content_sha256")
        assert fingerprint_content(payload) == claimed
        for entry in index["entries"]:
            metadata_path = generated / entry["metadata"]
            assert verify_preparation(metadata_path).metadata["artifact_fingerprint"] == entry[
                "artifact_fingerprint"
            ]


def test_project_units_error_names_the_offending_entry():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "part.stl"
        _write_ascii(source)
        project = root / "project.yaml"
        project.write_text(json.dumps({
            "schema_version": 1,
            "workstation": {"additional_collision_cad": []},
            "grippers": {}, "robots": {},
            "parts": {"unknown": {"cad": "part.stl"}},
        }), encoding="utf-8")
        try:
            prepare_project(project, root / "generated", project_root=root)
        except ValueError as error:
            assert "parts.unknown.cad" in str(error)
            assert "explicit units" in str(error)
        else:
            raise AssertionError("project CAD without explicit STL units was accepted")


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS  {test.__name__}")
    print(f"\n{len(tests)}/{len(tests)} passed")
