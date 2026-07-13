"""Acceptance tests for deterministic offline preprocessing and caching."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mujoco_sim.offline_tools.artifacts import (  # noqa: E402
    ARTIFACT_CATEGORIES,
    ArtifactCache,
    CacheCorruptionError,
    build_coverage_report,
    canonical_json_bytes,
    fingerprint_content,
    fingerprint_file,
    make_artifact_key,
)
from scripts.precompute_pipeline import precompute  # noqa: E402


def test_canonical_content_and_file_fingerprints():
    first = {"z": [3, 2, 1], "a": {"right": 2, "left": -0.0}}
    second = {"a": {"left": 0.0, "right": 2}, "z": [3, 2, 1]}
    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert fingerprint_content(first) == fingerprint_content(second)
    assert fingerprint_content({"x": [1, 2]}) != fingerprint_content({"x": [2, 1]})

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "asset.bin"
        path.write_bytes(b"same content regardless of file name or mtime")
        renamed = Path(directory) / "renamed.bin"
        renamed.write_bytes(path.read_bytes())
        os.utime(renamed, (1, 1))
        assert fingerprint_file(path) == fingerprint_file(renamed)


def test_artifact_keys_track_schema_versions_inputs_and_dependencies():
    mesh_v1 = make_artifact_key(
        "mesh", "connector", artifact_version="mesh-producer-1",
        input_fingerprints={"cad": "cad-a"}, parameters={"scale": 0.001},
    )
    mesh_v2 = make_artifact_key(
        "mesh", "connector", artifact_version="mesh-producer-2",
        input_fingerprints={"cad": "cad-a"}, parameters={"scale": 0.001},
    )
    grasp = make_artifact_key(
        "grasp", "connector-A", artifact_version=1,
        dependencies={"mesh": mesh_v1}, parameters={"samples": 800},
    )
    grasp_changed_dependency = make_artifact_key(
        "grasp", "connector-A", artifact_version=1,
        dependencies={"mesh": mesh_v2}, parameters={"samples": 800},
    )
    grasp_changed_schema = make_artifact_key(
        "grasp", "connector-A", artifact_version=1, schema_version=2,
        dependencies={"mesh": mesh_v1}, parameters={"samples": 800},
    )

    assert mesh_v1.digest != mesh_v2.digest
    assert grasp.digest != grasp_changed_dependency.digest
    assert grasp.digest != grasp_changed_schema.digest
    # Insertion order must not affect a key.
    reordered = make_artifact_key(
        "grasp", "order-check", artifact_version=1,
        input_fingerprints={"b": "2", "a": "1"},
        dependencies={"right": "r", "left": "l"},
    )
    ordered = make_artifact_key(
        "grasp", "order-check", artifact_version=1,
        input_fingerprints={"a": "1", "b": "2"},
        dependencies={"left": "l", "right": "r"},
    )
    assert reordered.digest == ordered.digest


def test_atomic_cache_get_or_compute_and_integrity_checks():
    with tempfile.TemporaryDirectory() as directory:
        cache = ArtifactCache(directory)
        key = make_artifact_key(
            "reachability", "robot-A", artifact_version="2",
            input_fingerprints={"robot": "gp7"}, parameters={"samples": 1000},
        )
        calls = []

        def compute():
            calls.append(True)
            return {"voxels": [3, 1, 2], "opaque": b"payload"}

        first = cache.get_or_compute(key, compute)
        second = cache.get_or_compute(key, lambda: (_ for _ in ()).throw(AssertionError("miss")))
        assert first == second == {"voxels": [3, 1, 2], "opaque": b"payload"}
        assert calls == [True]
        assert cache.contains(key)
        assert not list(Path(directory).rglob("*.tmp"))
        assert not list(Path(directory).rglob("*.lock"))

        path = cache.path_for(key)
        envelope = json.loads(path.read_text(encoding="utf-8"))
        envelope["value"]["voxels"][0] = 99
        path.write_text(json.dumps(envelope), encoding="utf-8")
        try:
            cache.get(key)
        except CacheCorruptionError:
            pass
        else:
            raise AssertionError("tampered cache content was accepted")


def test_coverage_report_is_complete_and_deterministic():
    required_a = {
        "coverage": [],
        "grasp": ["g2", "g1", "g1"],
        "mesh": ["m1"],
    }
    available_a = {
        "mesh": ["extra", "m1"],
        "grasp": ["g1"],
    }
    required_b = {"mesh": ["m1"], "grasp": ["g1", "g2"]}
    available_b = {"grasp": ["g1"], "mesh": ["m1", "extra"]}
    first = build_coverage_report(required_a, available_a, project_fingerprint="project")
    second = build_coverage_report(required_b, available_b, project_fingerprint="project")

    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert tuple(first["categories"]) == ARTIFACT_CATEGORIES
    assert not first["complete"]
    assert first["summary"] == {
        "required_count": 3, "covered_count": 2, "missing_count": 1,
        "fraction": 2 / 3,
    }
    assert first["categories"]["grasp"]["missing"] == ["g2"]
    assert first["categories"]["mesh"]["unexpected"] == ["extra"]
    assert first["categories"]["stable-pose"]["complete"]


def test_precompute_cli_core_snapshots_project_assets_and_runs_sorted_hooks():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "robot.urdf").write_text("<robot/>", encoding="utf-8")
        (root / "part.stl").write_bytes(b"solid part\nendsolid part\n")
        # JSON is valid YAML and keeps this test focused on preprocessing.
        manifest = {
            "schema_version": 1,
            "robots": {"A": {"model": "robot.urdf"}},
            "parts": {"part": {"cad": "part.stl"}},
        }
        project = root / "project.yaml"
        project.write_text(json.dumps(manifest), encoding="utf-8")
        cache_dir = root / "cache"
        order = []

        def z_hook(context):
            order.append("z")
            return {"project": context.metadata["project_fingerprint"]}

        def a_hook(context):
            order.append("a")
            return {"cache_root": context.cache.root.name}

        metadata = precompute(
            project, cache_dir, project_root=root,
            hooks=(("z-hook", z_hook), ("a-hook", a_hook)),
        )
        assert order == ["a", "z"]
        assert [item["field"] for item in metadata["assets"]] == [
            "parts/part/cad", "robots/A/model"
        ]
        assert list(metadata["hooks"]) == ["a-hook", "z-hook"]
        stored = json.loads((cache_dir / "project-metadata.json").read_text(encoding="utf-8"))
        assert stored == metadata

        # Formatting/key order do not affect the canonical project identity.
        project.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        repeated = precompute(project, cache_dir, project_root=root)
        assert repeated["project_fingerprint"] == metadata["project_fingerprint"]
        assert repeated["manifest_source_sha256"] != metadata["manifest_source_sha256"]


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS  {test.__name__}")
    print(f"\n{len(tests)}/{len(tests)} passed")
