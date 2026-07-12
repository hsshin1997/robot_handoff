"""Focused checks for alternate project/model/cache CLI threading."""
from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import xml.etree.ElementTree as ET

import mujoco
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mujoco_sim import pipeline  # noqa: E402
from mujoco_sim.project import DEFAULT_PROJECT  # noqa: E402
from mujoco_sim.sim import MODEL  # noqa: E402
from mujoco_sim import viewer, visualize_pipeline, visualize_reorientation_demo  # noqa: E402
from scripts import build_mujoco_scene, build_reachability  # noqa: E402


def test_all_entry_point_parsers_expose_selected_paths():
    build = build_mujoco_scene.build_parser().parse_args(
        ["--project", "p.yaml", "--output", "out.xml"])
    assert (build.project, build.output) == ("p.yaml", "out.xml")

    for parser in (
        pipeline.build_parser(),
        visualize_pipeline.build_parser(),
        visualize_reorientation_demo.build_parser(),
    ):
        args = parser.parse_args(
            ["--project", "p.yaml", "--model", "scene.xml", "--cache", "cache"])
        assert (args.project, args.model, args.cache) == ("p.yaml", "scene.xml", "cache")
    managed = viewer.build_parser().parse_args(
        ["--project", "p.yaml", "--model", "scene.xml"])
    assert (managed.project, managed.model) == ("p.yaml", "scene.xml")
    reachability = build_reachability.build_parser().parse_args(
        ["--project", "p.yaml", "--model", "scene.xml", "--out", "cache"])
    assert (reachability.project, reachability.model, reachability.out) == (
        "p.yaml", "scene.xml", "cache")


def test_help_checks_for_build_plan_and_visualization_entry_points():
    commands = (
        [sys.executable, str(ROOT / "scripts" / "build_mujoco_scene.py"), "--help"],
        [sys.executable, "-m", "mujoco_sim.pipeline", "--help"],
        [sys.executable, "-m", "mujoco_sim.visualize_pipeline", "--help"],
        [sys.executable, "-m", "mujoco_sim.visualize_reorientation_demo", "--help"],
        [sys.executable, "-m", "mujoco_sim.viewer", "--help"],
        [sys.executable, str(ROOT / "scripts" / "build_reachability.py"), "--help"],
    )
    for command in commands:
        completed = subprocess.run(
            command, cwd=ROOT, capture_output=True, text=True, check=False)
        assert completed.returncode == 0, (command, completed.stderr)
        assert "--project" in completed.stdout
        if any(str(part).endswith("build_mujoco_scene.py") for part in command):
            assert "--output" in completed.stdout
        elif any(str(part).endswith("build_reachability.py") for part in command):
            assert "--model" in completed.stdout
            assert "--out" in completed.stdout
        elif "mujoco_sim.viewer" not in command:
            assert "--model" in completed.stdout
            assert "--cache" in completed.stdout
        else:
            assert "--model" in completed.stdout


def test_alternate_project_and_output_build_a_self_contained_deterministic_scene():
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        manifest = yaml.safe_load(Path(DEFAULT_PROJECT).read_text(encoding="utf-8"))
        manifest["robots"]["A"]["initial_q"][0] = 0.123456789
        alternate_project = temporary / "alternate_project.yaml"
        alternate_project.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
        output = temporary / "compiled" / "alternate_scene.xml"

        built = Path(build_mujoco_scene.build_scene(alternate_project, output))
        assert built.resolve() == output.resolve()
        assert (output.parent / "generated_cad").is_dir()
        tree = ET.parse(output)
        qpos = [float(value) for value in tree.find("./keyframe/key").attrib["qpos"].split()]
        assert qpos[0] == 0.123456789  # proves the selected project was compiled

        for mesh in tree.findall("./asset/mesh"):
            referenced = (output.parent / mesh.attrib["file"]).resolve()
            assert referenced.is_file(), (mesh.attrib["name"], referenced)
        model = mujoco.MjModel.from_xml_path(str(output))
        assert model.nmesh == 28

        first_digest = hashlib.sha256(output.read_bytes()).hexdigest()
        build_mujoco_scene.build_scene(alternate_project, output)
        second_digest = hashlib.sha256(output.read_bytes()).hexdigest()
        assert first_digest == second_digest


def test_workstation_collision_mesh_routes_through_preprocessor_with_pose_and_scale():
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        manifest = yaml.safe_load(Path(DEFAULT_PROJECT).read_text(encoding="utf-8"))
        manifest["workstation"].update({
            "collision_cad": "parts/conn_header/conn_header_bin.stl",
            "collision_cad_units": "m",
            "collision_cad_world_pose": {
                "position_m": [0.1, -0.2, 0.3],
                "rpy_deg": [0.0, 0.0, 90.0],
            },
        })
        project = temporary / "mesh_collision_project.yaml"
        project.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
        output = temporary / "scene" / "mesh_collision.xml"
        build_mujoco_scene.build_scene(project, output)

        root = ET.parse(output).getroot()
        asset = root.find("./asset/mesh[@name='workstation_collision_mesh_00']")
        geom = root.find("./worldbody/geom[@name='workstation_collision_00']")
        assert asset is not None and geom is not None
        assert asset.attrib["scale"] == "1.0 1.0 1.0"
        assert (output.parent / asset.attrib["file"]).resolve().is_file()
        assert geom.attrib["class"] == "cell_collision"
        assert geom.attrib["pos"] == "0.1 -0.2 0.3"
        assert abs(float(geom.attrib["euler"].split()[2]) - 3.141592653589793 / 2) < 1e-15
        # Mesh routing replaces, rather than duplicates, the YAML box source.
        assert root.find("./worldbody/geom[@name='cell_collision_00']") is None
        mujoco.MjModel.from_xml_path(str(output))


def test_plan_and_execute_threads_paths_without_breaking_original_call_shape():
    constructed = {}

    class FakeSim:
        def __init__(self, model_path, project_path):
            constructed["sim"] = (model_path, project_path)

    class FakePlanner:
        def __init__(self, sim, known_start_pose=None, project_path=None, cache_dir=None):
            constructed["planner"] = (sim, known_start_pose, project_path, cache_dir)

        def plan(self, allow_regrasp, return_best):
            constructed["plan"] = (allow_regrasp, return_best)
            return SimpleNamespace(feasible=False)

    known = object()
    with patch.object(pipeline, "WorkcellSim", FakeSim), patch.object(
            pipeline, "HandoffPlanner", FakePlanner):
        report, result = pipeline.plan_and_execute(
            False, False, True, known,
            project_path="alternate.yaml", model_path="alternate.xml", cache_dir="cache-dir")
    assert constructed["sim"] == ("alternate.xml", "alternate.yaml")
    assert constructed["planner"][1:] == (known, "alternate.yaml", "cache-dir")
    assert constructed["plan"] == (False, True)
    assert report.feasible is False and result is None

    # Defaults remain the current API behavior for callers that pass nothing.
    with patch.object(pipeline, "WorkcellSim", FakeSim), patch.object(
            pipeline, "HandoffPlanner", FakePlanner):
        pipeline.plan_and_execute()
    assert constructed["sim"] == (MODEL, DEFAULT_PROJECT)


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS  {test.__name__}")
    print(f"\n{len(tests)}/{len(tests)} passed")
