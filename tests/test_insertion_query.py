"""Tests for runtime composition of connector-relative insertion grasps."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mujoco_sim.modeling.insertion_query import (  # noqa: E402
    attach_provisional_gp7_ik,
    bind_pcb_socket_contract,
    compose_insertion_pose_query,
    load_insertion_pose_library,
    normalize_correction_bounds,
    normalize_selection,
    normalize_world_frame,
    resolve_world_part_insert_pose,
)


CONNECTOR_SHA = "b" * 64


def _transform(translation=(0.0, 0.0, 0.0), rotation=None):
    value = np.eye(4)
    value[:3, :3] = np.eye(3) if rotation is None else rotation
    value[:3, 3] = translation
    return value


def _world_frame():
    return {
        "id": "synthetic_test_world",
        "calibration_fingerprint": "synthetic-calibration-v1",
        "calibration_source": "tests/test_insertion_query.py fixture",
    }


def _library_document():
    # +Z_I = -Y_P. Translation is omitted in this synthetic contract because it
    # does not affect insertion-axis consistency or E-target composition.
    T_I_P = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    return {
        "schema_version": 1,
        "project_id": "synthetic_connector",
        "config_sha256": "a" * 64,
        "asset_stats": {"part": {"sha256": CONNECTOR_SHA}},
        "task_geometry": {
            "insertion_axis_P": [0.0, -1.0, 0.0],
            "T_I_P": T_I_P.tolist(),
            "preinsert_distance_m": 0.04,
        },
        "candidates": [
            {
                "id": "g_preinsert_only",
                "library_index": 0,
                "status": "phase1_preinsert_only_candidate",
                "family": "close_z_approach_-y",
                "preinsert_compatible": True,
                "seated_compatible": False,
                "preinsert_task_rank": 1,
                "seated_task_rank": None,
                "required_aperture_m": 0.006,
                "quality": 0.7,
                "T_P_E": _transform([0.01, 0.02, 0.03]).tolist(),
            },
            {
                "id": "g_seated_best",
                "library_index": 1,
                "status": "phase1_seated_geometric_candidate",
                "family": "close_z_approach_+x",
                "preinsert_compatible": True,
                "seated_compatible": True,
                "preinsert_task_rank": 3,
                "seated_task_rank": 1,
                "required_aperture_m": 0.010,
                "quality": 0.9,
                "T_P_E": _transform([-0.01, 0.0, 0.02]).tolist(),
            },
            {
                "id": "g_seated_second",
                "library_index": 2,
                "status": "phase1_seated_geometric_candidate",
                "family": "close_y_approach_+z",
                "preinsert_compatible": True,
                "seated_compatible": True,
                "preinsert_task_rank": 2,
                "seated_task_rank": 2,
                "required_aperture_m": 0.008,
                "quality": 0.8,
                "T_P_E": _transform([0.02, -0.01, 0.01]).tolist(),
            },
            {
                "id": "g_rejected",
                "library_index": 3,
                "status": "rejected_contact_region",
                "family": "close_y_approach_-z",
                "preinsert_compatible": False,
                "seated_compatible": False,
                "preinsert_task_rank": None,
                "seated_task_rank": None,
                "required_aperture_m": 0.004,
                "quality": 0.2,
                "T_P_E": np.eye(4).tolist(),
            },
        ],
    }


def _write_library(directory: Path, document=None, name="pose_library.json") -> Path:
    path = directory / name
    value = _library_document() if document is None else document
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _socket_document():
    return {
        "schema_version": 1,
        "project_id": "synthetic_socket",
        "compatible_pose_library_project_ids": ["synthetic_connector"],
        "assets": {"connector": {"sha256": CONNECTOR_SHA}},
        "frames": {
            "B": {"insertion_direction": [0.0, -1.0, 0.0]},
            "P": {"short_tail_axis": [0.0, -1.0, 0.0]},
        },
        "T_B_P_insert": _transform([0.01, 0.0, -0.003]).tolist(),
    }


def _assert_value_error(function, text):
    try:
        function()
    except ValueError as error:
        assert text in str(error), str(error)
    else:
        raise AssertionError(f"expected ValueError containing {text!r}")


def test_default_insertion_requires_both_checks_and_uses_seated_rank():
    with tempfile.TemporaryDirectory() as directory:
        path = _write_library(Path(directory))
        library = load_insertion_pose_library(path)
        T_W_P = _transform([1.0, 2.0, 3.0])
        result = compose_insertion_pose_query(
            library,
            robot="b",
            T_W_P_insert=T_W_P,
            world_frame=_world_frame(),
            preinsert_distance_m=0.025,
        )
        expected_library_sha = hashlib.sha256(path.read_bytes()).hexdigest()

    assert result["robot"] == "B"
    assert result["claim_level"] == "composed_target"
    assert result["certified"] is False
    assert result["selection"] == {
        "purpose": "insertion",
        "preinsert_compatible": True,
        "seated_compatible": True,
    }
    assert result["eligible_candidate_count_before_limit"] == 2
    assert result["selected_candidate_count"] == 2
    assert result["selection_truncated"] is False
    assert result["continuous_complete"] is False
    # Seated rank, not pre-insert rank, controls insertion-purpose ordering.
    assert [item["grasp_id"] for item in result["targets"]] == [
        "g_seated_best", "g_seated_second",
    ]
    expected_pre = T_W_P.copy()
    # Native -Y is insertion, so pre-insert is +Y for identity T_W_P.
    expected_pre[1, 3] += 0.025
    assert np.allclose(result["T_W_P_preinsert"], expected_pre)
    seed = np.asarray(_library_document()["candidates"][1]["T_P_E"])
    assert np.allclose(result["targets"][0]["T_W_E_insert"], T_W_P @ seed)
    assert np.allclose(
        result["targets"][0]["T_W_E_preinsert"], expected_pre @ seed)
    assert result["pose_library"]["sha256"] == expected_library_sha
    assert result["pose_library"]["connector_sha256"] == CONNECTOR_SHA
    assert result["world_frame"] == _world_frame()


def test_preinsert_diagnostic_nulls_unsafe_insert_target_and_keeps_witness():
    with tempfile.TemporaryDirectory() as directory:
        library = load_insertion_pose_library(_write_library(Path(directory)))
        T_W_P = _transform([0.5, -0.2, 0.7])
        result = compose_insertion_pose_query(
            library,
            robot="A",
            T_W_P_insert=T_W_P,
            world_frame=_world_frame(),
            selection={"purpose": "preinsert_diagnostic"},
        )

    assert result["claim_level"] == "preinsert_diagnostic"
    # Diagnostic ordering follows pre-insert rank.
    assert [item["grasp_id"] for item in result["targets"]] == [
        "g_preinsert_only", "g_seated_second", "g_seated_best",
    ]
    pre_only = result["targets"][0]
    assert pre_only["claim_level"] == "preinsert_diagnostic"
    assert pre_only["T_W_E_insert"] is None
    seed = np.asarray(_library_document()["candidates"][0]["T_P_E"])
    assert np.allclose(
        pre_only["T_W_E_insert_nominal_witness"], T_W_P @ seed)
    for seated in result["targets"][1:]:
        assert seated["T_W_E_insert"] is not None
        assert seated["T_W_E_insert_nominal_witness"] is None


def test_board_socket_target_and_direct_target_are_mutually_exclusive():
    T_W_B = _transform([0.4, -0.2, 0.7])
    T_B_P = _transform([0.01, 0.02, -0.003])
    target, source = resolve_world_part_insert_pose(
        board_world_pose=T_W_B, T_B_P_insert=T_B_P)
    assert source == "board_world_pose_x_pcb_socket"
    assert np.allclose(target, T_W_B @ T_B_P)

    _assert_value_error(
        lambda: resolve_world_part_insert_pose(
            world_part_insert_pose=np.eye(4),
            board_world_pose=T_W_B,
            T_B_P_insert=T_B_P,
        ),
        "not both",
    )


def test_socket_contract_is_bound_by_project_asset_and_axis_semantics():
    with tempfile.TemporaryDirectory() as directory:
        library = load_insertion_pose_library(_write_library(Path(directory)))
        binding = bind_pcb_socket_contract(_socket_document(), library)
        assert binding.project_id == "synthetic_socket"
        assert np.allclose(binding.T_B_P_insert, _socket_document()["T_B_P_insert"])

        wrong_project = copy.deepcopy(_socket_document())
        wrong_project["compatible_pose_library_project_ids"] = ["other"]
        _assert_value_error(
            lambda: bind_pcb_socket_contract(wrong_project, library),
            "not compatible",
        )

        wrong_asset = copy.deepcopy(_socket_document())
        wrong_asset["assets"]["connector"]["sha256"] = "c" * 64
        _assert_value_error(
            lambda: bind_pcb_socket_contract(wrong_asset, library),
            "does not match",
        )

        wrong_mapping = copy.deepcopy(_socket_document())
        wrong_mapping["frames"]["B"]["insertion_direction"] = [0.0, 0.0, -1.0]
        _assert_value_error(
            lambda: bind_pcb_socket_contract(wrong_mapping, library),
            "does not map",
        )

        wrong_part_axis = copy.deepcopy(_socket_document())
        wrong_part_axis["frames"]["P"]["short_tail_axis"] = [1.0, 0.0, 0.0]
        _assert_value_error(
            lambda: bind_pcb_socket_contract(wrong_part_axis, library),
            "short_tail_axis",
        )


def test_selection_limit_reports_eligible_count_and_truncation():
    with tempfile.TemporaryDirectory() as directory:
        library = load_insertion_pose_library(_write_library(Path(directory)))
        result = compose_insertion_pose_query(
            library,
            robot="A",
            T_W_P_insert=np.eye(4),
            world_frame=_world_frame(),
            selection={
                "purpose": "preinsert_diagnostic",
                "max_candidates": 2,
            },
        )
    assert result["eligible_candidate_count_before_limit"] == 3
    assert result["selected_candidate_count"] == 2
    assert result["selection_truncated"] is True
    assert result["continuous_complete"] is False


def test_correction_bounds_are_insertion_frame_yaw_only_with_exact_order():
    with tempfile.TemporaryDirectory() as directory:
        library = load_insertion_pose_library(_write_library(Path(directory)))
        result = compose_insertion_pose_query(
            library,
            robot="A",
            T_W_P_insert=np.eye(4),
            world_frame=_world_frame(),
            correction_bounds={
                "lateral_m": 0.002,
                "axial_m": [-0.001, 0.0005],
                "yaw_deg": 3.0,
            },
        )
    assert result["correction_bounds_I"] == {
        "lateral_x_m": [-0.002, 0.002],
        "lateral_y_m": [-0.002, 0.002],
        "axial_m": [-0.001, 0.0005],
        "yaw_deg": [-3.0, 3.0],
    }
    contract = result["correction_contract"]
    assert contract["delta_definition"] == (
        "Delta_I = Trans_I(dx,dy,dz) @ RotZ_I(yaw)")
    assert contract["composition"] == (
        "T_W_E(delta) = T_W_I @ Delta_I @ T_I_P @ T_P_E")
    assert contract["side"] == (
        "Delta_I left-multiplies nominal T_I_P inside frame I")
    assert contract["rotation_axis"] == "+Z_I only"
    assert contract["rotation_pivot"] == "origin of insertion frame I"
    assert contract["point_operation_order"].startswith("yaw")
    _assert_value_error(
        lambda: normalize_correction_bounds({"roll_deg": 1.0}), "unknown")
    _assert_value_error(
        lambda: normalize_correction_bounds({"pitch_deg": 1.0}), "unknown")


def test_library_boolean_compatibility_and_rank_invariants_are_strict():
    invalid_documents = []

    string_boolean = _library_document()
    string_boolean["candidates"][0]["seated_compatible"] = "false"
    invalid_documents.append((string_boolean, "must be boolean"))

    seated_without_preinsert = _library_document()
    candidate = seated_without_preinsert["candidates"][1]
    candidate["preinsert_compatible"] = False
    candidate["preinsert_task_rank"] = None
    invalid_documents.append((seated_without_preinsert, "must also be"))

    rank_while_incompatible = _library_document()
    rank_while_incompatible["candidates"][3]["preinsert_task_rank"] = 4
    invalid_documents.append((rank_while_incompatible, "present iff"))

    missing_rank = _library_document()
    missing_rank["candidates"][0]["preinsert_task_rank"] = None
    invalid_documents.append((missing_rank, "present iff"))

    boolean_rank = _library_document()
    boolean_rank["candidates"][0]["preinsert_task_rank"] = True
    invalid_documents.append((boolean_rank, "positive integer"))

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for index, (document, expected) in enumerate(invalid_documents):
            path = _write_library(root, document, f"invalid_{index}.json")
            _assert_value_error(
                lambda path=path: load_insertion_pose_library(path), expected)


def test_selection_world_frame_and_acknowledgment_types_are_strict():
    _assert_value_error(
        lambda: normalize_selection({"preinsert_compatible": "false"}),
        "must be boolean",
    )
    _assert_value_error(
        lambda: normalize_selection({"max_candidates": True}),
        "positive integer",
    )
    _assert_value_error(
        lambda: normalize_world_frame({"id": "world"}), "requires")
    _assert_value_error(
        lambda: normalize_world_frame({
            "id": "world",
            "calibration_fingerprint": "",
            "calibration_source": "fixture",
        }),
        "non-empty",
    )

    with tempfile.TemporaryDirectory() as directory:
        library = load_insertion_pose_library(_write_library(Path(directory)))
        _assert_value_error(
            lambda: compose_insertion_pose_query(
                library,
                robot="A",
                T_W_P_insert=np.eye(4),
                world_frame={},
            ),
            "world_frame requires",
        )
        query = compose_insertion_pose_query(
            library,
            robot="A",
            T_W_P_insert=np.eye(4),
            world_frame=_world_frame(),
        )
        _assert_value_error(
            lambda: attach_provisional_gp7_ik(
                query,
                _FakeKinematics(),
                acknowledge_provisional_tcp="false",
            ),
            "must be boolean",
        )


class _FakeKinematics:
    def __init__(self):
        self.q = {"A": np.zeros(6), "B": np.zeros(6)}
        self.solution_calls = []

    def get_q(self, robot):
        return self.q[robot].copy()

    def set_q(self, robot, q):
        self.q[robot] = np.asarray(q, dtype=float).copy()

    def solutions(self, robot, target, **kwargs):
        target = np.asarray(target)
        self.solution_calls.append(target.copy())
        # Make only positive-world-X targets reachable, while exercising the
        # exact result protocol used by GP7Kinematics.
        if target[0, 3] <= 0.0:
            return []
        return [SimpleNamespace(
            q=np.full(6, target[0, 3]),
            position_error=1e-5,
            rotation_error=2e-5,
            iterations=4,
        )]

    def normalized_limit_margin(self, robot, q):
        return 0.75

    def manipulability(self, robot, q):
        return 0.12


def test_optional_insertion_ik_requires_ack_and_never_claims_path_or_collision():
    with tempfile.TemporaryDirectory() as directory:
        library = load_insertion_pose_library(_write_library(Path(directory)))
        query = compose_insertion_pose_query(
            library,
            robot="B",
            T_W_P_insert=_transform([1.0, 0.0, 0.0]),
            world_frame=_world_frame(),
        )
    _assert_value_error(
        lambda: attach_provisional_gp7_ik(
            query, _FakeKinematics(), acknowledge_provisional_tcp=False),
        "acknowledgment",
    )

    kinematics = _FakeKinematics()
    evaluated = attach_provisional_gp7_ik(
        query,
        kinematics,
        acknowledge_provisional_tcp=True,
        random_restarts=0,
        max_solutions=2,
    )
    assert len(kinematics.solution_calls) == 4
    assert evaluated["claim_level"] == "ik_only_provisional_tcp"
    assert evaluated["certified"] is False
    assert evaluated["ik_evaluation"]["performed"] is True
    assert evaluated["ik_evaluation"]["collision_checked"] is False
    assert evaluated["ik_evaluation"]["path_checked"] is False
    assert evaluated["ik_reachable_candidate_count"] == 2
    for target in evaluated["targets"]:
        assert target["claim_level"] == "ik_reachable_endpoints"
        assert target["ik"]["both_endpoints_reachable"] is True
        assert target["ik"]["same_branch_continuity_checked"] is False


def test_diagnostic_ik_skips_null_insert_target():
    with tempfile.TemporaryDirectory() as directory:
        library = load_insertion_pose_library(_write_library(Path(directory)))
        query = compose_insertion_pose_query(
            library,
            robot="B",
            T_W_P_insert=_transform([1.0, 0.0, 0.0]),
            world_frame=_world_frame(),
            selection={"purpose": "preinsert_diagnostic"},
        )
    kinematics = _FakeKinematics()
    evaluated = attach_provisional_gp7_ik(
        query,
        kinematics,
        acknowledge_provisional_tcp=True,
        random_restarts=0,
        max_solutions=2,
    )
    # Three pre-insert solves plus seated solves for only the two seated seeds.
    assert len(kinematics.solution_calls) == 5
    assert evaluated["ik_reachable_candidate_count"] == 3
    assert evaluated["ik_reachable_preinsert_candidate_count"] == 3
    assert evaluated["ik_reachable_insert_candidate_count"] == 2
    assert evaluated["ik_evaluation"]["diagnostic_insert_targets_skipped"] == 1
    pre_only = evaluated["targets"][0]
    assert pre_only["T_W_E_insert"] is None
    assert pre_only["claim_level"] == "ik_reachable_preinsert_diagnostic"
    assert pre_only["ik"]["insert_solutions"] is None
    assert "no executable" in pre_only["ik"]["insert_skipped_reason"]
    assert pre_only["ik"]["both_endpoints_reachable"] is False


def test_invalid_duplicate_ids_and_unknown_selection_keys_fail():
    document = _library_document()
    document["candidates"][1]["id"] = document["candidates"][0]["id"]
    with tempfile.TemporaryDirectory() as directory:
        path = _write_library(Path(directory), document, "duplicate.json")
        _assert_value_error(
            lambda: load_insertion_pose_library(path), "duplicate")

    _assert_value_error(
        lambda: normalize_selection({"unknown": True}), "unknown")


def _run_cli(config: Path, output: Path):
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "query_insertion_poses.py"),
            "--config", str(config),
            "--output", str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
    )


def _query_document(library: Path, socket: Path):
    return {
        "schema_version": 1,
        "pose_library": library.name,
        "robot": "A",
        "world_frame": _world_frame(),
        "board_world_pose": _transform([0.5, -0.4, 0.8]).tolist(),
        "pcb_socket": socket.name,
        "selection": {
            "purpose": "preinsert_diagnostic",
            "max_candidates": 1,
        },
        "solve_ik": False,
        "tcp_assumption": {"acknowledge_provisional": False},
    }


def test_cli_writes_bound_diagnostic_targets_without_loading_mujoco_scene():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        library = _write_library(root)
        socket = root / "socket.yaml"
        socket.write_text(
            yaml.safe_dump(_socket_document(), sort_keys=False), encoding="utf-8")
        output = root / "result.json"
        config = root / "query.yaml"
        config.write_text(
            yaml.safe_dump(_query_document(library, socket), sort_keys=False),
            encoding="utf-8",
        )
        completed = _run_cli(config, output)
        assert completed.returncode == 0, completed.stderr
        result = json.loads(output.read_text(encoding="utf-8"))
    assert result["target_source"] == "board_world_pose_x_pcb_socket"
    assert result["query"]["pcb_socket_project_id"] == "synthetic_socket"
    assert result["query"]["pcb_socket_connector_sha256"] == CONNECTOR_SHA
    assert result["pose_library"]["connector_sha256"] == CONNECTOR_SHA
    assert result["eligible_candidate_count_before_limit"] == 3
    assert result["selected_candidate_count"] == 1
    assert result["selection_truncated"] is True
    assert result["targets"][0]["T_W_E_insert"] is None
    assert result["ik_evaluation"]["performed"] is False
    assert result["excluded_checks"]


def test_cli_rejects_string_booleans_for_solve_ack_and_selection():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        library = _write_library(root)
        socket = root / "socket.yaml"
        socket.write_text(
            yaml.safe_dump(_socket_document(), sort_keys=False), encoding="utf-8")
        output = root / "result.json"

        cases = []
        solve_string = _query_document(library, socket)
        solve_string["solve_ik"] = "false"
        cases.append((solve_string, "solve_ik must be boolean"))

        ack_string = _query_document(library, socket)
        ack_string["tcp_assumption"]["acknowledge_provisional"] = "false"
        cases.append((ack_string, "acknowledge_provisional must be boolean"))

        selection_string = _query_document(library, socket)
        selection_string["selection"]["preinsert_compatible"] = "true"
        cases.append((selection_string, "preinsert_compatible must be boolean"))

        for index, (document, expected) in enumerate(cases):
            config = root / f"invalid_query_{index}.yaml"
            config.write_text(
                yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
            completed = _run_cli(config, output)
            assert completed.returncode != 0
            assert expected in completed.stderr, completed.stderr


def test_repository_socket_contract_matches_generated_library():
    library_path = (
        ROOT / "projects" / "connector_header_insertion" / "generated"
        / "grasps" / "phase1_pose_library.json"
    )
    socket_path = (
        ROOT / "projects" / "connector_header_insertion" / "config"
        / "pcb_socket.yaml"
    )
    library = load_insertion_pose_library(library_path)
    socket = yaml.safe_load(socket_path.read_text(encoding="utf-8"))
    binding = bind_pcb_socket_contract(socket, library)
    assert binding.connector_sha256 == library.connector_sha256
    assert library.project_id in binding.compatible_library_project_ids
    assert np.allclose(
        binding.T_B_P_insert[:3, :3] @ library.insertion_axis_P,
        binding.insertion_direction_B,
    )


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS  {test.__name__}")
    print(f"\n{len(tests)}/{len(tests)} passed")
