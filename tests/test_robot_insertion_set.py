"""Behavioral and adversarial tests for the robot insertion-set layer."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mujoco_sim.modeling.insertion_task_set import (  # noqa: E402
    artifact_sha256,
    certificate_binding_sha256,
)
from mujoco_sim.planner.robot_insertion_set import (  # noqa: E402
    _CERTIFICATE_HARD_GATES,
    _normalized_parameter_distance,
    build_robot_insertion_set,
    load_insertion_task_set,
    load_verified_continuous_robot_certificate,
    resolve_world_part_insert,
    sample_straight_insertion_path,
    select_task_cells_stratified,
)


def _transform(translation=(0.0, 0.0, 0.0)):
    value = np.eye(4)
    value[:3, 3] = translation
    return value


def _mode(mode_id):
    if mode_id == "housing_opposed_y":
        closing = np.array([0.0, 1.0, 0.0])
        u_axis = np.array([1.0, 0.0, 0.0])
        v_axis = np.array([0.0, 0.0, 1.0])
        zero = np.array([1.0, 0.0, 0.0])
        midplane, aperture = 0.01, 0.008
    else:
        closing = np.array([0.0, 0.0, 1.0])
        u_axis = np.array([1.0, 0.0, 0.0])
        v_axis = np.array([0.0, 1.0, 0.0])
        zero = np.array([1.0, 0.0, 0.0])
        midplane, aperture = 0.02, 0.006
    return {
        "id": mode_id,
        "description": "synthetic mode",
        "u_bounds_P_m": [0.0, 0.04],
        "v_bounds_P_m": [0.0, 0.02],
        "possible_aperture_range_m": [aperture, aperture],
        "cell_counts": {"u": 4, "v": 2, "roll": 4},
        "constructive_map": {
            "position_u_axis_P": u_axis.tolist(),
            "position_v_axis_P": v_axis.tolist(),
            "closing_axis_P": closing.tolist(),
            "contact_midplane_coordinate_P_m": midplane,
            "roll_zero_approach_axis_P": zero.tolist(),
            "positive_roll_quadrature_axis_P": np.cross(
                closing, zero).tolist(),
            "positive_roll_rule": "right_hand_about_closing_axis_P",
            "aperture_model": {
                "type": "constant",
                "value_m": aperture,
                "status": "synthetic",
            },
            "formula": "synthetic constructive map",
        },
    }


def _center_pose(mode, bounds):
    constructive = mode["constructive_map"]
    theta = {name: 0.5 * (limits[0] + limits[1])
             for name, limits in bounds.items()}
    closing = np.asarray(constructive["closing_axis_P"])
    zero = np.asarray(constructive["roll_zero_approach_axis_P"])
    quadrature = np.asarray(
        constructive["positive_roll_quadrature_axis_P"])
    approach = (np.cos(theta["roll_rad"]) * zero
                + np.sin(theta["roll_rad"]) * quadrature)
    rotation = np.column_stack((np.cross(closing, approach), closing, approach))
    position = (
        theta["u_P_m"] * np.asarray(constructive["position_u_axis_P"])
        + theta["v_P_m"] * np.asarray(constructive["position_v_axis_P"])
        + constructive["contact_midplane_coordinate_P_m"] * closing
    )
    T_P_E = np.eye(4)
    T_P_E[:3, :3] = rotation
    T_P_E[:3, 3] = position
    return {
        "theta": theta,
        "T_P_E": T_P_E.tolist(),
        "required_aperture_m": constructive["aperture_model"]["value_m"],
        "source": "contact_mode_constructive_map",
        "constructive_map_version": 1,
    }


def _cell(mode, u_index, v_index, roll_index, classification="UNRESOLVED"):
    u_step, v_step, r_step = 0.01, 0.01, np.pi / 2.0
    bounds = {
        "u_P_m": [u_index * u_step, (u_index + 1) * u_step],
        "v_P_m": [v_index * v_step, (v_index + 1) * v_step],
        "roll_rad": [-np.pi + roll_index * r_step,
                     -np.pi + (roll_index + 1) * r_step],
    }
    cell_id = f"{mode['id']}_u{u_index}_v{v_index}_r{roll_index}"
    return {
        "id": cell_id,
        "contact_mode": mode["id"],
        "grid_index": {"u": u_index, "v": v_index, "roll": roll_index},
        "bounds": bounds,
        "classification": classification,
        "classification_reason": "synthetic",
        "center_pose": _center_pose(mode, bounds),
        # Deliberately unrelated sampled witness: layer 2 must never evaluate it.
        "representative": {
            "seed_grasp_id": f"sample_{cell_id}",
            "T_P_E": _transform([1.5, 1.5, 1.5]).tolist(),
            "required_aperture_m": 0.001,
            "quality": -1.0,
        },
        "seed_witness_count": 1,
        "witnesses": [],
    }


def _seal_task_document(document):
    safe_ids = [cell["id"] for cell in document["cells"]
                if cell["classification"] == "SAFE"]
    rejected_ids = [cell["id"] for cell in document["cells"]
                    if cell["classification"] == "REJECTED"]
    unresolved_ids = [cell["id"] for cell in document["cells"]
                      if cell["classification"] == "UNRESOLVED"]
    document["safe_inner_cell_ids"] = safe_ids
    document["rejected_cell_ids"] = rejected_ids
    document["unresolved_cell_ids"] = unresolved_ids
    document["counts"] = {
        "cells": len(document["cells"]),
        "safe": len(safe_ids),
        "rejected": len(rejected_ids),
        "unresolved": len(unresolved_ids),
    }
    document.pop("whole_cell_task_certificates", None)
    base_binding = certificate_binding_sha256(document)
    for cell in document["cells"]:
        if cell["classification"] == "SAFE":
            cell["whole_cell_task_certificate"] = {
                "certificate_id": "synthetic_task_proof",
                "path": "synthetic-task-proof.json",
                "sha256": "d" * 64,
                "proved_constraints": ["synthetic_all_task_constraints"],
                "base_artifact_certificate_binding_sha256": base_binding,
            }
    document["whole_cell_task_certificates"] = {
        "verification_policy": "fail_closed_exact_file_and_artifact_binding",
        "base_artifact_certificate_binding_sha256": base_binding,
        "expected_bindings": {
            "base_artifact_certificate_binding_sha256": base_binding,
            "project_id": document["project_id"],
            "connector_sha256": document["task_identity"]["connector_sha256"],
        },
        "required_proved_constraints": ["synthetic_all_task_constraints"],
        "imports": ([{
            "certificate_id": "synthetic_task_proof",
            "path": "synthetic-task-proof.json",
            "sha256": "d" * 64,
            "promoted_cell_ids": safe_ids,
        }] if safe_ids else []),
        "promoted_safe_cell_count": len(safe_ids),
    }
    document["semantic_sha256"] = artifact_sha256(document)
    return document


def _document(*, dense=False):
    modes = [_mode("housing_opposed_y"), _mode("housing_opposed_z")]
    if dense:
        cells = [_cell(mode, u, v, roll)
                 for mode in modes
                 for u in range(4) for v in range(2) for roll in range(4)]
    else:
        cells = [
            _cell(modes[0], 1, 0, 1, "SAFE"),
            _cell(modes[1], 2, 1, 2, "UNRESOLVED"),
            _cell(modes[0], 0, 0, 0, "REJECTED"),
        ]
    document = {
        "schema_version": 1,
        "artifact_type": "robot_independent_insertion_task_set",
        "project_id": "synthetic_task_set",
        "task_identity": {
            "connector_sha256": "a" * 64,
            "pcb_sha256": "b" * 64,
        },
        "inputs": {},
        "parameterization": {
            "variables": [],
            "periodic_variables": ["roll_rad"],
            "contact_modes": modes,
        },
        "insertion_trajectory": {
            "type": "straight_fixed_orientation",
            "insertion_axis_P": [0.0, -1.0, 0.0],
            "preinsert_distance_m": 0.04,
            "T_B_P_insert": _transform([0.02, -0.01, -0.003]).tolist(),
        },
        "cells": cells,
    }
    return _seal_task_document(document)


def _write_json(directory, document, name="artifact.json"):
    path = directory / name
    path.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
    return path


@dataclass
class FakeResult:
    q: np.ndarray
    position_error: float = 1e-8
    rotation_error: float = 2e-8
    iterations: int = 3


class FakeKinematics:
    def __init__(self):
        self.lower = {"A": np.full(6, -2.0), "B": np.full(6, -2.0)}
        self.upper = {"A": np.full(6, 2.0), "B": np.full(6, 2.0)}
        self._q = {"A": np.zeros(6), "B": np.zeros(6)}

    def get_q(self, robot):
        return self._q[robot].copy()

    def set_q(self, robot, q):
        self._q[robot] = np.asarray(q, dtype=float).copy()

    def solutions(self, robot, target, **kwargs):
        y = float(np.asarray(target)[1, 3])
        return [FakeResult(np.array([sign * 0.4, y, 0.0, 0.0, 0.0, 0.0]))
                for sign in (1.0, -1.0)]

    def solve(self, robot, target, seed=None, **kwargs):
        q = np.asarray(seed, dtype=float).copy()
        q[1] = float(np.asarray(target)[1, 3])
        self._q[robot] = q.copy()
        return FakeResult(q)

    def normalized_limit_margin(self, robot, q):
        return float(np.min(1.0 - np.abs(np.asarray(q)) / 2.0))

    def singular_values(self, robot, q):
        return np.array([1.1, 1.0, 0.9, 0.8, 0.5, 0.2])


class FailingContinuationKinematics(FakeKinematics):
    def solve(self, robot, target, seed=None, **kwargs):
        return None


class SingularKinematics(FakeKinematics):
    def singular_values(self, robot, q):
        return np.array([1.1, 1.0, 0.9, 0.8, 0.5, 1e-8])


def _world_frame():
    return {
        "id": "synthetic_world",
        "calibration_fingerprint": "synthetic-calibration",
        "calibration_source": "tests/test_robot_insertion_set.py",
    }


def _build(task, kin, **overrides):
    arguments = dict(
        robot="B",
        T_W_P_insert=_transform([0.5, 0.1, 0.4]),
        world_frame=_world_frame(),
        target_source="unit_test",
        source_classifications=["SAFE", "UNRESOLVED"],
        path_sample_count=5,
        random_restarts=0,
        max_solutions=4,
        max_joint_step_rad=0.1,
        minimum_joint_limit_margin_rad=0.01,
        minimum_normalized_joint_limit_margin=0.01,
        minimum_sigma=0.1,
        tcp_calibrated=False,
        acknowledge_provisional_tcp=True,
    )
    arguments.update(overrides)
    return build_robot_insertion_set(task, kin, **arguments)


def _assert_value_error(function, text):
    try:
        function()
    except ValueError as error:
        assert text in str(error), str(error)
    else:
        raise AssertionError(f"expected ValueError containing {text!r}")


def test_strict_loader_uses_constructive_center_not_sampled_representative():
    with tempfile.TemporaryDirectory() as directory:
        task = load_insertion_task_set(
            _write_json(Path(directory), _document()))
    safe = next(cell for cell in task.cells if cell.source_classification == "SAFE")
    assert np.allclose(safe.T_P_E, safe.center_pose["T_P_E"])
    assert not np.allclose(
        safe.T_P_E, safe.sampled_representative["T_P_E"])
    assert task.semantic_sha256
    assert task.task_certificate_binding_sha256


def test_loader_rejects_semantic_tampering_and_partition_forgery():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        tampered = _document()
        tampered["cells"][0]["bounds"]["u_P_m"][0] += 0.001
        _assert_value_error(
            lambda: load_insertion_task_set(_write_json(root, tampered, "tampered.json")),
            "semantic_sha256",
        )
        forged = _document()
        forged["safe_inner_cell_ids"] = []
        forged["semantic_sha256"] = artifact_sha256(forged)
        _assert_value_error(
            lambda: load_insertion_task_set(_write_json(root, forged, "forged.json")),
            "partition",
        )


def test_loader_rejects_center_pose_that_disagrees_with_constructive_map():
    forged = _document()
    forged["cells"][0]["center_pose"]["T_P_E"][0][3] += 0.002
    # Refresh all outer digests so the semantic checks pass and the geometric
    # consistency check is the mechanism that rejects this adversarial input.
    forged = _seal_task_document(forged)
    with tempfile.TemporaryDirectory() as directory:
        _assert_value_error(
            lambda: load_insertion_task_set(
                _write_json(Path(directory), forged)),
            "constructive map",
        )


def test_stratified_limit_is_mode_balanced_space_filling_and_order_invariant():
    first = _document(dense=True)
    second = _document(dense=True)
    second["cells"].reverse()
    second = _seal_task_document(second)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        task_a = load_insertion_task_set(_write_json(root, first, "a.json"))
        task_b = load_insertion_task_set(_write_json(root, second, "b.json"))
    chosen_a, evidence_a, skipped_a = select_task_cells_stratified(
        task_a, source_classifications=["UNRESOLVED"], max_cells=8)
    chosen_b, evidence_b, skipped_b = select_task_cells_stratified(
        task_b, source_classifications=["UNRESOLVED"], max_cells=8)
    assert [cell.cell_id for cell in chosen_a] == [cell.cell_id for cell in chosen_b]
    assert evidence_a["quota_by_contact_mode"] == {
        "housing_opposed_y": 4, "housing_opposed_z": 4}
    assert evidence_a["strategy"] == "contact_mode_balanced_maximin_center_v1"
    assert evidence_a["distance_metric"] == (
        "euclidean_linear_u_v_circular_normalized_roll_v1")
    assert evidence_a["selected"] == evidence_b["selected"]
    assert skipped_a == skipped_b and len(skipped_a) == 56
    for mode in evidence_a["quota_by_contact_mode"]:
        points = np.asarray([
            item["normalized_center"] for item in evidence_a["selected"]
            if item["contact_mode"] == mode
        ])
        assert np.all(np.ptp(points, axis=0) >= 0.5), points


def test_normalized_roll_distance_is_circular_across_wrap_boundary():
    near_negative_pi = [0.25, 0.75, 0.01]
    near_positive_pi = [0.25, 0.75, 0.99]
    opposite_roll = [0.25, 0.75, 0.51]
    assert np.isclose(
        _normalized_parameter_distance(near_negative_pi, near_positive_pi),
        0.02,
    )
    assert np.isclose(
        _normalized_parameter_distance(near_negative_pi, opposite_roll),
        0.5,
    )


def test_center_path_witness_is_explicitly_provisional_and_restores_state():
    with tempfile.TemporaryDirectory() as directory:
        task = load_insertion_task_set(
            _write_json(Path(directory), _document()))
    kin = FakeKinematics()
    original_q = kin.get_q("B")
    result = _build(task, kin)
    assert result["certified"] is False
    assert result["certified_receiver_cell_ids"] == []
    assert len(result["provisional_center_path_witness_cell_ids"]) == 2
    assert result["numerically_unresolved_cell_ids"] == []
    assert "rejected_cell_ids" not in result
    for record in result["cells"]:
        assert record["robot_classification"] == (
            "PROVISIONAL_CENTER_PATH_WITNESS")
        assert record["evaluated_pose_source"] == "cell.center_pose"
        assert record["center_pose_only"] is True
        assert not np.allclose(
            record["center_pose"]["T_P_E"],
            record["sampled_representative"]["T_P_E"],
        )
    assert np.allclose(kin.get_q("B"), original_q)


def test_numerical_failures_remain_unresolved_not_rejected():
    with tempfile.TemporaryDirectory() as directory:
        task = load_insertion_task_set(
            _write_json(Path(directory), _document()))
    failed = _build(task, FailingContinuationKinematics())
    assert failed["provisional_center_path_witness_cell_ids"] == []
    assert len(failed["numerically_unresolved_cell_ids"]) == 2
    assert {record["robot_classification"] for record in failed["cells"]} == {
        "NO_WITNESS_CONTINUATION"}
    singular = _build(task, SingularKinematics())
    assert {record["robot_classification"] for record in singular["cells"]} == {
        "CENTER_PATH_NUMERIC_MARGIN_NOT_MET"}


def _certificate_document(task, target, execution_bindings):
    safe_id = task.safe_inner_cell_ids[0]
    document = {
        "schema_version": 1,
        "artifact_type": "continuous_robot_insertion_cell_certificate",
        "bindings": {
            "task_artifact_file_sha256": task.sha256,
            "task_artifact_semantic_sha256": task.semantic_sha256,
            "task_certificate_binding_sha256": (
                task.task_certificate_binding_sha256),
            "robot": "B",
            "world_frame": _world_frame(),
            "T_W_P_insert": target.tolist(),
            "tcp_calibration_fingerprint": "tcp-calibrated-v1",
            "project_sha256": execution_bindings["project_sha256"],
            "model_sha256": execution_bindings["model_sha256"],
        },
        "hard_gates": {key: True for key in _CERTIFICATE_HARD_GATES},
        "certified_cells": [{
            "cell_id": safe_id,
            "classification": "CERTIFIED_SAFE",
            "proof_sha256": "c" * 64,
            "minimum_joint_limit_margin_rad": 0.2,
            "minimum_normalized_joint_limit_margin": 0.3,
            "minimum_sigma": 0.1,
        }],
    }
    document["semantic_sha256"] = artifact_sha256(document)
    return document


def test_bound_external_continuous_certificate_enables_certified_cell():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        task = load_insertion_task_set(_write_json(root, _document(), "task.json"))
        target = _transform([0.5, 0.1, 0.4])
        execution = {"project_sha256": "1" * 64, "model_sha256": "2" * 64}
        certificate_path = _write_json(
            root, _certificate_document(task, target, execution), "certificate.json")
        expected = hashlib.sha256(certificate_path.read_bytes()).hexdigest()
        certificate = load_verified_continuous_robot_certificate(
            certificate_path,
            expected_file_sha256=expected,
            task_set=task,
            robot="B",
            T_W_P_insert=target,
            world_frame=_world_frame(),
            tcp_calibration_fingerprint="tcp-calibrated-v1",
            execution_bindings=execution,
        )
        result = _build(
            task,
            FakeKinematics(),
            tcp_calibrated=True,
            tcp_calibration_fingerprint="tcp-calibrated-v1",
            acknowledge_provisional_tcp=False,
            continuous_certificate=certificate,
        )
    assert result["certified"] is True
    assert result["certified_receiver_cell_ids"] == [task.safe_inner_cell_ids[0]]
    certified = next(record for record in result["cells"]
                     if record["robot_classification"] == "CERTIFIED_SAFE")
    assert certified["whole_parameter_cell_evaluated"] is True
    assert certified["external_continuous_certificate"]["proof_sha256"] == "c" * 64
    assert certified["certification"] == {
        "certified": True,
        "all_hard_gates_passed": True,
        "source": "external_continuous_robot_cell_certificate",
        "external_certificate_identity": {
            "artifact_type": "continuous_robot_insertion_cell_certificate",
            "path": str(certificate.path),
            "file_sha256": expected,
            "semantic_sha256": certificate.semantic_sha256,
        },
        "cell_proof_sha256": "c" * 64,
    }


def test_certified_cell_outside_numerical_selection_is_not_not_evaluated():
    dense = _document(dense=True)
    dense["cells"][0]["classification"] = "SAFE"
    dense = _seal_task_document(dense)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        task = load_insertion_task_set(_write_json(root, dense, "task.json"))
        target = _transform([0.5, 0.1, 0.4])
        execution = {"project_sha256": "1" * 64, "model_sha256": "2" * 64}
        certificate_path = _write_json(
            root, _certificate_document(task, target, execution), "certificate.json")
        certificate_sha = hashlib.sha256(certificate_path.read_bytes()).hexdigest()
        certificate = load_verified_continuous_robot_certificate(
            certificate_path,
            expected_file_sha256=certificate_sha,
            task_set=task,
            robot="B",
            T_W_P_insert=target,
            world_frame=_world_frame(),
            tcp_calibration_fingerprint="tcp-calibrated-v1",
            execution_bindings=execution,
        )
        result = _build(
            task,
            FakeKinematics(),
            max_cells=1,
            tcp_calibrated=True,
            tcp_calibration_fingerprint="tcp-calibrated-v1",
            acknowledge_provisional_tcp=False,
            continuous_certificate=certificate,
        )
    safe_id = task.safe_inner_cell_ids[0]
    numerically_selected = {
        item["cell_id"] for item in result["selection"]["stratification"]["selected"]}
    assert safe_id not in numerically_selected
    assert safe_id in result["certified_receiver_cell_ids"]
    assert safe_id not in result["not_evaluated_cell_ids"]
    assert result["summary"]["not_evaluated_eligible_count"] == 62


def test_continuous_certificate_fails_closed_on_hash_binding_and_gate_attacks():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        task = load_insertion_task_set(_write_json(root, _document(), "task.json"))
        target = _transform([0.5, 0.1, 0.4])
        execution = {"project_sha256": "1" * 64, "model_sha256": "2" * 64}
        document = _certificate_document(task, target, execution)
        path = _write_json(root, document, "certificate.json")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        _assert_value_error(
            lambda: load_verified_continuous_robot_certificate(
                path,
                expected_file_sha256="f" * 64,
                task_set=task,
                robot="B",
                T_W_P_insert=target,
                world_frame=_world_frame(),
                tcp_calibration_fingerprint="tcp-calibrated-v1",
                execution_bindings=execution,
            ),
            "file SHA-256 mismatch",
        )
        attacked = _certificate_document(task, target, execution)
        attacked["hard_gates"]["scene_and_other_arm_collision"] = False
        attacked["semantic_sha256"] = artifact_sha256(attacked)
        attacked_path = _write_json(root, attacked, "attacked.json")
        attacked_sha = hashlib.sha256(attacked_path.read_bytes()).hexdigest()
        _assert_value_error(
            lambda: load_verified_continuous_robot_certificate(
                attacked_path,
                expected_file_sha256=attacked_sha,
                task_set=task,
                robot="B",
                T_W_P_insert=target,
                world_frame=_world_frame(),
                tcp_calibration_fingerprint="tcp-calibrated-v1",
                execution_bindings=execution,
            ),
            "every hard gate true",
        )
        rebound = _certificate_document(task, target, execution)
        rebound["bindings"]["T_W_P_insert"][0][3] += 0.01
        rebound["semantic_sha256"] = artifact_sha256(rebound)
        rebound_path = _write_json(root, rebound, "rebound.json")
        rebound_sha = hashlib.sha256(rebound_path.read_bytes()).hexdigest()
        _assert_value_error(
            lambda: load_verified_continuous_robot_certificate(
                rebound_path,
                expected_file_sha256=rebound_sha,
                task_set=task,
                robot="B",
                T_W_P_insert=target,
                world_frame=_world_frame(),
                tcp_calibration_fingerprint="tcp-calibrated-v1",
                execution_bindings=execution,
            ),
            "target binding mismatch",
        )
        assert actual


def test_board_pose_composition_and_uncalibrated_acknowledgment():
    with tempfile.TemporaryDirectory() as directory:
        task = load_insertion_task_set(
            _write_json(Path(directory), _document()))
    T_W_B = _transform([0.4, -0.2, 0.7])
    target, source = resolve_world_part_insert(task, T_W_B=T_W_B)
    assert source == "T_W_B_x_T_B_P_insert"
    assert np.allclose(target, T_W_B @ task.T_B_P_insert)
    axis, path = sample_straight_insertion_path(
        target, task.insertion_axis_P, 0.04, 5)
    assert np.allclose(axis, [0.0, -1.0, 0.0])
    assert np.allclose(path[-1], target)
    _assert_value_error(
        lambda: _build(
            task, FakeKinematics(), acknowledge_provisional_tcp=False),
        "acknowledge_provisional_tcp",
    )


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"PASS  {test.__name__}")
    print(f"\n{len(tests)}/{len(tests)} passed")
