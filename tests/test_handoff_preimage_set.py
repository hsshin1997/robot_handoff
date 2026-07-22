"""Tests for conservative layer-3 handoff-preimage classification."""
from __future__ import annotations

import copy
import hashlib
import os
import json
from pathlib import Path
import sys
import tempfile


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from mujoco_sim.planner.handoff_preimage_set import build_handoff_preimage_set
from mujoco_sim.offline_tools.artifacts import fingerprint_content
from scripts.build_handoff_preimage_set import _evidence_catalog


TASK = "connector_task"
POSE = "receiver_pose"
RECEIVER_SHA = "e" * 64
MODEL_SHA = "f" * 64
PROJECT_SHA = "c" * 64
WORLD_ID = "workcell_test"
WORLD_CALIBRATION = "calibration-test-fingerprint"

DIRECT_CHECKS = (
    "dual_arm_trajectory_complete",
    "continuous_collision_free",
    "handoff_transition_valid",
    "robot_model_matched",
    "scene_matched",
)
PLACE_CHECKS = (
    "continuous_collision_free",
    "held_object_consistency",
    "stable_placement_valid",
    "robot_model_matched",
    "scene_matched",
)
PICK_CHECKS = (
    "continuous_collision_free",
    "grasp_acquisition_valid",
    "robot_model_matched",
    "scene_matched",
)
EMPTY_CHECKS = (
    "receiver_domain_covered",
    "current_domain_covered",
    "direct_search_complete",
    "transfer_search_complete",
    "continuous_constraints_certified",
)
L2_CERTIFICATE_HARD_GATES = (
    "calibrated_tcp",
    "bound_world_and_target",
    "whole_parameter_cell_coverage",
    "continuous_insertion_path",
    "complete_ik_branch_coverage",
    "joint_limits_and_singularity",
    "robot_self_collision",
    "scene_and_other_arm_collision",
    "gripper_part_pcb_fixture_collision",
    "calibration_uncertainty_envelope",
    "insertion_mechanics_and_contact",
)


def _context(receiver_sha=RECEIVER_SHA):
    return {
        "task_id": TASK,
        "receiver_pose_id": POSE,
        "receiver_artifact_sha256": receiver_sha,
        "robot": "B",
        "model_sha256": MODEL_SHA,
        "project_sha256": PROJECT_SHA,
        "world_frame_id": WORLD_ID,
        "world_calibration_fingerprint": WORLD_CALIBRATION,
    }


def _receiver(*, certified: bool = True):
    cell = {
        "id": "receiver_safe",
        "robot_classification": (
            "CERTIFIED_SAFE" if certified else "PROVISIONAL_PATH_WITNESS"),
        "certified": certified,
        "whole_parameter_cell_evaluated": certified,
        "external_continuous_certificate": ({
            "cell_id": "receiver_safe",
            "classification": "CERTIFIED_SAFE",
            "proof_sha256": "9" * 64,
            "minimum_joint_limit_margin_rad": 0.2,
            "minimum_normalized_joint_limit_margin": 0.3,
            "minimum_sigma": 0.1,
        } if certified else None),
        "certification": ({
            "certified": True,
            "all_hard_gates_passed": True,
            "source": "external_continuous_robot_cell_certificate",
            "external_certificate_identity": {
                "artifact_type": "continuous_robot_insertion_cell_certificate",
                "path": "/tmp/continuous-certificate.json",
                "file_sha256": "7" * 64,
                "semantic_sha256": "8" * 64,
            },
            "cell_proof_sha256": "9" * 64,
        } if certified else {
            "certified": False,
            "all_hard_gates_passed": False,
            "source": None,
            "external_certificate_identity": None,
            "cell_proof_sha256": None,
        }),
    }
    return {
        "schema_version": 1,
        "artifact_type": "robot_conditioned_insertion_path_set",
        "certified": certified,
        "certified_receiver_cell_ids": ["receiver_safe"] if certified else [],
        "provisional_center_path_witness_cell_ids": (
            [] if certified else ["receiver_safe"]),
        "certification": {
            "certified_receiver_cell_ids": (
                ["receiver_safe"] if certified else []),
            "tcp_calibrated": certified,
            "tcp_calibration_fingerprint": (
                "tcp-calibrated-v1" if certified else None),
            "external_continuous_certificate_supplied": certified,
        },
        "continuous_robot_cell_certificate": {
            "supplied": certified,
            "path": ("/tmp/continuous-certificate.json"
                     if certified else None),
            "file_sha256": "7" * 64 if certified else None,
            "semantic_sha256": "8" * 64 if certified else None,
            "hard_gates": ({
                name: True for name in L2_CERTIFICATE_HARD_GATES
            } if certified else None),
        },
        "robot": "B",
        "build": {
            "model_sha256": MODEL_SHA,
            "project_sha256": PROJECT_SHA,
        },
        "world_frame": {
            "id": WORLD_ID,
            "calibration_fingerprint": WORLD_CALIBRATION,
        },
        "cells": [cell],
    }


def _declarations():
    return {
        "schema_version": 1,
        "task_id": TASK,
        "receiver_pose_id": POSE,
        "max_reorientation_hops": 1,
        "current_grasp_domain": {
            "id": "current_domain",
            "complete": True,
            "classes": [{
                "class_id": "current_cell",
                "representative_grasp_ids": ["g_current"],
            }],
        },
        "direct_edges": [],
        "place_edges": [],
        "pick_edges": [],
        "exhaustiveness_certificates": [],
    }


def _catalog_record(payload, marker="a"):
    return {
        "expected_sha256_configured": True,
        "provenance_verified": True,
        "sha256": marker * 64,
        "payload": payload,
    }


def _direct_payload(giver="g_current", *, cover=True):
    return {
        "kind": "dual_arm_handoff_trajectory",
        **_context(),
        "giver_grasp": giver,
        "receiver_cell_id": "receiver_safe",
        "covered_current_class_ids": ["current_cell"] if cover else [],
        "trajectories": {
            "A": [[0.0, 0.0], [0.1, 0.2]],
            "B": [[0.0, 0.0], [-0.1, -0.2]],
        },
        "time_s": [0.0, 1.0],
    }


def _place_payload(grasp="g_current", *, cover=True):
    return {
        "kind": "single_arm_place_trajectory",
        **_context(),
        "grasp_id": grasp,
        "placement_id": "flat_stage_pose_0",
        "covered_current_class_ids": ["current_cell"] if cover else [],
        "trajectory": [[0.0, 0.0], [0.2, 0.1]],
    }


def _pick_payload(grasp="g_after"):
    return {
        "kind": "single_arm_pick_trajectory",
        **_context(),
        "grasp_id": grasp,
        "placement_id": "flat_stage_pose_0",
        "trajectory": [[0.2, 0.1], [0.4, 0.3]],
    }


def _hard_check_catalog(
    subject_id, names, extra_bindings, *, receiver_sha=RECEIVER_SHA,
    subject_sha="a" * 64,
):
    ids = {}
    catalog = {}
    markers = "123456789abcdef"
    for index, name in enumerate(names):
        evidence_id = f"{subject_id}:check:{name}"
        ids[name] = evidence_id
        catalog[evidence_id] = _catalog_record({
            "kind": "hard_check_record",
            "check_name": name,
            "subject_evidence_id": subject_id,
            "subject_evidence_sha256": subject_sha,
            "passed": True,
            "producer": {"name": "test_verifier", "version": "1.0"},
            **_context(receiver_sha),
            **extra_bindings,
        }, markers[index % len(markers)])
    return ids, catalog


def _class_hash_bindings(declarations):
    domain = declarations["current_grasp_domain"]
    raw = domain["classes"][0]
    class_content = {
        "class_id": raw["class_id"],
        "representative_grasp_ids": list(raw["representative_grasp_ids"]),
        "domain": raw.get("domain"),
    }
    return {
        "current_grasp_domain_sha256": fingerprint_content(domain),
        "current_class_sha256": fingerprint_content(class_content),
        "current_class_id": raw["class_id"],
    }


def _coverage_record(
    subject_id, operation, class_bindings, marker="d", subject_sha="a" * 64,
):
    return _catalog_record({
        "kind": "current_class_coverage_record",
        "operation": operation,
        "subject_evidence_id": subject_id,
        "subject_evidence_sha256": subject_sha,
        "whole_class_covered": True,
        "coverage_method": "certified interval enclosure",
        "producer": {"name": "test_coverage_verifier", "version": "1.0"},
        **_context(),
        **class_bindings,
    }, marker)


def _direct_case(*, giver="g_current", cover=True):
    declarations = _declarations()
    class_bindings = _class_hash_bindings(declarations)
    edge = {
        "edge_id": "d0", "giver_grasp": giver,
        "receiver_cell_id": "receiver_safe", "evidence_id": "direct",
        "cost": 1.0, "robustness": 0.8,
    }
    check_ids, checks = _hard_check_catalog(
        "direct", DIRECT_CHECKS,
        {"giver_grasp": giver, "receiver_cell_id": "receiver_safe",
         **class_bindings})
    edge["hard_check_evidence_ids"] = check_ids
    edge["current_class_coverage_evidence_ids"] = {
        "current_cell": "direct:coverage:current_cell"}
    declarations["direct_edges"] = [edge]
    payload = _direct_payload(giver, cover=cover)
    payload.update(class_bindings)
    catalog = {
        "direct": _catalog_record(payload),
        "direct:coverage:current_cell": _coverage_record(
            "direct", "direct", class_bindings),
        **checks,
    }
    return declarations, catalog


def test_provisional_receiver_witness_never_becomes_handoff_goal():
    declarations, catalog = _direct_case()
    result = build_handoff_preimage_set(
        _receiver(certified=False), declarations,
        evidence_catalog=catalog,
        receiver_source={"sha256": RECEIVER_SHA, "status": "LOADED"},
    )
    assert not result["sets"]["direct"]
    assert result["sets"]["unknown"][0]["status"] == "UNKNOWN"
    assert "receiver_safe" in result["receiver_insertion_set"][
        "provisional_path_witness_cell_ids"]
    assert not result["edge_evidence_audit"][0]["valid"]


def test_final_layer2_external_certificate_schema_admits_receiver_goal():
    result = build_handoff_preimage_set(
        _receiver(certified=True), _declarations(), evidence_catalog={},
        receiver_source={"sha256": RECEIVER_SHA, "status": "LOADED"})
    assert result["receiver_insertion_set"][
        "certified_receiver_cell_ids"] == ["receiver_safe"]
    assert not any("not_individually_certified" in issue for issue in
                   result["receiver_insertion_set"]["issues"])


def test_receiver_certificate_rejects_invented_or_incomplete_gate_vocabulary():
    receiver = _receiver(certified=True)
    receiver["continuous_robot_cell_certificate"]["hard_gates"] = {
        "continuous_collision_checked": True,
        "whole_parameter_cell_checked": True,
    }
    result = build_handoff_preimage_set(
        receiver, _declarations(), evidence_catalog={},
        receiver_source={"sha256": RECEIVER_SHA, "status": "LOADED"})
    assert result["receiver_insertion_set"][
        "certified_receiver_cell_ids"] == []
    assert "receiver_cell_not_individually_certified:receiver_safe" in result[
        "receiver_insertion_set"]["issues"]


def test_wrong_receiver_schema_or_artifact_type_cannot_produce_direct():
    for field, value, expected_issue in (
        ("schema_version", 2, "receiver_schema_version_invalid"),
        ("schema_version", True, "receiver_schema_version_invalid"),
        ("artifact_type", "wrong_type", "receiver_artifact_type_invalid"),
    ):
        receiver = _receiver(certified=True)
        receiver[field] = value
        declarations, catalog = _direct_case()
        result = build_handoff_preimage_set(
            receiver, declarations, evidence_catalog=catalog,
            receiver_source={"sha256": RECEIVER_SHA, "status": "LOADED"})
        assert not result["sets"]["direct"]
        assert result["sets"]["unknown"]
        assert result["receiver_insertion_set"][
            "certified_receiver_cell_ids"] == []
        assert expected_issue in result["receiver_insertion_set"]["issues"]


def test_direct_membership_requires_real_dual_arm_trajectory_and_class_coverage():
    declarations, catalog = _direct_case()
    payload = dict(catalog["direct"]["payload"])
    payload["trajectories"] = dict(payload["trajectories"])
    payload["trajectories"].pop("B")
    catalog["direct"] = _catalog_record(payload)
    rejected = build_handoff_preimage_set(
        _receiver(), declarations,
        evidence_catalog=catalog,
        receiver_source={"sha256": RECEIVER_SHA, "status": "LOADED"},
    )
    assert not rejected["sets"]["direct"]
    assert "robot_B_trajectory_missing" in rejected[
        "edge_evidence_audit"][0]["reasons"]

    declarations, catalog = _direct_case()
    accepted = build_handoff_preimage_set(
        _receiver(), declarations,
        evidence_catalog=catalog,
        receiver_source={"sha256": RECEIVER_SHA, "status": "LOADED"},
    )
    record = accepted["sets"]["direct"][0]
    assert record["status"] == "DIRECT"
    assert record["receiver_cell_id"] == "receiver_safe"
    assert record["plan"]["steps"][0]["evidence_id"] == "direct"
    assert accepted["certification"]["positive_membership_sound"] is True
    assert accepted["certification"]["coverage_certified"] is True

    declarations, catalog = _direct_case()
    catalog["direct:coverage:current_cell"]["payload"][
        "whole_class_covered"] = False
    not_covering = build_handoff_preimage_set(
        _receiver(), declarations,
        evidence_catalog=catalog,
        receiver_source={"sha256": RECEIVER_SHA, "status": "LOADED"},
    )
    assert not not_covering["sets"]["direct"]
    assert not_covering["sets"]["unknown"]


def test_transfer_requires_directed_place_pick_and_final_handoff_evidence():
    declarations = _declarations()
    class_bindings = _class_hash_bindings(declarations)
    direct_edge = {
        "edge_id": "handoff_after", "giver_grasp": "g_after",
        "receiver_cell_id": "receiver_safe", "evidence_id": "direct_after",
        "cost": 0.7, "robustness": 0.75,
    }
    place_edge = {
        "edge_id": "place_current", "grasp_id": "g_current",
        "placement_id": "flat_stage_pose_0", "evidence_id": "place",
        "cost": 0.4, "robustness": 0.7,
    }
    pick_edge = {
        "edge_id": "pick_after", "grasp_id": "g_after",
        "placement_id": "flat_stage_pose_0", "evidence_id": "pick",
        "cost": 0.5, "robustness": 0.8,
    }
    direct_ids, direct_checks = _hard_check_catalog(
        "direct_after", DIRECT_CHECKS,
        {"giver_grasp": "g_after", "receiver_cell_id": "receiver_safe"})
    place_ids, place_checks = _hard_check_catalog(
        "place", PLACE_CHECKS,
        {"grasp_id": "g_current", "placement_id": "flat_stage_pose_0",
         **class_bindings}, subject_sha="b" * 64)
    pick_ids, pick_checks = _hard_check_catalog(
        "pick", PICK_CHECKS,
        {"grasp_id": "g_after", "placement_id": "flat_stage_pose_0"},
        subject_sha="c" * 64)
    direct_edge["hard_check_evidence_ids"] = direct_ids
    place_edge["hard_check_evidence_ids"] = place_ids
    place_edge["current_class_coverage_evidence_ids"] = {
        "current_cell": "place:coverage:current_cell"}
    pick_edge["hard_check_evidence_ids"] = pick_ids
    declarations["direct_edges"] = [direct_edge]
    declarations["place_edges"] = [place_edge]
    declarations["pick_edges"] = [pick_edge]
    place_payload = _place_payload()
    place_payload.update(class_bindings)
    catalog = {
        "direct_after": _catalog_record(
            _direct_payload("g_after", cover=False), "a"),
        "place": _catalog_record(place_payload, "b"),
        "pick": _catalog_record(_pick_payload(), "c"),
        "place:coverage:current_cell": _coverage_record(
            "place", "place", class_bindings, subject_sha="b" * 64),
        **direct_checks,
        **place_checks,
        **pick_checks,
    }
    result = build_handoff_preimage_set(
        _receiver(), declarations, evidence_catalog=catalog,
        receiver_source={"sha256": RECEIVER_SHA, "status": "LOADED"})
    record = result["sets"]["reorientation"][0]
    assert record["status"] == "TRANSFER"
    assert [step["kind"] for step in record["plan"]["steps"]] == [
        "place", "regrasp", "handoff"]
    assert record["plan"]["reorientation_hops"] == 1
    assert record["plan"]["total_cost"] == 1.6
    assert record["plan"]["bottleneck_robustness"] == 0.7
    assert result["certification"]["positive_membership_sound"] is True
    assert result["certification"]["coverage_certified"] is True


def test_missing_edges_are_unknown_not_uncovered():
    result = build_handoff_preimage_set(
        _receiver(), _declarations(), evidence_catalog={})
    assert not result["sets"]["uncovered"]
    unknown = result["sets"]["unknown"][0]
    assert unknown["status"] == "UNKNOWN"
    assert "exhaustive_empty_preimage_certificate" in unknown["missing_inputs"]


def test_uncovered_requires_bound_exhaustive_certificate():
    declarations = _declarations()
    class_bindings = _class_hash_bindings(declarations)
    receiver_sha = "d" * 64
    empty_payload = {
        "kind": "certified_empty_handoff_preimage",
        **_context(receiver_sha),
        "class_id": "current_cell",
        "current_grasp_domain_id": "current_domain",
        **class_bindings,
    }
    check_ids, checks = _hard_check_catalog(
        "empty", EMPTY_CHECKS,
        {"class_id": "current_cell",
         "current_grasp_domain_id": "current_domain",
         **class_bindings},
        receiver_sha=receiver_sha)
    declarations["exhaustiveness_certificates"] = [{
        "class_id": "current_cell", "evidence_id": "empty",
        "hard_check_evidence_ids": check_ids,
    }]
    result = build_handoff_preimage_set(
        _receiver(), declarations,
        evidence_catalog={"empty": _catalog_record(empty_payload), **checks},
        receiver_source={"sha256": receiver_sha, "status": "LOADED"},
    )
    assert result["sets"]["uncovered"][0]["status"] == "UNCOVERED"
    assert not result["sets"]["unknown"]
    assert result["certification"]["positive_membership_sound"] is False
    assert result["certification"]["coverage_certified"] is True


def test_cli_catalog_does_not_trust_an_unpinned_file():
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        evidence_path = directory / "trajectory.json"
        evidence_path.write_text(json.dumps(_direct_payload()), encoding="utf-8")
        config_path = directory / "preimage.yaml"
        config_path.write_text("schema_version: 1\n", encoding="utf-8")
        unpinned = _evidence_catalog({
            "evidence_artifacts": [{
                "id": "direct", "path": "trajectory.json",
            }],
        }, config_path)
        assert not unpinned["direct"]["provenance_verified"]
        assert not unpinned["direct"]["expected_sha256_configured"]

        wrong = _evidence_catalog({
            "evidence_artifacts": [{
                "id": "direct", "path": "trajectory.json",
                "expected_sha256": "0" * 64,
            }],
        }, config_path)
        assert not wrong["direct"]["provenance_verified"]
        assert wrong["direct"]["expected_sha256_configured"]

        actual_sha = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
        pinned = _evidence_catalog({
            "evidence_artifacts": [{
                "id": "direct", "path": "trajectory.json",
                "expected_sha256": actual_sha,
            }],
        }, config_path)
        assert pinned["direct"]["provenance_verified"]
        assert pinned["direct"]["expected_sha256_configured"]


def test_garbage_ragged_nonnumeric_and_untimed_dual_paths_are_rejected():
    mutations = (
        ("ragged", lambda payload: payload["trajectories"].__setitem__(
            "A", [[0.0], [0.1, 0.2]]), "robot_A_inconsistent_dof:1"),
        ("string", lambda payload: payload["trajectories"].__setitem__(
            "A", [["not-a-joint", 0.0], [0.1, 0.2]]),
         "robot_A_nonfinite_or_nonnumeric_joint:0:0"),
        ("nan", lambda payload: payload["trajectories"].__setitem__(
            "A", [[float("nan"), 0.0], [0.1, 0.2]]),
         "robot_A_nonfinite_or_nonnumeric_joint:0:0"),
        ("missing_time", lambda payload: payload.pop("time_s"),
         "dual_arm_time_s_required"),
        ("reversed_time", lambda payload: payload.__setitem__(
            "time_s", [1.0, 0.0]), "dual_arm_time_s_not_strictly_increasing"),
    )
    for _name, mutate, expected_reason in mutations:
        declarations, catalog = _direct_case()
        payload = copy.deepcopy(catalog["direct"]["payload"])
        mutate(payload)
        catalog["direct"] = _catalog_record(payload)
        result = build_handoff_preimage_set(
            _receiver(), declarations, evidence_catalog=catalog,
            receiver_source={"sha256": RECEIVER_SHA, "status": "LOADED"})
        assert not result["sets"]["direct"]
        assert expected_reason in result["edge_evidence_audit"][0]["reasons"]


def test_bare_self_declared_checks_cannot_create_direct_membership():
    declarations, catalog = _direct_case()
    declarations["direct_edges"][0].pop("hard_check_evidence_ids")
    catalog["direct"]["payload"]["checks"] = {
        name: True for name in DIRECT_CHECKS
    }
    result = build_handoff_preimage_set(
        _receiver(), declarations, evidence_catalog=catalog,
        receiver_source={"sha256": RECEIVER_SHA, "status": "LOADED"})
    assert not result["sets"]["direct"]
    assert "hard_check_evidence_ids_missing" in result[
        "edge_evidence_audit"][0]["reasons"]


def test_truthy_check_and_model_or_world_mismatch_are_rejected():
    declarations, catalog = _direct_case()
    first_check = next(iter(
        declarations["direct_edges"][0]["hard_check_evidence_ids"].values()))
    catalog[first_check]["payload"]["passed"] = 1
    truthy = build_handoff_preimage_set(
        _receiver(), declarations, evidence_catalog=catalog,
        receiver_source={"sha256": RECEIVER_SHA, "status": "LOADED"})
    assert not truthy["sets"]["direct"]
    assert any("hard_check_not_passed" in reason for reason in
               truthy["edge_evidence_audit"][0]["reasons"])

    declarations, catalog = _direct_case()
    catalog["direct"]["payload"]["model_sha256"] = "0" * 64
    catalog["direct"]["payload"]["world_frame_id"] = "wrong-world"
    mismatch = build_handoff_preimage_set(
        _receiver(), declarations, evidence_catalog=catalog,
        receiver_source={"sha256": RECEIVER_SHA, "status": "LOADED"})
    reasons = mismatch["edge_evidence_audit"][0]["reasons"]
    assert "binding_mismatch:model_sha256" in reasons
    assert "binding_mismatch:world_frame_id" in reasons
    assert not mismatch["sets"]["direct"]


def test_repinning_trajectory_cannot_reuse_checks_bound_to_old_digest():
    declarations, catalog = _direct_case()
    replacement = copy.deepcopy(catalog["direct"]["payload"])
    replacement["trajectories"]["A"][1][0] += 0.01
    # Simulate an attacker updating the configured trajectory digest while
    # leaving all previously validated hard-check artifacts untouched.
    catalog["direct"] = _catalog_record(replacement, "b")
    result = build_handoff_preimage_set(
        _receiver(), declarations, evidence_catalog=catalog,
        receiver_source={"sha256": RECEIVER_SHA, "status": "LOADED"})
    assert not result["sets"]["direct"]
    assert any("hard_check_subject_sha256_mismatch" in reason for reason in
               result["edge_evidence_audit"][0]["reasons"])


def test_domain_or_representative_mutation_invalidates_whole_class_evidence():
    mutations = (
        lambda domain: domain.__setitem__("measurement_revision", 2),
        lambda domain: domain["classes"][0].__setitem__(
            "domain", {"u": [0.0, 0.1], "rho": [-0.2, 0.2]}),
        lambda domain: domain["classes"][0][
            "representative_grasp_ids"].append("g_other"),
    )
    for mutate in mutations:
        declarations, catalog = _direct_case()
        mutate(declarations["current_grasp_domain"])
        result = build_handoff_preimage_set(
            _receiver(), declarations, evidence_catalog=catalog,
            receiver_source={"sha256": RECEIVER_SHA, "status": "LOADED"})
        assert not result["sets"]["direct"]
        reasons = result["current_class_coverage_audit"][0]["reasons"]
        assert any("class_binding_mismatch" in reason
                   or "coverage_binding_mismatch" in reason
                   for reason in reasons)
        assert result["certification"]["positive_membership_sound"] is False


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS  {test.__name__}")
    print(f"\n{len(tests)}/{len(tests)} passed")
