"""Evidence-backed handoff preimage and transfer-stage classification.

This module deliberately does *not* infer feasibility from proximity, an IK
endpoint, or the absence of a failed search.  A current-grasp domain enters
the direct or transfer preimage only when every edge in the corresponding
task path carries a validated trajectory artifact.  A domain is labelled
``UNCOVERED`` only with a separate exhaustive emptiness certificate; all
other gaps remain ``UNKNOWN``.

The continuous geometry and robot insertion layers are upstream of this
classifier.  Their certified receiver cells are task-graph goals.  Provisional
path witnesses are retained as diagnostics but can never become handoff
edges here.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from numbers import Real
from typing import Any, Mapping, Sequence

from ..offline_tools.artifacts import fingerprint_content


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "handoff_preimage_set"

_CERTIFIED_RECEIVER_LABELS = {
    "CERTIFIED_SAFE",
    "CERTIFIED_INSERTION_SAFE",
    "CERTIFIED_PATH_SAFE",
}
# Exact projection of robot_insertion_set._CERTIFICATE_HARD_GATES for schema 1.
# A subset, superset, or renamed gate vocabulary is not the same certificate.
_L2_CERTIFICATE_HARD_GATES = frozenset({
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
})
_DIRECT_CHECKS = (
    "dual_arm_trajectory_complete",
    "continuous_collision_free",
    "handoff_transition_valid",
    "robot_model_matched",
    "scene_matched",
)
_PLACE_CHECKS = (
    "continuous_collision_free",
    "held_object_consistency",
    "stable_placement_valid",
    "robot_model_matched",
    "scene_matched",
)
_PICK_CHECKS = (
    "continuous_collision_free",
    "grasp_acquisition_valid",
    "robot_model_matched",
    "scene_matched",
)
_EMPTY_CHECKS = (
    "receiver_domain_covered",
    "current_domain_covered",
    "direct_search_complete",
    "transfer_search_complete",
    "continuous_constraints_certified",
)


def _as_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _as_identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _metric(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return result


def _trajectory_reasons(value: Any, label: str) -> tuple[list[str], int | None]:
    """Validate a finite rectangular numeric joint trajectory."""
    if (not isinstance(value, Sequence)
            or isinstance(value, (str, bytes))):
        return [f"{label}_trajectory_missing"], None
    if len(value) < 2:
        return [f"{label}_trajectory_requires_at_least_two_knots"], None
    dof: int | None = None
    reasons: list[str] = []
    for knot_index, knot in enumerate(value):
        if (not isinstance(knot, Sequence)
                or isinstance(knot, (str, bytes)) or len(knot) == 0):
            reasons.append(f"{label}_knot_not_numeric_vector:{knot_index}")
            continue
        if dof is None:
            dof = len(knot)
        elif len(knot) != dof:
            reasons.append(f"{label}_inconsistent_dof:{knot_index}")
        for joint_index, joint in enumerate(knot):
            if (isinstance(joint, bool) or not isinstance(joint, Real)
                    or not math.isfinite(float(joint))):
                reasons.append(
                    f"{label}_nonfinite_or_nonnumeric_joint:"
                    f"{knot_index}:{joint_index}")
    return sorted(set(reasons)), (None if reasons else dof)


def _time_reasons(
    value: Any,
    knot_count: int,
    label: str,
    *,
    required: bool,
) -> list[str]:
    if value is None:
        return [f"{label}_time_s_required"] if required else []
    if (not isinstance(value, Sequence)
            or isinstance(value, (str, bytes)) or len(value) != knot_count):
        return [f"{label}_time_s_length_mismatch"]
    times: list[float] = []
    for index, raw in enumerate(value):
        if (isinstance(raw, bool) or not isinstance(raw, Real)
                or not math.isfinite(float(raw))):
            return [f"{label}_time_s_not_finite_numeric:{index}"]
        times.append(float(raw))
    if any(right <= left for left, right in zip(times, times[1:])):
        return [f"{label}_time_s_not_strictly_increasing"]
    return []


def _sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


@dataclass(frozen=True)
class _EvidenceVerdict:
    evidence_id: str
    valid: bool
    reasons: tuple[str, ...]
    sha256: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "valid": self.valid,
            "reasons": list(self.reasons),
            "sha256": self.sha256,
        }


def _provenance_reasons(raw: Mapping[str, Any], label: str) -> list[str]:
    reasons: list[str] = []
    if raw.get("expected_sha256_configured") is not True:
        reasons.append(f"{label}_expected_sha256_not_configured")
    if raw.get("provenance_verified") is not True:
        reasons.append(f"{label}_provenance_not_verified")
    if not _sha256(raw.get("sha256")):
        reasons.append(f"{label}_sha256_missing_or_invalid")
    return reasons


def _hard_check_reasons(
    catalog: Mapping[str, Any],
    subject_evidence_id: str,
    subject_evidence_sha256: str | None,
    hard_check_evidence_ids: Any,
    required_checks: Sequence[str],
    bindings: Mapping[str, str],
) -> list[str]:
    """Validate independently pinned hard-check records.

    Bare booleans in a trajectory payload are deliberately ignored.  Every
    mandatory gate must name a separate evidence artifact whose configured
    digest matches and whose payload binds the exact trajectory and context.
    """
    if not isinstance(hard_check_evidence_ids, Mapping):
        return ["hard_check_evidence_ids_missing"]
    reasons: list[str] = []
    for check_name in required_checks:
        check_id = hard_check_evidence_ids.get(check_name)
        if not isinstance(check_id, str) or not check_id:
            reasons.append(f"hard_check_evidence_missing:{check_name}")
            continue
        raw = catalog.get(check_id)
        if not isinstance(raw, Mapping):
            reasons.append(f"hard_check_record_missing:{check_name}")
            continue
        reasons.extend(
            f"{check_name}:{reason}"
            for reason in _provenance_reasons(raw, "hard_check")
        )
        payload = raw.get("payload")
        if not isinstance(payload, Mapping):
            reasons.append(f"hard_check_payload_missing:{check_name}")
            continue
        if payload.get("kind") != "hard_check_record":
            reasons.append(f"hard_check_kind_invalid:{check_name}")
        if payload.get("check_name") != check_name:
            reasons.append(f"hard_check_name_mismatch:{check_name}")
        if payload.get("subject_evidence_id") != subject_evidence_id:
            reasons.append(f"hard_check_subject_mismatch:{check_name}")
        if (subject_evidence_sha256 is None
                or payload.get("subject_evidence_sha256")
                != subject_evidence_sha256):
            reasons.append(f"hard_check_subject_sha256_mismatch:{check_name}")
        # `is True` rejects truthy integers and strings.
        if payload.get("passed") is not True:
            reasons.append(f"hard_check_not_passed:{check_name}")
        producer = payload.get("producer")
        if (not isinstance(producer, Mapping)
                or not isinstance(producer.get("name"), str)
                or not producer.get("name")
                or not isinstance(producer.get("version"), str)
                or not producer.get("version")):
            reasons.append(f"hard_check_producer_missing:{check_name}")
        for field, expected in bindings.items():
            if payload.get(field) != expected:
                reasons.append(f"hard_check_binding_mismatch:{check_name}:{field}")
    return reasons


def _validate_evidence(
    catalog: Mapping[str, Any],
    evidence_id: str,
    *,
    expected_kind: str,
    required_checks: Sequence[str],
    hard_check_evidence_ids: Any,
    bindings: Mapping[str, str],
    trajectory_mode: str | None,
) -> _EvidenceVerdict:
    reasons: list[str] = []
    raw = catalog.get(evidence_id)
    if not isinstance(raw, Mapping):
        return _EvidenceVerdict(
            evidence_id, False, ("evidence_record_missing",), None)
    fingerprint = raw.get("sha256")
    reasons.extend(_provenance_reasons(raw, "evidence"))
    if not _sha256(fingerprint):
        fingerprint = None
    payload = raw.get("payload")
    if not isinstance(payload, Mapping):
        reasons.append("evidence_payload_missing")
        return _EvidenceVerdict(
            evidence_id, False, tuple(reasons), fingerprint)
    if payload.get("kind") != expected_kind:
        reasons.append(f"expected_evidence_kind:{expected_kind}")
    for field, expected in bindings.items():
        if payload.get(field) != expected:
            reasons.append(f"binding_mismatch:{field}")
    reasons.extend(_hard_check_reasons(
        catalog, evidence_id, fingerprint, hard_check_evidence_ids,
        required_checks, bindings))

    if trajectory_mode == "dual":
        trajectories = payload.get("trajectories")
        if not isinstance(trajectories, Mapping):
            reasons.append("dual_arm_trajectories_missing")
        else:
            path_a = trajectories.get("A", trajectories.get("robot_A"))
            path_b = trajectories.get("B", trajectories.get("robot_B"))
            reasons_a, _ = _trajectory_reasons(path_a, "robot_A")
            reasons_b, _ = _trajectory_reasons(path_b, "robot_B")
            reasons.extend(reasons_a)
            reasons.extend(reasons_b)
            if not reasons_a and not reasons_b:
                if len(path_a) != len(path_b):
                    reasons.append("dual_arm_knot_count_mismatch")
                else:
                    reasons.extend(_time_reasons(
                        payload.get("time_s"), len(path_a),
                        "dual_arm", required=True))
    elif trajectory_mode == "single":
        path = payload.get("trajectory")
        path_reasons, _ = _trajectory_reasons(path, "single_arm")
        if path_reasons:
            trajectories = payload.get("trajectories")
            path = (trajectories.get("A") if isinstance(trajectories, Mapping)
                    else None)
            path_reasons, _ = _trajectory_reasons(path, "single_arm")
        reasons.extend(path_reasons)
        if not path_reasons:
            reasons.extend(_time_reasons(
                payload.get("time_s"), len(path),
                "single_arm", required=False))

    return _EvidenceVerdict(
        evidence_id, not reasons, tuple(reasons), fingerprint)


def _receiver_cell_certificate_valid(
    artifact: Mapping[str, Any],
    cell: Mapping[str, Any],
    cell_id: str,
) -> bool:
    """Verify the exact final layer-2 external-certificate projection."""
    root_certification = artifact.get("certification")
    external = artifact.get("continuous_robot_cell_certificate")
    cell_certification = cell.get("certification")
    proof = cell.get("external_continuous_certificate")
    expected_cell_certificate_fields = {
        "certified",
        "all_hard_gates_passed",
        "source",
        "external_certificate_identity",
        "cell_proof_sha256",
    }
    identity = (cell_certification.get("external_certificate_identity")
                if isinstance(cell_certification, Mapping) else None)
    if (artifact.get("certified") is not True
            or not isinstance(root_certification, Mapping)
            or root_certification.get("tcp_calibrated") is not True
            or root_certification.get(
                "external_continuous_certificate_supplied") is not True
            or not isinstance(root_certification.get(
                "tcp_calibration_fingerprint"), str)
            or not root_certification.get("tcp_calibration_fingerprint")
            or not isinstance(external, Mapping)
            or external.get("supplied") is not True
            or not _sha256(external.get("file_sha256"))
            or not _sha256(external.get("semantic_sha256"))
            or not isinstance(external.get("hard_gates"), Mapping)
            or set(external["hard_gates"]) != _L2_CERTIFICATE_HARD_GATES
            or any(value is not True
                   for value in external["hard_gates"].values())
            or cell.get("certified") is not True
            or cell.get("whole_parameter_cell_evaluated") is not True
            or not isinstance(cell_certification, Mapping)
            or set(cell_certification) != expected_cell_certificate_fields
            or cell_certification.get("certified") is not True
            or cell_certification.get("all_hard_gates_passed") is not True
            or cell_certification.get("source")
            != "external_continuous_robot_cell_certificate"
            or not isinstance(identity, Mapping)
            or set(identity) != {
                "artifact_type", "path", "file_sha256", "semantic_sha256"}
            or identity.get("artifact_type")
            != "continuous_robot_insertion_cell_certificate"
            or not isinstance(identity.get("path"), str)
            or not identity.get("path")
            or identity.get("path") != external.get("path")
            or identity.get("file_sha256") != external.get("file_sha256")
            or identity.get("semantic_sha256")
            != external.get("semantic_sha256")
            or not _sha256(cell_certification.get("cell_proof_sha256"))
            or not isinstance(proof, Mapping)
            or proof.get("cell_id") != cell_id
            or proof.get("classification") != "CERTIFIED_SAFE"
            or not _sha256(proof.get("proof_sha256"))
            or proof.get("proof_sha256")
            != cell_certification.get("cell_proof_sha256")):
        return False
    for field in (
        "minimum_joint_limit_margin_rad",
        "minimum_normalized_joint_limit_margin",
        "minimum_sigma",
    ):
        value = proof.get(field)
        if (isinstance(value, bool) or not isinstance(value, Real)
                or not math.isfinite(float(value)) or float(value) <= 0.0):
            return False
    return True


def _receiver_goals(
    artifact: Mapping[str, Any] | None,
) -> tuple[dict[str, Mapping[str, Any]], list[str], list[str]]:
    """Extract only individually certified receiver cells.

    The root list is necessary but insufficient.  Each listed cell must also
    carry a certified-safe classification and an explicit certificate marker.
    This fail-closed rule prevents a provisional layer-2 witness from being
    promoted to a handoff goal by a naming mistake.
    """
    issues: list[str] = []
    diagnostics: list[str] = []
    if artifact is None:
        return {}, ["receiver_insertion_set_missing"], diagnostics
    schema = artifact.get("schema_version")
    if (isinstance(schema, bool) or not isinstance(schema, int)
            or schema != 1):
        issues.append("receiver_schema_version_invalid")
    if artifact.get("artifact_type") != "robot_conditioned_insertion_path_set":
        issues.append("receiver_artifact_type_invalid")
    if issues:
        issues.append("no_certified_receiver_cells")
        return {}, sorted(set(issues)), diagnostics
    declared = artifact.get("certified_receiver_cell_ids")
    if not isinstance(declared, list):
        issues.append("certified_receiver_cell_ids_missing")
        declared = []
    cells = artifact.get("cells")
    if not isinstance(cells, list):
        issues.append("receiver_cells_missing")
        cells = []
    by_id: dict[str, Mapping[str, Any]] = {}
    for cell in cells:
        if not isinstance(cell, Mapping) or not isinstance(cell.get("id"), str):
            issues.append("receiver_cell_record_invalid")
            continue
        by_id[cell["id"]] = cell
    goals: dict[str, Mapping[str, Any]] = {}
    for cell_id in declared:
        if not isinstance(cell_id, str) or cell_id not in by_id:
            issues.append(f"certified_receiver_cell_not_found:{cell_id}")
            continue
        cell = by_id[cell_id]
        classification = cell.get("robot_classification")
        if (classification not in _CERTIFIED_RECEIVER_LABELS
                or not _receiver_cell_certificate_valid(
                    artifact, cell, cell_id)):
            issues.append(f"receiver_cell_not_individually_certified:{cell_id}")
            continue
        goals[cell_id] = cell
    for key in (
        "provisional_center_path_witness_cell_ids",
        "robot_path_witness_cell_ids",
    ):
        provisional = artifact.get(key, [])
        if isinstance(provisional, list):
            diagnostics.extend(str(value) for value in provisional)
    if not goals:
        issues.append("no_certified_receiver_cells")
    return goals, sorted(set(issues)), sorted(set(diagnostics))


def _current_classes(
    declarations: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], bool, str, str | None, list[str]]:
    issues: list[str] = []
    domain = declarations.get("current_grasp_domain")
    if not isinstance(domain, Mapping):
        return [], False, "missing", None, ["current_grasp_domain_missing"]
    domain_id = str(domain.get("id", "current_grasp_domain"))
    try:
        domain_sha256 = fingerprint_content(domain)
    except (TypeError, ValueError):
        domain_sha256 = None
        issues.append("current_grasp_domain_not_canonical_json")
    complete = domain.get("complete") is True
    if not complete:
        issues.append("current_grasp_domain_not_complete")
    raw_classes = domain.get("classes")
    if not isinstance(raw_classes, list) or not raw_classes:
        return (
            [], complete, domain_id, domain_sha256,
            issues + ["current_grasp_classes_missing"])
    classes: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in raw_classes:
        if not isinstance(raw, Mapping):
            issues.append("current_grasp_class_invalid")
            continue
        try:
            class_id = _as_identifier(raw.get("class_id"), "class_id")
        except ValueError:
            issues.append("current_grasp_class_id_invalid")
            continue
        representatives = raw.get(
            "representative_grasp_ids", raw.get("grasp_ids"))
        if (not isinstance(representatives, list) or not representatives
                or not all(isinstance(value, str) and value
                           for value in representatives)):
            issues.append(f"representative_grasp_ids_invalid:{class_id}")
            continue
        if len(set(representatives)) != len(representatives):
            issues.append(f"representative_grasp_ids_duplicate:{class_id}")
            continue
        if class_id in seen:
            issues.append(f"duplicate_current_grasp_class:{class_id}")
            continue
        seen.add(class_id)
        class_content = {
            "class_id": class_id,
            "representative_grasp_ids": list(representatives),
            "domain": raw.get("domain"),
        }
        try:
            class_sha256 = fingerprint_content(class_content)
        except (TypeError, ValueError):
            issues.append(f"current_grasp_class_not_canonical_json:{class_id}")
            continue
        classes.append({
            **class_content,
            "class_content_sha256": class_sha256,
        })
    return (
        classes, complete, domain_id, domain_sha256,
        sorted(set(issues)))


def _evidence_payload(catalog: Mapping[str, Any], evidence_id: str) -> Mapping[str, Any]:
    record = catalog.get(evidence_id, {})
    payload = record.get("payload", {}) if isinstance(record, Mapping) else {}
    return payload if isinstance(payload, Mapping) else {}


def _class_coverage_verdict(
    edge: Mapping[str, Any],
    catalog: Mapping[str, Any],
    *,
    class_id: str,
    current_grasp_domain_sha256: str | None,
    current_class_sha256: str,
    context_bindings: Mapping[str, str],
    operation: str,
) -> _EvidenceVerdict:
    """Verify externally evidenced whole-cell coverage for an initial edge."""
    coverage_ids = edge.get("current_class_coverage_evidence_ids")
    coverage_id = (coverage_ids.get(class_id)
                   if isinstance(coverage_ids, Mapping) else None)
    if not isinstance(coverage_id, str) or not coverage_id:
        return _EvidenceVerdict(
            str(coverage_id or "missing"), False,
            ("current_class_coverage_evidence_missing",), None)
    if current_grasp_domain_sha256 is None:
        return _EvidenceVerdict(
            coverage_id, False,
            ("current_grasp_domain_sha256_missing",), None)
    subject_id = str(edge.get("evidence_id", ""))
    subject_raw = catalog.get(subject_id)
    subject_sha = (subject_raw.get("sha256")
                   if isinstance(subject_raw, Mapping) else None)
    class_bindings = {
        "current_grasp_domain_sha256": current_grasp_domain_sha256,
        "current_class_sha256": current_class_sha256,
        "current_class_id": class_id,
    }
    reasons: list[str] = []
    subject_payload = _evidence_payload(catalog, subject_id)
    for field, expected in class_bindings.items():
        if subject_payload.get(field) != expected:
            reasons.append(f"subject_class_binding_mismatch:{field}")
    hard_checks = edge.get("hard_check_evidence_ids")
    if not isinstance(hard_checks, Mapping):
        reasons.append("hard_check_evidence_ids_missing_for_class_binding")
    else:
        for check_name, check_id in hard_checks.items():
            raw = catalog.get(check_id)
            payload = (raw.get("payload")
                       if isinstance(raw, Mapping) else None)
            if not isinstance(payload, Mapping):
                reasons.append(f"hard_check_class_payload_missing:{check_name}")
                continue
            for field, expected in class_bindings.items():
                if payload.get(field) != expected:
                    reasons.append(
                        f"hard_check_class_binding_mismatch:{check_name}:{field}")

    raw = catalog.get(coverage_id)
    if not isinstance(raw, Mapping):
        reasons.append("current_class_coverage_record_missing")
        return _EvidenceVerdict(
            coverage_id, False, tuple(sorted(set(reasons))), None)
    reasons.extend(_provenance_reasons(raw, "class_coverage"))
    fingerprint = raw.get("sha256")
    if not _sha256(fingerprint):
        fingerprint = None
    payload = raw.get("payload")
    if not isinstance(payload, Mapping):
        reasons.append("current_class_coverage_payload_missing")
    else:
        expected_bindings = {**context_bindings, **class_bindings}
        if payload.get("kind") != "current_class_coverage_record":
            reasons.append("current_class_coverage_kind_invalid")
        if payload.get("operation") != operation:
            reasons.append("current_class_coverage_operation_mismatch")
        if payload.get("subject_evidence_id") != subject_id:
            reasons.append("current_class_coverage_subject_mismatch")
        if subject_sha is None or payload.get(
                "subject_evidence_sha256") != subject_sha:
            reasons.append("current_class_coverage_subject_sha256_mismatch")
        if payload.get("whole_class_covered") is not True:
            reasons.append("current_class_not_whole_covered")
        if (not isinstance(payload.get("coverage_method"), str)
                or not payload.get("coverage_method")):
            reasons.append("current_class_coverage_method_missing")
        producer = payload.get("producer")
        if (not isinstance(producer, Mapping)
                or not isinstance(producer.get("name"), str)
                or not producer.get("name")
                or not isinstance(producer.get("version"), str)
                or not producer.get("version")):
            reasons.append("current_class_coverage_producer_missing")
        for field, expected in expected_bindings.items():
            if payload.get(field) != expected:
                reasons.append(f"current_class_coverage_binding_mismatch:{field}")
    return _EvidenceVerdict(
        coverage_id, not reasons, tuple(sorted(set(reasons))), fingerprint)


def _required_context_bindings(
    receiver_artifact: Mapping[str, Any] | None,
    *,
    task_id: str,
    receiver_pose_id: str,
    receiver_artifact_sha256: str | None,
) -> tuple[dict[str, str], list[str]]:
    """Build the immutable context every positive edge must reproduce."""
    bindings = {
        "task_id": task_id,
        "receiver_pose_id": receiver_pose_id,
    }
    issues: list[str] = []
    if receiver_artifact_sha256 is None:
        issues.append("receiver_artifact_sha256_missing")
    else:
        bindings["receiver_artifact_sha256"] = receiver_artifact_sha256
    artifact = receiver_artifact or {}
    build = artifact.get("build")
    world = artifact.get("world_frame")
    required = {
        "robot": artifact.get("robot"),
        "model_sha256": (build.get("model_sha256")
                         if isinstance(build, Mapping) else None),
        "project_sha256": (build.get("project_sha256")
                           if isinstance(build, Mapping) else None),
        "world_frame_id": (world.get("id")
                           if isinstance(world, Mapping) else None),
        "world_calibration_fingerprint": (
            world.get("calibration_fingerprint")
            if isinstance(world, Mapping) else None),
    }
    for field, value in required.items():
        if not isinstance(value, str) or not value:
            issues.append(f"receiver_context_binding_missing:{field}")
        else:
            bindings[field] = value
    return bindings, sorted(set(issues))


def _edge_verdicts(
    declarations: Mapping[str, Any],
    catalog: Mapping[str, Any],
    receiver_goals: Mapping[str, Any],
    *,
    context_bindings: Mapping[str, str],
    context_issues: Sequence[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    valid_direct: list[dict[str, Any]] = []
    valid_place: list[dict[str, Any]] = []
    valid_pick: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []

    specs = (
        ("direct_edges", "dual_arm_handoff_trajectory", _DIRECT_CHECKS, "dual"),
        ("place_edges", "single_arm_place_trajectory", _PLACE_CHECKS, "single"),
        ("pick_edges", "single_arm_pick_trajectory", _PICK_CHECKS, "single"),
    )
    for list_name, kind, checks, trajectory_mode in specs:
        values = declarations.get(list_name, [])
        if not isinstance(values, list):
            audit.append({
                "edge_id": list_name,
                "edge_type": list_name,
                "valid": False,
                "reasons": ["edge_list_must_be_a_list"],
            })
            continue
        for raw in values:
            reasons: list[str] = []
            if not isinstance(raw, Mapping):
                audit.append({
                    "edge_id": "invalid",
                    "edge_type": list_name,
                    "valid": False,
                    "reasons": ["edge_record_must_be_a_mapping"],
                })
                continue
            edge_id = str(raw.get("edge_id", "invalid"))
            evidence_id = str(raw.get("evidence_id", ""))
            bindings = dict(context_bindings)
            reasons.extend(context_issues)
            if list_name == "direct_edges":
                giver = raw.get("giver_grasp")
                receiver = raw.get("receiver_cell_id")
                if not isinstance(giver, str) or not giver:
                    reasons.append("giver_grasp_missing")
                if not isinstance(receiver, str) or not receiver:
                    reasons.append("receiver_cell_id_missing")
                elif receiver not in receiver_goals:
                    reasons.append("receiver_cell_not_certified")
                bindings.update({
                    "giver_grasp": str(giver),
                    "receiver_cell_id": str(receiver),
                })
            else:
                grasp = raw.get("grasp_id")
                placement = raw.get("placement_id")
                if not isinstance(grasp, str) or not grasp:
                    reasons.append("grasp_id_missing")
                if not isinstance(placement, str) or not placement:
                    reasons.append("placement_id_missing")
                bindings.update({
                    "grasp_id": str(grasp),
                    "placement_id": str(placement),
                })
            verdict = _validate_evidence(
                catalog, evidence_id,
                expected_kind=kind,
                required_checks=checks,
                hard_check_evidence_ids=raw.get("hard_check_evidence_ids"),
                bindings=bindings,
                trajectory_mode=trajectory_mode,
            )
            reasons.extend(verdict.reasons)
            normalized = dict(raw)
            try:
                normalized["cost"] = _metric(raw.get("cost", 0.0), "edge cost")
                normalized["robustness"] = _metric(
                    raw.get("robustness", 0.0), "edge robustness")
            except (TypeError, ValueError) as error:
                reasons.append(str(error))
            normalized["edge_id"] = edge_id
            normalized["evidence_id"] = evidence_id
            normalized["evidence_sha256"] = verdict.sha256
            check_ids = raw.get("hard_check_evidence_ids")
            normalized["hard_check_evidence_ids"] = (
                dict(check_ids) if isinstance(check_ids, Mapping) else {})
            coverage_ids = raw.get("current_class_coverage_evidence_ids")
            normalized["current_class_coverage_evidence_ids"] = (
                dict(coverage_ids) if isinstance(coverage_ids, Mapping) else {})
            is_valid = not reasons
            audit.append({
                "edge_id": edge_id,
                "edge_type": list_name.removesuffix("_edges"),
                "valid": is_valid,
                "evidence": verdict.to_dict(),
                "reasons": sorted(set(reasons)),
            })
            if is_valid:
                if list_name == "direct_edges":
                    valid_direct.append(normalized)
                elif list_name == "place_edges":
                    valid_place.append(normalized)
                else:
                    valid_pick.append(normalized)
    order = lambda edge: str(edge["edge_id"])
    return (
        sorted(valid_direct, key=order),
        sorted(valid_place, key=order),
        sorted(valid_pick, key=order),
        sorted(audit, key=lambda item: (item["edge_type"], item["edge_id"])),
    )


def _direct_plan(
    grasp_ids: Sequence[str],
    class_id: str,
    direct_edges: Sequence[Mapping[str, Any]],
    catalog: Mapping[str, Any],
    current_grasp_domain_sha256: str | None,
    current_class_sha256: str,
    context_bindings: Mapping[str, str],
    coverage_audit: list[dict[str, Any]],
) -> dict[str, Any] | None:
    candidates = []
    for edge in direct_edges:
        if edge["giver_grasp"] not in grasp_ids:
            continue
        coverage = _class_coverage_verdict(
            edge, catalog,
            class_id=class_id,
            current_grasp_domain_sha256=current_grasp_domain_sha256,
            current_class_sha256=current_class_sha256,
            context_bindings=context_bindings,
            operation="direct",
        )
        coverage_audit.append({
            "class_id": class_id,
            "edge_id": edge["edge_id"],
            "operation": "direct",
            **coverage.to_dict(),
        })
        if not coverage.valid:
            continue
        candidates.append({**edge, "class_coverage": coverage})
    if not candidates:
        return None
    edge = min(candidates, key=lambda item: (
        item["cost"], -item["robustness"], item["edge_id"]))
    return {
        "mode": "direct",
        "receiver_cell_id": edge["receiver_cell_id"],
        "total_cost": edge["cost"],
        "bottleneck_robustness": edge["robustness"],
        "steps": [{
            "kind": "handoff",
            "source": edge["giver_grasp"],
            "target": edge["receiver_cell_id"],
            "edge_id": edge["edge_id"],
            "evidence_id": edge["evidence_id"],
            "evidence_sha256": edge["evidence_sha256"],
            "hard_check_evidence_ids": edge["hard_check_evidence_ids"],
            "current_class_coverage_evidence": edge[
                "class_coverage"].to_dict(),
            "cost": edge["cost"],
            "robustness": edge["robustness"],
        }],
    }


def _transfer_plan(
    grasp_ids: Sequence[str],
    class_id: str,
    direct_edges: Sequence[Mapping[str, Any]],
    place_edges: Sequence[Mapping[str, Any]],
    pick_edges: Sequence[Mapping[str, Any]],
    catalog: Mapping[str, Any],
    max_hops: int,
    current_grasp_domain_sha256: str | None,
    current_class_sha256: str,
    context_bindings: Mapping[str, str],
    coverage_audit: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Dijkstra search over directed place--pick transitions."""
    transitions: dict[str, list[dict[str, Any]]] = {}
    for place in place_edges:
        for pick in pick_edges:
            if (place["placement_id"] != pick["placement_id"]
                    or place["grasp_id"] == pick["grasp_id"]):
                continue
            transitions.setdefault(place["grasp_id"], []).append({
                "source": place["grasp_id"],
                "target": pick["grasp_id"],
                "placement_id": place["placement_id"],
                "place": place,
                "pick": pick,
                "cost": place["cost"] + pick["cost"],
                "robustness": min(place["robustness"], pick["robustness"]),
            })
    for values in transitions.values():
        values.sort(key=lambda item: (
            item["cost"], -item["robustness"], item["placement_id"],
            item["target"]))

    direct_by_giver: dict[str, list[Mapping[str, Any]]] = {}
    for edge in direct_edges:
        direct_by_giver.setdefault(edge["giver_grasp"], []).append(edge)
    candidates: list[dict[str, Any]] = []
    # Heap key: cost, hops, negative bottleneck, deterministic signature.
    heap: list[tuple[float, int, float, tuple[str, ...], str, tuple[dict[str, Any], ...], frozenset[str]]] = []
    for grasp in sorted(set(grasp_ids)):
        heapq.heappush(heap, (0.0, 0, -math.inf, (), grasp, (), frozenset({grasp})))
    while heap:
        cost, hops, negative_robustness, signature, grasp, path, visited = heapq.heappop(heap)
        if hops > 0:
            for direct in direct_by_giver.get(grasp, []):
                direct_robustness = direct["robustness"]
                bottleneck = min(-negative_robustness, direct_robustness)
                candidates.append({
                    "path": path,
                    "direct": direct,
                    "cost": cost + direct["cost"],
                    "hops": hops,
                    "robustness": bottleneck,
                    "signature": signature + (direct["edge_id"],),
                })
        if hops >= max_hops:
            continue
        for transition in transitions.get(grasp, []):
            target = transition["target"]
            if target in visited:
                continue
            # Only the first place motion must certify the whole current cell.
            if hops == 0:
                coverage = _class_coverage_verdict(
                    transition["place"], catalog,
                    class_id=class_id,
                    current_grasp_domain_sha256=current_grasp_domain_sha256,
                    current_class_sha256=current_class_sha256,
                    context_bindings=context_bindings,
                    operation="place",
                )
                coverage_audit.append({
                    "class_id": class_id,
                    "edge_id": transition["place"]["edge_id"],
                    "operation": "place",
                    **coverage.to_dict(),
                })
                if not coverage.valid:
                    continue
                transition = {**transition, "class_coverage": coverage}
            robustness = (transition["robustness"] if hops == 0 else
                          min(-negative_robustness, transition["robustness"]))
            edge_signature = (
                transition["place"]["edge_id"],
                transition["pick"]["edge_id"],
            )
            heapq.heappush(heap, (
                cost + transition["cost"],
                hops + 1,
                -robustness,
                signature + edge_signature,
                target,
                path + (transition,),
                visited | {target},
            ))
    if not candidates:
        return None
    selected = min(candidates, key=lambda item: (
        item["cost"], item["hops"], -item["robustness"], item["signature"]))
    steps: list[dict[str, Any]] = []
    for transition in selected["path"]:
        place, pick = transition["place"], transition["pick"]
        steps.extend(({
            "kind": "place",
            "source": transition["source"],
            "target": transition["placement_id"],
            "edge_id": place["edge_id"],
            "evidence_id": place["evidence_id"],
            "evidence_sha256": place["evidence_sha256"],
            "hard_check_evidence_ids": place["hard_check_evidence_ids"],
            "current_class_coverage_evidence": (
                transition.get("class_coverage").to_dict()
                if transition.get("class_coverage") is not None else None),
            "cost": place["cost"],
            "robustness": place["robustness"],
        }, {
            "kind": "regrasp",
            "source": transition["placement_id"],
            "target": transition["target"],
            "edge_id": pick["edge_id"],
            "evidence_id": pick["evidence_id"],
            "evidence_sha256": pick["evidence_sha256"],
            "hard_check_evidence_ids": pick["hard_check_evidence_ids"],
            "cost": pick["cost"],
            "robustness": pick["robustness"],
        }))
    direct = selected["direct"]
    steps.append({
        "kind": "handoff",
        "source": direct["giver_grasp"],
        "target": direct["receiver_cell_id"],
        "edge_id": direct["edge_id"],
        "evidence_id": direct["evidence_id"],
        "evidence_sha256": direct["evidence_sha256"],
        "hard_check_evidence_ids": direct["hard_check_evidence_ids"],
        "cost": direct["cost"],
        "robustness": direct["robustness"],
    })
    return {
        "mode": "transfer_stage",
        "receiver_cell_id": direct["receiver_cell_id"],
        "total_cost": selected["cost"],
        "bottleneck_robustness": selected["robustness"],
        "reorientation_hops": selected["hops"],
        "steps": steps,
    }


def _empty_certificate(
    class_id: str,
    declarations: Mapping[str, Any],
    catalog: Mapping[str, Any],
    *,
    context_bindings: Mapping[str, str],
    context_issues: Sequence[str],
    current_domain_id: str,
    current_grasp_domain_sha256: str | None,
    current_class_sha256: str,
) -> _EvidenceVerdict | None:
    certificates = declarations.get("exhaustiveness_certificates", [])
    if not isinstance(certificates, list):
        return None
    for record in certificates:
        if not isinstance(record, Mapping) or record.get("class_id") != class_id:
            continue
        evidence_id = str(record.get("evidence_id", ""))
        bindings = {
            **context_bindings,
            "class_id": class_id,
            "current_grasp_domain_id": current_domain_id,
            "current_class_id": class_id,
            "current_class_sha256": current_class_sha256,
        }
        if current_grasp_domain_sha256 is not None:
            bindings["current_grasp_domain_sha256"] = (
                current_grasp_domain_sha256)
        else:
            context_issues = tuple(context_issues) + (
                "current_grasp_domain_sha256_missing",)
        verdict = _validate_evidence(
            catalog, evidence_id,
            expected_kind="certified_empty_handoff_preimage",
            required_checks=_EMPTY_CHECKS,
            hard_check_evidence_ids=record.get("hard_check_evidence_ids"),
            bindings=bindings,
            trajectory_mode=None,
        )
        if context_issues:
            return _EvidenceVerdict(
                verdict.evidence_id,
                False,
                tuple(sorted(set(verdict.reasons) | set(context_issues))),
                verdict.sha256,
            )
        return verdict
    return None


def _plan_evidence_ids(plan: Mapping[str, Any]) -> list[str]:
    values: set[str] = set()
    for step in plan.get("steps", []):
        if not isinstance(step, Mapping):
            continue
        evidence_id = step.get("evidence_id")
        if isinstance(evidence_id, str) and evidence_id:
            values.add(evidence_id)
        checks = step.get("hard_check_evidence_ids")
        if isinstance(checks, Mapping):
            values.update(
                value for value in checks.values()
                if isinstance(value, str) and value)
        coverage = step.get("current_class_coverage_evidence")
        if isinstance(coverage, Mapping):
            value = coverage.get("evidence_id")
            if isinstance(value, str) and value:
                values.add(value)
    return sorted(values)


def build_handoff_preimage_set(
    receiver_insertion_set: Mapping[str, Any] | None,
    declarations: Mapping[str, Any],
    *,
    evidence_catalog: Mapping[str, Any] | None = None,
    receiver_source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify current-grasp cells into an evidence-backed preimage set."""
    declarations = _as_mapping(declarations, "preimage declarations")
    if int(declarations.get("schema_version", 0)) != SCHEMA_VERSION:
        raise ValueError("preimage declarations schema_version must be 1")
    task_id = _as_identifier(declarations.get("task_id"), "task_id")
    receiver_pose_id = _as_identifier(
        declarations.get("receiver_pose_id"), "receiver_pose_id")
    max_hops = int(declarations.get("max_reorientation_hops", 1))
    if max_hops < 0:
        raise ValueError("max_reorientation_hops must be non-negative")
    catalog = evidence_catalog or {}
    if not isinstance(catalog, Mapping):
        raise ValueError("evidence_catalog must be a mapping")

    goals, receiver_issues, provisional_ids = _receiver_goals(
        receiver_insertion_set)
    (classes, domain_complete, domain_id, domain_sha256,
     domain_issues) = _current_classes(declarations)
    source = dict(receiver_source or {})
    receiver_sha = source.get("sha256")
    if not _sha256(receiver_sha):
        receiver_sha = None
    context_bindings, context_issues = _required_context_bindings(
        receiver_insertion_set,
        task_id=task_id,
        receiver_pose_id=receiver_pose_id,
        receiver_artifact_sha256=receiver_sha,
    )
    direct_edges, place_edges, pick_edges, edge_audit = _edge_verdicts(
        declarations, catalog, goals,
        context_bindings=context_bindings,
        context_issues=context_issues)

    sets: dict[str, list[dict[str, Any]]] = {
        "direct": [],
        "reorientation": [],
        "uncovered": [],
        "unknown": [],
    }
    class_coverage_audit: list[dict[str, Any]] = []
    for current in classes:
        class_id = current["class_id"]
        grasps = current["representative_grasp_ids"]
        common = {
            "task_id": task_id,
            "receiver_pose_id": receiver_pose_id,
            "class_id": class_id,
            "representative_grasp_ids": grasps,
            "domain": current["domain"],
            "current_grasp_domain_sha256": domain_sha256,
            "current_class_sha256": current["class_content_sha256"],
        }
        direct = _direct_plan(
            grasps, class_id, direct_edges, catalog,
            domain_sha256, current["class_content_sha256"],
            context_bindings, class_coverage_audit)
        transfer = None if direct is not None else _transfer_plan(
            grasps, class_id, direct_edges, place_edges, pick_edges,
            catalog, max_hops, domain_sha256,
            current["class_content_sha256"], context_bindings,
            class_coverage_audit)
        empty = _empty_certificate(
            class_id, declarations, catalog,
            context_bindings=context_bindings,
            context_issues=context_issues,
            current_domain_id=domain_id,
            current_grasp_domain_sha256=domain_sha256,
            current_class_sha256=current["class_content_sha256"],
        )
        if (empty is not None and empty.valid
                and (direct is not None or transfer is not None)):
            sets["unknown"].append({
                **common,
                "status": "UNKNOWN",
                "reason": "conflicting_positive_and_empty_certificates",
                "evidence_ids": sorted(
                    set(_plan_evidence_ids(direct or transfer))
                    | {empty.evidence_id}),
                "missing_inputs": [],
            })
        elif direct is not None:
            sets["direct"].append({
                **common,
                "status": "DIRECT",
                "reason": "validated_dual_arm_handoff_to_certified_receiver",
                "receiver_cell_id": direct["receiver_cell_id"],
                "evidence_ids": _plan_evidence_ids(direct),
                "plan": direct,
                "missing_inputs": [],
            })
        elif transfer is not None:
            sets["reorientation"].append({
                **common,
                "status": "TRANSFER",
                "reason": "validated_place_regrasp_and_handoff_path",
                "receiver_cell_id": transfer["receiver_cell_id"],
                "evidence_ids": _plan_evidence_ids(transfer),
                "plan": transfer,
                "missing_inputs": [],
            })
        elif empty is not None and empty.valid:
            sets["uncovered"].append({
                **common,
                "status": "UNCOVERED",
                "reason": "certified_empty_handoff_preimage",
                "evidence_ids": [empty.evidence_id],
                "exhaustiveness_certificate": empty.to_dict(),
                "missing_inputs": [],
            })
        else:
            missing = list(receiver_issues)
            if not direct_edges:
                missing.append("validated_dual_arm_handoff_trajectory")
            elif any(edge["giver_grasp"] in grasps for edge in direct_edges):
                missing.append(
                    "externally_verified_current_class_direct_coverage")
            if max_hops > 0 and not place_edges:
                missing.append("validated_transfer_place_trajectory")
            elif max_hops > 0 and any(
                    edge["grasp_id"] in grasps for edge in place_edges):
                missing.append(
                    "externally_verified_current_class_place_coverage")
            if max_hops > 0 and not pick_edges:
                missing.append("validated_transfer_pick_trajectory")
            if empty is None:
                missing.append("exhaustive_empty_preimage_certificate")
            elif not empty.valid:
                missing.extend(empty.reasons)
            sets["unknown"].append({
                **common,
                "status": "UNKNOWN",
                "reason": ("no_certified_receiver_goal" if not goals else
                           "no_validated_path_and_no_empty_certificate"),
                "evidence_ids": [],
                "missing_inputs": sorted(set(missing)),
            })

    all_records = sum(sets.values(), [])
    classified_ids = [record["class_id"] for record in all_records]
    positive_count = len(sets["direct"]) + len(sets["reorientation"])
    positive_membership_sound = positive_count > 0
    verified_classification_count = positive_count + len(sets["uncovered"])
    missing_global = sorted(set(
        receiver_issues + domain_issues + list(context_issues)))
    if not catalog:
        missing_global.append("trajectory_evidence_catalog_empty")
    partition_complete = len(classified_ids) == len(classes) == len(set(classified_ids))
    coverage_certified = bool(
        partition_complete
        and domain_complete
        and not sets["unknown"]
        and not missing_global
        and verified_classification_count == len(classes)
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "method": "evidence_backed_direct_first_handoff_preimage",
        "evidence_contract_version": 2,
        "task_id": task_id,
        "receiver_pose_id": receiver_pose_id,
        "required_evidence_context": dict(context_bindings),
        "claim_semantics": {
            "DIRECT": "validated synchronized dual-arm trajectory to an individually certified receiver cell",
            "TRANSFER": "validated place, regrasp, and dual-arm handoff trajectories",
            "UNCOVERED": "explicit certified-empty preimage over the declared domains",
            "UNKNOWN": "insufficient or conflicting evidence; absence of an edge is not infeasibility",
        },
        "evidence_contract": {
            "trajectory_and_hard_check_binding": [
                "subject_evidence_id",
                "subject_evidence_sha256",
            ],
            "current_class_binding": [
                "current_grasp_domain_sha256",
                "current_class_sha256",
                "current_class_id",
            ],
            "initial_edge_whole_class_evidence": (
                "current_class_coverage_evidence_ids[class_id] -> pinned "
                "current_class_coverage_record"),
            "receiver_cell_certificate": (
                "layer-2 cell.certification external certificate identity "
                "plus matching cell proof"),
        },
        "receiver_insertion_set": {
            **source,
            "certified_receiver_cell_ids": sorted(goals),
            "provisional_path_witness_cell_ids": provisional_ids,
            "issues": receiver_issues,
        },
        "current_grasp_domain": {
            "id": domain_id,
            "sha256": domain_sha256,
            "complete": domain_complete,
            "class_count": len(classes),
            "issues": domain_issues,
        },
        "sets": sets,
        "edge_evidence_audit": edge_audit,
        "current_class_coverage_audit": sorted(
            class_coverage_audit,
            key=lambda item: (
                item["class_id"], item["operation"], item["edge_id"])),
        "summary": {
            "class_count": len(classes),
            "direct_count": len(sets["direct"]),
            "reorientation_count": len(sets["reorientation"]),
            "uncovered_count": len(sets["uncovered"]),
            "unknown_count": len(sets["unknown"]),
            "positive_preimage_count": positive_count,
        },
        "certification": {
            "positive_membership_sound": positive_membership_sound,
            "verified_classification_count": verified_classification_count,
            "partition_complete": partition_complete,
            "coverage_certified": coverage_certified,
            "missing_inputs": sorted(set(missing_global)),
            "limitations": [
                "Provisional receiver witnesses are diagnostic only.",
                "No nearest-neighbor or lookup-table interpolation creates an edge.",
                "Every positive edge requires an explicitly pinned trajectory artifact and independently pinned hard-check records.",
                "UNCOVERED requires an exhaustive certificate; failed or absent search remains UNKNOWN.",
            ],
        },
    }


__all__ = [
    "ARTIFACT_TYPE",
    "SCHEMA_VERSION",
    "build_handoff_preimage_set",
]
