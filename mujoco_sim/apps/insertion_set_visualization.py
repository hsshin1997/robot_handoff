"""Build an interactive, CAD-backed view of the insertion pose sets.

The visualization is deliberately a view of generated evidence, not another
feasibility checker.  It shows every constructive cell center in the finite
task-space cover, decorates the subset evaluated by the robot layer, and
reports the handoff-preimage classification.  The full continuous cells remain
available in the source JSON artifacts.

All geometry is converted to millimetres for display only.  Transform inputs
and generated planning artifacts keep the repository-wide SI convention.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from ..modeling.insertion_grasps import load_scaled_binary_stl
from ..modeling.insertion_task_set import artifact_sha256
from ..offline_tools.artifacts import fingerprint_content
from ..planner.handoff_preimage_set import _receiver_cell_certificate_valid


PROJECT_RELATIVE = Path("projects/connector_header_insertion")
DEFAULT_FRAGMENT_RELATIVE = Path(
    ".codex/visualizations/2026/07/21/connector-header-feasible-set/"
    "insertion-feasible-set.html"
)
DEFAULT_STANDALONE_RELATIVE = (
    PROJECT_RELATIVE / "generated/visualization/insertion-feasible-set.html"
)


def _load_yaml(path: Path) -> Mapping[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def _load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_declared_files(root: Path, value: Any, checks: list[str]) -> None:
    """Recursively verify every declared path/file-digest pair."""
    if isinstance(value, Mapping):
        path_value = value.get("path")
        digest = value.get("file_sha256", value.get("sha256"))
        if isinstance(path_value, str) and isinstance(digest, str):
            source = _resolve_repo_path(root, path_value)
            if not source.is_file():
                raise ValueError(f"declared provenance file is missing: {source}")
            actual = _sha256_file(source)
            if actual != digest:
                raise ValueError(
                    f"stale provenance for {source}: expected {digest}, got {actual}"
                )
            checks.append(f"file:{path_value}")
        for child in value.values():
            _validate_declared_files(root, child, checks)
    elif isinstance(value, list):
        for child in value:
            _validate_declared_files(root, child, checks)


def _identifier_set(value: Any, *, label: str) -> set[str]:
    if (
        not isinstance(value, list)
        or not all(isinstance(item, str) and item for item in value)
        or len(value) != len(set(value))
    ):
        raise ValueError(f"{label} must be a duplicate-free string list")
    return set(value)


def _digest_string(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _validate_robot_partition(
    root: Path,
    *,
    task_set: Mapping[str, Any],
    robot_set: Mapping[str, Any],
    checks: list[str],
) -> None:
    """Reject inconsistent layer-2 status overlays before assigning colors."""
    task_cells_raw = task_set.get("cells")
    robot_cells_raw = robot_set.get("cells")
    if not isinstance(task_cells_raw, list) or not isinstance(robot_cells_raw, list):
        raise ValueError("task and robot cells must be lists")
    task_cells = {
        str(cell.get("id")): cell
        for cell in task_cells_raw
        if isinstance(cell, Mapping) and isinstance(cell.get("id"), str)
    }
    if len(task_cells) != len(task_cells_raw):
        raise ValueError("layer-1 cells have missing or duplicate IDs")
    robot_cells = {
        str(cell.get("id")): cell
        for cell in robot_cells_raw
        if isinstance(cell, Mapping) and isinstance(cell.get("id"), str)
    }
    if len(robot_cells) != len(robot_cells_raw):
        raise ValueError("layer-2 cells have missing or duplicate IDs")

    root_keys = {
        "CERTIFIED_SAFE": "certified_receiver_cell_ids",
        "PROVISIONAL_CENTER_PATH_WITNESS": (
            "provisional_center_path_witness_cell_ids"
        ),
    }
    no_witness = {
        "NO_WITNESS_AT_PREINSERT",
        "NO_WITNESS_CONTINUATION",
        "CENTER_PATH_NUMERIC_MARGIN_NOT_MET",
    }
    expected: dict[str, set[str]] = {
        classification: set() for classification in root_keys
    }
    expected["NUMERICALLY_UNRESOLVED"] = set()
    for cell_id, cell in robot_cells.items():
        source = task_cells.get(cell_id)
        if source is None:
            raise ValueError(f"layer-2 cell is absent from layer 1: {cell_id}")
        if cell.get("source_classification") != source.get("classification"):
            raise ValueError(f"layer-2 source classification mismatch: {cell_id}")
        if cell.get("center_pose") != source.get("center_pose"):
            raise ValueError(f"layer-2 center pose mismatch: {cell_id}")
        classification = cell.get("robot_classification")
        if classification in root_keys:
            expected[str(classification)].add(cell_id)
        elif classification in no_witness:
            expected["NUMERICALLY_UNRESOLVED"].add(cell_id)
        else:
            raise ValueError(
                f"unknown layer-2 robot classification for {cell_id}: "
                f"{classification}"
            )
        certified = classification == "CERTIFIED_SAFE"
        if cell.get("certified") is not certified:
            raise ValueError(f"layer-2 certified flag mismatch: {cell_id}")
        complete_count = cell.get("complete_discrete_branch_count")
        branch_count = cell.get("accepted_provisional_branch_count")
        branch_ids = cell.get("accepted_provisional_branch_ids")
        if (
            isinstance(complete_count, bool)
            or not isinstance(complete_count, int)
            or complete_count < 0
            or isinstance(branch_count, bool)
            or not isinstance(branch_count, int)
            or branch_count < 0
            or branch_count > complete_count
            or not isinstance(branch_ids, list)
            or len(branch_ids) != branch_count
            or not all(isinstance(item, str) and item for item in branch_ids)
            or len(branch_ids) != len(set(branch_ids))
        ):
            raise ValueError(f"layer-2 branch accounting mismatch: {cell_id}")
        if classification == "PROVISIONAL_CENTER_PATH_WITNESS" and branch_count == 0:
            raise ValueError(f"layer-2 witness has no complete branch: {cell_id}")
        if classification in no_witness and branch_count != 0:
            raise ValueError(f"layer-2 no-witness cell has a complete branch: {cell_id}")
        if certified and not _receiver_cell_certificate_valid(
            robot_set, cell, cell_id
        ):
            raise ValueError(
                f"layer-2 certified cell lacks exact certificate evidence: {cell_id}"
            )

    declared: list[set[str]] = []
    for classification, key in root_keys.items():
        values = _identifier_set(robot_set.get(key), label=f"layer-2 {key}")
        if values != expected[classification]:
            raise ValueError(f"layer-2 {key} is inconsistent with cells")
        declared.append(values)
    numeric_values = _identifier_set(
        robot_set.get("numerically_unresolved_cell_ids"),
        label="layer-2 numerically_unresolved_cell_ids",
    )
    if numeric_values != expected["NUMERICALLY_UNRESOLVED"]:
        raise ValueError("layer-2 numerically unresolved IDs are inconsistent")
    declared.append(numeric_values)
    if any(left & right for index, left in enumerate(declared)
           for right in declared[index + 1:]):
        raise ValueError("layer-2 result ID lists overlap")

    selection = robot_set.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("layer-2 selection metadata is missing")
    source_classes = selection.get("source_classifications")
    if not isinstance(source_classes, list):
        raise ValueError("layer-2 source classifications are missing")
    eligible_ids = {
        cell_id for cell_id, cell in task_cells.items()
        if cell.get("classification") in set(source_classes)
    }
    not_evaluated = _identifier_set(
        robot_set.get("not_evaluated_cell_ids"),
        label="layer-2 not_evaluated_cell_ids",
    )
    if set(robot_cells) & not_evaluated:
        raise ValueError("layer-2 evaluated and not-evaluated IDs overlap")
    if set(robot_cells) | not_evaluated != eligible_ids:
        raise ValueError("layer-2 evaluated/not-evaluated partition is incomplete")

    summary = robot_set.get("summary")
    if not isinstance(summary, Mapping):
        raise ValueError("layer-2 summary is missing")
    expected_summary = {
        "source_cell_count": len(task_cells),
        "eligible_source_cell_count": len(eligible_ids),
        "provisional_center_path_witness_count": len(
            expected["PROVISIONAL_CENTER_PATH_WITNESS"]
        ),
        "numerically_unresolved_count": len(expected["NUMERICALLY_UNRESOLVED"]),
        "not_evaluated_eligible_count": len(not_evaluated),
        "certified_receiver_cell_count": len(expected["CERTIFIED_SAFE"]),
    }
    for key, value in expected_summary.items():
        if summary.get(key) != value:
            raise ValueError(f"layer-2 summary mismatch: {key}")
    root_certification = robot_set.get("certification")
    if (
        not isinstance(root_certification, Mapping)
        or set(root_certification.get("certified_receiver_cell_ids", []))
        != expected["CERTIFIED_SAFE"]
        or robot_set.get("certified") is not bool(expected["CERTIFIED_SAFE"])
    ):
        raise ValueError("layer-2 root certification state is inconsistent")
    certificate = robot_set.get("continuous_robot_cell_certificate")
    if expected["CERTIFIED_SAFE"]:
        if not isinstance(certificate, Mapping) or certificate.get("supplied") is not True:
            raise ValueError("layer-2 certified cells lack a root certificate")
        _validate_declared_files(root, certificate, checks)
    checks.append("layer2:internal_cell_partition")


def _validate_handoff_partition(
    *,
    robot_set: Mapping[str, Any],
    handoff_set: Mapping[str, Any],
    checks: list[str],
) -> None:
    sets = handoff_set.get("sets")
    if not isinstance(sets, Mapping):
        raise ValueError("layer-3 sets mapping is missing")
    status_by_key = {
        "direct": "DIRECT",
        "reorientation": "TRANSFER",
        "uncovered": "UNCOVERED",
        "unknown": "UNKNOWN",
    }
    class_ids: list[str] = []
    counts: dict[str, int] = {}
    for key, status in status_by_key.items():
        records = sets.get(key)
        if not isinstance(records, list):
            raise ValueError(f"layer-3 {key} set must be a list")
        counts[key] = len(records)
        for record in records:
            if (
                not isinstance(record, Mapping)
                or record.get("status") != status
                or not isinstance(record.get("class_id"), str)
            ):
                raise ValueError(f"layer-3 {key} record is malformed")
            class_ids.append(str(record["class_id"]))
    if len(class_ids) != len(set(class_ids)):
        raise ValueError("layer-3 current-grasp classes overlap")
    domain = handoff_set.get("current_grasp_domain")
    if not isinstance(domain, Mapping) or domain.get("class_count") != len(class_ids):
        raise ValueError("layer-3 current-domain partition is incomplete")
    summary = handoff_set.get("summary")
    if not isinstance(summary, Mapping):
        raise ValueError("layer-3 summary is missing")
    expected_summary = {
        "class_count": len(class_ids),
        "direct_count": counts["direct"],
        "reorientation_count": counts["reorientation"],
        "uncovered_count": counts["uncovered"],
        "unknown_count": counts["unknown"],
        "positive_preimage_count": counts["direct"] + counts["reorientation"],
    }
    for key, value in expected_summary.items():
        if summary.get(key) != value:
            raise ValueError(f"layer-3 summary mismatch: {key}")
    receiver = handoff_set.get("receiver_insertion_set")
    if not isinstance(receiver, Mapping):
        raise ValueError("layer-3 receiver metadata is missing")
    receiver_certified = _identifier_set(
        receiver.get("certified_receiver_cell_ids"),
        label="layer-3 certified receiver IDs",
    )
    robot_certified = _identifier_set(
        robot_set.get("certified_receiver_cell_ids"),
        label="layer-2 certified receiver IDs",
    )
    if receiver_certified != robot_certified:
        raise ValueError("layer-3 certified receiver IDs do not match layer 2")
    certification = handoff_set.get("certification")
    if not isinstance(certification, Mapping):
        raise ValueError("layer-3 certification metadata is missing")
    if handoff_set.get("evidence_contract_version") != 2:
        raise ValueError("layer-3 evidence contract version is not supported")
    domain_sha = domain.get("sha256")
    if not _digest_string(domain_sha):
        raise ValueError("layer-3 current-domain digest is invalid")

    edge_audit_raw = handoff_set.get("edge_evidence_audit")
    coverage_audit_raw = handoff_set.get("current_class_coverage_audit")
    if not isinstance(edge_audit_raw, list) or not isinstance(coverage_audit_raw, list):
        raise ValueError("layer-3 evidence audits are missing")
    edge_audit: dict[str, Mapping[str, Any]] = {}
    for item in edge_audit_raw:
        if not isinstance(item, Mapping) or not isinstance(item.get("edge_id"), str):
            raise ValueError("layer-3 edge audit record is malformed")
        if item["edge_id"] in edge_audit:
            raise ValueError("layer-3 edge audit IDs are duplicated")
        edge_audit[str(item["edge_id"])] = item

    positives = list(sets["direct"]) + list(sets["reorientation"])
    for record in positives:
        class_id = str(record["class_id"])
        if (
            record.get("task_id") != handoff_set.get("task_id")
            or record.get("receiver_pose_id") != handoff_set.get("receiver_pose_id")
            or record.get("current_grasp_domain_sha256") != domain_sha
            or record.get("receiver_cell_id") not in receiver_certified
        ):
            raise ValueError(f"layer-3 positive context mismatch: {class_id}")
        class_content = {
            "class_id": class_id,
            "representative_grasp_ids": record.get("representative_grasp_ids"),
            "domain": record.get("domain"),
        }
        if record.get("current_class_sha256") != fingerprint_content(class_content):
            raise ValueError(f"layer-3 current-class digest mismatch: {class_id}")
        evidence_ids = _identifier_set(
            record.get("evidence_ids"),
            label=f"layer-3 positive evidence IDs for {class_id}",
        )
        if not evidence_ids or record.get("missing_inputs") != []:
            raise ValueError(f"layer-3 positive evidence is incomplete: {class_id}")
        plan = record.get("plan")
        expected_mode = "direct" if record.get("status") == "DIRECT" else "transfer_stage"
        if (
            not isinstance(plan, Mapping)
            or plan.get("mode") != expected_mode
            or plan.get("receiver_cell_id") != record.get("receiver_cell_id")
            or not isinstance(plan.get("steps"), list)
            or not plan["steps"]
        ):
            raise ValueError(f"layer-3 positive plan is malformed: {class_id}")
        coverage_count = 0
        for step in plan["steps"]:
            if not isinstance(step, Mapping):
                raise ValueError(f"layer-3 plan step is malformed: {class_id}")
            evidence_id = step.get("evidence_id")
            evidence_sha = step.get("evidence_sha256")
            checks_for_step = step.get("hard_check_evidence_ids")
            if (
                not isinstance(evidence_id, str)
                or evidence_id not in evidence_ids
                or not _digest_string(evidence_sha)
                or not isinstance(checks_for_step, Mapping)
                or not checks_for_step
                or not all(
                    isinstance(value, str) and value in evidence_ids
                    for value in checks_for_step.values()
                )
            ):
                raise ValueError(f"layer-3 step evidence is incomplete: {class_id}")
            audit = edge_audit.get(str(step.get("edge_id")))
            verdict = audit.get("evidence") if isinstance(audit, Mapping) else None
            if (
                not isinstance(audit, Mapping)
                or audit.get("valid") is not True
                or audit.get("reasons") != []
                or not isinstance(verdict, Mapping)
                or verdict.get("valid") is not True
                or verdict.get("evidence_id") != evidence_id
                or verdict.get("sha256") != evidence_sha
            ):
                raise ValueError(f"layer-3 step audit is inconsistent: {class_id}")
            coverage = step.get("current_class_coverage_evidence")
            if coverage is not None:
                coverage_count += 1
                if (
                    not isinstance(coverage, Mapping)
                    or coverage.get("valid") is not True
                    or coverage.get("reasons") != []
                    or coverage.get("evidence_id") not in evidence_ids
                    or not _digest_string(coverage.get("sha256"))
                ):
                    raise ValueError(
                        f"layer-3 class-coverage evidence is invalid: {class_id}"
                    )
                matching = [
                    item for item in coverage_audit_raw
                    if isinstance(item, Mapping)
                    and item.get("class_id") == class_id
                    and item.get("edge_id") == step.get("edge_id")
                    and item.get("evidence_id") == coverage.get("evidence_id")
                    and item.get("sha256") == coverage.get("sha256")
                    and item.get("valid") is True
                    and item.get("reasons") == []
                ]
                if len(matching) != 1:
                    raise ValueError(
                        f"layer-3 class-coverage audit is inconsistent: {class_id}"
                    )
        if coverage_count != 1:
            raise ValueError(
                f"layer-3 positive plan lacks one whole-class proof: {class_id}"
            )

    for record in sets["uncovered"]:
        class_id = str(record["class_id"])
        evidence_ids = _identifier_set(
            record.get("evidence_ids"),
            label=f"layer-3 empty-preimage evidence IDs for {class_id}",
        )
        verdict = record.get("exhaustiveness_certificate")
        if (
            record.get("current_grasp_domain_sha256") != domain_sha
            or not _digest_string(record.get("current_class_sha256"))
            or not isinstance(verdict, Mapping)
            or verdict.get("valid") is not True
            or verdict.get("reasons") != []
            or verdict.get("evidence_id") not in evidence_ids
            or not _digest_string(verdict.get("sha256"))
        ):
            raise ValueError(f"layer-3 empty-preimage evidence is invalid: {class_id}")

    positive_count = expected_summary["positive_preimage_count"]
    verified_count = positive_count + counts["uncovered"]
    if certification.get("positive_membership_sound") is not bool(positive_count):
        raise ValueError("layer-3 positive-membership soundness flag is inconsistent")
    if certification.get("verified_classification_count") != verified_count:
        raise ValueError("layer-3 verified-classification count is inconsistent")
    if certification.get("partition_complete") is not True:
        raise ValueError("layer-3 partition-complete flag is inconsistent")
    missing_inputs = certification.get("missing_inputs")
    if not isinstance(missing_inputs, list):
        raise ValueError("layer-3 missing-input list is malformed")
    expected_coverage = bool(
        domain.get("complete") is True
        and counts["unknown"] == 0
        and verified_count == len(class_ids)
        and not missing_inputs
    )
    if certification.get("coverage_certified") is not expected_coverage:
        raise ValueError("layer-3 coverage-certified flag is inconsistent")
    checks.append("layer3:internal_class_partition")


def _validate_artifact_chain(
    root: Path,
    *,
    task_set: Mapping[str, Any],
    task_set_path: Path,
    robot_set: Mapping[str, Any] | None,
    robot_set_path: Path,
    handoff_set: Mapping[str, Any] | None,
    socket_config: Mapping[str, Any],
) -> list[str]:
    checks: list[str] = []
    if task_set.get("artifact_type") != "robot_independent_insertion_task_set":
        raise ValueError("visualization requires a layer-1 task-set artifact")
    if int(task_set.get("schema_version", 0)) != 1:
        raise ValueError("unsupported layer-1 schema_version")
    semantic = task_set.get("semantic_sha256")
    if not isinstance(semantic, str) or artifact_sha256(task_set) != semantic:
        raise ValueError("layer-1 semantic_sha256 is missing or invalid")
    checks.append("layer1:semantic_sha256")
    _validate_declared_files(root, task_set.get("inputs", {}), checks)

    task_socket = np.asarray(
        task_set.get("insertion_trajectory", {}).get("T_B_P_insert"), dtype=float
    )
    configured_socket = np.asarray(socket_config.get("T_B_P_insert"), dtype=float)
    if (
        task_socket.shape != (4, 4)
        or configured_socket.shape != (4, 4)
        or not np.allclose(task_socket, configured_socket, atol=1e-12, rtol=0.0)
    ):
        raise ValueError("layer-1 and pcb_socket T_B_P_insert do not match")
    checks.append("layer1:socket_transform")

    task_raw_sha = _sha256_file(task_set_path)
    if robot_set is not None:
        if robot_set.get("artifact_type") != "robot_conditioned_insertion_path_set":
            raise ValueError("visualization requires a layer-2 robot-set artifact")
        if int(robot_set.get("schema_version", 0)) != 1:
            raise ValueError("unsupported layer-2 schema_version")
        source = robot_set.get("source_task_set", {})
        if source.get("file_sha256", source.get("sha256")) != task_raw_sha:
            raise ValueError("robot artifact is not bound to the selected task artifact")
        if source.get("artifact_type") != task_set.get("artifact_type"):
            raise ValueError("robot/task artifact types do not match")
        if source.get("semantic_sha256") != task_set.get("semantic_sha256"):
            raise ValueError("robot/task semantic digests do not match")
        if source.get("project_id") != task_set.get("project_id"):
            raise ValueError("robot/task project identities do not match")
        task_binding = task_set.get("whole_cell_task_certificates", {}).get(
            "base_artifact_certificate_binding_sha256"
        )
        if source.get("task_certificate_binding_sha256") != task_binding:
            raise ValueError("robot/task certificate bindings do not match")
        checks.append("layer2:source_task_set_sha256")
        _validate_robot_partition(
            root, task_set=task_set, robot_set=robot_set, checks=checks
        )
    if handoff_set is not None:
        if handoff_set.get("artifact_type") != "handoff_preimage_set":
            raise ValueError("visualization requires a layer-3 handoff-preimage artifact")
        if int(handoff_set.get("schema_version", 0)) != 1:
            raise ValueError("unsupported layer-3 schema_version")
        receiver = handoff_set.get("receiver_insertion_set", {})
        robot_raw_sha = _sha256_file(robot_set_path) if robot_set is not None else None
        if robot_raw_sha is None or receiver.get("sha256") != robot_raw_sha:
            raise ValueError("handoff artifact is not bound to the selected robot artifact")
        if receiver.get("status") != "LOADED":
            raise ValueError("handoff receiver artifact was not loaded successfully")
        checks.append("layer3:receiver_artifact_sha256")
        _validate_handoff_partition(
            robot_set=robot_set, handoff_set=handoff_set, checks=checks
        )
    return checks


def _matrix_mm(value: Any, *, label: str) -> list[list[float]]:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{label} must be a finite 4x4 matrix")
    result = matrix.copy()
    result[:3, 3] *= 1000.0
    return np.round(result, 6).tolist()


def _resolve_repo_path(root: Path, value: str | Path) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else root / candidate


def _surface_points_mm(
    path: Path,
    *,
    scale_to_m: float,
    count: int,
    seed: int,
) -> list[list[float]]:
    """Return a deterministic area-weighted surface sample from a binary STL."""
    mesh = load_scaled_binary_stl(path, scale_to_m=scale_to_m)
    usable = mesh.areas > 0.0
    if not np.any(usable):
        raise ValueError(f"{path} has no positive-area triangles")
    triangles = mesh.triangles[usable]
    weights = mesh.areas[usable]
    weights = weights / np.sum(weights)
    generator = np.random.default_rng(seed)
    indices = generator.choice(len(triangles), size=count, replace=True, p=weights)
    selected = triangles[indices]
    r1 = generator.random(count)
    r2 = generator.random(count)
    root_r1 = np.sqrt(r1)
    barycentric = np.column_stack(
        (1.0 - root_r1, root_r1 * (1.0 - r2), root_r1 * r2)
    )
    points_m = np.einsum("ni,nij->nj", barycentric, selected)
    return np.round(points_m * 1000.0, 3).tolist()


def _task_pose_records(task_set: Mapping[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    cells = task_set.get("cells", [])
    if not isinstance(cells, Sequence):
        raise ValueError("task-set cells must be a sequence")
    for cell in cells:
        if not isinstance(cell, Mapping):
            continue
        center_pose = cell.get("center_pose")
        sampled = cell.get("representative")
        if not isinstance(sampled, Mapping):
            sampled = {}
        pose_source = "constructive_cell_center"
        if not isinstance(center_pose, Mapping):
            # Compatibility with the first layer-1 artifact revision.
            center_pose = sampled
            pose_source = "sampled_representative_fallback"
        transform = center_pose.get("T_P_E")
        if transform is None:
            continue
        bounds = cell.get("bounds") if isinstance(cell.get("bounds"), Mapping) else {}
        claims = [
            str(item.get("claim"))
            for item in cell.get("witnesses", [])
            if isinstance(item, Mapping) and item.get("claim") is not None
        ]
        record = {
            "id": str(cell.get("id", sampled.get("seed_grasp_id", "pose"))),
            "seed_grasp_id": str(sampled.get("seed_grasp_id", "")),
            "classification": str(cell.get("classification", "UNRESOLVED")),
            "classification_reason": str(cell.get("classification_reason", "")),
            "contact_mode": str(cell.get("contact_mode", "unknown")),
            "T_P_E": _matrix_mm(transform, label="task cell-center T_P_E"),
            "required_aperture_mm": round(
                1000.0 * float(center_pose.get("required_aperture_m", 0.0)), 4
            ),
            "quality": (
                None if sampled.get("quality") is None
                else round(float(sampled["quality"]), 6)
            ),
            "pose_source": pose_source,
            "has_sample_witness": bool(sampled),
            "sample_status": str(sampled.get("seed_status", "NONE")),
            "sample_witness_claims": claims,
            "bounds": {
                "u_mm": [round(1000.0 * float(item), 3) for item in bounds.get("u_P_m", [])],
                "v_mm": [round(1000.0 * float(item), 3) for item in bounds.get("v_P_m", [])],
                "roll_deg": [round(np.degrees(float(item)), 2) for item in bounds.get("roll_rad", [])],
            },
            "robot_classification": "NOT_EVALUATED",
            "robot_branch_count": 0,
            "handoff_status": "NOT_EVALUATED",
        }
        records.append(record)
    return records


def _phase1_pose_records(library: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Compatibility fallback used before a layer-1 artifact is generated."""
    records: list[dict[str, Any]] = []
    for candidate in library.get("candidates", []):
        if not isinstance(candidate, Mapping) or not candidate.get("preinsert_compatible"):
            continue
        records.append(
            {
                "id": str(candidate.get("id", "pose")),
                "seed_grasp_id": str(candidate.get("id", "")),
                "classification": "UNRESOLVED",
                "classification_reason": "PHASE1_PREINSERT_SEED_ONLY",
                "contact_mode": str(candidate.get("family", "unknown")),
                "T_P_E": _matrix_mm(candidate["T_P_E"], label="phase-1 T_P_E"),
                "required_aperture_mm": round(
                    1000.0 * float(candidate.get("required_aperture_m", 0.0)), 4
                ),
                "quality": round(float(candidate.get("quality", 0.0)), 6),
                "pose_source": "phase1_sample",
                "has_sample_witness": True,
                "sample_status": str(candidate.get("status", "UNKNOWN")),
                "sample_witness_claims": ["PHASE1_PREINSERT_SEED_ONLY"],
                "bounds": {"u_mm": [], "v_mm": [], "roll_deg": []},
                "robot_classification": "NOT_EVALUATED",
                "robot_branch_count": 0,
                "handoff_status": "NOT_EVALUATED",
            }
        )
    return records


def _decorate_robot(
    records: list[dict[str, Any]],
    robot_set: Mapping[str, Any] | None,
) -> None:
    if not robot_set:
        return
    robot_cells = {
        str(cell.get("id")): cell
        for cell in robot_set.get("cells", [])
        if isinstance(cell, Mapping) and cell.get("id") is not None
    }
    for record in records:
        cell = robot_cells.get(record["id"])
        if cell is None:
            continue
        record["robot_classification"] = str(
            cell.get("robot_classification", "EVALUATED_UNKNOWN")
        )
        record["robot_branch_count"] = int(
            cell.get(
                "accepted_provisional_branch_count",
                cell.get("complete_discrete_branch_count", 0),
            )
        )
        record["robot_certified"] = bool(cell.get("certified", False))


def _decorate_handoff(
    records: list[dict[str, Any]],
    handoff_set: Mapping[str, Any] | None,
) -> None:
    if not handoff_set:
        return
    sets = handoff_set.get("sets", {})
    if not isinstance(sets, Mapping):
        return
    active = []
    for set_name in ("direct", "reorientation", "uncovered", "unknown"):
        if isinstance(sets.get(set_name), list) and sets[set_name]:
            active.append("TRANSFER" if set_name == "reorientation" else set_name.upper())
    global_status = "+".join(active) if active else "EMPTY_DOMAIN"
    # Layer 3 classifies current/donor grasps, not receiver task cells.  Do not
    # incorrectly join donor representative IDs onto receiver cell IDs.
    for record in records:
        record["handoff_status"] = f"GLOBAL_{global_status}"


def _status_counts(records: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        value = str(record.get(key, "UNKNOWN"))
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def build_visualization_data(
    root: Path,
    *,
    task_set_path: Path,
    robot_set_path: Path,
    handoff_set_path: Path,
    point_budget: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    project = root / PROJECT_RELATIVE
    grasp_config = _load_yaml(project / "config/grasp_generation.yaml")
    socket_config = _load_yaml(project / "config/pcb_socket.yaml")
    budgets = {
        "board": 5200,
        "connector": 2800,
        "body": 4200,
        "finger": 2600,
    }
    if point_budget:
        budgets.update({key: int(value) for key, value in point_budget.items()})

    task_set_document: Mapping[str, Any] | None = None
    if task_set_path.exists():
        task_set_document = _load_json(task_set_path)
        records = _task_pose_records(task_set_document)
        source_name = "layer_1_constructive_cell_centers"
        task_counts = task_set_document.get("counts", {})
        trajectory = task_set_document.get("insertion_trajectory", {})
        preinsert_distance_m = float(trajectory.get("preinsert_distance_m", 0.04))
    else:
        library_path = project / "generated/grasps/phase1_pose_library.json"
        library = _load_json(library_path)
        records = _phase1_pose_records(library)
        source_name = "phase1_preinsert_seed_fallback"
        task_counts = {
            "cells": len(records), "safe": 0, "rejected": 0,
            "unresolved": len(records),
        }
        preinsert_distance_m = float(
            library.get("task_geometry", {}).get("preinsert_distance_m", 0.04)
        )
    if not records:
        raise ValueError("no representative insertion poses are available to visualize")

    robot_set = _load_json(robot_set_path) if robot_set_path.exists() else None
    handoff_set = _load_json(handoff_set_path) if handoff_set_path.exists() else None
    if task_set_document is None and (robot_set is not None or handoff_set is not None):
        raise ValueError(
            "phase-1 fallback cannot be overlaid with layer-2 or layer-3 artifacts"
        )
    provenance_checks = (
        _validate_artifact_chain(
            root,
            task_set=task_set_document,
            task_set_path=task_set_path,
            robot_set=robot_set,
            robot_set_path=robot_set_path,
            handoff_set=handoff_set,
            socket_config=socket_config,
        )
        if task_set_document is not None
        else ["phase1_fallback:no_cross_layer_overlay_guarantee"]
    )
    # The displayed CAD is independently checked against the same authored
    # contracts used to register it; a valid generated artifact must not mask
    # a later replacement of an STL on disk.
    _validate_declared_files(root, socket_config.get("assets", {}), provenance_checks)
    _validate_declared_files(root, grasp_config, provenance_checks)
    _decorate_robot(records, robot_set)
    _decorate_handoff(records, handoff_set)

    assets = socket_config["assets"]
    pcb_asset = assets["pcb"]
    connector_asset = assets["connector"]
    stl_scale = float(pcb_asset["scale_to_m"])
    board_points = _surface_points_mm(
        _resolve_repo_path(root, pcb_asset["path"]),
        scale_to_m=stl_scale,
        count=budgets["board"],
        seed=2026072101,
    )
    connector_points = _surface_points_mm(
        _resolve_repo_path(root, connector_asset["path"]),
        scale_to_m=float(connector_asset["scale_to_m"]),
        count=budgets["connector"],
        seed=2026072102,
    )

    gripper_config = grasp_config["gripper"]
    components = []
    for index, component in enumerate(gripper_config["components"]):
        count = budgets["body"] if component["name"] == "body" else budgets["finger"]
        components.append(
            {
                "name": str(component["name"]),
                "points_C_mm": _surface_points_mm(
                    _resolve_repo_path(root, component["asset"]["path"]),
                    scale_to_m=float(grasp_config["assets"]["stl_scale_to_m"]),
                    count=count,
                    seed=2026072110 + index,
                ),
                "T_G_C_reference": _matrix_mm(
                    component["T_G_C_reference"], label="T_G_C_reference"
                ),
                "aperture_multiplier": float(component["aperture_multiplier"]),
            }
        )

    board_bounds = pcb_asset["bounds_min_m"], pcb_asset["bounds_max_m"]
    top_z_mm = 1000.0 * float(socket_config["board"]["nominal_top_surface_z_B_m"])
    board_outline = [
        [1000.0 * board_bounds[0][0], 1000.0 * board_bounds[0][1], top_z_mm],
        [1000.0 * board_bounds[1][0], 1000.0 * board_bounds[0][1], top_z_mm],
        [1000.0 * board_bounds[1][0], 1000.0 * board_bounds[1][1], top_z_mm],
        [1000.0 * board_bounds[0][0], 1000.0 * board_bounds[1][1], top_z_mm],
    ]
    insertion_axis_P = np.asarray(
        grasp_config["task"]["insertion_axis_P"], dtype=float
    )
    T_B_P = np.asarray(socket_config["T_B_P_insert"], dtype=float)
    insertion_axis_B = T_B_P[:3, :3] @ insertion_axis_P

    handoff_summary = handoff_set.get("summary", {}) if handoff_set else {}
    robot_summary = robot_set.get("summary", {}) if robot_set else {}
    default_pose_index = next(
        (
            index for index, record in enumerate(records)
            if record.get("robot_classification") in (
                "PROVISIONAL_PATH_WITNESS",
                "PROVISIONAL_CENTER_PATH_WITNESS",
                "CERTIFIED_SAFE",
            )
        ),
        next(
            (
                index for index, record in enumerate(records)
                if record.get("classification") in ("SAFE", "UNRESOLVED")
            ),
            0,
        ),
    )
    if handoff_set is not None:
        provenance_label = "layers 1–3 chain verified"
        provenance_claim = (
            "Layer-1 source files and the layer-1 → layer-2 → layer-3 artifact "
            "hash chain were verified before rendering."
        )
    elif robot_set is not None:
        provenance_label = "layers 1–2 chain verified"
        provenance_claim = (
            "Layer-1 source files and the layer-1 → layer-2 artifact binding "
            "were verified before rendering."
        )
    elif task_set_document is not None:
        provenance_label = "layer 1 provenance verified"
        provenance_claim = (
            "Layer-1 source-file provenance was verified; no downstream "
            "artifact was supplied."
        )
    else:
        provenance_label = "phase-1 fallback"
        provenance_claim = "This fallback view has no cross-layer provenance guarantee."
    artifact_hashes = {
        "layer1_file_sha256": (
            _sha256_file(task_set_path) if task_set_document is not None else None
        ),
        "layer1_semantic_sha256": (
            task_set_document.get("semantic_sha256")
            if task_set_document is not None else None
        ),
        "layer2_file_sha256": (
            _sha256_file(robot_set_path) if robot_set is not None else None
        ),
        "layer3_file_sha256": (
            _sha256_file(handoff_set_path) if handoff_set is not None else None
        ),
    }
    return {
        "schema_version": 1,
        "display_units": "millimetres",
        "pose_source": source_name,
        "provenance": {
            "verified": task_set_document is not None,
            "label": provenance_label,
            "checks": provenance_checks,
            "artifact_hashes": artifact_hashes,
        },
        "task_counts": task_counts,
        "task_representative_counts": _status_counts(records, "classification"),
        "robot_counts": robot_summary,
        "robot_representative_counts": _status_counts(records, "robot_classification"),
        "handoff_counts": handoff_summary,
        "poses": records,
        "default_pose_index": default_pose_index,
        "geometry": {
            "board_points_B_mm": board_points,
            "board_outline_B_mm": np.round(np.asarray(board_outline), 3).tolist(),
            "socket_hole_centers_B_mm": np.round(
                1000.0 * np.asarray(socket_config["socket"]["hole_centers_B_m"]), 3
            ).tolist(),
            "socket_hole_radius_mm": round(
                1000.0 * float(socket_config["socket"]["nominal_hole_radius_m"]), 4
            ),
            "connector_points_P_mm": connector_points,
            "gripper_components": components,
            "T_B_P_insert": _matrix_mm(socket_config["T_B_P_insert"], label="T_B_P_insert"),
            "T_G_E": _matrix_mm(gripper_config["T_G_E"], label="T_G_E"),
            "reference_aperture_mm": round(
                1000.0 * float(gripper_config["reference_aperture_m"]), 5
            ),
            "opening_axis_G": [float(item) for item in gripper_config["opening_axis_G"]],
            "insertion_axis_B": np.round(insertion_axis_B, 9).tolist(),
            "preinsert_distance_mm": round(1000.0 * preinsert_distance_m, 3),
        },
        "claim_note": (
            "Pose glyphs are constructive cell-center poses, not the continuous "
            "cells themselves. UNRESOLVED is an outer search set only within the "
            "authored contact modes, not a safety certificate. The selected gripper "
            "uses the supplied body and two finger STL files. "
            + provenance_claim
        ),
    }


_FRAGMENT_TEMPLATE = r'''<section id="insertion-feasible-vis" aria-labelledby="ifv-title">
<style>
#insertion-feasible-vis{--ifv-safe:light-dark(#15803d,#4ade80);--ifv-task-safe:light-dark(#0f766e,#5eead4);--ifv-robot:light-dark(#2563eb,#60a5fa);--ifv-no-witness:light-dark(#6d28d9,#a78bfa);--ifv-unresolved:light-dark(#b45309,#fbbf24);--ifv-rejected:light-dark(#9f1239,#fb7185);--ifv-board:light-dark(#64748b,#94a3b8);--ifv-part:light-dark(#7c3aed,#c4b5fd);--ifv-gripper:light-dark(#0f172a,#e2e8f0);color:var(--color-text-primary);font:400 14px/1.45 var(--font-sans,ui-sans-serif,system-ui,sans-serif);display:block;max-width:1120px;margin:0 auto}
#insertion-feasible-vis *{box-sizing:border-box}
#insertion-feasible-vis .ifv-head{padding:4px 2px 14px}
#insertion-feasible-vis .ifv-kicker{color:var(--color-text-secondary);font-size:12px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;margin:0 0 4px}
#insertion-feasible-vis h2{font-size:clamp(22px,4vw,34px);line-height:1.08;letter-spacing:-.025em;margin:0 0 8px}
#insertion-feasible-vis .ifv-sub{color:var(--color-text-secondary);max-width:78ch;margin:0}
#insertion-feasible-vis .ifv-badges{display:flex;flex-wrap:wrap;gap:7px;margin-top:12px}
#insertion-feasible-vis .ifv-badge{border:1px solid var(--color-border-secondary);border-radius:999px;background:var(--color-background-secondary);padding:5px 9px;font-size:12px;font-variant-numeric:tabular-nums}
#insertion-feasible-vis .ifv-grid{display:grid;grid-template-columns:minmax(0,1fr) 286px;gap:12px;align-items:stretch}
#insertion-feasible-vis .ifv-card{background:var(--color-background-primary);border:1px solid var(--color-border-secondary);border-radius:14px;box-shadow:var(--shadow-sm);overflow:hidden}
#insertion-feasible-vis .ifv-viewport{position:relative;min-height:560px;background:var(--color-background-secondary)}
#insertion-feasible-vis canvas{display:block;width:100%;height:560px;cursor:grab;touch-action:none;outline:none}
#insertion-feasible-vis canvas:active{cursor:grabbing}
#insertion-feasible-vis canvas:focus-visible{outline:2px solid var(--color-ring);outline-offset:-3px}
#insertion-feasible-vis .ifv-overlay{position:absolute;left:10px;bottom:10px;display:flex;flex-wrap:wrap;gap:6px;max-width:calc(100% - 20px);pointer-events:none}
#insertion-feasible-vis .ifv-legend{background:color-mix(in srgb,var(--color-background-primary) 88%,transparent);border:1px solid var(--color-border-secondary);border-radius:8px;padding:5px 7px;font-size:11px;display:flex;align-items:center;gap:5px}
#insertion-feasible-vis .ifv-dot{width:8px;height:8px;border-radius:50%;display:inline-block;color:var(--ifv-unresolved);background:currentColor}
#insertion-feasible-vis .ifv-dot.robot{color:var(--ifv-robot)}#insertion-feasible-vis .ifv-dot.no-witness{color:var(--ifv-no-witness);border-radius:1px;transform:rotate(45deg)}#insertion-feasible-vis .ifv-dot.task-safe{color:var(--ifv-task-safe)}#insertion-feasible-vis .ifv-dot.safe{color:var(--ifv-safe)}#insertion-feasible-vis .ifv-dot.rejected{color:var(--ifv-rejected)}
#insertion-feasible-vis .ifv-controls{padding:14px;display:flex;flex-direction:column;gap:14px}
#insertion-feasible-vis .ifv-control{display:grid;gap:6px}
#insertion-feasible-vis label,#insertion-feasible-vis .ifv-label{font-size:12px;color:var(--color-text-secondary);font-weight:650}
#insertion-feasible-vis input[type=range]{width:100%;accent-color:var(--ifv-robot)}
#insertion-feasible-vis .ifv-row{display:flex;gap:6px;flex-wrap:wrap}
#insertion-feasible-vis button{appearance:none;border:1px solid var(--color-border-secondary);border-radius:8px;background:var(--color-background-secondary);color:var(--color-text-primary);padding:7px 9px;font:inherit;font-size:12px;cursor:pointer}
#insertion-feasible-vis button:hover{background:var(--color-background-tertiary)}#insertion-feasible-vis button[aria-pressed=true]{border-color:var(--ifv-robot);color:var(--ifv-robot);font-weight:700}
#insertion-feasible-vis .ifv-checks{display:grid;grid-template-columns:1fr 1fr;gap:7px 9px}
#insertion-feasible-vis .ifv-checks label{display:flex;align-items:center;gap:6px;font-weight:500;color:var(--color-text-primary)}
#insertion-feasible-vis .ifv-pose-name{font-family:var(--font-mono,ui-monospace,monospace);font-size:11px;word-break:break-all;background:var(--color-background-secondary);border-radius:7px;padding:7px}
#insertion-feasible-vis dl{display:grid;grid-template-columns:1fr auto;gap:5px 9px;margin:0;font-size:12px}
#insertion-feasible-vis dt{color:var(--color-text-secondary)}#insertion-feasible-vis dd{margin:0;text-align:right;font-variant-numeric:tabular-nums;max-width:155px;word-break:break-word}
#insertion-feasible-vis .ifv-layers{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:10px}
#insertion-feasible-vis .ifv-layer{padding:12px 13px;background:var(--color-background-primary);border:1px solid var(--color-border-secondary);border-radius:12px}
#insertion-feasible-vis .ifv-layer h3{font-size:13px;margin:0 0 4px}.ifv-layer p{font-size:12px;color:var(--color-text-secondary);margin:0}
#insertion-feasible-vis .ifv-number{font-size:21px;font-weight:760;letter-spacing:-.02em;margin:3px 0;color:var(--color-text-primary)}
#insertion-feasible-vis .ifv-note{font-size:11px;color:var(--color-text-secondary);margin:10px 2px 0}
#insertion-feasible-vis .ifv-swatch{position:absolute;visibility:hidden}.ifv-safe-swatch{color:var(--ifv-safe)}.ifv-task-safe-swatch{color:var(--ifv-task-safe)}.ifv-robot-swatch{color:var(--ifv-robot)}.ifv-no-witness-swatch{color:var(--ifv-no-witness)}.ifv-unresolved-swatch{color:var(--ifv-unresolved)}.ifv-rejected-swatch{color:var(--ifv-rejected)}.ifv-board-swatch{color:var(--ifv-board)}.ifv-part-swatch{color:var(--ifv-part)}.ifv-gripper-swatch{color:var(--ifv-gripper)}
@media(max-width:760px){#insertion-feasible-vis .ifv-grid{grid-template-columns:1fr}#insertion-feasible-vis .ifv-viewport,#insertion-feasible-vis canvas{min-height:430px;height:430px}#insertion-feasible-vis .ifv-layers{grid-template-columns:1fr}.ifv-controls{display:grid!important;grid-template-columns:1fr 1fr}.ifv-controls .ifv-details{grid-column:1/-1}}
@media(prefers-reduced-motion:reduce){#insertion-feasible-vis *{scroll-behavior:auto!important}}
</style>
<span class="ifv-swatch ifv-safe-swatch"></span><span class="ifv-swatch ifv-task-safe-swatch"></span><span class="ifv-swatch ifv-robot-swatch"></span><span class="ifv-swatch ifv-no-witness-swatch"></span><span class="ifv-swatch ifv-unresolved-swatch"></span><span class="ifv-swatch ifv-rejected-swatch"></span><span class="ifv-swatch ifv-board-swatch"></span><span class="ifv-swatch ifv-part-swatch"></span><span class="ifv-swatch ifv-gripper-swatch"></span>
<header class="ifv-head"><p class="ifv-kicker">Connector header · three-layer feasible set</p><h2 id="ifv-title">Insertion pose set on the actual PCB socket</h2><p class="ifv-sub">Drag to orbit, scroll to zoom, and move along the insertion path. Every small glyph is the constructive center of one bounded task cell; the detailed CAD gripper is shown for the selected center pose.</p><div class="ifv-badges" id="ifv-badges"></div></header>
<div class="ifv-grid">
  <div class="ifv-card ifv-viewport"><canvas id="ifv-canvas" tabindex="0" role="img" aria-label="Interactive three-dimensional view of the PCB, connector header, gripper, and insertion-cell center poses"></canvas><div class="ifv-overlay"><span class="ifv-legend"><i class="ifv-dot"></i>not robot-evaluated</span><span class="ifv-legend"><i class="ifv-dot robot"></i>robot path witness</span><span class="ifv-legend"><i class="ifv-dot no-witness"></i>evaluated, no witness</span><span class="ifv-legend"><i class="ifv-dot task-safe"></i>task-safe only</span><span class="ifv-legend"><i class="ifv-dot safe"></i>robot-certified</span><span class="ifv-legend"><i class="ifv-dot rejected"></i>task rejected</span></div></div>
  <aside class="ifv-card ifv-controls" aria-label="Visualization controls">
    <div class="ifv-control"><label for="ifv-travel">Insertion travel <output id="ifv-travel-value">0%</output></label><input id="ifv-travel" type="range" min="0" max="100" value="0" step="1"><div class="ifv-row"><button data-stage="0">Pre-insert</button><button data-stage="100">Seated</button></div></div>
    <div class="ifv-control"><label for="ifv-pose">Cell-center pose <output id="ifv-pose-value"></output></label><input id="ifv-pose" type="range" min="0" value="0" step="1"><div class="ifv-pose-name" id="ifv-pose-name"></div></div>
    <div class="ifv-control"><span class="ifv-label">View</span><div class="ifv-row"><button data-view="iso" aria-pressed="true">Orbit</button><button data-view="top" aria-pressed="false">Top</button><button data-view="side" aria-pressed="false">Side</button><button id="ifv-fit">Fit</button></div></div>
    <div class="ifv-checks"><label><input id="ifv-show-all" type="checkbox" checked>all poses</label><label><input id="ifv-show-rejected" type="checkbox">rejected</label><label><input id="ifv-show-board" type="checkbox" checked>PCB</label><label><input id="ifv-show-part" type="checkbox" checked>connector</label><label><input id="ifv-show-gripper" type="checkbox" checked>gripper</label><label><input id="ifv-auto-fit" type="checkbox" checked>auto fit</label></div>
    <div class="ifv-details"><dl><dt>Task cell</dt><dd id="ifv-task-status"></dd><dt>Robot layer</dt><dd id="ifv-robot-status"></dd><dt>Handoff preimage</dt><dd id="ifv-handoff-status"></dd><dt>Contact mode</dt><dd id="ifv-contact"></dd><dt>Aperture</dt><dd id="ifv-aperture"></dd><dt>Sample evidence</dt><dd id="ifv-sample"></dd><dt>Seed quality</dt><dd id="ifv-quality"></dd><dt>Cell bounds</dt><dd id="ifv-bounds"></dd></dl></div>
  </aside>
</div>
<div class="ifv-layers"><article class="ifv-layer"><h3>1 · Task-space set</h3><div class="ifv-number" id="ifv-task-number"></div><p>Certified inner cells / unresolved outer cells inside the authored contact-mode domain.</p></article><article class="ifv-layer"><h3>2 · Robot-conditioned set</h3><div class="ifv-number" id="ifv-robot-number"></div><p>Same-branch GP7 discrete center-path witnesses. They remain provisional until TCP calibration and collision proof.</p></article><article class="ifv-layer"><h3>3 · Handoff preimage</h3><div class="ifv-number" id="ifv-handoff-number"></div><p>This is a global current/donor-grasp partition, not a status of each receiver glyph.</p></article></div>
<p class="ifv-note" id="ifv-note"></p>
</section>
<script>(()=>{"use strict";const DATA=__DATA_JSON__;const root=document.getElementById("insertion-feasible-vis"),canvas=document.getElementById("ifv-canvas"),ctx=canvas.getContext("2d");const $=id=>document.getElementById(id);const css=n=>getComputedStyle(root.querySelector(n)).color;let C={safe:css(".ifv-safe-swatch"),taskSafe:css(".ifv-task-safe-swatch"),robot:css(".ifv-robot-swatch"),nowitness:css(".ifv-no-witness-swatch"),unresolved:css(".ifv-unresolved-swatch"),rejected:css(".ifv-rejected-swatch"),board:css(".ifv-board-swatch"),part:css(".ifv-part-swatch"),gripper:css(".ifv-gripper-swatch")};let state={yaw:-.72,pitch:.72,zoom:1,stage:0,pose:DATA.default_pose_index||0,drag:false,moved:false,last:[0,0],fit:null};const G=DATA.geometry,P=DATA.poses;
const clone=M=>M.map(r=>r.slice()),mul=(A,B)=>A.map((r,i)=>B[0].map((_,j)=>r.reduce((s,v,k)=>s+v*B[k][j],0))),point=(M,p)=>[M[0][0]*p[0]+M[0][1]*p[1]+M[0][2]*p[2]+M[0][3],M[1][0]*p[0]+M[1][1]*p[1]+M[1][2]*p[2]+M[1][3],M[2][0]*p[0]+M[2][1]*p[1]+M[2][2]*p[2]+M[2][3]],inv=M=>{let R=[[M[0][0],M[1][0],M[2][0]],[M[0][1],M[1][1],M[2][1]],[M[0][2],M[1][2],M[2][2]]],t=[M[0][3],M[1][3],M[2][3]],o=[[R[0][0],R[0][1],R[0][2],-(R[0][0]*t[0]+R[0][1]*t[1]+R[0][2]*t[2])],[R[1][0],R[1][1],R[1][2],-(R[1][0]*t[0]+R[1][1]*t[1]+R[1][2]*t[2])],[R[2][0],R[2][1],R[2][2],-(R[2][0]*t[0]+R[2][1]*t[1]+R[2][2]*t[2])],[0,0,0,1]];return o};
function TBP(){let M=clone(G.T_B_P_insert),rise=G.preinsert_distance_mm*(1-state.stage);for(let i=0;i<3;i++)M[i][3]-=G.insertion_axis_B[i]*rise;return M}function selectedP(){let pose=P[state.pose],TPG=mul(pose.T_P_E,inv(G.T_G_E)),out=[];for(const comp of G.gripper_components){let T=clone(comp.T_G_C_reference),delta=comp.aperture_multiplier*(pose.required_aperture_mm-G.reference_aperture_mm);for(let i=0;i<3;i++)T[i][3]+=G.opening_axis_G[i]*delta;let M=mul(TPG,T);for(const p of comp.points_C_mm)out.push(point(M,p))}return out}let selectedGripper=selectedP();
function cameraBasis(){let cy=Math.cos(state.yaw),sy=Math.sin(state.yaw),cp=Math.cos(state.pitch),sp=Math.sin(state.pitch);return{right:[-sy,cy,0],up:[-sp*cy,-sp*sy,cp],view:[cp*cy,cp*sy,sp]}}function dot(a,b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]}let activeBasis=null;function project(p){let q=[p[0]-state.fit.center[0],p[1]-state.fit.center[1],p[2]-state.fit.center[2]],b=activeBasis||cameraBasis();return[x0+dot(q,b.right)*state.fit.scale*state.zoom,y0-dot(q,b.up)*state.fit.scale*state.zoom,dot(q,b.view)]}let x0=0,y0=0;
function allBounds(){let M=TBP(),pts=G.board_outline_B_mm.concat(G.board_points_B_mm.slice(0,200));if($("ifv-show-part").checked)for(const p of G.connector_points_P_mm.slice(0,500))pts.push(point(M,p));if($("ifv-show-gripper").checked)for(let i=0;i<selectedGripper.length;i+=8)pts.push(point(M,selectedGripper[i]));let mn=[Infinity,Infinity,Infinity],mx=[-Infinity,-Infinity,-Infinity];for(const p of pts)for(let i=0;i<3;i++){mn[i]=Math.min(mn[i],p[i]);mx[i]=Math.max(mx[i],p[i])}return{mn,mx}}
function fit(){let b=allBounds(),center=b.mn.map((v,i)=>(v+b.mx[i])/2),basis=cameraBasis(),corners=[];for(const x of [b.mn[0],b.mx[0]])for(const y of [b.mn[1],b.mx[1]])for(const z of [b.mn[2],b.mx[2]])corners.push([x-center[0],y-center[1],z-center[2]]);let sx=Math.max(...corners.map(p=>Math.abs(dot(p,basis.right))))*2||1,sy=Math.max(...corners.map(p=>Math.abs(dot(p,basis.up))))*2||1;state.fit={center,scale:.88*Math.min(canvas.clientWidth/sx,canvas.clientHeight/sy)};state.zoom=1}
function line3(a,b,color,width=1,alpha=1){let A=project(a),B=project(b);ctx.globalAlpha=alpha;ctx.strokeStyle=color;ctx.lineWidth=width;ctx.beginPath();ctx.moveTo(A[0],A[1]);ctx.lineTo(B[0],B[1]);ctx.stroke();ctx.globalAlpha=1}function queueCloud(points,M,color,size,alpha,queue){for(const p of points){let s=project(M?point(M,p):p);queue.push([s[0],s[1],s[2],color,size,alpha])}}function paintClouds(queue){queue.sort((a,b)=>a[2]-b[2]);for(const s of queue){ctx.fillStyle=s[3];ctx.globalAlpha=s[5];ctx.fillRect(s[0],s[1],s[4],s[4])}ctx.globalAlpha=1}function robotWitness(p){return p.robot_classification==="PROVISIONAL_PATH_WITNESS"||p.robot_classification==="PROVISIONAL_CENTER_PATH_WITNESS"}function noRobotWitness(p){return p.robot_classification==="NO_IK_AT_PREINSERT"||p.robot_classification==="NO_CONTINUOUS_IK_BRANCH"||p.robot_classification==="NUMERIC_MARGIN_REJECTED"||p.robot_classification==="NO_WITNESS_AT_PREINSERT"||p.robot_classification==="NO_WITNESS_CONTINUATION"||p.robot_classification==="CENTER_PATH_NUMERIC_MARGIN_NOT_MET"}function statusColor(p){if(p.robot_certified||p.robot_classification==="CERTIFIED_SAFE")return C.safe;if(robotWitness(p))return C.robot;if(noRobotWitness(p))return C.nowitness;if(p.classification==="REJECTED")return C.rejected;if(p.classification==="SAFE")return C.taskSafe;return C.unresolved}
function hole(center,r,M){let last=null,first=null;for(let i=0;i<=18;i++){let a=2*Math.PI*i/18,p=point(M,[center[0]+r*Math.cos(a),center[1]+r*Math.sin(a),center[2]+.2]);if(!first)first=p;if(last)line3(last,p,C.rejected,1.2,.9);last=p}}
function drawPoseGlyph(p,M,selected=false){let E=mul(M,p.T_P_E),o=point(E,[0,0,0]),z=point(E,[0,0,selected?15:7]),ym=point(E,[0,selected?8:3,0]),yp=point(E,[0,selected?-8:-3,0]),color=statusColor(p);line3(o,z,color,selected?2.6:1,selected?1:.55);line3(ym,yp,color,selected?2:1,selected?1:.45);let s=project(o);ctx.strokeStyle=color;ctx.lineWidth=selected?2:1.3;if(noRobotWitness(p)){ctx.beginPath();ctx.moveTo(s[0]-3,s[1]-3);ctx.lineTo(s[0]+3,s[1]+3);ctx.moveTo(s[0]+3,s[1]-3);ctx.lineTo(s[0]-3,s[1]+3);ctx.stroke()}else if(robotWitness(p)||p.robot_certified){ctx.beginPath();ctx.arc(s[0],s[1],selected?4:2.5,0,2*Math.PI);ctx.stroke()}if(selected){line3(o,point(E,[12,0,0]),"#ef4444",2);line3(o,point(E,[0,12,0]),"#22c55e",2);line3(o,point(E,[0,0,12]),"#3b82f6",2)}}
function draw(){let rect=canvas.getBoundingClientRect(),dpr=Math.min(devicePixelRatio||1,2);if(canvas.width!==Math.round(rect.width*dpr)||canvas.height!==Math.round(rect.height*dpr)){canvas.width=Math.round(rect.width*dpr);canvas.height=Math.round(rect.height*dpr);ctx.setTransform(dpr,0,0,dpr,0,0);if(!state.fit)fit()}x0=rect.width/2;y0=rect.height/2;activeBasis=cameraBasis();ctx.clearRect(0,0,rect.width,rect.height);let M=TBP(),dots=[];if($("ifv-show-board").checked)queueCloud(G.board_points_B_mm,null,C.board,1,.16,dots);if($("ifv-show-part").checked)queueCloud(G.connector_points_P_mm,M,C.part,1.35,.75,dots);if($("ifv-show-gripper").checked)queueCloud(selectedGripper,M,C.gripper,1,.48,dots);paintClouds(dots);if($("ifv-show-board").checked){let O=G.board_outline_B_mm;for(let i=0;i<4;i++)line3(O[i],O[(i+1)%4],C.board,1.4,.8);for(const h of G.socket_hole_centers_B_mm)hole(h,G.socket_hole_radius_mm,[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])}if($("ifv-show-all").checked){for(let i=0;i<P.length;i++){if(i===state.pose)continue;if(P[i].classification==="REJECTED"&&!$("ifv-show-rejected").checked)continue;drawPoseGlyph(P[i],M,false)}}drawPoseGlyph(P[state.pose],M,true);activeBasis=null}
let drawPending=false;function requestDraw(){if(drawPending)return;drawPending=true;requestAnimationFrame(()=>{drawPending=false;draw()})}
function updateDetails(){let p=P[state.pose],b=p.bounds||{},range=a=>a&&a.length===2?`${a[0]}…${a[1]}`:"—";$("ifv-pose").max=P.length-1;$("ifv-pose").value=state.pose;$("ifv-pose-value").textContent=`${state.pose+1} / ${P.length}`;$("ifv-pose-name").textContent=p.id;$("ifv-task-status").textContent=p.classification;$("ifv-task-status").style.color=statusColor(p);$("ifv-robot-status").textContent=p.robot_classification+(p.robot_branch_count?` (${p.robot_branch_count} branches)`:"");$("ifv-handoff-status").textContent=p.handoff_status.replace("GLOBAL_","global ");$("ifv-contact").textContent=p.contact_mode;$("ifv-aperture").textContent=`${p.required_aperture_mm.toFixed(3)} mm`;$("ifv-sample").textContent=p.has_sample_witness?`${p.sample_status}${p.sample_witness_claims.length?`; ${p.sample_witness_claims.join(", ")}`:""}`:"none";$("ifv-quality").textContent=p.quality===null?"—":p.quality.toFixed(3);$("ifv-bounds").textContent=`u ${range(b.u_mm)}; v ${range(b.v_mm)}; r ${range(b.roll_deg)}`;selectedGripper=selectedP();if($("ifv-auto-fit").checked)fit();requestDraw()}
function setup(){let tc=DATA.task_counts||{},rc=DATA.robot_counts||{},hc=DATA.handoff_counts||{},w=rc.provisional_center_path_witness_count??rc.provisional_path_witness_count??0,n=rc.numerically_unresolved_count??rc.rejected_representative_count??0,e=rc.numerical_center_evaluated_count??rc.evaluated_cell_count??0,c=rc.certified_receiver_cell_count??0,source=DATA.pose_source.replaceAll("_"," "),provenance=DATA.provenance&&DATA.provenance.label?DATA.provenance.label:(DATA.provenance&&DATA.provenance.verified?"artifact chain verified":"fallback: no hash chain");$("ifv-badges").innerHTML=`<span class="ifv-badge">${P.length.toLocaleString()} cell centers</span><span class="ifv-badge">${(tc.unresolved||0).toLocaleString()} unresolved</span><span class="ifv-badge">${e.toLocaleString()} robot-evaluated</span><span class="ifv-badge">${w.toLocaleString()} robot witnesses</span><span class="ifv-badge">${c.toLocaleString()} certified</span><span class="ifv-badge">${source}</span><span class="ifv-badge">${provenance}</span>`;$("ifv-task-number").textContent=`${tc.safe||0} safe / ${tc.unresolved||0} unresolved`;$("ifv-robot-number").textContent=`${w} witness / ${n} no witness`;$("ifv-handoff-number").textContent=`${hc.positive_preimage_count||0} positive / ${hc.unknown_count||0} unknown`;$("ifv-note").textContent=DATA.claim_note;$("ifv-travel").addEventListener("input",e=>{state.stage=+e.target.value/100;$("ifv-travel-value").textContent=`${e.target.value}%`;if($("ifv-auto-fit").checked)fit();requestDraw()});for(const b of root.querySelectorAll("[data-stage]"))b.addEventListener("click",()=>{$("ifv-travel").value=b.dataset.stage;$("ifv-travel").dispatchEvent(new Event("input"))});$("ifv-pose").addEventListener("input",e=>{state.pose=+e.target.value;updateDetails()});for(const b of root.querySelectorAll("[data-view]"))b.addEventListener("click",()=>{root.querySelectorAll("[data-view]").forEach(x=>x.setAttribute("aria-pressed","false"));b.setAttribute("aria-pressed","true");if(b.dataset.view==="top"){state.yaw=-Math.PI/2;state.pitch=Math.PI/2-.001}else if(b.dataset.view==="side"){state.yaw=0;state.pitch=.001}else{state.yaw=-.72;state.pitch=.72}fit();requestDraw()});$("ifv-fit").addEventListener("click",()=>{fit();requestDraw()});for(const id of ["ifv-show-all","ifv-show-rejected","ifv-show-board","ifv-show-part","ifv-show-gripper"])$(id).addEventListener("change",()=>{if($("ifv-auto-fit").checked)fit();requestDraw()});let ro=new ResizeObserver(()=>{state.fit=null;fit();requestDraw()});ro.observe(canvas);canvas.addEventListener("pointerdown",e=>{state.drag=true;state.moved=false;state.last=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId)});canvas.addEventListener("pointermove",e=>{if(!state.drag)return;let dx=e.clientX-state.last[0],dy=e.clientY-state.last[1];state.moved=state.moved||Math.abs(dx)+Math.abs(dy)>2;state.yaw-=dx*.008;state.pitch=Math.max(-1.45,Math.min(1.55,state.pitch+dy*.008));state.last=[e.clientX,e.clientY];if($("ifv-auto-fit").checked)fit();requestDraw()});canvas.addEventListener("pointerup",e=>{state.drag=false;if(state.moved)return;let r=canvas.getBoundingClientRect(),q=[e.clientX-r.left,e.clientY-r.top],M=TBP(),best=-1,dist=100;for(let i=0;i<P.length;i++){if(P[i].classification==="REJECTED"&&!$("ifv-show-rejected").checked)continue;let o=project(point(mul(M,P[i].T_P_E),[0,0,0])),d=Math.hypot(o[0]-q[0],o[1]-q[1]);if(d<dist){dist=d;best=i}}if(best>=0&&dist<14){state.pose=best;updateDetails()}});canvas.addEventListener("wheel",e=>{e.preventDefault();state.zoom=Math.max(.25,Math.min(6,state.zoom*Math.exp(-e.deltaY*.001)));$("ifv-auto-fit").checked=false;requestDraw()},{passive:false});updateDetails()}
setup();})();</script>'''


def render_fragment(data: Mapping[str, Any]) -> str:
    payload = json.dumps(data, separators=(",", ":"), ensure_ascii=True)
    result = _FRAGMENT_TEMPLATE.replace("__DATA_JSON__", payload)
    if len(result.encode("utf-8")) >= 2_000_000:
        raise ValueError("visualization fragment exceeds the 2 MB inline limit")
    return result


def render_standalone(fragment: str) -> str:
    """Wrap the inline fragment in a portable, directly openable document."""
    return """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark">
<title>Connector Header Insertion Feasible Set</title>
<style>
:root{color-scheme:light dark;--color-background-primary:light-dark(#fff,#181818);--color-background-secondary:light-dark(#f8fafc,#202020);--color-background-tertiary:light-dark(#eef2f7,#2a2a2a);--color-text-primary:light-dark(#111827,#f8fafc);--color-text-secondary:light-dark(#526071,#b8c0cc);--color-border-secondary:light-dark(#d9e0e8,#383838);--color-ring:#3b82f6;--shadow-sm:0 1px 2px rgb(0 0 0/.08);--font-sans:ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;--font-mono:ui-monospace,SFMono-Regular,Menlo,monospace}
html,body{margin:0;min-height:100%;background:var(--color-background-primary);color:var(--color-text-primary)}body{padding:clamp(12px,3vw,34px)}
</style>
</head>
<body>
""" + fragment + """
</body>
</html>
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the connector insertion feasible-set visualization."
    )
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--task-set", type=Path)
    parser.add_argument("--robot-set", type=Path)
    parser.add_argument("--handoff-set", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--standalone-output",
        type=Path,
        help="directly openable HTML (default: project generated/visualization)",
    )
    parser.add_argument("--compact", action="store_true", help="Use a smaller CAD point budget")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root.resolve()
    project = root / PROJECT_RELATIVE
    task_set = args.task_set or project / "generated/sets/insertion_task_set.json"
    robot_set = args.robot_set or project / "generated/sets/robot_insertion_set.json"
    handoff_set = args.handoff_set or project / "generated/sets/handoff_preimage_set.json"
    output = args.output or root / DEFAULT_FRAGMENT_RELATIVE
    standalone_output = args.standalone_output or root / DEFAULT_STANDALONE_RELATIVE
    budget = (
        {"board": 800, "connector": 450, "body": 650, "finger": 400}
        if args.compact
        else {"board": 1600, "connector": 900, "body": 1300, "finger": 800}
    )
    data = build_visualization_data(
        root,
        task_set_path=task_set.resolve(),
        robot_set_path=robot_set.resolve(),
        handoff_set_path=handoff_set.resolve(),
        point_budget=budget,
    )
    fragment = render_fragment(data)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(fragment, encoding="utf-8")
    standalone_output.parent.mkdir(parents=True, exist_ok=True)
    standalone_output.write_text(render_standalone(fragment), encoding="utf-8")
    print(output)
    print(standalone_output)
    print(f"poses={len(data['poses'])} bytes={output.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
