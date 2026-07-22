"""Target-conditioned GP7 intersection of an insertion task set.

Layer 1 describes robot-independent grasp cells in the connector frame.  This
module composes each selected cell's constructive center ``T_P_E`` with a
runtime seated connector pose, samples the straight insertion path, and tracks
the same numerical GP7 IK seed through every sample. Sampled library witnesses
are retained only as evidence and never used as the evaluation pose.

The result is deliberately a *path-witness set*, not a certified feasible set.
Numerical GP7 IK is not branch-complete, path samples do not prove the
continuous interval between them, and this module performs no scene collision
checking. A local witness therefore receives
``PROVISIONAL_CENTER_PATH_WITNESS``. ``CERTIFIED_SAFE`` can enter only through
an external continuous whole-cell certificate with exact content and runtime
bindings and all hard gates true.

Frame convention: ``T_X_Y`` maps coordinates from frame Y into frame X, so
``T_W_E(s) = T_W_P(s) @ T_P_E``.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..core.se3 import validate_transform


_CLASSIFICATIONS = ("SAFE", "REJECTED", "UNRESOLVED")
_ROBOT_CLASSIFICATIONS = frozenset((
    "CERTIFIED_SAFE",
    "PROVISIONAL_CENTER_PATH_WITNESS",
    "NO_WITNESS_AT_PREINSERT",
    "NO_WITNESS_CONTINUATION",
    "CENTER_PATH_NUMERIC_MARGIN_NOT_MET",
))
_TASK_ARTIFACT_TYPE = "robot_independent_insertion_task_set"
_CERTIFICATE_ARTIFACT_TYPE = "continuous_robot_insertion_cell_certificate"
_CERTIFICATE_HARD_GATES = frozenset((
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
))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_array(value: Any) -> list[Any]:
    return np.asarray(value, dtype=float).tolist()


def _semantic_sha256(document: Mapping[str, Any]) -> str:
    normalized = dict(document)
    normalized.pop("semantic_sha256", None)
    payload = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _task_certificate_binding_sha256(document: Mapping[str, Any]) -> str:
    normalized = deepcopy(dict(document))
    normalized.pop("semantic_sha256", None)
    normalized.pop("whole_cell_task_certificates", None)
    inputs = normalized.get("inputs")
    if isinstance(inputs, dict):
        task_config = inputs.get("task_set_config")
        if isinstance(task_config, dict):
            task_config.pop("sha256", None)
    payload = json.dumps(
        normalized, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_digest(value: Any, *, label: str) -> str:
    digest = _nonempty(value, label=label)
    if (len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _nonempty(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _positive_float(value: Any, *, label: str, allow_zero: bool = False) -> float:
    result = float(value)
    valid = result >= 0.0 if allow_zero else result > 0.0
    if not np.isfinite(result) or not valid:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{label} must be finite and {qualifier}")
    return result


def _positive_int(value: Any, *, label: str, minimum: int = 1) -> int:
    if (not isinstance(value, (int, np.integer))
            or isinstance(value, (bool, np.bool_)) or int(value) < minimum):
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return int(value)


def _unit(value: Any, *, label: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{label} must be a finite three-vector")
    norm = float(np.linalg.norm(vector))
    if norm <= np.finfo(float).eps * 64.0:
        raise ValueError(f"{label} must be nonzero")
    return vector / norm


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _strict_schema_version(document: Mapping[str, Any], *, label: str) -> None:
    value = document.get("schema_version")
    if not isinstance(value, int) or isinstance(value, bool) or value != 1:
        raise ValueError(f"{label} schema_version must be integer 1")


def _bounds_pair(value: Any, *, label: str) -> tuple[float, float]:
    array = np.asarray(value, dtype=float)
    if array.shape != (2,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must be a finite [minimum, maximum] pair")
    lower, upper = map(float, array)
    if not lower < upper:
        raise ValueError(f"{label} requires minimum < maximum")
    return lower, upper


def _center_constructive_transform(
    theta: Mapping[str, Any],
    constructive_map: Mapping[str, Any],
) -> tuple[np.ndarray, float]:
    closing = _unit(
        constructive_map.get("closing_axis_P"),
        label="constructive_map.closing_axis_P",
    )
    zero = _unit(
        constructive_map.get("roll_zero_approach_axis_P"),
        label="constructive_map.roll_zero_approach_axis_P",
    )
    quadrature = _unit(
        constructive_map.get("positive_roll_quadrature_axis_P"),
        label="constructive_map.positive_roll_quadrature_axis_P",
    )
    expected_quadrature = np.cross(closing, zero)
    if not np.allclose(quadrature, expected_quadrature, atol=1e-9, rtol=0.0):
        raise ValueError(
            "constructive_map positive-roll quadrature must equal "
            "cross(closing_axis_P, roll_zero_approach_axis_P)")
    if constructive_map.get("positive_roll_rule") != (
            "right_hand_about_closing_axis_P"):
        raise ValueError("constructive_map positive_roll_rule is unsupported")
    u_axis = _unit(
        constructive_map.get("position_u_axis_P"),
        label="constructive_map.position_u_axis_P",
    )
    v_axis = _unit(
        constructive_map.get("position_v_axis_P"),
        label="constructive_map.position_v_axis_P",
    )
    if abs(float(u_axis @ v_axis)) > 1e-9:
        raise ValueError("constructive-map position axes must be orthogonal")
    if (abs(float(u_axis @ closing)) > 1e-9
            or abs(float(v_axis @ closing)) > 1e-9
            or abs(float(zero @ closing)) > 1e-9):
        raise ValueError("constructive-map tangent/approach axes must oppose closing")
    midplane = float(constructive_map.get(
        "contact_midplane_coordinate_P_m"))
    if not np.isfinite(midplane):
        raise ValueError("constructive-map midplane coordinate must be finite")
    aperture_model = _mapping(
        constructive_map.get("aperture_model"),
        label="constructive_map.aperture_model",
    )
    if set(aperture_model) not in (
            {"type", "value_m"}, {"type", "value_m", "status"}):
        raise ValueError("constructive-map aperture model has unexpected fields")
    if aperture_model["type"] != "constant":
        raise ValueError("only constant constructive-map aperture is supported")
    if "status" in aperture_model:
        _nonempty(aperture_model["status"], label="aperture_model.status")
    aperture = _positive_float(
        aperture_model["value_m"], label="constructive-map aperture")
    roll = float(theta["roll_rad"])
    approach = np.cos(roll) * zero + np.sin(roll) * quadrature
    approach = _unit(approach, label="constructive center approach")
    rotation = np.column_stack((
        np.cross(closing, approach),
        closing,
        approach,
    ))
    position = (
        float(theta["u_P_m"]) * u_axis
        + float(theta["v_P_m"]) * v_axis
        + midplane * closing
    )
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = position
    return validate_transform(transform), aperture


@dataclass(frozen=True)
class InsertionTaskCell:
    """One source cell and its constructive parameter-box center pose."""

    cell_id: str
    source_classification: str
    contact_mode: str
    bounds: Mapping[str, Any]
    center_pose: Mapping[str, Any]
    sampled_representative: Mapping[str, Any] | None
    T_P_E: np.ndarray
    center_theta: np.ndarray
    grid_index: Mapping[str, int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "cell_id", _nonempty(self.cell_id, label="cell id"))
        classification = str(self.source_classification).upper()
        if classification not in set(_CLASSIFICATIONS):
            raise ValueError(
                f"cell {self.cell_id!r} classification must be one of "
                f"{sorted(_CLASSIFICATIONS)}")
        object.__setattr__(self, "source_classification", classification)
        object.__setattr__(
            self, "contact_mode",
            _nonempty(self.contact_mode, label=f"cell {self.cell_id} contact_mode"),
        )
        bounds = dict(self.bounds)
        if set(bounds) != {"u_P_m", "v_P_m", "roll_rad"}:
            raise ValueError(f"cell {self.cell_id!r} bounds have unexpected fields")
        normalized_bounds = {
            name: list(_bounds_pair(value, label=f"cell {self.cell_id} {name}"))
            for name, value in bounds.items()
        }
        object.__setattr__(self, "bounds", normalized_bounds)
        theta = np.asarray(self.center_theta, dtype=float)
        if theta.shape != (3,) or not np.all(np.isfinite(theta)):
            raise ValueError(f"cell {self.cell_id!r} center theta is invalid")
        for index, name in enumerate(("u_P_m", "v_P_m", "roll_rad")):
            lower, upper = normalized_bounds[name]
            if not lower <= theta[index] <= upper:
                raise ValueError(f"cell {self.cell_id!r} center {name} is outside bounds")
            if not np.isclose(
                    theta[index], 0.5 * (lower + upper), atol=1e-10, rtol=0.0):
                raise ValueError(f"cell {self.cell_id!r} {name} is not the cell center")
        theta.setflags(write=False)
        object.__setattr__(self, "center_theta", theta)
        transform = validate_transform(self.T_P_E)
        transform.setflags(write=False)
        object.__setattr__(self, "T_P_E", transform)
        center_pose = deepcopy(dict(self.center_pose))
        center_pose["T_P_E"] = _json_array(transform)
        object.__setattr__(self, "center_pose", center_pose)
        representative = self.sampled_representative
        object.__setattr__(
            self,
            "sampled_representative",
            None if representative is None else deepcopy(dict(representative)),
        )
        grid = dict(self.grid_index)
        if set(grid) != {"u", "v", "roll"} or any(
                not isinstance(value, int) or isinstance(value, bool) or value < 0
                for value in grid.values()):
            raise ValueError(f"cell {self.cell_id!r} grid_index is invalid")
        object.__setattr__(self, "grid_index", grid)


@dataclass(frozen=True)
class InsertionTaskSet:
    """Validated robot-independent layer-1 artifact view."""

    path: Path
    sha256: str
    semantic_sha256: str
    task_certificate_binding_sha256: str
    artifact_type: str
    project_id: str
    insertion_axis_P: np.ndarray
    preinsert_distance_m: float
    T_B_P_insert: np.ndarray | None
    cells: tuple[InsertionTaskCell, ...]
    safe_inner_cell_ids: tuple[str, ...]
    rejected_cell_ids: tuple[str, ...]
    unresolved_cell_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "artifact_type",
            _nonempty(self.artifact_type, label="layer-1 artifact_type"),
        )
        object.__setattr__(self, "sha256", _sha256_digest(
            self.sha256, label="layer-1 file SHA-256"))
        object.__setattr__(self, "semantic_sha256", _sha256_digest(
            self.semantic_sha256, label="layer-1 semantic SHA-256"))
        object.__setattr__(self, "task_certificate_binding_sha256", _sha256_digest(
            self.task_certificate_binding_sha256,
            label="layer-1 task certificate binding SHA-256",
        ))
        if self.artifact_type != _TASK_ARTIFACT_TYPE:
            raise ValueError(f"layer-1 artifact_type must be {_TASK_ARTIFACT_TYPE!r}")
        object.__setattr__(self, "project_id", _nonempty(
            self.project_id, label="layer-1 project_id"))
        axis = _unit(self.insertion_axis_P, label="insertion_axis_P")
        axis.setflags(write=False)
        object.__setattr__(self, "insertion_axis_P", axis)
        object.__setattr__(
            self, "preinsert_distance_m",
            _positive_float(
                self.preinsert_distance_m, label="preinsert_distance_m"),
        )
        if self.T_B_P_insert is not None:
            socket = validate_transform(self.T_B_P_insert)
            socket.setflags(write=False)
            object.__setattr__(self, "T_B_P_insert", socket)
        identifiers = [cell.cell_id for cell in self.cells]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("layer-1 artifact contains duplicate cell IDs")


@dataclass(frozen=True)
class VerifiedContinuousRobotCertificate:
    """Opaque result of the strict external-certificate verifier."""

    path: Path
    file_sha256: str
    semantic_sha256: str
    bindings: Mapping[str, Any]
    hard_gates: Mapping[str, bool]
    certified_cells: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "file_sha256", _sha256_digest(
            self.file_sha256, label="certificate file SHA-256"))
        object.__setattr__(self, "semantic_sha256", _sha256_digest(
            self.semantic_sha256, label="certificate semantic SHA-256"))
        if set(self.hard_gates) != _CERTIFICATE_HARD_GATES or any(
                value is not True for value in self.hard_gates.values()):
            raise ValueError("verified certificate hard gates are incomplete")
        object.__setattr__(self, "bindings", deepcopy(dict(self.bindings)))
        object.__setattr__(self, "hard_gates", deepcopy(dict(self.hard_gates)))
        object.__setattr__(self, "certified_cells", tuple(
            deepcopy(dict(record)) for record in self.certified_cells))


def _classification_id_list(
    document: Mapping[str, Any], key: str,
) -> tuple[str, ...]:
    value = document.get(key)
    if not isinstance(value, list) or any(
            not isinstance(item, str) or not item for item in value):
        raise ValueError(f"layer-1 {key} must be a list of cell IDs")
    if len(set(value)) != len(value):
        raise ValueError(f"layer-1 {key} contains duplicate IDs")
    return tuple(value)


def load_insertion_task_set(path: str | Path) -> InsertionTaskSet:
    """Strictly load and cross-check the robot-independent layer-1 artifact."""
    source = Path(path).expanduser().resolve()
    with source.open(encoding="utf-8") as stream:
        document = json.load(
            stream,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {token!r}")),
        )
    document = _mapping(document, label="layer-1 artifact root")
    _strict_schema_version(document, label="layer-1 artifact")
    if document.get("artifact_type") != _TASK_ARTIFACT_TYPE:
        raise ValueError(f"layer-1 artifact_type must be {_TASK_ARTIFACT_TYPE!r}")
    embedded_semantic = _sha256_digest(
        document.get("semantic_sha256"), label="layer-1 semantic_sha256")
    computed_semantic = _semantic_sha256(document)
    if embedded_semantic != computed_semantic:
        raise ValueError("layer-1 semantic_sha256 does not match document content")
    trajectory = document.get("insertion_trajectory")
    trajectory = _mapping(trajectory, label="layer-1 insertion trajectory")
    axis = trajectory.get("insertion_axis_P")
    distance = trajectory.get("preinsert_distance_m")
    socket = trajectory.get("T_B_P_insert")
    if axis is None or distance is None or socket is None:
        raise ValueError(
            "layer-1 insertion trajectory requires insertion_axis_P and "
            "preinsert_distance_m and T_B_P_insert")
    if trajectory.get("type") != "straight_fixed_orientation":
        raise ValueError("layer-1 insertion trajectory type is unsupported")

    parameterization = _mapping(
        document.get("parameterization"), label="layer-1 parameterization")
    mode_records = parameterization.get("contact_modes")
    if not isinstance(mode_records, list) or not mode_records:
        raise ValueError("layer-1 contact_modes must be a non-empty list")
    modes: dict[str, Mapping[str, Any]] = {}
    for index, raw_mode in enumerate(mode_records):
        mode = _mapping(raw_mode, label=f"contact_modes[{index}]")
        mode_id = _nonempty(mode.get("id"), label=f"contact_modes[{index}].id")
        if mode_id in modes:
            raise ValueError("layer-1 contact_modes contain duplicate IDs")
        constructive = _mapping(
            mode.get("constructive_map"),
            label=f"contact mode {mode_id} constructive_map",
        )
        if constructive.get("constructive_map_version", 1) != 1:
            raise ValueError(f"contact mode {mode_id} map version is unsupported")
        modes[mode_id] = mode

    records = document.get("cells")
    if not isinstance(records, list):
        raise ValueError("layer-1 cells must be a list")
    cells: list[InsertionTaskCell] = []
    for index, raw in enumerate(records):
        record = _mapping(raw, label=f"layer-1 cells[{index}]")
        cell_id = _nonempty(record.get("id"), label=f"cells[{index}].id")
        classification = _nonempty(
            record.get("classification"),
            label=f"cell {cell_id} classification",
        ).upper()
        contact_mode = _nonempty(
            record.get("contact_mode"), label=f"cell {cell_id} contact_mode")
        if contact_mode not in modes:
            raise ValueError(f"cell {cell_id!r} references unknown contact_mode")
        bounds = _mapping(
            record.get("bounds"), label=f"cell {cell_id!r} bounds")
        center = _mapping(
            record.get("center_pose"), label=f"cell {cell_id!r} center_pose")
        if center.get("source") != "contact_mode_constructive_map":
            raise ValueError(f"cell {cell_id!r} center_pose source is invalid")
        if center.get("constructive_map_version") != 1:
            raise ValueError(f"cell {cell_id!r} center_pose map version is invalid")
        theta_map = _mapping(
            center.get("theta"), label=f"cell {cell_id!r} center theta")
        if set(theta_map) != {"u_P_m", "v_P_m", "roll_rad"}:
            raise ValueError(f"cell {cell_id!r} center theta fields are invalid")
        theta = np.array([
            float(theta_map["u_P_m"]),
            float(theta_map["v_P_m"]),
            float(theta_map["roll_rad"]),
        ])
        mode = modes[contact_mode]
        expected_transform, expected_aperture = _center_constructive_transform(
            theta_map,
            _mapping(mode["constructive_map"], label="constructive_map"),
        )
        supplied_transform = validate_transform(np.asarray(
            center.get("T_P_E"), dtype=float))
        if not np.allclose(
                supplied_transform, expected_transform, atol=2e-9, rtol=0.0):
            raise ValueError(
                f"cell {cell_id!r} center T_P_E disagrees with constructive map")
        supplied_aperture = _positive_float(
            center.get("required_aperture_m"),
            label=f"cell {cell_id!r} center aperture",
        )
        if not np.isclose(
                supplied_aperture, expected_aperture, atol=1e-10, rtol=0.0):
            raise ValueError(
                f"cell {cell_id!r} center aperture disagrees with constructive map")
        sampled = record.get("representative")
        if sampled is not None:
            sampled = _mapping(
                sampled, label=f"cell {cell_id!r} sampled representative")
        grid_index = _mapping(
            record.get("grid_index"), label=f"cell {cell_id!r} grid_index")
        cell_counts = _mapping(
            mode.get("cell_counts"),
            label=f"contact mode {contact_mode} cell_counts",
        )
        if set(cell_counts) != {"u", "v", "roll"}:
            raise ValueError(f"contact mode {contact_mode} cell_counts are invalid")
        for axis_name in ("u", "v", "roll"):
            count = cell_counts[axis_name]
            index_value = grid_index.get(axis_name)
            if (not isinstance(count, int) or isinstance(count, bool) or count <= 0
                    or not isinstance(index_value, int)
                    or isinstance(index_value, bool)
                    or not 0 <= index_value < count):
                raise ValueError(f"cell {cell_id!r} grid index/count is invalid")
        mode_u = _bounds_pair(
            mode.get("u_bounds_P_m"), label=f"contact mode {contact_mode} u bounds")
        mode_v = _bounds_pair(
            mode.get("v_bounds_P_m"), label=f"contact mode {contact_mode} v bounds")
        expected_bounds = {}
        for name, domain, grid_name in (
            ("u_P_m", mode_u, "u"),
            ("v_P_m", mode_v, "v"),
            ("roll_rad", (-np.pi, np.pi), "roll"),
        ):
            count = int(cell_counts[grid_name])
            cell_index = int(grid_index[grid_name])
            width = (domain[1] - domain[0]) / count
            expected_bounds[name] = [
                domain[0] + cell_index * width,
                domain[0] + (cell_index + 1) * width,
            ]
            if not np.allclose(
                    np.asarray(bounds[name], dtype=float),
                    expected_bounds[name], atol=2e-9, rtol=0.0):
                raise ValueError(
                    f"cell {cell_id!r} bounds disagree with grid partition")
        cells.append(InsertionTaskCell(
            cell_id=cell_id,
            source_classification=classification,
            contact_mode=contact_mode,
            bounds=bounds,
            center_pose=center,
            sampled_representative=sampled,
            T_P_E=supplied_transform,
            center_theta=theta,
            grid_index=grid_index,
        ))

    classification_lists = {
        "SAFE": _classification_id_list(document, "safe_inner_cell_ids"),
        "REJECTED": _classification_id_list(document, "rejected_cell_ids"),
        "UNRESOLVED": _classification_id_list(document, "unresolved_cell_ids"),
    }
    listed = [cell_id for name in _CLASSIFICATIONS
              for cell_id in classification_lists[name]]
    cell_ids = [cell.cell_id for cell in cells]
    if len(listed) != len(set(listed)):
        raise ValueError("layer-1 classification ID lists overlap")
    if set(listed) != set(cell_ids):
        raise ValueError("layer-1 classification ID lists do not partition cells")
    actual_by_classification = {
        name: {cell.cell_id for cell in cells
               if cell.source_classification == name}
        for name in _CLASSIFICATIONS
    }
    for name in _CLASSIFICATIONS:
        expected_order = tuple(
            cell.cell_id for cell in cells
            if cell.source_classification == name)
        if classification_lists[name] != expected_order:
            raise ValueError(
                f"layer-1 {name} ID list disagrees with cell classifications")
    counts = _mapping(document.get("counts"), label="layer-1 counts")
    expected_counts = {
        "cells": len(cells),
        "safe": len(classification_lists["SAFE"]),
        "rejected": len(classification_lists["REJECTED"]),
        "unresolved": len(classification_lists["UNRESOLVED"]),
    }
    for key, expected in expected_counts.items():
        value = counts.get(key)
        if (not isinstance(value, int) or isinstance(value, bool)
                or value != expected):
            raise ValueError(f"layer-1 counts.{key} is inconsistent")
    task_certificates = _mapping(
        document.get("whole_cell_task_certificates"),
        label="layer-1 whole_cell_task_certificates",
    )
    base_binding = _sha256_digest(
        task_certificates.get("base_artifact_certificate_binding_sha256"),
        label="layer-1 base artifact certificate binding",
    )
    if (not classification_lists["SAFE"]
            and base_binding != _task_certificate_binding_sha256(document)):
        raise ValueError("layer-1 task-certificate base binding is inconsistent")
    if task_certificates.get("verification_policy") != (
            "fail_closed_exact_file_and_artifact_binding"):
        raise ValueError("layer-1 task-certificate verification policy is invalid")
    identity = _mapping(
        document.get("task_identity"), label="layer-1 task_identity")
    connector_sha = _sha256_digest(
        identity.get("connector_sha256"), label="layer-1 connector SHA-256")
    _sha256_digest(identity.get("pcb_sha256"), label="layer-1 PCB SHA-256")
    expected_task_bindings = {
        "base_artifact_certificate_binding_sha256": base_binding,
        "project_id": document.get("project_id"),
        "connector_sha256": connector_sha,
    }
    if task_certificates.get("expected_bindings") != expected_task_bindings:
        raise ValueError("layer-1 task-certificate expected bindings are inconsistent")
    required_proofs = task_certificates.get("required_proved_constraints")
    if (not isinstance(required_proofs, list) or not required_proofs
            or any(not isinstance(item, str) or not item for item in required_proofs)
            or len(set(required_proofs)) != len(required_proofs)):
        raise ValueError("layer-1 required task proofs are invalid")
    promoted_count = task_certificates.get("promoted_safe_cell_count")
    if (not isinstance(promoted_count, int) or isinstance(promoted_count, bool)
            or promoted_count != len(classification_lists["SAFE"])):
        raise ValueError("layer-1 promoted SAFE count is inconsistent")
    imports = task_certificates.get("imports")
    if not isinstance(imports, list):
        raise ValueError("layer-1 task-certificate imports must be a list")
    promoted_from_imports: set[str] = set()
    imports_by_id: dict[str, Mapping[str, Any]] = {}
    for index, raw_import in enumerate(imports):
        imported = _mapping(raw_import, label=f"task certificate import {index}")
        if set(imported) != {
                "certificate_id", "path", "sha256", "promoted_cell_ids"}:
            raise ValueError("layer-1 task-certificate import fields are invalid")
        certificate_id = _nonempty(
            imported["certificate_id"], label="task certificate_id")
        if certificate_id in imports_by_id:
            raise ValueError("layer-1 task certificates contain duplicate IDs")
        _nonempty(imported["path"], label="task certificate path")
        _sha256_digest(imported["sha256"], label="task certificate SHA-256")
        promoted = imported["promoted_cell_ids"]
        if (not isinstance(promoted, list)
                or any(not isinstance(item, str) or not item for item in promoted)
                or len(set(promoted)) != len(promoted)):
            raise ValueError("task certificate promoted IDs are invalid")
        if promoted_from_imports.intersection(promoted):
            raise ValueError("task certificate imports promote duplicate cells")
        promoted_from_imports.update(promoted)
        imports_by_id[certificate_id] = imported
    if promoted_from_imports != set(classification_lists["SAFE"]):
        raise ValueError("task certificate imports do not exactly cover SAFE cells")
    records_by_id = {str(record["id"]): record for record in records}
    for cell in cells:
        if cell.source_classification != "SAFE":
            continue
        original = records_by_id[cell.cell_id]
        proof = _mapping(
            original.get("whole_cell_task_certificate"),
            label=f"SAFE cell {cell.cell_id} task certificate",
        )
        if proof.get("base_artifact_certificate_binding_sha256") != base_binding:
            raise ValueError(f"SAFE cell {cell.cell_id!r} has an unbound proof")
        certificate_id = proof.get("certificate_id")
        if certificate_id not in imports_by_id:
            raise ValueError(f"SAFE cell {cell.cell_id!r} proof import is unknown")
        imported = imports_by_id[certificate_id]
        if (proof.get("path") != imported["path"]
                or proof.get("sha256") != imported["sha256"]
                or cell.cell_id not in imported["promoted_cell_ids"]):
            raise ValueError(f"SAFE cell {cell.cell_id!r} proof metadata mismatch")
        proved = proof.get("proved_constraints")
        if (not isinstance(proved, list)
                or not set(required_proofs).issubset(proved)):
            raise ValueError(f"SAFE cell {cell.cell_id!r} proof obligations incomplete")

    return InsertionTaskSet(
        path=source,
        sha256=_sha256_file(source),
        semantic_sha256=embedded_semantic,
        task_certificate_binding_sha256=base_binding,
        artifact_type=str(document["artifact_type"]),
        project_id=document.get("project_id"),
        insertion_axis_P=np.asarray(axis, dtype=float),
        preinsert_distance_m=float(distance),
        T_B_P_insert=np.asarray(socket, dtype=float),
        cells=tuple(cells),
        safe_inner_cell_ids=classification_lists["SAFE"],
        rejected_cell_ids=classification_lists["REJECTED"],
        unresolved_cell_ids=classification_lists["UNRESOLVED"],
    )


def resolve_world_part_insert(
    task_set: InsertionTaskSet,
    *,
    T_W_P_insert: np.ndarray | None = None,
    T_W_B: np.ndarray | None = None,
    T_B_P_insert: np.ndarray | None = None,
) -> tuple[np.ndarray, str]:
    """Resolve either a direct connector pose or measured board pose."""
    direct = T_W_P_insert is not None
    board = T_W_B is not None
    if direct == board:
        raise ValueError("provide exactly one of T_W_P_insert or T_W_B")
    if direct:
        if T_B_P_insert is not None:
            raise ValueError("T_B_P_insert cannot accompany direct T_W_P_insert")
        return validate_transform(T_W_P_insert), "direct_T_W_P_insert"
    socket = task_set.T_B_P_insert if T_B_P_insert is None else T_B_P_insert
    if socket is None:
        raise ValueError(
            "board-pose target requires T_B_P_insert in layer 1 or config")
    return (
        validate_transform(T_W_B) @ validate_transform(socket),
        "T_W_B_x_T_B_P_insert",
    )


def sample_straight_insertion_path(
    T_W_P_insert: np.ndarray,
    insertion_axis_P: np.ndarray,
    preinsert_distance_m: float,
    sample_count: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Sample fixed-orientation part poses from pre-insert through seating."""
    seated = validate_transform(T_W_P_insert)
    axis_W = seated[:3, :3] @ _unit(
        insertion_axis_P, label="insertion_axis_P")
    distance = _positive_float(
        preinsert_distance_m, label="preinsert_distance_m")
    count = _positive_int(sample_count, label="path_sample_count", minimum=2)
    samples: list[np.ndarray] = []
    for progress in np.linspace(0.0, 1.0, count):
        pose = seated.copy()
        pose[:3, 3] -= (1.0 - float(progress)) * distance * axis_W
        samples.append(pose)
    return axis_W, samples


def _joint_metrics(kinematics: Any, robot: str, q: np.ndarray) -> dict[str, Any]:
    q = np.asarray(q, dtype=float)
    lower = np.asarray(kinematics.lower[robot], dtype=float)
    upper = np.asarray(kinematics.upper[robot], dtype=float)
    lower_margin = q - lower
    upper_margin = upper - q
    absolute_margin = float(np.min(np.minimum(lower_margin, upper_margin)))
    span = upper - lower
    normalized = float(np.min(
        2.0 * np.minimum(q - lower, upper - q) / span))
    if hasattr(kinematics, "normalized_limit_margin"):
        normalized = float(kinematics.normalized_limit_margin(robot, q))
    singular_values = np.asarray(
        kinematics.singular_values(robot, q), dtype=float)
    if singular_values.ndim != 1 or not singular_values.size:
        raise ValueError("kinematics.singular_values must return a non-empty vector")
    return {
        "joint_lower_margin_rad": _json_array(lower_margin),
        "joint_upper_margin_rad": _json_array(upper_margin),
        "joint_limit_margin_rad": absolute_margin,
        "normalized_joint_limit_margin": normalized,
        "singular_values": _json_array(singular_values),
        "sigma_min": float(np.min(singular_values)),
        "manipulability": float(np.prod(singular_values)),
    }


def _ik_record(result: Any, metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "q_rad": _json_array(result.q),
        "position_error_m": float(result.position_error),
        "rotation_error_rad": float(result.rotation_error),
        "iterations": int(result.iterations),
        **deepcopy(dict(metrics)),
    }


def _branch_summary(
    branch_id: str,
    samples: list[dict[str, Any]],
    *,
    complete: bool,
    failure_sample_index: int | None,
    failure_reason: str | None,
) -> dict[str, Any]:
    q_values = [np.asarray(sample["q_rad"], dtype=float) for sample in samples]
    maximum_step = max(
        (float(np.max(np.abs(second - first)))
         for first, second in zip(q_values, q_values[1:])),
        default=0.0,
    )
    return {
        "branch_id": branch_id,
        "complete_at_all_discrete_samples": complete,
        "continuous_between_samples_proven": False,
        "failure_sample_index": failure_sample_index,
        "failure_reason": failure_reason,
        "sample_count_reached": len(samples),
        "q_path_rad": [sample["q_rad"] for sample in samples],
        "samples": samples,
        "margins": {
            "minimum_joint_limit_margin_rad": min(
                sample["joint_limit_margin_rad"] for sample in samples),
            "minimum_normalized_joint_limit_margin": min(
                sample["normalized_joint_limit_margin"] for sample in samples),
            "minimum_sigma": min(sample["sigma_min"] for sample in samples),
            "minimum_manipulability": min(
                sample["manipulability"] for sample in samples),
            "maximum_joint_step_rad": maximum_step,
            "maximum_position_error_m": max(
                sample["position_error_m"] for sample in samples),
            "maximum_rotation_error_rad": max(
                sample["rotation_error_rad"] for sample in samples),
        },
    }


def _deterministic_rng(
    task_sha256: str,
    robot: str,
    cell_id: str,
    target: np.ndarray,
) -> np.random.Generator:
    digest = hashlib.sha256(
        b"robot-insertion-set-ik-v1\0"
        + task_sha256.encode("ascii")
        + robot.encode("ascii")
        + cell_id.encode("utf-8")
        + np.round(target, 10).tobytes()
    ).digest()
    return np.random.default_rng(
        int.from_bytes(digest[:8], "little", signed=False))


def _evaluate_cell(
    cell: InsertionTaskCell,
    *,
    task_set: InsertionTaskSet,
    kinematics: Any,
    robot: str,
    T_W_P_path: list[np.ndarray],
    random_restarts: int,
    max_solutions: int,
    position_tolerance_m: float,
    rotation_tolerance_rad: float,
    max_joint_step_rad: float,
    minimum_joint_limit_margin_rad: float,
    minimum_normalized_joint_limit_margin: float,
    minimum_sigma: float,
) -> dict[str, Any]:
    T_W_E_path = [pose @ cell.T_P_E for pose in T_W_P_path]
    initial = kinematics.solutions(
        robot,
        T_W_E_path[0],
        random_restarts=random_restarts,
        max_solutions=max_solutions,
        rng=_deterministic_rng(
            task_set.sha256, robot, cell.cell_id, T_W_E_path[0]),
        position_tolerance=position_tolerance_m,
        rotation_tolerance=rotation_tolerance_rad,
    )
    branches: list[dict[str, Any]] = []
    for branch_index, initial_result in enumerate(initial):
        first_q = np.asarray(initial_result.q, dtype=float)
        samples = [_ik_record(
            initial_result, _joint_metrics(kinematics, robot, first_q))]
        previous = first_q
        failure_index: int | None = None
        failure_reason: str | None = None
        for sample_index, target in enumerate(T_W_E_path[1:], start=1):
            result = kinematics.solve(
                robot,
                target,
                seed=previous,
                position_tolerance=position_tolerance_m,
                rotation_tolerance=rotation_tolerance_rad,
            )
            if result is None:
                failure_index = sample_index
                failure_reason = "seeded_ik_failed"
                break
            q = np.asarray(result.q, dtype=float)
            step = float(np.max(np.abs(q - previous)))
            if step > max_joint_step_rad:
                failure_index = sample_index
                failure_reason = "joint_branch_jump_exceeds_threshold"
                break
            samples.append(_ik_record(
                result, _joint_metrics(kinematics, robot, q)))
            previous = q
        branches.append(_branch_summary(
            f"{cell.cell_id}:branch_{branch_index:02d}",
            samples,
            complete=failure_index is None,
            failure_sample_index=failure_index,
            failure_reason=failure_reason,
        ))

    complete = [branch for branch in branches
                if branch["complete_at_all_discrete_samples"]]

    def passes_margins(branch: Mapping[str, Any]) -> bool:
        margins = branch["margins"]
        return bool(
            margins["minimum_joint_limit_margin_rad"]
            >= minimum_joint_limit_margin_rad
            and margins["minimum_normalized_joint_limit_margin"]
            >= minimum_normalized_joint_limit_margin
            and margins["minimum_sigma"] >= minimum_sigma
            and margins["maximum_joint_step_rad"] <= max_joint_step_rad
        )

    accepted = [branch for branch in complete if passes_margins(branch)]
    if accepted:
        classification = "PROVISIONAL_CENTER_PATH_WITNESS"
    elif complete:
        classification = "CENTER_PATH_NUMERIC_MARGIN_NOT_MET"
    elif not initial:
        classification = "NO_WITNESS_AT_PREINSERT"
    else:
        classification = "NO_WITNESS_CONTINUATION"
    assert classification in _ROBOT_CLASSIFICATIONS

    return {
        "id": cell.cell_id,
        "source_classification": cell.source_classification,
        "contact_mode": cell.contact_mode,
        "bounds": deepcopy(dict(cell.bounds)),
        "center_pose": deepcopy(dict(cell.center_pose)),
        "sampled_representative": deepcopy(cell.sampled_representative),
        "evaluated_pose_source": "cell.center_pose",
        "center_pose_only": True,
        "whole_parameter_cell_evaluated": False,
        "robot_classification": classification,
        "certified": False,
        "certification": {
            "certified": False,
            "all_hard_gates_passed": False,
            "source": None,
            "external_certificate_identity": None,
            "cell_proof_sha256": None,
        },
        "T_W_E_preinsert": _json_array(T_W_E_path[0]),
        "T_W_E_insert": _json_array(T_W_E_path[-1]),
        "T_W_E_path": [_json_array(pose) for pose in T_W_E_path],
        "enumerated_preinsert_branch_count": len(initial),
        "complete_discrete_branch_count": len(complete),
        "accepted_provisional_branch_count": len(accepted),
        "accepted_provisional_branch_ids": [
            branch["branch_id"] for branch in accepted],
        "branches": branches,
    }


def _normalized_center_coordinates(
    cells: Sequence[InsertionTaskCell],
) -> dict[str, np.ndarray]:
    lower = np.array([
        min(cell.bounds["u_P_m"][0] for cell in cells),
        min(cell.bounds["v_P_m"][0] for cell in cells),
        min(cell.bounds["roll_rad"][0] for cell in cells),
    ])
    upper = np.array([
        max(cell.bounds["u_P_m"][1] for cell in cells),
        max(cell.bounds["v_P_m"][1] for cell in cells),
        max(cell.bounds["roll_rad"][1] for cell in cells),
    ])
    span = upper - lower
    if np.any(span <= 0.0):
        raise ValueError("eligible cell domain has zero-width stratification axis")
    return {
        cell.cell_id: (np.asarray(cell.center_theta) - lower) / span
        for cell in cells
    }


def _normalized_parameter_distance(first: Any, second: Any) -> float:
    """Euclidean ``u,v`` distance with circular normalized-roll distance."""
    first_value = np.asarray(first, dtype=float)
    second_value = np.asarray(second, dtype=float)
    if (first_value.shape != (3,) or second_value.shape != (3,)
            or not np.all(np.isfinite(first_value))
            or not np.all(np.isfinite(second_value))):
        raise ValueError("normalized parameter points must be finite three-vectors")
    delta = np.abs(first_value - second_value)
    if np.any(delta > 1.0 + 1e-12):
        raise ValueError("normalized parameter points must lie in [0, 1]")
    delta[2] = min(float(delta[2]), 1.0 - float(delta[2]))
    return float(np.linalg.norm(delta))


def select_task_cells_stratified(
    task_set: InsertionTaskSet,
    *,
    source_classifications: Sequence[str],
    max_cells: int | None,
) -> tuple[list[InsertionTaskCell], dict[str, Any], list[str]]:
    """Select deterministic, space-filling centers across mode/u/v/roll.

    Quotas are balanced round-robin across contact modes.  Within each mode a
    deterministic maximin design starts nearest the domain center and then
    repeatedly selects the point farthest from the selected normalized centers.
    Cell IDs are used only to break exact geometric ties.
    """
    selection = [str(value).upper() for value in source_classifications]
    if not selection or any(value not in set(_CLASSIFICATIONS)
                            for value in selection):
        raise ValueError(
            f"source_classifications must use {sorted(_CLASSIFICATIONS)}")
    if len(set(selection)) != len(selection):
        raise ValueError("source_classifications contains duplicates")
    if "REJECTED" in selection:
        raise ValueError("source_classifications cannot include REJECTED cells")
    eligible = [cell for cell in task_set.cells
                if cell.source_classification in set(selection)]
    grouped: dict[str, list[InsertionTaskCell]] = {}
    for cell in eligible:
        grouped.setdefault(cell.contact_mode, []).append(cell)
    for cells in grouped.values():
        cells.sort(key=lambda item: (
            item.grid_index["u"],
            item.grid_index["v"],
            item.grid_index["roll"],
            item.cell_id,
        ))
    modes = sorted(grouped)
    eligible_by_mode = {mode: len(grouped[mode]) for mode in modes}

    if max_cells is None or max_cells >= len(eligible):
        selected = [cell for mode in modes for cell in grouped[mode]]
        evidence = [{
            "selection_rank": rank,
            "cell_id": cell.cell_id,
            "contact_mode": cell.contact_mode,
            "grid_index": dict(cell.grid_index),
            "center_theta": {
                name: float(cell.center_theta[index])
                for index, name in enumerate(("u_P_m", "v_P_m", "roll_rad"))
            },
            "normalized_center": None,
            "minimum_distance_to_prior_in_mode": None,
        } for rank, cell in enumerate(selected, start=1)]
        return selected, {
            "strategy": "all_eligible_cells_grid_order_v1",
            "eligible_by_contact_mode": eligible_by_mode,
            "quota_by_contact_mode": dict(eligible_by_mode),
            "selected": evidence,
        }, []

    limit = _positive_int(max_cells, label="max_cells")
    quotas = {mode: 0 for mode in modes}
    remaining_budget = min(limit, len(eligible))
    while remaining_budget:
        progressed = False
        for mode in modes:
            if remaining_budget == 0:
                break
            if quotas[mode] < len(grouped[mode]):
                quotas[mode] += 1
                remaining_budget -= 1
                progressed = True
        if not progressed:
            break

    chosen_by_mode: dict[str, list[InsertionTaskCell]] = {}
    normalized_by_mode: dict[str, dict[str, np.ndarray]] = {}
    distance_by_id: dict[str, float | None] = {}
    for mode in modes:
        candidates = grouped[mode]
        normalized = _normalized_center_coordinates(candidates)
        normalized_by_mode[mode] = normalized
        chosen: list[InsertionTaskCell] = []
        remaining = list(candidates)
        while len(chosen) < quotas[mode]:
            if not chosen:
                cell = min(
                    remaining,
                    key=lambda item: (
                        _normalized_parameter_distance(
                            normalized[item.cell_id], [0.5, 0.5, 0.5]),
                        item.cell_id,
                    ),
                )
                distance_by_id[cell.cell_id] = None
            else:
                selected_points = [normalized[item.cell_id] for item in chosen]

                def minimum_distance(item: InsertionTaskCell) -> float:
                    point = normalized[item.cell_id]
                    return min(_normalized_parameter_distance(point, selected)
                               for selected in selected_points)

                cell = sorted(
                    remaining,
                    key=lambda item: (-minimum_distance(item), item.cell_id),
                )[0]
                distance_by_id[cell.cell_id] = minimum_distance(cell)
            chosen.append(cell)
            remaining.remove(cell)
        chosen_by_mode[mode] = chosen

    # Interleave mode strata so every prefix remains mode-balanced.
    selected = []
    for depth in range(max(quotas.values(), default=0)):
        for mode in modes:
            if depth < len(chosen_by_mode[mode]):
                selected.append(chosen_by_mode[mode][depth])
    selected_ids = {cell.cell_id for cell in selected}
    not_evaluated = sorted(
        cell.cell_id for cell in eligible if cell.cell_id not in selected_ids)
    evidence = []
    for rank, cell in enumerate(selected, start=1):
        normalized = normalized_by_mode[cell.contact_mode][cell.cell_id]
        evidence.append({
            "selection_rank": rank,
            "cell_id": cell.cell_id,
            "contact_mode": cell.contact_mode,
            "grid_index": dict(cell.grid_index),
            "center_theta": {
                name: float(cell.center_theta[index])
                for index, name in enumerate(("u_P_m", "v_P_m", "roll_rad"))
            },
            "normalized_center": _json_array(normalized),
            "minimum_distance_to_prior_in_mode": distance_by_id[cell.cell_id],
        })
    return selected, {
        "strategy": "contact_mode_balanced_maximin_center_v1",
        "distance_metric": "euclidean_linear_u_v_circular_normalized_roll_v1",
        "roll_distance": "min(abs(delta_roll), 1-abs(delta_roll))",
        "tie_breaker": "cell_id_only_after_equal_geometric_score",
        "eligible_by_contact_mode": eligible_by_mode,
        "quota_by_contact_mode": quotas,
        "selected": evidence,
    }, not_evaluated


def load_verified_continuous_robot_certificate(
    path: str | Path,
    *,
    expected_file_sha256: str,
    task_set: InsertionTaskSet,
    robot: str,
    T_W_P_insert: np.ndarray,
    world_frame: Mapping[str, Any],
    tcp_calibration_fingerprint: str,
    execution_bindings: Mapping[str, str],
) -> VerifiedContinuousRobotCertificate:
    """Verify an externally produced continuous robot-cell certificate.

    Trust is opt-in and content-addressed.  The expected file hash comes from
    configuration outside the certificate; the embedded semantic hash catches
    accidental/non-byte-preserving changes; bindings then pin the proof to the
    exact task set, robot, target, world calibration, TCP, project, and model.
    Every hard gate must be explicitly true.
    """
    source = Path(path).expanduser().resolve()
    expected_file = _sha256_digest(
        expected_file_sha256, label="continuous certificate expected SHA-256")
    actual_file = _sha256_file(source)
    if actual_file != expected_file:
        raise ValueError("continuous certificate file SHA-256 mismatch")
    with source.open(encoding="utf-8") as stream:
        document = json.load(
            stream,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {token!r}")),
        )
    document = _mapping(document, label="continuous certificate root")
    _strict_schema_version(document, label="continuous certificate")
    if document.get("artifact_type") != _CERTIFICATE_ARTIFACT_TYPE:
        raise ValueError(
            f"continuous certificate artifact_type must be "
            f"{_CERTIFICATE_ARTIFACT_TYPE!r}")
    embedded = _sha256_digest(
        document.get("semantic_sha256"),
        label="continuous certificate semantic_sha256",
    )
    if embedded != _semantic_sha256(document):
        raise ValueError("continuous certificate semantic_sha256 mismatch")

    binding = _mapping(
        document.get("bindings"), label="continuous certificate bindings")
    expected_binding_keys = {
        "task_artifact_file_sha256",
        "task_artifact_semantic_sha256",
        "task_certificate_binding_sha256",
        "robot",
        "world_frame",
        "T_W_P_insert",
        "tcp_calibration_fingerprint",
        "project_sha256",
        "model_sha256",
    }
    if set(binding) != expected_binding_keys:
        raise ValueError("continuous certificate binding fields are not exact")
    robot_name = str(robot).upper()
    expected_world = {
        key: _nonempty(world_frame[key], label=f"world_frame.{key}")
        for key in ("id", "calibration_fingerprint", "calibration_source")
    }
    supplied_world = _mapping(
        binding["world_frame"], label="continuous certificate world_frame")
    if dict(supplied_world) != expected_world:
        raise ValueError("continuous certificate world-frame binding mismatch")
    supplied_target = validate_transform(np.asarray(
        binding["T_W_P_insert"], dtype=float))
    expected_target = validate_transform(T_W_P_insert)
    if not np.array_equal(supplied_target, expected_target):
        raise ValueError("continuous certificate target binding mismatch")
    required_bindings = {
        "task_artifact_file_sha256": task_set.sha256,
        "task_artifact_semantic_sha256": task_set.semantic_sha256,
        "task_certificate_binding_sha256": (
            task_set.task_certificate_binding_sha256),
        "robot": robot_name,
        "tcp_calibration_fingerprint": _nonempty(
            tcp_calibration_fingerprint,
            label="tcp_calibration_fingerprint",
        ),
        "project_sha256": _sha256_digest(
            execution_bindings.get("project_sha256"),
            label="execution project_sha256",
        ),
        "model_sha256": _sha256_digest(
            execution_bindings.get("model_sha256"),
            label="execution model_sha256",
        ),
    }
    for key, expected in required_bindings.items():
        if binding[key] != expected:
            raise ValueError(f"continuous certificate {key} binding mismatch")

    gates = _mapping(
        document.get("hard_gates"), label="continuous certificate hard_gates")
    if set(gates) != _CERTIFICATE_HARD_GATES:
        raise ValueError("continuous certificate hard-gate fields are not exact")
    if any(value is not True for value in gates.values()):
        raise ValueError("continuous certificate requires every hard gate true")

    records = document.get("certified_cells")
    if not isinstance(records, list):
        raise ValueError("continuous certificate certified_cells must be a list")
    safe_ids = set(task_set.safe_inner_cell_ids)
    normalized_records = []
    seen: set[str] = set()
    expected_record_keys = {
        "cell_id",
        "classification",
        "proof_sha256",
        "minimum_joint_limit_margin_rad",
        "minimum_normalized_joint_limit_margin",
        "minimum_sigma",
    }
    for index, raw in enumerate(records):
        record = _mapping(
            raw, label=f"continuous certificate certified_cells[{index}]")
        if set(record) != expected_record_keys:
            raise ValueError("continuous certified-cell fields are not exact")
        cell_id = _nonempty(
            record["cell_id"], label=f"certified_cells[{index}].cell_id")
        if cell_id in seen:
            raise ValueError("continuous certificate has duplicate cell IDs")
        seen.add(cell_id)
        if cell_id not in safe_ids:
            raise ValueError(
                f"continuous certificate cell {cell_id!r} is not layer-1 SAFE")
        if record["classification"] != "CERTIFIED_SAFE":
            raise ValueError("continuous certificate classification must be CERTIFIED_SAFE")
        normalized_records.append({
            "cell_id": cell_id,
            "classification": "CERTIFIED_SAFE",
            "proof_sha256": _sha256_digest(
                record["proof_sha256"],
                label=f"certified cell {cell_id} proof_sha256",
            ),
            "minimum_joint_limit_margin_rad": _positive_float(
                record["minimum_joint_limit_margin_rad"],
                label=f"certified cell {cell_id} joint margin",
            ),
            "minimum_normalized_joint_limit_margin": _positive_float(
                record["minimum_normalized_joint_limit_margin"],
                label=f"certified cell {cell_id} normalized margin",
            ),
            "minimum_sigma": _positive_float(
                record["minimum_sigma"],
                label=f"certified cell {cell_id} sigma",
            ),
        })
    return VerifiedContinuousRobotCertificate(
        path=source,
        file_sha256=actual_file,
        semantic_sha256=embedded,
        bindings=binding,
        hard_gates=gates,
        certified_cells=tuple(normalized_records),
    )


def build_robot_insertion_set(
    task_set: InsertionTaskSet,
    kinematics: Any,
    *,
    robot: str,
    T_W_P_insert: np.ndarray,
    world_frame: Mapping[str, Any],
    target_source: str,
    source_classifications: Sequence[str] = ("SAFE",),
    max_cells: int | None = None,
    path_sample_count: int = 11,
    random_restarts: int = 18,
    max_solutions: int = 8,
    position_tolerance_m: float = 7e-4,
    rotation_tolerance_rad: float = np.radians(0.35),
    max_joint_step_rad: float = 0.35,
    minimum_joint_limit_margin_rad: float = 0.03,
    minimum_normalized_joint_limit_margin: float = 0.02,
    minimum_sigma: float = 1e-4,
    tcp_calibrated: bool = False,
    tcp_calibration_fingerprint: str | None = None,
    acknowledge_provisional_tcp: bool = False,
    continuous_certificate: VerifiedContinuousRobotCertificate | None = None,
) -> dict[str, Any]:
    """Intersect task-cell center poses with sampled GP7 IK paths.

    Numerical center paths remain provisional.  ``CERTIFIED_SAFE`` is possible
    only for cells carried by a :class:`VerifiedContinuousRobotCertificate`.
    """
    robot_name = str(robot).upper()
    if robot_name not in ("A", "B"):
        raise ValueError("robot must be A or B")
    if not isinstance(tcp_calibrated, bool):
        raise ValueError("tcp_calibrated must be boolean")
    if not isinstance(acknowledge_provisional_tcp, bool):
        raise ValueError("acknowledge_provisional_tcp must be boolean")
    if not tcp_calibrated and not acknowledge_provisional_tcp:
        raise ValueError(
            "uncalibrated TCP requires acknowledge_provisional_tcp=True")
    if tcp_calibrated and not tcp_calibration_fingerprint:
        raise ValueError(
            "calibrated TCP requires a calibration fingerprint")
    if continuous_certificate is not None:
        if not isinstance(
                continuous_certificate, VerifiedContinuousRobotCertificate):
            raise ValueError(
                "continuous_certificate must come from the strict verifier")
        if not tcp_calibrated:
            raise ValueError("continuous certificate requires calibrated TCP")
    world_metadata = _mapping(world_frame, label="world_frame")
    for key in ("id", "calibration_fingerprint", "calibration_source"):
        if key not in world_metadata:
            raise ValueError(f"world_frame requires {key}")

    selection = [str(value).upper() for value in source_classifications]
    cells, stratification, not_evaluated_ids = select_task_cells_stratified(
        task_set,
        source_classifications=selection,
        max_cells=max_cells,
    )
    eligible_count = sum(
        cell.source_classification in set(selection) for cell in task_set.cells)

    count = _positive_int(
        path_sample_count, label="path_sample_count", minimum=2)
    restarts = _positive_int(
        random_restarts, label="random_restarts", minimum=0)
    solutions = _positive_int(max_solutions, label="max_solutions")
    position_tolerance = _positive_float(
        position_tolerance_m, label="position_tolerance_m")
    rotation_tolerance = _positive_float(
        rotation_tolerance_rad, label="rotation_tolerance_rad")
    jump = _positive_float(max_joint_step_rad, label="max_joint_step_rad")
    joint_margin = _positive_float(
        minimum_joint_limit_margin_rad,
        label="minimum_joint_limit_margin_rad", allow_zero=True)
    normalized_margin = _positive_float(
        minimum_normalized_joint_limit_margin,
        label="minimum_normalized_joint_limit_margin", allow_zero=True)
    sigma = _positive_float(
        minimum_sigma, label="minimum_sigma", allow_zero=True)

    seated = validate_transform(T_W_P_insert)
    axis_W, T_W_P_path = sample_straight_insertion_path(
        seated,
        task_set.insertion_axis_P,
        task_set.preinsert_distance_m,
        count,
    )
    reference_q = np.asarray(kinematics.get_q(robot_name), dtype=float).copy()
    try:
        results = []
        for cell in cells:
            kinematics.set_q(robot_name, reference_q)
            results.append(_evaluate_cell(
                cell,
                task_set=task_set,
                kinematics=kinematics,
                robot=robot_name,
                T_W_P_path=T_W_P_path,
                random_restarts=restarts,
                max_solutions=solutions,
                position_tolerance_m=position_tolerance,
                rotation_tolerance_rad=rotation_tolerance,
                max_joint_step_rad=jump,
                minimum_joint_limit_margin_rad=joint_margin,
                minimum_normalized_joint_limit_margin=normalized_margin,
                minimum_sigma=sigma,
            ))
    finally:
        kinematics.set_q(robot_name, reference_q)

    certificate_records = (
        {} if continuous_certificate is None else {
            record["cell_id"]: dict(record)
            for record in continuous_certificate.certified_cells
        }
    )
    certificate_identity = (
        None if continuous_certificate is None else {
            "artifact_type": _CERTIFICATE_ARTIFACT_TYPE,
            "path": str(continuous_certificate.path),
            "file_sha256": continuous_certificate.file_sha256,
            "semantic_sha256": continuous_certificate.semantic_sha256,
        }
    )

    def certified_cell_mapping(certificate_record: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "certified": True,
            "all_hard_gates_passed": True,
            "source": "external_continuous_robot_cell_certificate",
            "external_certificate_identity": deepcopy(certificate_identity),
            "cell_proof_sha256": certificate_record["proof_sha256"],
        }

    task_cells_by_id = {cell.cell_id: cell for cell in task_set.cells}
    result_by_id = {record["id"]: record for record in results}
    for cell_id, certificate_record in certificate_records.items():
        if cell_id in result_by_id:
            record = result_by_id[cell_id]
            record["numerical_center_classification"] = record[
                "robot_classification"]
            record["robot_classification"] = "CERTIFIED_SAFE"
            record["certified"] = True
            record["center_pose_only"] = False
            record["whole_parameter_cell_evaluated"] = True
            record["evaluated_pose_source"] = (
                "cell.center_pose_diagnostic_plus_external_continuous_cell_certificate")
            record["external_continuous_certificate"] = certificate_record
            record["certification"] = certified_cell_mapping(certificate_record)
            continue
        cell = task_cells_by_id[cell_id]
        T_W_E_path = [pose @ cell.T_P_E for pose in T_W_P_path]
        record = {
            "id": cell.cell_id,
            "source_classification": cell.source_classification,
            "contact_mode": cell.contact_mode,
            "bounds": deepcopy(dict(cell.bounds)),
            "center_pose": deepcopy(dict(cell.center_pose)),
            "sampled_representative": deepcopy(cell.sampled_representative),
            "evaluated_pose_source": "external_continuous_cell_certificate",
            "center_pose_only": False,
            "whole_parameter_cell_evaluated": True,
            "robot_classification": "CERTIFIED_SAFE",
            "numerical_center_classification": None,
            "certified": True,
            "certification": certified_cell_mapping(certificate_record),
            "T_W_E_preinsert": _json_array(T_W_E_path[0]),
            "T_W_E_insert": _json_array(T_W_E_path[-1]),
            "T_W_E_path": [_json_array(pose) for pose in T_W_E_path],
            "enumerated_preinsert_branch_count": 0,
            "complete_discrete_branch_count": 0,
            "accepted_provisional_branch_count": 0,
            "accepted_provisional_branch_ids": [],
            "branches": [],
            "external_continuous_certificate": certificate_record,
        }
        results.append(record)
        result_by_id[cell_id] = record

    certified_ids = sorted(certificate_records)
    if certified_ids:
        certified_id_set = set(certified_ids)
        not_evaluated_ids = [cell_id for cell_id in not_evaluated_ids
                             if cell_id not in certified_id_set]
    witnesses = [record for record in results
                 if record["robot_classification"]
                 == "PROVISIONAL_CENTER_PATH_WITNESS"]
    numerically_unresolved = [record for record in results
                              if record["robot_classification"] in {
                                  "NO_WITNESS_AT_PREINSERT",
                                  "NO_WITNESS_CONTINUATION",
                                  "CENTER_PATH_NUMERIC_MARGIN_NOT_MET",
                              }]
    receiver_witnesses = [{
        "cell_id": record["id"],
        "source_classification": record["source_classification"],
        "T_P_E": record["center_pose"]["T_P_E"],
        "T_W_E_preinsert": record["T_W_E_preinsert"],
        "T_W_E_insert": record["T_W_E_insert"],
        "accepted_provisional_branch_ids": (
            record["accepted_provisional_branch_ids"]),
        "certified": False,
    } for record in witnesses]

    failed_requirements = [
        "scene_collision_checked",
        "continuous_between_path_samples_proven",
        "whole_parameter_cell_robot_feasibility_proven_by_local_solver",
        "complete_analytic_IK_branch_enumeration",
    ]
    if not tcp_calibrated:
        failed_requirements.insert(0, "calibrated_flange_to_E_transform")

    return {
        "schema_version": 1,
        "artifact_type": "robot_conditioned_insertion_path_set",
        "claim_level": (
            "mixed_certified_cells_and_provisional_center_witnesses"
            if certified_ids else "provisional_discrete_center_path_witness_set"
        ),
        "certified": bool(certified_ids),
        "certification": {
            "certified_receiver_cell_ids": certified_ids,
            "tcp_calibrated": tcp_calibrated,
            "tcp_calibration_fingerprint": tcp_calibration_fingerprint,
            "external_continuous_certificate_supplied": (
                continuous_certificate is not None),
            "local_numerical_scene_collision_checked": False,
            "local_continuous_between_path_samples_proven": False,
            "local_whole_parameter_cells_checked": False,
            "local_analytic_ik_branch_complete": False,
            "local_numerical_failed_requirements": failed_requirements,
        },
        "robot": robot_name,
        "world_frame": {
            key: _nonempty(world_metadata[key], label=f"world_frame.{key}")
            for key in ("id", "calibration_fingerprint", "calibration_source")
        },
        "source_task_set": {
            "path": str(task_set.path),
            "file_sha256": task_set.sha256,
            "semantic_sha256": task_set.semantic_sha256,
            "task_certificate_binding_sha256": (
                task_set.task_certificate_binding_sha256),
            "artifact_type": task_set.artifact_type,
            "project_id": task_set.project_id,
        },
        "frame_contract": {
            "T_P_E": "maps gripper contact/TCP frame E into connector frame P",
            "T_W_E": "T_W_P(s) @ T_P_E",
            "path": "preinsert is opposite insertion_axis_W; final sample is seated",
        },
        "target_source": _nonempty(target_source, label="target_source"),
        "T_W_P_preinsert": _json_array(T_W_P_path[0]),
        "T_W_P_insert": _json_array(T_W_P_path[-1]),
        "T_W_P_path": [_json_array(pose) for pose in T_W_P_path],
        "insertion_axis_P": _json_array(task_set.insertion_axis_P),
        "insertion_axis_W": _json_array(axis_W),
        "preinsert_distance_m": task_set.preinsert_distance_m,
        "path_sample_count": count,
        "path_parameter_values": np.linspace(0.0, 1.0, count).tolist(),
        "selection": {
            "source_classifications": selection,
            "eligible_cell_count_before_limit": eligible_count,
            "numerical_center_evaluated_cell_count": len(cells),
            "truncated": bool(not_evaluated_ids),
            "stratification": stratification,
        },
        "ik_settings": {
            "solver": "seeded_numerical_GP7_FK_verified",
            "random_restarts": restarts,
            "max_preinsert_solutions": solutions,
            "position_tolerance_m": position_tolerance,
            "rotation_tolerance_rad": rotation_tolerance,
            "max_joint_step_rad": jump,
            "minimum_joint_limit_margin_rad": joint_margin,
            "minimum_normalized_joint_limit_margin": normalized_margin,
            "minimum_sigma": sigma,
        },
        "continuous_robot_cell_certificate": (
            {
                "supplied": False,
                "path": None,
                "file_sha256": None,
                "semantic_sha256": None,
                "hard_gates": None,
            }
            if continuous_certificate is None else {
                "supplied": True,
                "path": str(continuous_certificate.path),
                "file_sha256": continuous_certificate.file_sha256,
                "semantic_sha256": continuous_certificate.semantic_sha256,
                "hard_gates": dict(continuous_certificate.hard_gates),
            }
        ),
        "certified_receiver_cell_ids": certified_ids,
        "provisional_center_path_witness_cell_ids": [
            record["id"] for record in witnesses],
        "numerically_unresolved_cell_ids": [
            record["id"] for record in numerically_unresolved],
        "not_evaluated_cell_ids": not_evaluated_ids,
        "provisional_center_path_witnesses": receiver_witnesses,
        "cells": results,
        "summary": {
            "source_cell_count": len(task_set.cells),
            "eligible_source_cell_count": eligible_count,
            "numerical_center_evaluated_count": len(cells),
            "provisional_center_path_witness_count": len(witnesses),
            "numerically_unresolved_count": len(numerically_unresolved),
            "not_evaluated_eligible_count": len(not_evaluated_ids),
            "certified_receiver_cell_count": len(certified_ids),
        },
        "warnings": [
            "A center-pose IK path does not certify its continuous grasp cell.",
            "Sampled representative poses are retained as evidence only and never evaluated here.",
            "IK continuity is checked only at discrete path samples.",
            "No robot, gripper, connector, PCB, fixture, or other-arm collision was checked.",
            "Numerical multi-seed IK does not enumerate all GP7 branches analytically.",
        ],
    }


__all__ = [
    "InsertionTaskCell",
    "InsertionTaskSet",
    "VerifiedContinuousRobotCertificate",
    "build_robot_insertion_set",
    "load_insertion_task_set",
    "load_verified_continuous_robot_certificate",
    "resolve_world_part_insert",
    "sample_straight_insertion_path",
    "select_task_cells_stratified",
]
