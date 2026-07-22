"""Compose reusable part-relative insertion grasps with a world target.

This module intentionally stops at transform composition and optional numerical
IK.  It does not perform robot/gripper collision checking, Cartesian path
validation, or insertion mechanics.  Keeping that boundary explicit prevents a
reachable TCP endpoint from being mistaken for an executable insertion.

Frame convention follows the rest of the MuJoCo stack: ``T_X_Y`` maps points
from frame ``Y`` into frame ``X``.  A library grasp is ``T_P_E`` and therefore
the corresponding world TCP/contact-frame target is ``T_W_P @ T_P_E``.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..core.se3 import inverse, validate_transform


_EPS = np.finfo(float).eps
_CORRECTION_KEYS = (
    "lateral_x_m",
    "lateral_y_m",
    "axial_m",
    "yaw_deg",
)
_SELECTION_PURPOSES = ("insertion", "preinsert_diagnostic")


def _unit(vector: Any, *, label: str) -> np.ndarray:
    value = np.asarray(vector, dtype=float)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError(f"{label} must be a finite three-vector")
    norm = float(np.linalg.norm(value))
    if norm <= 64.0 * _EPS:
        raise ValueError(f"{label} must be nonzero")
    return value / norm


def _finite_nonnegative(value: Any, *, label: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return result


def _strict_bool(value: Any, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be boolean")
    return value


def _optional_positive_int(value: Any, *, label: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{label} must be a positive integer or null")
    return value


def _nonempty_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_array(value: np.ndarray) -> list:
    return np.asarray(value, dtype=float).tolist()


def normalize_world_frame(value: Mapping[str, Any]) -> dict[str, str]:
    """Require an explicit world-frame identity and calibration provenance."""
    if not isinstance(value, Mapping):
        raise ValueError("world_frame must be a mapping")
    required = ("id", "calibration_fingerprint", "calibration_source")
    missing = [name for name in required if name not in value]
    if missing:
        raise ValueError(f"world_frame requires {', '.join(missing)}")
    return {
        name: _nonempty_string(value[name], label=f"world_frame.{name}")
        for name in required
    }


@dataclass(frozen=True)
class InsertionGraspSeed:
    """Runtime subset of one generated connector-relative grasp record."""

    grasp_id: str
    library_index: int
    status: str
    family: str
    T_P_E: np.ndarray
    required_aperture_m: float
    quality: float
    preinsert_compatible: bool
    seated_compatible: bool
    preinsert_task_rank: int | None
    seated_task_rank: int | None

    def __post_init__(self) -> None:
        if not isinstance(self.grasp_id, str) or not self.grasp_id:
            raise ValueError("grasp ID must be a non-empty string")
        if int(self.library_index) < 0:
            raise ValueError("library_index must be non-negative")
        if not isinstance(self.status, str) or not self.status:
            raise ValueError("grasp status must be a non-empty string")
        if not isinstance(self.family, str) or not self.family:
            raise ValueError("grasp family must be a non-empty string")
        if not isinstance(self.preinsert_compatible, bool):
            raise ValueError("preinsert_compatible must be boolean")
        if not isinstance(self.seated_compatible, bool):
            raise ValueError("seated_compatible must be boolean")
        if self.seated_compatible and not self.preinsert_compatible:
            raise ValueError("seated-compatible grasp must also be preinsert-compatible")
        transform = validate_transform(self.T_P_E)
        transform.setflags(write=False)
        object.__setattr__(self, "T_P_E", transform)
        object.__setattr__(
            self,
            "required_aperture_m",
            _finite_nonnegative(
                self.required_aperture_m, label="required_aperture_m",
            ),
        )
        quality = float(self.quality)
        if not np.isfinite(quality):
            raise ValueError("grasp quality must be finite")
        object.__setattr__(self, "quality", quality)
        for name in ("preinsert_task_rank", "seated_task_rank"):
            rank = getattr(self, name)
            if rank is not None and (
                    not isinstance(rank, int) or isinstance(rank, bool)
                    or rank <= 0):
                raise ValueError(
                    f"{name} must be a positive integer when supplied")
        if ((self.preinsert_task_rank is not None)
                != self.preinsert_compatible):
            raise ValueError(
                "preinsert_task_rank must be present iff preinsert_compatible is true"
            )
        if ((self.seated_task_rank is not None)
                != self.seated_compatible):
            raise ValueError(
                "seated_task_rank must be present iff seated_compatible is true"
            )


@dataclass(frozen=True)
class InsertionPoseLibrary:
    """Validated runtime view of a phase-1 insertion-grasp library."""

    path: Path
    file_sha256: str
    project_id: str
    config_sha256: str | None
    connector_sha256: str
    insertion_axis_P: np.ndarray
    T_I_P: np.ndarray
    preinsert_distance_m: float
    candidates: tuple[InsertionGraspSeed, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "project_id",
            _nonempty_string(self.project_id, label="pose library project_id"),
        )
        for name in ("file_sha256", "connector_sha256"):
            value = getattr(self, name)
            if (not isinstance(value, str) or len(value) != 64
                    or any(character not in "0123456789abcdef" for character in value)):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        axis = _unit(self.insertion_axis_P, label="insertion_axis_P")
        axis.setflags(write=False)
        object.__setattr__(self, "insertion_axis_P", axis)
        transform = validate_transform(self.T_I_P)
        transform.setflags(write=False)
        object.__setattr__(self, "T_I_P", transform)
        distance = float(self.preinsert_distance_m)
        if not np.isfinite(distance) or distance <= 0.0:
            raise ValueError("preinsert_distance_m must be positive and finite")
        object.__setattr__(self, "preinsert_distance_m", distance)
        if len({item.grasp_id for item in self.candidates}) != len(self.candidates):
            raise ValueError("pose library contains duplicate grasp IDs")


@dataclass(frozen=True)
class PCBSocketBinding:
    """Socket transform cryptographically and semantically bound to a library."""

    project_id: str
    compatible_library_project_ids: tuple[str, ...]
    connector_sha256: str
    T_B_P_insert: np.ndarray
    insertion_direction_B: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "project_id",
            _nonempty_string(self.project_id, label="PCB socket project_id"),
        )
        if not self.compatible_library_project_ids:
            raise ValueError(
                "PCB socket requires compatible_pose_library_project_ids"
            )
        if any(not isinstance(item, str) or not item
               for item in self.compatible_library_project_ids):
            raise ValueError(
                "compatible_pose_library_project_ids must contain non-empty strings"
            )
        digest = self.connector_sha256
        if (not isinstance(digest, str) or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)):
            raise ValueError("PCB socket connector SHA must be a lowercase SHA-256 digest")
        transform = validate_transform(self.T_B_P_insert)
        transform.setflags(write=False)
        object.__setattr__(self, "T_B_P_insert", transform)
        axis = _unit(
            self.insertion_direction_B, label="PCB socket insertion_direction_B")
        axis.setflags(write=False)
        object.__setattr__(self, "insertion_direction_B", axis)


def bind_pcb_socket_contract(
    value: Mapping[str, Any],
    library: InsertionPoseLibrary,
) -> PCBSocketBinding:
    """Validate socket/library identity and insertion-axis semantics."""
    if not isinstance(value, Mapping):
        raise ValueError("PCB socket contract root must be a mapping")
    if int(value.get("schema_version", 0)) != 1:
        raise ValueError("unsupported PCB socket schema_version")
    try:
        compatible = value["compatible_pose_library_project_ids"]
        if (not isinstance(compatible, list)
                or any(not isinstance(item, str) or not item for item in compatible)):
            raise ValueError(
                "compatible_pose_library_project_ids must be a list of non-empty strings"
            )
        binding = PCBSocketBinding(
            project_id=value["project_id"],
            compatible_library_project_ids=tuple(compatible),
            connector_sha256=value["assets"]["connector"]["sha256"],
            T_B_P_insert=np.asarray(value["T_B_P_insert"], dtype=float),
            insertion_direction_B=np.asarray(
                value["frames"]["B"]["insertion_direction"], dtype=float),
        )
        socket_tail_axis = _unit(
            value["frames"]["P"]["short_tail_axis"],
            label="PCB socket frames.P.short_tail_axis",
        )
    except KeyError as error:
        raise ValueError(f"PCB socket contract is missing {error.args[0]}") from error

    if library.project_id not in binding.compatible_library_project_ids:
        raise ValueError(
            f"PCB socket is not compatible with pose library project_id "
            f"{library.project_id!r}"
        )
    if binding.connector_sha256 != library.connector_sha256:
        raise ValueError(
            "PCB socket connector SHA does not match the pose library connector SHA"
        )
    if not np.allclose(
            socket_tail_axis, library.insertion_axis_P, atol=1e-9, rtol=0.0):
        raise ValueError(
            "PCB socket short_tail_axis does not match library insertion_axis_P"
        )
    mapped_axis = binding.T_B_P_insert[:3, :3] @ library.insertion_axis_P
    if not np.allclose(
            mapped_axis, binding.insertion_direction_B, atol=1e-9, rtol=0.0):
        raise ValueError(
            "PCB socket R_B_P does not map library insertion_axis_P to the "
            "declared board insertion direction"
        )
    return binding


def load_insertion_pose_library(path: str | Path) -> InsertionPoseLibrary:
    """Load and validate the fields required by the runtime pose query."""
    source = Path(path).resolve()
    with source.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError("pose library root must be a JSON object")
    if int(value.get("schema_version", 0)) != 1:
        raise ValueError("unsupported pose library schema_version")
    task = value.get("task_geometry")
    records = value.get("candidates")
    if not isinstance(task, dict) or not isinstance(records, list):
        raise ValueError("pose library requires task_geometry and candidates")

    candidates = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"candidates[{index}] must be an object")
        try:
            candidates.append(InsertionGraspSeed(
                grasp_id=record["id"],
                library_index=int(record.get("library_index", index)),
                status=record["status"],
                family=record["family"],
                T_P_E=np.asarray(record["T_P_E"], dtype=float),
                required_aperture_m=float(record["required_aperture_m"]),
                quality=float(record["quality"]),
                preinsert_compatible=_strict_bool(
                    record["preinsert_compatible"],
                    label=f"candidates[{index}].preinsert_compatible",
                ),
                seated_compatible=_strict_bool(
                    record["seated_compatible"],
                    label=f"candidates[{index}].seated_compatible",
                ),
                preinsert_task_rank=_optional_positive_int(
                    record.get("preinsert_task_rank"),
                    label=f"candidates[{index}].preinsert_task_rank",
                ),
                seated_task_rank=_optional_positive_int(
                    record.get("seated_task_rank"),
                    label=f"candidates[{index}].seated_task_rank",
                ),
            ))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid candidates[{index}]: {error}") from error

    try:
        return InsertionPoseLibrary(
            path=source,
            file_sha256=_sha256_file(source),
            project_id=str(value["project_id"]),
            config_sha256=(None if value.get("config_sha256") is None
                           else str(value["config_sha256"])),
            connector_sha256=str(value["asset_stats"]["part"]["sha256"]),
            insertion_axis_P=np.asarray(task["insertion_axis_P"], dtype=float),
            T_I_P=np.asarray(task["T_I_P"], dtype=float),
            preinsert_distance_m=float(task["preinsert_distance_m"]),
            candidates=tuple(candidates),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid pose-library task contract: {error}") from error


def world_part_insert_from_board(
    board_world_pose: np.ndarray,
    T_B_P_insert: np.ndarray,
) -> np.ndarray:
    """Compose ``T_W_P_insert = T_W_B @ T_B_P_insert``."""
    return validate_transform(board_world_pose) @ validate_transform(T_B_P_insert)


def resolve_world_part_insert_pose(
    *,
    world_part_insert_pose: np.ndarray | None = None,
    board_world_pose: np.ndarray | None = None,
    T_B_P_insert: np.ndarray | None = None,
) -> tuple[np.ndarray, str]:
    """Resolve exactly one direct-part or board/socket target declaration."""
    direct = world_part_insert_pose is not None
    board_fields = board_world_pose is not None or T_B_P_insert is not None
    if direct and board_fields:
        raise ValueError(
            "provide either world_part_insert_pose or board_world_pose plus "
            "T_B_P_insert, not both"
        )
    if direct:
        return validate_transform(world_part_insert_pose), "world_part_insert_pose"
    if board_world_pose is None or T_B_P_insert is None:
        raise ValueError(
            "target requires world_part_insert_pose or both board_world_pose "
            "and T_B_P_insert"
        )
    return (
        world_part_insert_from_board(board_world_pose, T_B_P_insert),
        "board_world_pose_x_pcb_socket",
    )


def _bound_pair(value: Any, *, label: str) -> list[float]:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        magnitude = abs(float(array))
        array = np.array([-magnitude, magnitude])
    if array.shape != (2,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must be a finite scalar or [minimum, maximum]")
    if array[0] > array[1]:
        raise ValueError(f"{label} minimum must not exceed its maximum")
    return [float(array[0]), float(array[1])]


def normalize_correction_bounds(
    value: Mapping[str, Any] | None,
) -> dict[str, list[float]]:
    """Normalize translation and +Z_I yaw bounds without sampling them."""
    supplied = {} if value is None else dict(value)
    unknown = set(supplied) - set(_CORRECTION_KEYS) - {"lateral_m"}
    if unknown:
        raise ValueError(f"unknown correction bound keys: {sorted(unknown)}")
    if "lateral_m" in supplied and (
            "lateral_x_m" in supplied or "lateral_y_m" in supplied):
        raise ValueError(
            "correction_bounds.lateral_m cannot be combined with axis-specific "
            "lateral bounds"
        )
    if "lateral_m" in supplied:
        supplied["lateral_x_m"] = supplied["lateral_m"]
        supplied["lateral_y_m"] = supplied["lateral_m"]
    supplied.pop("lateral_m", None)
    return {
        key: _bound_pair(supplied.get(key, 0.0), label=f"correction_bounds.{key}")
        for key in _CORRECTION_KEYS
    }


def select_grasp_seeds(
    library: InsertionPoseLibrary,
    selection: Mapping[str, Any] | None = None,
) -> list[InsertionGraspSeed]:
    """Apply deterministic boolean/status/family selection to library seeds."""
    selected, _, _, _ = _select_grasp_seeds(library, selection)
    return selected


def _select_grasp_seeds(
    library: InsertionPoseLibrary,
    selection: Mapping[str, Any] | None,
) -> tuple[list[InsertionGraspSeed], int, bool, dict[str, Any]]:
    policy = normalize_selection(selection)
    statuses = None if "statuses" not in policy else set(policy["statuses"])
    families = None if "families" not in policy else set(policy["families"])
    eligible = []
    for seed in library.candidates:
        if ("preinsert_compatible" in policy
                and seed.preinsert_compatible != policy["preinsert_compatible"]):
            continue
        if ("seated_compatible" in policy
                and seed.seated_compatible != policy["seated_compatible"]):
            continue
        if statuses is not None and seed.status not in statuses:
            continue
        if families is not None and seed.family not in families:
            continue
        eligible.append(seed)

    def rank(item: InsertionGraspSeed) -> tuple:
        primary = (item.seated_task_rank
                   if policy["purpose"] == "insertion"
                   else item.preinsert_task_rank)
        # Purpose invariants guarantee the corresponding rank is present.
        assert primary is not None
        return primary, item.library_index, item.grasp_id

    eligible.sort(key=rank)
    eligible_count = len(eligible)
    output = eligible
    if "max_candidates" in policy:
        output = output[:policy["max_candidates"]]
    truncated = len(output) < eligible_count
    return output, eligible_count, truncated, policy


def normalize_selection(
    selection: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Validate and JSON-normalize a pose-library selection mapping."""
    policy = {} if selection is None else dict(selection)
    allowed = {
        "purpose", "preinsert_compatible", "seated_compatible", "statuses",
        "families", "max_candidates",
    }
    unknown = set(policy) - allowed
    if unknown:
        raise ValueError(f"unknown selection keys: {sorted(unknown)}")
    purpose = policy.get("purpose", "insertion")
    if purpose not in _SELECTION_PURPOSES:
        raise ValueError(
            f"selection.purpose must be one of {list(_SELECTION_PURPOSES)}"
        )
    policy["purpose"] = purpose
    for name in ("preinsert_compatible", "seated_compatible"):
        if name in policy:
            policy[name] = _strict_bool(
                policy[name], label=f"selection.{name}")
    if purpose == "insertion":
        for name in ("preinsert_compatible", "seated_compatible"):
            if name in policy and policy[name] is not True:
                raise ValueError(
                    f"selection purpose insertion requires {name}: true"
                )
            policy[name] = True
    else:
        if ("preinsert_compatible" in policy
                and policy["preinsert_compatible"] is not True):
            raise ValueError(
                "selection purpose preinsert_diagnostic requires "
                "preinsert_compatible: true"
            )
        policy["preinsert_compatible"] = True
    for name in ("statuses", "families"):
        if name not in policy:
            continue
        raw = policy[name]
        if isinstance(raw, str):
            values = [raw]
        elif isinstance(raw, (list, tuple)):
            values = list(raw)
        else:
            raise ValueError(
                f"selection.{name} must be a string or list of strings")
        if not values or any(not isinstance(item, str) or not item for item in values):
            raise ValueError(f"selection.{name} must contain non-empty strings")
        policy[name] = values
    if "max_candidates" in policy:
        if (not isinstance(policy["max_candidates"], int)
                or isinstance(policy["max_candidates"], bool)):
            raise ValueError("selection.max_candidates must be a positive integer")
        maximum = policy["max_candidates"]
        if maximum <= 0:
            raise ValueError("selection.max_candidates must be positive")
        policy["max_candidates"] = maximum
    return policy


def compose_insertion_pose_query(
    library: InsertionPoseLibrary,
    *,
    robot: str,
    T_W_P_insert: np.ndarray,
    world_frame: Mapping[str, Any],
    target_source: str = "world_part_insert_pose",
    preinsert_distance_m: float | None = None,
    correction_bounds: Mapping[str, Any] | None = None,
    selection: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compose every selected ``T_P_E`` into nominal world E targets."""
    robot_name = str(robot).upper()
    if robot_name not in ("A", "B"):
        raise ValueError("robot must be A or B")
    target = validate_transform(T_W_P_insert)
    distance = (library.preinsert_distance_m if preinsert_distance_m is None
                else float(preinsert_distance_m))
    if not np.isfinite(distance) or distance <= 0.0:
        raise ValueError("preinsert_distance_m must be positive and finite")

    axis_W = target[:3, :3] @ library.insertion_axis_P
    preinsert = target.copy()
    preinsert[:3, 3] -= distance * axis_W
    T_W_I = target @ inverse(library.T_I_P)
    # The library contract says +Z_I is the insertion direction. Fail if a
    # malformed library makes its two equivalent representations disagree.
    if not np.allclose(T_W_I[:3, 2], axis_W, atol=1e-7, rtol=0.0):
        raise ValueError(
            "pose library T_I_P is inconsistent with insertion_axis_P"
        )

    seeds, eligible_count, truncated, selection_policy = _select_grasp_seeds(
        library, selection)
    purpose = selection_policy["purpose"]
    targets = []
    for seed in seeds:
        nominal_insert = target @ seed.T_P_E
        diagnostic_only = not seed.seated_compatible
        targets.append({
            "grasp_id": seed.grasp_id,
            "library_index": seed.library_index,
            "source_status": seed.status,
            "family": seed.family,
            "claim_level": (
                "preinsert_diagnostic" if diagnostic_only else "composed_target"
            ),
            "required_aperture_m": seed.required_aperture_m,
            "quality": seed.quality,
            "preinsert_compatible": seed.preinsert_compatible,
            "seated_compatible": seed.seated_compatible,
            "T_P_E": _json_array(seed.T_P_E),
            "T_W_E_preinsert": _json_array(preinsert @ seed.T_P_E),
            # A preinsert-only record must never look like an executable seated
            # target. Preserve the arithmetic result only as an explicitly
            # non-executable diagnostic witness.
            "T_W_E_insert": (None if diagnostic_only
                              else _json_array(nominal_insert)),
            "T_W_E_insert_nominal_witness": (
                _json_array(nominal_insert) if diagnostic_only else None
            ),
            "ik": None,
        })

    return {
        "schema_version": 1,
        "claim_level": (
            "preinsert_diagnostic" if purpose == "preinsert_diagnostic"
            else "composed_target"
        ),
        "certified": False,
        "robot": robot_name,
        "world_frame": normalize_world_frame(world_frame),
        "pose_library": {
            "path": str(library.path),
            "sha256": library.file_sha256,
            "project_id": library.project_id,
            "config_sha256": library.config_sha256,
            "connector_sha256": library.connector_sha256,
            "candidate_count": len(library.candidates),
        },
        "target_source": target_source,
        "frame_contract": {
            "grasp": "T_P_E maps E into connector frame P",
            "composition": "T_W_E = T_W_P @ T_P_E",
            "correction_frame": "I; bounds are recorded but not swept by this query",
        },
        "T_W_I": _json_array(T_W_I),
        "T_W_P_insert": _json_array(target),
        "T_W_P_preinsert": _json_array(preinsert),
        "insertion_axis_P": _json_array(library.insertion_axis_P),
        "insertion_axis_W": _json_array(axis_W),
        "preinsert_distance_m": distance,
        "correction_bounds_I": normalize_correction_bounds(correction_bounds),
        "correction_contract": {
            "delta_definition": "Delta_I = Trans_I(dx,dy,dz) @ RotZ_I(yaw)",
            "composition": "T_W_E(delta) = T_W_I @ Delta_I @ T_I_P @ T_P_E",
            "side": "Delta_I left-multiplies nominal T_I_P inside frame I",
            "rotation_axis": "+Z_I only",
            "rotation_pivot": "origin of insertion frame I",
            "point_operation_order": (
                "yaw about the I origin first, then translation expressed in "
                "nominal I axes"
            ),
            "evaluated": False,
        },
        "selection": selection_policy,
        "eligible_candidate_count_before_limit": eligible_count,
        "selected_candidate_count": len(targets),
        "selection_truncated": truncated,
        "continuous_complete": False,
        "targets": targets,
        "ik_evaluation": {
            "performed": False,
            "claim_level": None,
            "tcp_status": "not_evaluated",
        },
        "excluded_checks": [
            "robot, gripper, part, PCB, fixture, and other-arm collision",
            "same-branch Cartesian insertion path continuity",
            "jaw aperture actuation and pad contact",
            "correction-envelope sweep",
            "insertion force, friction, compliance, and uncertainty",
        ],
        "warnings": [
            "Composed targets are transform arithmetic only, not executable poses.",
            "The connector phase-1 library uses provisional gripper/TCP assumptions.",
            "The finite sampled library is not complete over the continuous grasp family.",
        ],
    }


def _ik_solution_record(kinematics: Any, robot: str, result: Any) -> dict[str, Any]:
    q = np.asarray(result.q, dtype=float)
    record = {
        "q": _json_array(q),
        "position_error_m": float(result.position_error),
        "rotation_error_rad": float(result.rotation_error),
        "iterations": int(result.iterations),
    }
    if hasattr(kinematics, "normalized_limit_margin"):
        record["normalized_joint_limit_margin"] = float(
            kinematics.normalized_limit_margin(robot, q))
    if hasattr(kinematics, "manipulability"):
        record["manipulability"] = float(kinematics.manipulability(robot, q))
    return record


def attach_provisional_gp7_ik(
    query: Mapping[str, Any],
    kinematics: Any,
    *,
    acknowledge_provisional_tcp: bool,
    random_restarts: int = 18,
    max_solutions: int = 8,
    position_tolerance_m: float = 7e-4,
    rotation_tolerance_rad: float = np.radians(0.35),
) -> dict[str, Any]:
    """Attach FK-verified endpoint IK, explicitly without collision/path claims.

    The caller must acknowledge that the compiled ``<robot>_tcp`` may not be a
    calibrated realization of the supplied long-finger gripper's contact frame
    ``E``.  This keeps the optional check useful for software/reachability work
    without silently upgrading it to physical certification.
    """
    if not isinstance(acknowledge_provisional_tcp, bool):
        raise ValueError("acknowledge_provisional_tcp must be boolean")
    if acknowledge_provisional_tcp is not True:
        raise ValueError(
            "--solve-ik requires explicit provisional TCP acknowledgment; "
            "the supplied gripper has no calibrated flange-to-E transform"
        )
    restarts = int(random_restarts)
    maximum = int(max_solutions)
    if restarts < 0 or maximum <= 0:
        raise ValueError("IK restart count must be non-negative and max_solutions positive")
    position_tolerance = float(position_tolerance_m)
    rotation_tolerance = float(rotation_tolerance_rad)
    if (not np.isfinite(position_tolerance) or position_tolerance <= 0.0
            or not np.isfinite(rotation_tolerance) or rotation_tolerance <= 0.0):
        raise ValueError("IK tolerances must be positive and finite")

    output = deepcopy(dict(query))
    robot = str(output["robot"]).upper()
    reference_q = np.asarray(kinematics.get_q(robot), dtype=float).copy()

    def solve(target: np.ndarray, grasp_id: str, stage: str):
        kinematics.set_q(robot, reference_q)
        digest = hashlib.sha256(
            b"insertion-query-ik-v1\0"
            + robot.encode("ascii")
            + grasp_id.encode("utf-8")
            + stage.encode("ascii")
            + np.round(target, 10).tobytes()
        ).digest()
        rng = np.random.default_rng(
            int.from_bytes(digest[:8], "little", signed=False))
        results = kinematics.solutions(
            robot,
            target,
            random_restarts=restarts,
            max_solutions=maximum,
            rng=rng,
            position_tolerance=position_tolerance,
            rotation_tolerance=rotation_tolerance,
        )
        return [_ik_solution_record(kinematics, robot, result) for result in results]

    reachable = 0
    preinsert_reachable = 0
    insert_reachable = 0
    try:
        for record in output["targets"]:
            pre = solve(
                np.asarray(record["T_W_E_preinsert"], dtype=float),
                record["grasp_id"],
                "preinsert",
            )
            insert_target = record["T_W_E_insert"]
            if insert_target is None:
                insert = None
                insert_skipped_reason = (
                    "preinsert-only diagnostic has no executable T_W_E_insert"
                )
            else:
                insert = solve(
                    np.asarray(insert_target, dtype=float),
                    record["grasp_id"],
                    "insert",
                )
                insert_skipped_reason = None
            pre_ok = bool(pre)
            insert_ok = bool(insert) if insert is not None else False
            if pre_ok:
                preinsert_reachable += 1
            if insert_ok:
                insert_reachable += 1
            purpose_ok = pre_ok if insert is None else bool(pre_ok and insert_ok)
            if purpose_ok:
                reachable += 1
                record["claim_level"] = (
                    "ik_reachable_preinsert_diagnostic"
                    if insert is None else "ik_reachable_endpoints"
                )
            record["ik"] = {
                "claim_level": "ik_only_provisional_tcp",
                "purpose_endpoint_requirement_reachable": purpose_ok,
                "both_endpoints_reachable": bool(pre_ok and insert_ok),
                "preinsert_solutions": pre,
                "insert_solutions": insert,
                "insert_skipped_reason": insert_skipped_reason,
                "same_branch_continuity_checked": False,
                "collision_checked": False,
                "path_checked": False,
            }
    finally:
        kinematics.set_q(robot, reference_q)

    output["claim_level"] = "ik_only_provisional_tcp"
    output["certified"] = False
    output["ik_reachable_candidate_count"] = reachable
    output["ik_reachable_preinsert_candidate_count"] = preinsert_reachable
    output["ik_reachable_insert_candidate_count"] = insert_reachable
    output["ik_evaluation"] = {
        "performed": True,
        "claim_level": "ik_only_provisional_tcp",
        "tcp_status": "provisional_or_unverified",
        "provisional_tcp_acknowledged": True,
        "random_restarts": restarts,
        "max_solutions": maximum,
        "position_tolerance_m": position_tolerance,
        "rotation_tolerance_rad": rotation_tolerance,
        "collision_checked": False,
        "path_checked": False,
        "diagnostic_insert_targets_skipped": sum(
            item["T_W_E_insert"] is None for item in output["targets"]),
    }
    output["warnings"].extend([
        "IK targets the compiled robot TCP site, whose flange-to-E calibration is unverified.",
        "Endpoint IK does not prove a common branch, collision-free path, or insertion feasibility.",
    ])
    return output


__all__ = [
    "InsertionGraspSeed",
    "InsertionPoseLibrary",
    "PCBSocketBinding",
    "attach_provisional_gp7_ik",
    "bind_pcb_socket_contract",
    "compose_insertion_pose_query",
    "load_insertion_pose_library",
    "normalize_correction_bounds",
    "normalize_selection",
    "normalize_world_frame",
    "resolve_world_part_insert_pose",
    "select_grasp_seeds",
    "world_part_insert_from_board",
]
