"""Deterministic previews of sampled parallel-jaw grasp candidates.

The renderer consumes the machine-readable output of
``scripts/generate_parallel_jaw_grasps.py`` and produces candidate-aligned
orthographic views.  It deliberately draws only geometry represented by the
generator:

* the actual prepared CAD triangle surface;
* the two ideal contacts and their normals;
* the ideal contact frame ``E``;
* rectangular contact pads; and
* wireframe jaw centerlines plus the idealized palm-depth datum.

It does not invent a finite physical palm, finger thickness, gripper TCP, or
collision model.  The preview is therefore explanatory evidence for sampled
object-geometry candidates, not a feasibility certificate.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import struct
from typing import Any, Mapping, Sequence
import zlib

import numpy as np

from ..core.se3 import validate_transform
from ..offline_tools.artifacts import atomic_write_bytes
from .cad_preprocess import prepare_cad, verify_preparation
from .part_mesh import load_prepared_triangle_mesh


ROOT = Path(__file__).resolve().parents[2]
SOURCE_ARTIFACT_TYPE = "sampled_parallel_jaw_grasp_candidates"
RELIABLE_CLAIM_LEVEL = "resolution_qualified_object_geometry_candidate"
UNRELIABLE_CLAIM_LEVEL = "unreliable_mesh_sampled_candidate"
ARTIFACT_TYPE = "parallel_jaw_grasp_candidate_preview"
DEFAULT_GENERATED_ROOT = ROOT / "build" / "parallel_jaw_grasps" / "cad"
MAX_DISPLAYED_CANDIDATES = 8


_FONT_3X5 = {
    " ": "000/000/000/000/000",
    "-": "000/000/111/000/000",
    ".": "000/000/000/000/010",
    ":": "000/010/000/010/000",
    "/": "001/001/010/100/100",
    "(": "010/100/100/100/010",
    ")": "010/001/001/001/010",
    "+": "000/010/111/010/000",
    "=": "000/111/000/111/000",
    "%": "101/001/010/100/101",
    "?": "110/001/010/000/010",
    "_": "000/000/000/000/111",
    "0": "111/101/101/101/111",
    "1": "010/110/010/010/111",
    "2": "110/001/111/100/111",
    "3": "110/001/110/001/110",
    "4": "101/101/111/001/001",
    "5": "111/100/110/001/110",
    "6": "011/100/111/101/111",
    "7": "111/001/010/010/010",
    "8": "111/101/111/101/111",
    "9": "111/101/111/001/110",
    "A": "010/101/111/101/101",
    "B": "110/101/110/101/110",
    "C": "011/100/100/100/011",
    "D": "110/101/101/101/110",
    "E": "111/100/110/100/111",
    "F": "111/100/110/100/100",
    "G": "011/100/101/101/011",
    "H": "101/101/111/101/101",
    "I": "111/010/010/010/111",
    "J": "001/001/001/101/010",
    "K": "101/101/110/101/101",
    "L": "100/100/100/100/111",
    "M": "101/111/111/101/101",
    "N": "101/111/111/111/101",
    "O": "010/101/101/101/010",
    "P": "110/101/110/100/100",
    "Q": "010/101/101/111/011",
    "R": "110/101/110/101/101",
    "S": "011/100/010/001/110",
    "T": "111/010/010/010/010",
    "U": "101/101/101/101/111",
    "V": "101/101/101/101/010",
    "W": "101/101/111/111/101",
    "X": "101/101/010/101/101",
    "Y": "101/101/010/010/010",
    "Z": "111/001/010/100/111",
}


@dataclass(frozen=True)
class _Candidate:
    record: Mapping[str, Any]
    candidate_id: str
    index: int
    source_rank: int
    T_P_E: np.ndarray
    contacts_P: np.ndarray
    normals_P: np.ndarray
    closing_P: np.ndarray
    approach_P: np.ndarray
    opening_m: float
    quality: float


@dataclass(frozen=True)
class _ValidatedDocument:
    value: Mapping[str, Any]
    candidates: tuple[_Candidate, ...]
    cad: Mapping[str, Any]
    gripper: Mapping[str, Any]
    opening_range_m: np.ndarray
    pad_size_m: np.ndarray
    finger_depth_m: float
    reliable_input: bool
    not_checked: tuple[str, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate_identifier(
    transform: np.ndarray,
    contacts: np.ndarray,
    opening_m: float,
) -> str:
    identity = {
        "T_P_E": np.round(np.asarray(transform, dtype=float), 10).tolist(),
        "contacts_P_m": np.round(
            np.asarray(contacts, dtype=float), 10).tolist(),
        "required_opening_m": round(float(opening_m), 10),
    }
    payload = json.dumps(
        identity,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "grasp_" + hashlib.sha256(payload).hexdigest()[:16]


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _finite_array(
    value: Any,
    shape: tuple[int, ...],
    *,
    label: str,
) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must be a finite array with shape {shape}")
    return result


def _finite_scalar(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _load_json(path: Path) -> tuple[Mapping[str, Any], bytes]:
    payload = path.read_bytes()
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON in {path}: {error}") from error
    return _mapping(value, label="candidate document"), payload


def _validate_document(
    document: Mapping[str, Any],
    *,
    allow_unreliable_input: bool,
) -> _ValidatedDocument:
    if not isinstance(allow_unreliable_input, bool):
        raise TypeError("allow_unreliable_input must be a boolean")
    if document.get("schema_version") != 1:
        raise ValueError("candidate document schema_version must be 1")
    if document.get("artifact_type") != SOURCE_ARTIFACT_TYPE:
        raise ValueError(
            f"input must have artifact_type={SOURCE_ARTIFACT_TYPE}"
        )
    if document.get("continuous_exhaustive") is not False:
        raise ValueError("candidate document must declare continuous_exhaustive=false")
    candidate_cap_applied = document.get("candidate_cap_applied")
    all_returned = document.get(
        "all_deduplicated_accepted_candidates_returned")
    if not isinstance(candidate_cap_applied, bool) or not isinstance(
        all_returned, bool
    ):
        raise ValueError(
            "candidate cap/completeness flags must be boolean")
    if candidate_cap_applied == all_returned:
        raise ValueError(
            "candidate cap/completeness flags must be logical opposites")
    sampling = _mapping(document.get("sampling"), label="sampling")
    for key in (
        "surface_samples",
        "closing_directions_per_surface",
        "approaches_per_contact_pair",
    ):
        value = sampling.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"sampling.{key} must be a positive integer")
    maximum_candidates = sampling.get("max_candidates")
    if candidate_cap_applied:
        if (
            isinstance(maximum_candidates, bool)
            or not isinstance(maximum_candidates, int)
            or maximum_candidates <= 0
        ):
            raise ValueError(
                "a configured candidate cap requires positive "
                "sampling.max_candidates"
            )
    elif maximum_candidates is not None:
        raise ValueError(
            "sampling.max_candidates must be null when no cap is configured")
    feasibility = _mapping(
        document.get("feasibility_contract"),
        label="feasibility_contract",
    )
    raw_not_checked = feasibility.get("not_checked")
    if (
        not isinstance(raw_not_checked, list)
        or any(not isinstance(item, str) or not item for item in raw_not_checked)
    ):
        raise ValueError(
            "feasibility_contract.not_checked must contain non-empty strings")
    not_checked = tuple(raw_not_checked)

    claim_level = document.get("claim_level")
    if claim_level not in (RELIABLE_CLAIM_LEVEL, UNRELIABLE_CLAIM_LEVEL):
        raise ValueError(f"unsupported candidate claim_level {claim_level!r}")
    cad = _mapping(document.get("cad"), label="cad")
    topology = _mapping(cad.get("topology_audit"), label="cad.topology_audit")
    for key in (
        "normal_ray_assumptions_accepted",
        "closed_consistently_wound_two_manifold",
    ):
        if not isinstance(topology.get(key), bool):
            raise ValueError(
                f"cad.topology_audit.{key} must be boolean")
    topology_zero_fields = (
        "degenerate_face_count",
        "boundary_edge_count",
        "nonmanifold_edge_count",
        "inconsistent_paired_edge_orientation_count",
        "zero_or_unresolved_volume_component_count",
    )
    topology_counts: dict[str, int] = {}
    for key in topology_zero_fields:
        value = topology.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                f"cad.topology_audit.{key} must be a non-negative integer")
        topology_counts[key] = value
    mixed_winding = topology.get("mixed_component_winding_signs")
    if not isinstance(mixed_winding, bool):
        raise ValueError(
            "cad.topology_audit.mixed_component_winding_signs must be boolean")
    topology_reliable = (
        topology.get("normal_ray_assumptions_accepted") is True
        and topology.get("closed_consistently_wound_two_manifold") is True
        and all(value == 0 for value in topology_counts.values())
        and mixed_winding is False
    )
    expected_claim = (
        RELIABLE_CLAIM_LEVEL
        if topology_reliable
        else UNRELIABLE_CLAIM_LEVEL
    )
    if claim_level != expected_claim:
        raise ValueError(
            "candidate claim_level contradicts its topology audit")
    reliable_input = topology_reliable
    if not reliable_input and not allow_unreliable_input:
        raise ValueError(
            "candidate document has unreliable mesh/contact semantics; pass "
            "allow_unreliable_input=True (CLI: --allow-unreliable-input) only "
            "for a diagnostic preview"
        )

    gripper = _mapping(document.get("gripper_model"), label="gripper_model")
    if gripper.get("type") != "ideal_symmetric_parallel_jaw":
        raise ValueError("unsupported gripper_model.type")
    opening_range = _finite_array(
        gripper.get("opening_range_m"),
        (2,),
        label="gripper_model.opening_range_m",
    )
    if opening_range[0] < 0.0 or opening_range[1] <= opening_range[0]:
        raise ValueError("gripper opening range must satisfy 0 <= min < max")
    pad_size = _finite_array(
        gripper.get("pad_size_m"),
        (2,),
        label="gripper_model.pad_size_m",
    )
    if np.any(pad_size <= 0.0):
        raise ValueError("gripper pad dimensions must be positive")
    finger_depth = _finite_scalar(
        gripper.get("finger_tip_to_palm_depth_m"),
        label="gripper_model.finger_tip_to_palm_depth_m",
    )
    if finger_depth <= 0.0:
        raise ValueError("gripper finger depth must be positive")
    friction = _finite_scalar(
        gripper.get("friction_coefficient"),
        label="gripper_model.friction_coefficient",
    )
    if friction < 0.0:
        raise ValueError("gripper friction coefficient must be non-negative")

    bounds_min = _finite_array(
        cad.get("bounds_min_P_m"), (3,), label="cad.bounds_min_P_m")
    bounds_max = _finite_array(
        cad.get("bounds_max_P_m"), (3,), label="cad.bounds_max_P_m")
    if np.any(bounds_max < bounds_min):
        raise ValueError("CAD bounds are invalid")
    coordinate_scale = max(
        float(np.linalg.norm(bounds_max - bounds_min)),
        float(opening_range[1]),
        finger_depth,
        1e-6,
    )
    position_tolerance = max(2e-10, 2e-8 * coordinate_scale)

    raw_candidates = document.get("candidates")
    if not isinstance(raw_candidates, list):
        raise ValueError("candidates must be an array")
    candidate_count = document.get("candidate_count")
    if (
        isinstance(candidate_count, bool)
        or not isinstance(candidate_count, int)
        or candidate_count != len(raw_candidates)
    ):
        raise ValueError("candidate_count must match candidates")
    if candidate_cap_applied and candidate_count > maximum_candidates:
        raise ValueError("candidate_count exceeds sampling.max_candidates")
    if not raw_candidates:
        raise ValueError("candidate document contains no candidates to display")

    parsed: list[_Candidate] = []
    seen_ids: set[str] = set()
    seen_indices: set[int] = set()
    for source_rank, raw in enumerate(raw_candidates):
        record = _mapping(raw, label=f"candidates[{source_rank}]")
        candidate_id = record.get("id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError(f"candidates[{source_rank}].id must be non-empty")
        if candidate_id in seen_ids:
            raise ValueError(f"duplicate candidate id {candidate_id!r}")
        seen_ids.add(candidate_id)
        index = record.get("index")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError(
                f"candidates[{source_rank}].index must be non-negative")
        if index in seen_indices:
            raise ValueError(f"duplicate candidate index {index}")
        seen_indices.add(index)
        if index != source_rank:
            raise ValueError(
                f"candidates[{source_rank}].index must equal its generator "
                "output rank"
            )

        try:
            transform = validate_transform(
                np.asarray(record.get("T_P_E"), dtype=float),
                atol=2e-8,
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"candidates[{source_rank}].T_P_E is invalid: {error}"
            ) from error
        contacts = _finite_array(
            record.get("contact_points_P_m"),
            (2, 3),
            label=f"candidates[{source_rank}].contact_points_P_m",
        )
        normals = _finite_array(
            record.get("contact_normals_P"),
            (2, 3),
            label=f"candidates[{source_rank}].contact_normals_P",
        )
        normal_lengths = np.linalg.norm(normals, axis=1)
        if not np.allclose(normal_lengths, 1.0, atol=2e-8, rtol=0.0):
            raise ValueError(
                f"candidates[{source_rank}] contact normals must be unit")
        closing = _finite_array(
            record.get("closing_direction_P"),
            (3,),
            label=f"candidates[{source_rank}].closing_direction_P",
        )
        approach = _finite_array(
            record.get("approach_direction_P"),
            (3,),
            label=f"candidates[{source_rank}].approach_direction_P",
        )
        if not math.isclose(
            float(np.linalg.norm(closing)), 1.0, rel_tol=0.0, abs_tol=2e-8
        ):
            raise ValueError(
                f"candidates[{source_rank}] closing direction must be unit")
        if not math.isclose(
            float(np.linalg.norm(approach)), 1.0, rel_tol=0.0, abs_tol=2e-8
        ):
            raise ValueError(
                f"candidates[{source_rank}] approach direction must be unit")
        opening = _finite_scalar(
            record.get("required_opening_m"),
            label=f"candidates[{source_rank}].required_opening_m",
        )
        quality = _finite_scalar(
            record.get("quality"),
            label=f"candidates[{source_rank}].quality",
        )
        if opening <= 0.0:
            raise ValueError(
                f"candidates[{source_rank}].required_opening_m must be positive")
        quality_components = {
            "quality": quality,
            "antipodal_quality": _finite_scalar(
                record.get("antipodal_quality"),
                label=f"candidates[{source_rank}].antipodal_quality",
            ),
            "support_quality": _finite_scalar(
                record.get("support_quality"),
                label=f"candidates[{source_rank}].support_quality",
            ),
            "opening_margin": _finite_scalar(
                record.get("opening_margin"),
                label=f"candidates[{source_rank}].opening_margin",
            ),
        }
        if any(
            value < -2e-8 or value > 1.0 + 2e-8
            for value in quality_components.values()
        ):
            raise ValueError(
                f"candidates[{source_rank}] quality terms must lie in [0, 1]")
        palm_clearance = _finite_scalar(
            record.get("idealized_palm_clearance_m"),
            label=f"candidates[{source_rank}].idealized_palm_clearance_m",
        )
        if (
            palm_clearance < -position_tolerance
            or palm_clearance > finger_depth + position_tolerance
        ):
            raise ValueError(
                f"candidates[{source_rank}] palm clearance is inconsistent "
                "with finger depth"
            )
        if (
            opening < opening_range[0] - position_tolerance
            or opening > opening_range[1] + position_tolerance
        ):
            raise ValueError(
                f"candidates[{source_rank}] opening is outside gripper range")

        rotation = transform[:3, :3]
        origin = transform[:3, 3]
        midpoint = np.mean(contacts, axis=0)
        separation = contacts[1] - contacts[0]
        separation_length = float(np.linalg.norm(separation))
        contacts_E = (contacts - origin) @ rotation
        expected_contacts_E = np.array([
            [0.0, -0.5 * opening, 0.0],
            [0.0, +0.5 * opening, 0.0],
        ])
        if not np.allclose(origin, midpoint, atol=position_tolerance, rtol=0.0):
            raise ValueError(
                f"candidates[{source_rank}] E origin is not contact midpoint")
        if not math.isclose(
            separation_length,
            opening,
            rel_tol=2e-8,
            abs_tol=position_tolerance,
        ):
            raise ValueError(
                f"candidates[{source_rank}] opening disagrees with contacts")
        if not np.allclose(
            contacts_E,
            expected_contacts_E,
            atol=position_tolerance,
            rtol=0.0,
        ):
            raise ValueError(
                f"candidates[{source_rank}] contacts are inconsistent with "
                "the serialized E frame"
            )
        if not np.allclose(rotation[:, 1], closing, atol=2e-8, rtol=0.0):
            raise ValueError(
                f"candidates[{source_rank}] +Y_E disagrees with closing direction")
        if not np.allclose(rotation[:, 2], approach, atol=2e-8, rtol=0.0):
            raise ValueError(
                f"candidates[{source_rank}] +Z_E disagrees with approach direction")
        if not np.allclose(
            separation / separation_length,
            closing,
            atol=2e-8,
            rtol=0.0,
        ):
            raise ValueError(
                f"candidates[{source_rank}] contact order disagrees with +Y_E")
        alignment_0 = float(normals[0] @ (-closing))
        alignment_1 = float(normals[1] @ closing)
        friction_cosine = 1.0 / math.sqrt(1.0 + friction * friction)
        declared_antipodal = quality_components["antipodal_quality"]
        if not math.isclose(
            min(alignment_0, alignment_1),
            declared_antipodal,
            rel_tol=2e-8,
            abs_tol=2e-8,
        ):
            raise ValueError(
                f"candidates[{source_rank}] antipodal quality disagrees "
                "with contact normals"
            )
        if min(alignment_0, alignment_1) + 1e-10 < friction_cosine:
            raise ValueError(
                f"candidates[{source_rank}] contact normals violate the "
                "declared Coulomb friction cone"
            )
        normalized_antipodal = float(np.clip(
            (declared_antipodal - friction_cosine)
            / max(1.0 - friction_cosine, 1e-12),
            0.0,
            1.0,
        ))
        expected_quality = (
            0.45 * normalized_antipodal
            + 0.25 * quality_components["support_quality"]
            + 0.15 * quality_components["opening_margin"]
            + 0.15 * float(np.clip(
                palm_clearance / finger_depth, 0.0, 1.0))
        )
        if not math.isclose(
            quality,
            expected_quality,
            rel_tol=2e-8,
            abs_tol=2e-8,
        ):
            raise ValueError(
                f"candidates[{source_rank}] quality disagrees with its "
                "serialized components"
            )
        expected_id = _candidate_identifier(transform, contacts, opening)
        if candidate_id != expected_id:
            raise ValueError(
                f"candidates[{source_rank}].id does not match its pose/contact "
                "identity"
            )

        parsed.append(_Candidate(
            record=record,
            candidate_id=candidate_id,
            index=index,
            source_rank=source_rank,
            T_P_E=transform,
            contacts_P=contacts,
            normals_P=normals,
            closing_P=closing,
            approach_P=approach,
            opening_m=opening,
            quality=quality,
        ))

    return _ValidatedDocument(
        value=document,
        candidates=tuple(parsed),
        cad=cad,
        gripper=gripper,
        opening_range_m=opening_range,
        pad_size_m=pad_size,
        finger_depth_m=finger_depth,
        reliable_input=reliable_input,
        not_checked=not_checked,
    )


def _pose_display_distance(
    first: _Candidate,
    second: _Candidate,
    *,
    position_scale_m: float,
    opening_scale_m: float,
) -> float:
    position = min(
        float(np.linalg.norm(
            first.T_P_E[:3, 3] - second.T_P_E[:3, 3]
        )) / position_scale_m,
        1.0,
    )
    closing_dot = float(np.clip(
        abs(first.closing_P @ second.closing_P), 0.0, 1.0))
    closing = (2.0 / math.pi) * math.acos(closing_dot)
    approach_dot = float(np.clip(
        first.approach_P @ second.approach_P, -1.0, 1.0))
    approach = math.acos(approach_dot) / math.pi
    opening = min(
        abs(first.opening_m - second.opening_m) / opening_scale_m,
        1.0,
    )
    return math.sqrt(
        (position * position
         + closing * closing
         + approach * approach
         + opening * opening)
        / 4.0
    )


def _select_candidates(
    validated: _ValidatedDocument,
    *,
    count: int,
    selection_mode: str,
    candidate_ids: Sequence[str],
) -> tuple[tuple[_Candidate, ...], dict[str, Any]]:
    requested_ids = tuple(candidate_ids)
    if requested_ids:
        if len(requested_ids) > MAX_DISPLAYED_CANDIDATES:
            raise ValueError(
                f"at most {MAX_DISPLAYED_CANDIDATES} candidate IDs may be displayed")
        if any(not isinstance(value, str) or not value for value in requested_ids):
            raise ValueError("candidate IDs must be non-empty strings")
        if len(set(requested_ids)) != len(requested_ids):
            raise ValueError("candidate IDs must be unique")
        by_id = {
            candidate.candidate_id: candidate
            for candidate in validated.candidates
        }
        missing = [value for value in requested_ids if value not in by_id]
        if missing:
            raise ValueError(
                "candidate IDs were not found: " + ", ".join(missing))
        selected = tuple(by_id[value] for value in requested_ids)
        metadata = {
            "mode": "explicit_ids",
            "policy": "exact candidate IDs in caller-supplied order",
            "ranking_intent": "explicit inspection; no ranking inferred",
            "requested_candidate_ids": list(requested_ids),
        }
    else:
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or not 1 <= count <= MAX_DISPLAYED_CANDIDATES
        ):
            raise ValueError(
                f"count must be an integer in [1, {MAX_DISPLAYED_CANDIDATES}]")
        normalized_mode = {
            "ranked": "ranked",
            "pose-diverse": "pose_diverse",
            "pose_diverse": "pose_diverse",
            "coverage": "pose_diverse",
        }.get(str(selection_mode))
        if normalized_mode is None:
            raise ValueError("selection_mode must be ranked or pose_diverse")
        retained_count = min(count, len(validated.candidates))
        if normalized_mode == "ranked":
            selected = validated.candidates[:retained_count]
            metadata = {
                "mode": "ranked",
                "policy": "first candidates in generator output order",
                "ranking_intent": (
                    "generator object-only heuristic order; not a grasp "
                    "success probability or physical-feasibility ranking"
                ),
            }
        else:
            cad_extent = _finite_array(
                validated.cad.get("extent_P_m"),
                (3,),
                label="cad.extent_P_m",
            )
            position_scale = max(float(np.linalg.norm(cad_extent)), 1e-12)
            opening_scale = max(
                float(np.ptp(validated.opening_range_m)), 1e-12)
            selected_indices = [0]
            remaining = set(range(1, len(validated.candidates)))
            while remaining and len(selected_indices) < retained_count:
                best_index = -1
                best_distance = -1.0
                for index in sorted(remaining):
                    candidate = validated.candidates[index]
                    minimum_distance = min(
                        _pose_display_distance(
                            candidate,
                            validated.candidates[old],
                            position_scale_m=position_scale,
                            opening_scale_m=opening_scale,
                        )
                        for old in selected_indices
                    )
                    if minimum_distance > best_distance + 1e-15:
                        best_index = index
                        best_distance = minimum_distance
                selected_indices.append(best_index)
                remaining.remove(best_index)
            selected = tuple(
                validated.candidates[index] for index in selected_indices)
            metadata = {
                "mode": "pose_diverse",
                "policy": (
                    "seed source rank 0 then greedy max-min display distance"
                ),
                "ranking_intent": (
                    "deterministic display coverage; not a new feasibility "
                    "or grasp-quality ranking"
                ),
                "distance_metric": {
                    "form": (
                        "root-mean-square of normalized position, unoriented "
                        "closing-axis angle, oriented approach-axis angle, "
                        "and opening difference"
                    ),
                    "position_scale_m": position_scale,
                    "closing_axis_sign_invariant": True,
                    "opposite_approach_distinct": True,
                    "opening_scale_m": opening_scale,
                    "equal_component_weights": True,
                },
            }

    metadata.update({
        "source_candidate_count": len(validated.candidates),
        "displayed_candidate_count": len(selected),
        "displayed": [
            {
                "id": candidate.candidate_id,
                "index": candidate.index,
                "source_rank": candidate.source_rank,
            }
            for candidate in selected
        ],
    })
    return selected, metadata


def select_candidate_records(
    document: Mapping[str, Any],
    *,
    count: int = 4,
    selection_mode: str = "pose_diverse",
    candidate_ids: Sequence[str] = (),
    allow_unreliable_input: bool = False,
) -> dict[str, Any]:
    """Return deterministic display candidates without changing source claims."""
    validated = _validate_document(
        document,
        allow_unreliable_input=allow_unreliable_input,
    )
    selected, metadata = _select_candidates(
        validated,
        count=count,
        selection_mode=selection_mode,
        candidate_ids=candidate_ids,
    )
    return {
        "selection": metadata,
        "candidates": [dict(candidate.record) for candidate in selected],
    }


def ideal_parallel_jaw_glyph_E(
    required_opening_m: float,
    pad_size_m: Sequence[float],
    finger_tip_to_palm_depth_m: float,
) -> dict[str, Any]:
    """Return the exact ideal wireframe glyph in contact frame ``E``.

    The two pad rectangles have no thickness.  The jaw and palm records are
    centerlines/data, not physical collision solids.
    """
    opening = _finite_scalar(
        required_opening_m, label="required_opening_m")
    pad = _finite_array(pad_size_m, (2,), label="pad_size_m")
    depth = _finite_scalar(
        finger_tip_to_palm_depth_m,
        label="finger_tip_to_palm_depth_m",
    )
    if opening <= 0.0 or np.any(pad <= 0.0) or depth <= 0.0:
        raise ValueError("glyph dimensions must be positive")
    half_opening = 0.5 * opening
    half_width = 0.5 * float(pad[0])
    half_height = 0.5 * float(pad[1])
    contacts = np.array([
        [0.0, -half_opening, 0.0],
        [0.0, +half_opening, 0.0],
    ])
    rectangles = []
    for y_value in (-half_opening, +half_opening):
        rectangles.append(np.array([
            [-half_width, y_value, -half_height],
            [+half_width, y_value, -half_height],
            [+half_width, y_value, +half_height],
            [-half_width, y_value, +half_height],
        ]))
    jaw_centerlines = np.array([
        [[0.0, -half_opening, 0.0],
         [0.0, -half_opening, -depth]],
        [[0.0, +half_opening, 0.0],
         [0.0, +half_opening, -depth]],
    ])
    palm_depth_line = np.array([
        [0.0, -half_opening, -depth],
        [0.0, +half_opening, -depth],
    ])
    return {
        "frame": "E",
        "contacts_E_m": contacts.tolist(),
        "pad_rectangles_E_m": np.stack(rectangles).tolist(),
        "jaw_centerlines_E_m": jaw_centerlines.tolist(),
        "palm_depth_line_E_m": palm_depth_line.tolist(),
        "palm_depth_datum": {
            "boundary_equation": f"z_E = {-depth:.12g}",
            "accepted_surface_side": f"z_E >= {-depth:.12g}",
            "jaw_slab_y_bounds_m": [-half_opening, +half_opening],
            "rendered_as": "finite centerline datum only",
            "x_E_unrestricted_in_generator_hard_check": True,
        },
        "physical_finger_or_palm_solids_defined": False,
    }


def _load_prepared_mesh(
    validated: _ValidatedDocument,
    *,
    generated_root: Path,
    cad_override: Path | None,
):
    cad = validated.cad
    expected_hash = cad.get("sha256")
    if (
        not isinstance(expected_hash, str)
        or len(expected_hash) != 64
        or any(character not in "0123456789abcdef" for character in expected_hash)
    ):
        raise ValueError("cad.sha256 must be a SHA-256 hex digest")
    fingerprint = cad.get("artifact_fingerprint")
    if (
        not isinstance(fingerprint, str)
        or len(fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in fingerprint)
    ):
        raise ValueError("cad.artifact_fingerprint must be a SHA-256 hex digest")

    source_value = cad.get("path")
    if not isinstance(source_value, str) or not source_value:
        raise ValueError("cad.path must be non-empty")
    source = cad_override.resolve() if cad_override is not None else Path(
        source_value).resolve()
    if cad_override is not None:
        if not source.is_file():
            raise FileNotFoundError(f"CAD override was not found: {source}")
        if _sha256(source) != expected_hash:
            raise ValueError("CAD override does not match cad.sha256")
    elif source.is_file() and _sha256(source) != expected_hash:
        raise ValueError("cad.path no longer matches cad.sha256")

    metadata_path = generated_root.resolve() / fingerprint / "metadata.json"
    if metadata_path.is_file():
        preparation = verify_preparation(metadata_path)
    else:
        if not source.is_file():
            raise FileNotFoundError(
                "prepared CAD cache and original CAD are both unavailable; "
                "supply --generated-root or a hash-matching --cad override"
            )
        if source.suffix.lower() in (".step", ".stp"):
            raise FileNotFoundError(
                "prepared STEP tessellation is unavailable at the requested "
                "generated root; use the same --generated-root used during "
                "grasp generation"
            )
        scale = _finite_array(
            cad.get("scale_to_m"), (3,), label="cad.scale_to_m")
        preparation = prepare_cad(
            source,
            generated_root,
            scale_to_m=scale.tolist(),
            role="parallel-jaw-grasp-source",
            static_assembly=False,
        )
        if preparation.metadata["artifact_fingerprint"] != fingerprint:
            raise ValueError(
                "reprepared CAD fingerprint does not match candidate document")

    metadata = preparation.metadata
    if metadata.get("artifact_fingerprint") != fingerprint:
        raise ValueError("prepared CAD fingerprint mismatch")
    prepared_source = _mapping(metadata.get("source"), label="prepared source")
    if prepared_source.get("sha256") != expected_hash:
        raise ValueError("prepared CAD source hash mismatch")
    expected_format = cad.get("format")
    if expected_format not in ("stl", "obj", "step", "stp"):
        raise ValueError("cad.format must be stl, obj, step, or stp")
    if prepared_source.get("format") != expected_format:
        raise ValueError("prepared CAD format disagrees with document")
    expected_scale = _finite_array(
        cad.get("scale_to_m"), (3,), label="cad.scale_to_m")
    prepared_scale = _finite_array(
        prepared_source.get("scale_to_m"),
        (3,),
        label="prepared source.scale_to_m",
    )
    if not np.allclose(
        prepared_scale, expected_scale, atol=0.0, rtol=0.0
    ):
        raise ValueError("prepared CAD scale disagrees with document")
    mesh = load_prepared_triangle_mesh(preparation)
    expected_triangles = cad.get("triangle_count")
    if (
        isinstance(expected_triangles, bool)
        or not isinstance(expected_triangles, int)
        or expected_triangles != len(mesh.triangles)
    ):
        raise ValueError("prepared CAD triangle count disagrees with document")
    expected_min = _finite_array(
        cad.get("bounds_min_P_m"), (3,), label="cad.bounds_min_P_m")
    expected_max = _finite_array(
        cad.get("bounds_max_P_m"), (3,), label="cad.bounds_max_P_m")
    bounds_scale = max(float(np.linalg.norm(expected_max - expected_min)), 1e-9)
    if not np.allclose(
        mesh.bounds_min, expected_min, atol=1e-8 * bounds_scale, rtol=0.0
    ) or not np.allclose(
        mesh.bounds_max, expected_max, atol=1e-8 * bounds_scale, rtol=0.0
    ):
        raise ValueError("prepared CAD bounds disagree with document")
    return mesh, preparation, source


def _encode_png(rgb: np.ndarray) -> bytes:
    image = np.asarray(rgb, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("PNG image must have shape (height, width, 3)")
    height, width, _ = image.shape
    scanlines = b"".join(
        b"\x00" + np.ascontiguousarray(image[row]).tobytes()
        for row in range(height)
    )

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(
            ">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(scanlines, level=7))
        + chunk(b"IEND", b"")
    )


def _draw_disk(
    image: np.ndarray,
    point: Sequence[float],
    radius: int,
    color: Sequence[int],
) -> None:
    height, width = image.shape[:2]
    x, y = (int(round(float(value))) for value in point)
    radius = max(1, int(radius))
    x0, x1 = max(0, x - radius), min(width, x + radius + 1)
    y0, y1 = max(0, y - radius), min(height, y + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.ogrid[y0:y1, x0:x1]
    mask = (xx - x) ** 2 + (yy - y) ** 2 <= radius**2
    image[y0:y1, x0:x1][mask] = np.asarray(color, dtype=np.uint8)


def _draw_line(
    image: np.ndarray,
    first: Sequence[float],
    second: Sequence[float],
    color: Sequence[int],
    *,
    thickness: int = 1,
) -> None:
    start = np.asarray(first, dtype=float)
    end = np.asarray(second, dtype=float)
    count = max(int(np.max(np.abs(end - start))), 1) + 1
    for point in np.linspace(start, end, count):
        _draw_disk(image, point, thickness, color)


def _draw_dashed_line(
    image: np.ndarray,
    first: Sequence[float],
    second: Sequence[float],
    color: Sequence[int],
    *,
    thickness: int = 1,
    dash_px: float = 7.0,
) -> None:
    start = np.asarray(first, dtype=float)
    end = np.asarray(second, dtype=float)
    length = float(np.linalg.norm(end - start))
    if length <= 0.0:
        return
    direction = (end - start) / length
    for beginning in np.arange(0.0, length, 2.0 * dash_px):
        finish = min(beginning + dash_px, length)
        _draw_line(
            image,
            start + beginning * direction,
            start + finish * direction,
            color,
            thickness=thickness,
        )


def _draw_arrow(
    image: np.ndarray,
    first: Sequence[float],
    second: Sequence[float],
    color: Sequence[int],
    *,
    thickness: int = 1,
) -> None:
    start = np.asarray(first, dtype=float)
    end = np.asarray(second, dtype=float)
    _draw_line(image, start, end, color, thickness=thickness)
    vector = end - start
    length = float(np.linalg.norm(vector))
    if length <= 2.0:
        return
    direction = vector / length
    side = np.array([-direction[1], direction[0]])
    head = min(9.0, 0.3 * length)
    for sign in (-1.0, 1.0):
        _draw_line(
            image,
            end,
            end - head * direction + sign * 0.5 * head * side,
            color,
            thickness=thickness,
        )


def _fill_triangle_with_depth(
    image: np.ndarray,
    depth_buffer: np.ndarray,
    triangle: np.ndarray,
    vertex_depths: np.ndarray,
    color: Sequence[int],
) -> None:
    """Rasterize one triangle with perspective-free barycentric depth."""
    points = np.asarray(triangle, dtype=float)
    depths = np.asarray(vertex_depths, dtype=float)
    if points.shape != (3, 2) or depths.shape != (3,):
        raise ValueError("triangle/depth inputs have invalid shape")
    first_edge = points[1] - points[0]
    second_edge = points[2] - points[0]
    denominator = float(
        first_edge[0] * second_edge[1]
        - first_edge[1] * second_edge[0]
    )
    if abs(denominator) < 0.2:
        return
    height, width = image.shape[:2]
    x0 = max(0, int(math.floor(float(np.min(points[:, 0])))))
    x1 = min(width - 1, int(math.ceil(float(np.max(points[:, 0])))))
    y0 = max(0, int(math.floor(float(np.min(points[:, 1])))))
    y1 = min(height - 1, int(math.ceil(float(np.max(points[:, 1])))))
    if x0 > x1 or y0 > y1:
        return
    yy, xx = np.mgrid[y0:y1 + 1, x0:x1 + 1]
    relative_x = xx + 0.5 - points[0, 0]
    relative_y = yy + 0.5 - points[0, 1]
    weight_1 = (
        relative_x * second_edge[1]
        - second_edge[0] * relative_y
    ) / denominator
    weight_2 = (
        first_edge[0] * relative_y
        - relative_x * first_edge[1]
    ) / denominator
    weight_0 = 1.0 - weight_1 - weight_2
    inside = (
        (weight_0 >= -1e-9)
        & (weight_1 >= -1e-9)
        & (weight_2 >= -1e-9)
    )
    interpolated_depth = (
        weight_0 * depths[0]
        + weight_1 * depths[1]
        + weight_2 * depths[2]
    )
    old_depth = depth_buffer[y0:y1 + 1, x0:x1 + 1]
    visible = inside & (interpolated_depth > old_depth)
    if not np.any(visible):
        return
    old_depth[visible] = interpolated_depth[visible]
    image[y0:y1 + 1, x0:x1 + 1][visible] = np.asarray(
        color, dtype=np.uint8)


def _draw_text(
    image: np.ndarray,
    position: Sequence[int],
    text: str,
    color: Sequence[int],
    *,
    scale: int = 2,
) -> None:
    x_start, y_start = (int(value) for value in position)
    cursor = x_start
    for character in text.upper():
        pattern = _FONT_3X5.get(character, _FONT_3X5["?"]).split("/")
        for row, bits in enumerate(pattern):
            for column, bit in enumerate(bits):
                if bit == "1":
                    x0 = cursor + column * scale
                    y0 = y_start + row * scale
                    image[
                        max(0, y0):min(image.shape[0], y0 + scale),
                        max(0, x0):min(image.shape[1], x0 + scale),
                    ] = np.asarray(color, dtype=np.uint8)
        cursor += 4 * scale


def _draw_rectangle_border(
    image: np.ndarray,
    rectangle: tuple[int, int, int, int],
    color: Sequence[int],
) -> None:
    left, top, right, bottom = rectangle
    _draw_line(image, (left, top), (right, top), color)
    _draw_line(image, (right, top), (right, bottom), color)
    _draw_line(image, (right, bottom), (left, bottom), color)
    _draw_line(image, (left, bottom), (left, top), color)


_VIEWS = (
    {
        "name": "XY",
        "horizontal_axis": 0,
        "vertical_axis": 1,
        "depth_axis": 2,
        "depth_sign": 1.0,
        "view_direction": "-Z_E",
        "out_of_page_axis": "+Z_E",
    },
    {
        "name": "XZ",
        "horizontal_axis": 0,
        "vertical_axis": 2,
        "depth_axis": 1,
        "depth_sign": -1.0,
        "view_direction": "+Y_E",
        "out_of_page_axis": "-Y_E",
    },
    {
        "name": "YZ",
        "horizontal_axis": 1,
        "vertical_axis": 2,
        "depth_axis": 0,
        "depth_sign": 1.0,
        "view_direction": "-X_E",
        "out_of_page_axis": "+X_E",
    },
)


def _projector(
    rectangle: tuple[int, int, int, int],
    *,
    center_E: np.ndarray,
    pixels_per_m: float,
    horizontal_axis: int,
    vertical_axis: int,
):
    left, top, right, bottom = rectangle
    center_pixel = np.array([
        0.5 * (left + right),
        0.5 * (top + bottom),
    ])

    def project(points_E: np.ndarray) -> np.ndarray:
        points = np.asarray(points_E, dtype=float)
        result = np.empty(points.shape[:-1] + (2,), dtype=float)
        result[..., 0] = (
            center_pixel[0]
            + pixels_per_m
            * (points[..., horizontal_axis] - center_E[horizontal_axis])
        )
        result[..., 1] = (
            center_pixel[1]
            - pixels_per_m
            * (points[..., vertical_axis] - center_E[vertical_axis])
        )
        return result

    return project


def _area_stratified_triangle_indices(
    areas: np.ndarray,
    maximum: int | None,
) -> np.ndarray:
    count = len(areas)
    if maximum is None or maximum >= count:
        return np.arange(count, dtype=int)
    if maximum <= 0:
        raise ValueError("max_render_triangles must be positive or None")
    cumulative = np.cumsum(np.asarray(areas, dtype=float))
    total = float(cumulative[-1])
    if total <= 0.0:
        return np.linspace(0, count - 1, maximum, dtype=int)
    targets = (np.arange(maximum, dtype=float) + 0.5) * total / maximum
    return np.unique(np.searchsorted(cumulative, targets, side="right"))


def _nice_scale_length(span_m: float) -> float:
    target = max(0.18 * span_m, 1e-9)
    exponent = math.floor(math.log10(target))
    base = 10.0**exponent
    candidates = [base, 2.0 * base, 5.0 * base, 10.0 * base]
    usable = [value for value in candidates if value <= target * 1.001]
    return max(usable) if usable else base


def _scale_label(length_m: float) -> str:
    if length_m < 1.0:
        value = length_m * 1000.0
        return f"{value:.3g} MM"
    return f"{length_m:.3g} M"


def _render_candidate_view(
    image: np.ndarray,
    rectangle: tuple[int, int, int, int],
    *,
    triangles_E: np.ndarray,
    normals_E: np.ndarray,
    contacts_E: np.ndarray,
    contact_normals_E: np.ndarray,
    glyph: Mapping[str, Any],
    center_E: np.ndarray,
    pixels_per_m: float,
    axis_length_m: float,
    normal_length_m: float,
    view: Mapping[str, Any],
) -> None:
    left, top, right, bottom = rectangle
    image[top:bottom, left:right] = [14, 19, 27]
    _draw_rectangle_border(image, rectangle, [55, 68, 84])
    title = f"{view['name']}  LOOK {view['view_direction']}"
    _draw_text(image, (left + 12, top + 10), title, [210, 220, 232], scale=2)
    plot_rectangle = (left + 10, top + 34, right - 10, bottom - 10)
    horizontal = int(view["horizontal_axis"])
    vertical = int(view["vertical_axis"])
    depth = int(view["depth_axis"])
    depth_sign = float(view["depth_sign"])
    project = _projector(
        plot_rectangle,
        center_E=center_E,
        pixels_per_m=pixels_per_m,
        horizontal_axis=horizontal,
        vertical_axis=vertical,
    )

    projected_triangles = project(triangles_E)
    vertex_depths = depth_sign * triangles_E[:, :, depth]
    mean_depths = np.mean(vertex_depths, axis=1)
    depth_min = float(np.min(mean_depths))
    depth_span = max(float(np.ptp(mean_depths)), 1e-12)
    depth_buffer = np.full(image.shape[:2], -np.inf, dtype=float)
    for index in range(len(triangles_E)):
        facing = abs(float(normals_E[index, depth]))
        height_fraction = (mean_depths[index] - depth_min) / depth_span
        shade = int(round(66 + 58 * facing + 28 * height_fraction))
        shade = max(52, min(172, shade))
        _fill_triangle_with_depth(
            image,
            depth_buffer,
            projected_triangles[index],
            vertex_depths[index],
            [shade, shade + 5, shade + 12],
        )

    cyan = [54, 201, 238]
    green = [67, 160, 71]
    blue = [30, 136, 229]
    red = [229, 57, 53]
    yellow = [255, 214, 64]
    orange = [255, 152, 45]
    white = [245, 248, 252]

    rectangles = np.asarray(glyph["pad_rectangles_E_m"], dtype=float)
    for pad_rectangle in rectangles:
        pixels = project(pad_rectangle)
        for index in range(4):
            _draw_line(
                image,
                pixels[index],
                pixels[(index + 1) % 4],
                cyan,
                thickness=2,
            )
    jaw_centerlines = np.asarray(glyph["jaw_centerlines_E_m"], dtype=float)
    for centerline in jaw_centerlines:
        pixels = project(centerline)
        _draw_line(image, pixels[0], pixels[1], cyan, thickness=2)
    palm_line = project(np.asarray(
        glyph["palm_depth_line_E_m"], dtype=float))
    _draw_dashed_line(
        image, palm_line[0], palm_line[1], cyan, thickness=2)

    contact_pixels = project(contacts_E)
    _draw_dashed_line(
        image, contact_pixels[0], contact_pixels[1], green, thickness=1)
    if float(np.linalg.norm(contact_pixels[1] - contact_pixels[0])) > 5.0:
        _draw_arrow(
            image, contact_pixels[0], contact_pixels[1], green, thickness=1)

    actuation_length = 0.75 * normal_length_m
    for sign, contact in ((-1.0, contacts_E[0]), (1.0, contacts_E[1])):
        outside = contact + np.array([0.0, sign * actuation_length, 0.0])
        outside_pixel, contact_pixel = project(
            np.stack((outside, contact)))
        if float(np.linalg.norm(outside_pixel - contact_pixel)) > 4.0:
            _draw_arrow(
                image, outside_pixel, contact_pixel, orange, thickness=1)

    for contact, normal in zip(contacts_E, contact_normals_E):
        segment = np.stack((contact, contact + normal_length_m * normal))
        pixels = project(segment)
        if float(np.linalg.norm(pixels[1] - pixels[0])) > 4.0:
            _draw_arrow(image, pixels[0], pixels[1], yellow, thickness=1)

    origin = np.zeros(3)
    axis_colors = (red, green, blue)
    axis_names = ("+X", "+Y", "+Z")
    for axis_index, (color, name) in enumerate(zip(axis_colors, axis_names)):
        endpoint = np.zeros(3)
        endpoint[axis_index] = axis_length_m
        pixels = project(np.stack((origin, endpoint)))
        if float(np.linalg.norm(pixels[1] - pixels[0])) <= 4.0:
            _draw_disk(image, pixels[0], 7, color)
            _draw_disk(image, pixels[0], 4, [14, 19, 27])
            if depth_sign > 0.0:
                _draw_disk(image, pixels[0], 2, color)
            else:
                arm = 3.0
                _draw_line(
                    image,
                    pixels[0] + [-arm, -arm],
                    pixels[0] + [arm, arm],
                    color,
                    thickness=1,
                )
                _draw_line(
                    image,
                    pixels[0] + [-arm, arm],
                    pixels[0] + [arm, -arm],
                    color,
                    thickness=1,
                )
            _draw_text(
                image,
                (int(pixels[0, 0] + 9), int(pixels[0, 1] + 5)),
                name,
                color,
                scale=1,
            )
        else:
            _draw_arrow(image, pixels[0], pixels[1], color, thickness=2)
            _draw_text(
                image,
                (int(pixels[1, 0] + 5), int(pixels[1, 1] - 3)),
                name,
                color,
                scale=1,
            )

    for contact_index, point in enumerate(contact_pixels):
        _draw_disk(image, point, 5, [12, 16, 23])
        _draw_disk(image, point, 3, white)
        _draw_text(
            image,
            (int(point[0] + 7), int(point[1] - 7 + 12 * contact_index)),
            f"C{contact_index}",
            white,
            scale=1,
        )

    scale_length = _nice_scale_length(
        (right - left - 20) / pixels_per_m)
    scale_pixels = scale_length * pixels_per_m
    scale_start = np.array([left + 24.0, bottom - 23.0])
    _draw_line(
        image, scale_start, scale_start + [scale_pixels, 0],
        white, thickness=1)
    for x_value in (scale_start[0], scale_start[0] + scale_pixels):
        _draw_line(
            image,
            (x_value, scale_start[1] - 4),
            (x_value, scale_start[1] + 4),
            white,
            thickness=1,
        )
    _draw_text(
        image,
        (int(scale_start[0]), int(scale_start[1] - 15)),
        _scale_label(scale_length),
        white,
        scale=1,
    )


def render_parallel_jaw_candidate_image(
    candidate_path: str | Path,
    output_png: str | Path,
    *,
    count: int = 4,
    selection_mode: str = "pose_diverse",
    candidate_ids: Sequence[str] = (),
    generated_root: str | Path = DEFAULT_GENERATED_ROOT,
    cad_path: str | Path | None = None,
    width: int = 1600,
    row_height: int = 320,
    max_render_triangles: int | None = None,
    allow_unreliable_input: bool = False,
) -> dict[str, Any]:
    """Write a multi-view PNG and return exact companion metadata."""
    candidate_file = Path(candidate_path).resolve()
    output_file = Path(output_png).resolve()
    if not candidate_file.is_file():
        raise FileNotFoundError(
            f"candidate JSON was not found: {candidate_file}")
    if output_file.suffix.lower() != ".png":
        raise ValueError("output_png must name a .png file")
    if output_file == candidate_file:
        raise ValueError("output PNG must not overwrite the candidate JSON")
    if isinstance(width, bool) or not isinstance(width, int) or width < 1200:
        raise ValueError("width must be an integer >= 1200")
    if (
        isinstance(row_height, bool)
        or not isinstance(row_height, int)
        or row_height < 260
    ):
        raise ValueError("row_height must be an integer >= 260")
    if max_render_triangles is not None and (
        isinstance(max_render_triangles, bool)
        or not isinstance(max_render_triangles, int)
        or max_render_triangles <= 0
    ):
        raise ValueError("max_render_triangles must be positive or None")
    if not isinstance(allow_unreliable_input, bool):
        raise TypeError("allow_unreliable_input must be a boolean")

    document, candidate_payload = _load_json(candidate_file)
    validated = _validate_document(
        document,
        allow_unreliable_input=allow_unreliable_input,
    )
    selected, selection_metadata = _select_candidates(
        validated,
        count=count,
        selection_mode=selection_mode,
        candidate_ids=candidate_ids,
    )
    mesh, preparation, resolved_cad = _load_prepared_mesh(
        validated,
        generated_root=Path(generated_root),
        cad_override=(None if cad_path is None else Path(cad_path)),
    )
    protected_paths = {
        candidate_file,
        preparation.metadata_path.resolve(),
        resolved_cad.resolve(),
    }
    prepared_visual = _mapping(
        preparation.metadata.get("visual"), label="prepared visual")
    prepared_chunks = prepared_visual.get("chunks")
    if not isinstance(prepared_chunks, list):
        raise ValueError("prepared visual.chunks must be an array")
    for chunk_index, chunk_value in enumerate(prepared_chunks):
        chunk = _mapping(
            chunk_value, label=f"prepared visual.chunks[{chunk_index}]")
        relative_path = chunk.get("path")
        if not isinstance(relative_path, str) or not relative_path:
            raise ValueError("prepared visual chunk path must be non-empty")
        chunk_path = (
            preparation.artifact_dir / relative_path
        ).resolve()
        try:
            chunk_path.relative_to(preparation.artifact_dir.resolve())
        except ValueError as error:
            raise ValueError("prepared visual chunk escapes its artifact") from error
        protected_paths.add(chunk_path)
    if output_file in protected_paths:
        raise ValueError(
            "output PNG collides with a source CAD, candidate JSON, or "
            "prepared-CAD metadata path"
        )
    triangle_indices = _area_stratified_triangle_indices(
        mesh.areas, max_render_triangles)
    rendered_triangles_P = mesh.triangles[triangle_indices]
    rendered_normals_P = mesh.normals[triangle_indices]

    header_height = 140
    footer_height = 100
    height = header_height + row_height * len(selected) + footer_height
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[:] = [18, 24, 33]
    image[:header_height] = [26, 35, 47]
    image[-footer_height:] = [26, 35, 47]
    white = [235, 241, 247]
    muted = [154, 169, 187]
    warning = [255, 190, 70]
    _draw_text(
        image,
        (38, 22),
        "SAMPLED OBJECT-GEOMETRY GRASP CANDIDATES",
        white,
        scale=4,
    )
    _draw_text(
        image,
        (40, 68),
        "CANDIDATE-ALIGNED ORTHOGRAPHIC VIEWS IN IDEAL CONTACT FRAME E",
        muted,
        scale=2,
    )
    _draw_text(
        image, (40, 99), "+X PAD WIDTH", [229, 57, 53], scale=2)
    _draw_text(
        image, (226, 99), "+Y C0 TO C1", [67, 160, 71], scale=2)
    _draw_text(
        image, (398, 99), "+Z PALM TO CONTACT", [30, 136, 229], scale=2)
    _draw_text(
        image, (650, 99), "YELLOW NORMAL", [255, 214, 64], scale=2)
    _draw_text(
        image, (840, 99), "ORANGE JAW CLOSE", [255, 152, 45], scale=2)
    display_warnings = [
        f"DISPLAYING {len(selected)} OF {len(validated.candidates)}",
    ]
    if not validated.reliable_input:
        display_warnings.append("UNRELIABLE MESH INPUT")
    if document.get("candidate_cap_applied") is True:
        display_warnings.append("SOURCE CAP CONFIGURED")
    if len(triangle_indices) != len(mesh.triangles):
        display_warnings.append("CAD DISPLAY SUBSET")
    warning_x = width - 292
    for warning_index, warning_text in enumerate(display_warnings):
        _draw_text(
            image,
            (warning_x, 20 + 16 * warning_index),
            warning_text,
            warning,
            scale=1,
        )

    left_margin = 30
    right_margin = 30
    information_width = 270
    gap = 14
    views_left = left_margin + information_width
    available = width - views_left - right_margin - 2 * gap
    view_width = available // 3
    if view_width < 260:
        raise ValueError("width leaves less than 260 px per orthographic view")

    displayed_metadata: list[dict[str, Any]] = []
    for display_index, candidate in enumerate(selected):
        row_top = header_height + display_index * row_height
        row_bottom = row_top + row_height
        if display_index % 2:
            image[row_top:row_bottom] = [20, 27, 37]
        _draw_line(
            image,
            (left_margin, row_top),
            (width - right_margin, row_top),
            [48, 61, 77],
        )

        rotation = candidate.T_P_E[:3, :3]
        origin_P = candidate.T_P_E[:3, 3]
        triangles_E = (rendered_triangles_P - origin_P) @ rotation
        normals_E = rendered_normals_P @ rotation
        all_vertices_E = (mesh.vertices - origin_P) @ rotation
        contacts_E = (candidate.contacts_P - origin_P) @ rotation
        contact_normals_E = candidate.normals_P @ rotation
        glyph = ideal_parallel_jaw_glyph_E(
            candidate.opening_m,
            validated.pad_size_m,
            validated.finger_depth_m,
        )
        glyph_points = np.concatenate((
            np.asarray(glyph["pad_rectangles_E_m"], dtype=float).reshape(-1, 3),
            np.asarray(glyph["jaw_centerlines_E_m"], dtype=float).reshape(-1, 3),
            np.asarray(glyph["palm_depth_line_E_m"], dtype=float).reshape(-1, 3),
        ))
        base_points = np.concatenate((all_vertices_E, glyph_points, contacts_E))
        base_span = max(float(np.max(np.ptp(base_points, axis=0))), 1e-9)
        axis_length = max(
            0.16 * base_span,
            0.65 * float(np.min(validated.pad_size_m)),
        )
        normal_length = max(
            0.12 * base_span,
            0.45 * float(np.min(validated.pad_size_m)),
        )
        semantic_points = [base_points]
        for axis_index in range(3):
            endpoint = np.zeros(3)
            endpoint[axis_index] = axis_length
            semantic_points.append(endpoint[None, :])
        semantic_points.append(
            contacts_E + normal_length * contact_normals_E)
        actuation = np.array([
            contacts_E[0] + [0.0, -0.75 * normal_length, 0.0],
            contacts_E[1] + [0.0, +0.75 * normal_length, 0.0],
        ])
        semantic_points.append(actuation)
        content_points = np.concatenate(semantic_points)
        bounds_min = np.min(content_points, axis=0)
        bounds_max = np.max(content_points, axis=0)
        center_E = 0.5 * (bounds_min + bounds_max)
        span = max(float(np.max(bounds_max - bounds_min)), 1e-9)

        info_x = left_margin + 10
        info_y = row_top + 28
        display_label = candidate.candidate_id.upper()
        _draw_text(
            image,
            (info_x, info_y),
            f"CANDIDATE {display_index + 1}",
            white,
            scale=3,
        )
        _draw_text(
            image, (info_x, info_y + 38), display_label, warning, scale=2)
        _draw_text(
            image,
            (info_x, info_y + 66),
            f"SOURCE RANK {candidate.source_rank}",
            muted,
            scale=2,
        )
        _draw_text(
            image,
            (info_x, info_y + 92),
            f"INDEX {candidate.index}",
            muted,
            scale=2,
        )
        _draw_text(
            image,
            (info_x, info_y + 118),
            f"OPENING {candidate.opening_m * 1000.0:.3f} MM",
            white,
            scale=2,
        )
        _draw_text(
            image,
            (info_x, info_y + 144),
            f"QUALITY {candidate.quality:.4f}",
            white,
            scale=2,
        )
        _draw_text(
            image,
            (info_x, info_y + 180),
            "OBJECT-ONLY HEURISTIC",
            muted,
            scale=1,
        )
        _draw_text(
            image,
            (info_x, info_y + 198),
            "NOT SUCCESS PROBABILITY",
            muted,
            scale=1,
        )

        view_records = []
        for view_index, view in enumerate(_VIEWS):
            left = views_left + view_index * (view_width + gap)
            rectangle = (
                left,
                row_top + 14,
                left + view_width,
                row_bottom - 14,
            )
            plot_width = rectangle[2] - rectangle[0] - 20
            plot_height = rectangle[3] - rectangle[1] - 44
            pixels_per_m = 0.86 * min(plot_width, plot_height) / span
            _render_candidate_view(
                image,
                rectangle,
                triangles_E=triangles_E,
                normals_E=normals_E,
                contacts_E=contacts_E,
                contact_normals_E=contact_normals_E,
                glyph=glyph,
                center_E=center_E,
                pixels_per_m=pixels_per_m,
                axis_length_m=axis_length,
                normal_length_m=normal_length,
                view=view,
            )
            view_records.append({
                "name": view["name"],
                "horizontal_axis": f"+{'XYZ'[int(view['horizontal_axis'])]}_E",
                "vertical_axis": f"+{'XYZ'[int(view['vertical_axis'])]}_E",
                "view_direction": view["view_direction"],
                "out_of_page_axis": view["out_of_page_axis"],
                "pixels_per_m": pixels_per_m,
            })

        displayed_metadata.append({
            "id": candidate.candidate_id,
            "index": candidate.index,
            "source_rank": candidate.source_rank,
            "T_P_E": candidate.T_P_E.tolist(),
            "contact_points_P_m": candidate.contacts_P.tolist(),
            "contact_normals_P": candidate.normals_P.tolist(),
            "required_opening_m": candidate.opening_m,
            "quality": candidate.quality,
            "quality_semantics": "object-only heuristic score",
            "glyph_geometry": glyph,
            "view_content_bounds_E_m": [
                bounds_min.tolist(),
                bounds_max.tolist(),
            ],
            "axis_glyph_length_m": axis_length,
            "normal_glyph_length_m": normal_length,
            "views": view_records,
        })

    footer_top = height - footer_height
    _draw_text(
        image,
        (38, footer_top + 18),
        "IDEAL CONTACTS PAD RECTANGLES AND PALM-DEPTH DATUM",
        warning,
        scale=2,
    )
    _draw_text(
        image,
        (38, footer_top + 48),
        "NOT PHYSICAL GRIPPER COLLISION GEOMETRY OR A PHYSICAL TCP",
        white,
        scale=2,
    )
    _draw_text(
        image,
        (38, footer_top + 74),
        "NOT CERTIFIED  FULL COLLISION / APPROACH SWEEP / ROBOT REACHABILITY / TASK FEASIBILITY",
        white,
        scale=2,
    )

    png_payload = _encode_png(image)
    atomic_write_bytes(output_file, png_payload)
    source_path_label = (
        candidate_file.relative_to(ROOT).as_posix()
        if candidate_file.is_relative_to(ROOT)
        else str(candidate_file)
    )
    prepared_metadata_path = preparation.metadata_path
    return {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "claim_level": "visualization_only",
        "diagnostic_unreliable_input": not validated.reliable_input,
        "output_image": {
            "path": str(output_file),
            "sha256": hashlib.sha256(png_payload).hexdigest(),
            "size_bytes": len(png_payload),
        },
        "source_candidates": {
            "path": source_path_label,
            "sha256": hashlib.sha256(candidate_payload).hexdigest(),
            "artifact_type": document["artifact_type"],
            "claim_level": document["claim_level"],
            "candidate_count": len(validated.candidates),
            "candidate_cap_applied": document["candidate_cap_applied"],
            "all_deduplicated_accepted_candidates_returned": document[
                "all_deduplicated_accepted_candidates_returned"
            ],
            "continuous_exhaustive": document["continuous_exhaustive"],
            "sampling": document["sampling"],
        },
        "cad": {
            "path": str(resolved_cad),
            "sha256": validated.cad["sha256"],
            "artifact_fingerprint": validated.cad["artifact_fingerprint"],
            "prepared_metadata_path": str(prepared_metadata_path),
            "source_triangle_count": len(mesh.triangles),
            "rendered_triangle_count": len(triangle_indices),
            "complete_triangle_projection": (
                len(triangle_indices) == len(mesh.triangles)
            ),
            "render_triangle_selection": (
                "all triangles"
                if len(triangle_indices) == len(mesh.triangles)
                else "area-stratified deterministic display subset"
            ),
        },
        "selection": selection_metadata,
        "view": {
            "type": "candidate_aligned_orthographic_gallery",
            "frame": "each candidate ideal contact frame E",
            "image_size_px": [width, height],
            "row_height_px": row_height,
            "display_warnings": display_warnings,
            "projections": [
                {
                    "name": view["name"],
                    "horizontal_axis": f"+{'XYZ'[int(view['horizontal_axis'])]}_E",
                    "vertical_axis": f"+{'XYZ'[int(view['vertical_axis'])]}_E",
                    "view_direction": view["view_direction"],
                    "out_of_page_axis": view["out_of_page_axis"],
                }
                for view in _VIEWS
            ],
        },
        "displayed_candidates": displayed_metadata,
        "render_semantics": {
            "part_geometry": (
                "complete prepared CAD triangle surface"
                if len(triangle_indices) == len(mesh.triangles)
                else "recorded area-stratified display-only CAD triangle subset"
            ),
            "contact_geometry": "exact serialized ideal contacts and normals",
            "gripper_geometry": (
                "zero-thickness pad rectangles and schematic U centerlines"
            ),
            "palm_geometry": (
                "idealized palm-depth datum only; no finite palm solid"
            ),
            "frame": (
                "E is the ideal contact-midpoint frame, not a physical TCP"
            ),
            "display_selection_is_visualization_only": True,
            "displayed_candidates_are_exact_source_records": True,
        },
        "certification": {
            "physical_gripper_geometry_shown": False,
            "physical_tcp_shown": False,
            "full_part_gripper_collision_checked": False,
            "approach_sweep_checked": False,
            "environment_collision_checked": False,
            "robot_reachability_checked": False,
            "task_feasibility_checked": False,
        },
        "limitations": list(validated.not_checked) + [
            "preview glyph is not a calibrated physical gripper model",
            "candidate selection is for display coverage only",
        ],
    }


__all__ = [
    "ARTIFACT_TYPE",
    "DEFAULT_GENERATED_ROOT",
    "ideal_parallel_jaw_glyph_E",
    "render_parallel_jaw_candidate_image",
    "select_candidate_records",
]
