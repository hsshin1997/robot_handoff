"""CPU rendering of representative poses from a continuous grasp map.

This module is deliberately a *view* of the object-only set-valued map.  It
selects a few deterministic points for display, but it does not discretize the
map or add any collision, visibility, PCB-clearance, or robot-reachability
claim.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import struct
from typing import Any, Mapping, Sequence
import zlib

import numpy as np

from ..core.se3 import validate_transform
from .two_finger_grasp_map import SCOPE, load_scaled_binary_stl


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_TYPE = "top_down_representative_grasp_candidates"


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
    "_": "000/000/000/000/111",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _finite_array(value: Any, shape: tuple[int, ...], *, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must be a finite array with shape {shape}")
    return result


def _unit(value: Any, *, label: str) -> np.ndarray:
    vector = _finite_array(value, (3,), label=label)
    norm = float(np.linalg.norm(vector))
    if norm <= 64.0 * np.finfo(float).eps:
        raise ValueError(f"{label} must be nonzero")
    return vector / norm


def _load_document(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {path}: {error}") from error
    document = _mapping(value, label="grasp map")
    if document.get("artifact_type") != "two_finger_continuous_grasp_map":
        raise ValueError("input must be a two_finger_continuous_grasp_map")
    if document.get("scope") != SCOPE or document.get("insertion_safe") is not False:
        raise ValueError(
            f"input must have scope={SCOPE} and insertion_safe=false"
        )
    families = document.get("families")
    if not isinstance(families, list) or not families:
        raise ValueError("grasp map must contain at least one family")
    return document


def _polygon_area(vertices: np.ndarray) -> float:
    return 0.5 * abs(float(np.sum(
        vertices[:, 0] * np.roll(vertices[:, 1], -1)
        - vertices[:, 1] * np.roll(vertices[:, 0], -1)
    )))


def _domain_vertices(domain: Mapping[str, Any], *, label: str) -> np.ndarray:
    vertices = np.asarray(domain.get("vertices_uv_m"), dtype=float)
    if (
        vertices.ndim != 2
        or vertices.shape[1:] != (2,)
        or len(vertices) < 3
        or not np.all(np.isfinite(vertices))
    ):
        raise ValueError(f"{label}.vertices_uv_m must have shape (N>=3, 2)")
    area = _polygon_area(vertices)
    if area <= 0.0:
        raise ValueError(f"{label} must have positive area")
    declared = domain.get("area_m2")
    if not isinstance(declared, (int, float)) or not math.isfinite(float(declared)):
        raise ValueError(f"{label}.area_m2 must be finite")
    if not math.isclose(area, float(declared), rel_tol=1e-7, abs_tol=1e-14):
        raise ValueError(f"{label}.area_m2 does not match its vertices")
    return vertices


def _merge_intervals(
    intervals: Sequence[tuple[float, float]],
    *,
    tolerance: float = 1e-12,
) -> list[tuple[float, float]]:
    ordered = sorted((min(a, b), max(a, b)) for a, b in intervals)
    result: list[tuple[float, float]] = []
    for start, end in ordered:
        if not result or start > result[-1][1] + tolerance:
            result.append((start, end))
        else:
            result[-1] = (result[-1][0], max(result[-1][1], end))
    return result


def _quantile_on_intervals(
    intervals: Sequence[tuple[float, float]],
    fraction: float,
) -> float:
    merged = _merge_intervals(intervals)
    lengths = [end - start for start, end in merged]
    total = float(sum(lengths))
    if total <= 0.0:
        raise ValueError("family projection has zero length")
    target = min(max(float(fraction), 0.0), 1.0) * total
    traversed = 0.0
    for (start, end), length in zip(merged, lengths):
        if target <= traversed + length or end == merged[-1][1]:
            return start + min(max(target - traversed, 0.0), length)
        traversed += length
    return merged[-1][1]


def _polygon_cross_section(
    vertices: np.ndarray,
    *,
    long_axis: int,
    coordinate: float,
    tolerance: float = 1e-11,
) -> tuple[float, float] | None:
    short_axis = 1 - long_axis
    values: list[float] = []
    for index, first in enumerate(vertices):
        second = vertices[(index + 1) % len(vertices)]
        first_delta = float(first[long_axis] - coordinate)
        second_delta = float(second[long_axis] - coordinate)
        if abs(first_delta) <= tolerance:
            values.append(float(first[short_axis]))
        if first_delta * second_delta < -tolerance * tolerance:
            alpha = -first_delta / (second_delta - first_delta)
            values.append(float(
                first[short_axis] + alpha * (second[short_axis] - first[short_axis])
            ))
        elif abs(second_delta) <= tolerance:
            values.append(float(second[short_axis]))
    if len(values) < 2:
        return None
    return min(values), max(values)


def _evaluate_serialized_family(
    document: Mapping[str, Any],
    family: Mapping[str, Any],
    *,
    u_m: float,
    v_m: float,
    roll_rad: float,
) -> dict[str, Any]:
    inputs = _mapping(document.get("inputs"), label="inputs")
    T_W_P = validate_transform(np.asarray(inputs.get("T_W_P_insert"), dtype=float))
    insertion_axis = _unit(
        inputs.get("insertion_axis_P"), label="inputs.insertion_axis_P")
    parameterization = _mapping(
        family.get("parameterization"), label="family.parameterization")
    origin = _finite_array(
        parameterization.get("parameter_origin_P_m"),
        (3,),
        label="parameter_origin_P_m",
    )
    u_axis = _unit(parameterization.get("u_axis_P"), label="u_axis_P")
    v_axis = _unit(parameterization.get("v_axis_P"), label="v_axis_P")
    closing = _unit(family.get("closing_axis_P"), label="closing_axis_P")
    basis = np.column_stack((u_axis, v_axis, closing))
    if not np.allclose(basis.T @ basis, np.eye(3), atol=2e-8, rtol=0.0):
        raise ValueError("serialized family basis is not orthonormal")

    planes = _mapping(family.get("planes"), label="family.planes")
    contacts: list[np.ndarray] = []
    q = origin + float(u_m) * u_axis + float(v_m) * v_axis
    for name in ("negative", "positive"):
        plane = _mapping(planes.get(name), label=f"planes.{name}")
        normal = _unit(plane.get("normal_P"), label=f"planes.{name}.normal_P")
        offset = float(plane.get("offset_m"))
        if not math.isfinite(offset):
            raise ValueError(f"planes.{name}.offset_m must be finite")
        denominator = float(normal @ closing)
        if abs(denominator) <= 64.0 * np.finfo(float).eps:
            raise ValueError(f"planes.{name} is parallel to the closing ray")
        contacts.append(q + ((offset - float(normal @ q)) / denominator) * closing)
    contacts_P = np.stack(contacts)
    aperture = float((contacts_P[1] - contacts_P[0]) @ closing)
    if aperture <= 0.0:
        raise ValueError("serialized contact ordering gives non-positive aperture")

    aperture_map = _mapping(family.get("aperture_map"), label="aperture_map")
    coefficients = _mapping(
        aperture_map.get("coefficients"), label="aperture coefficients")
    declared_aperture = (
        float(coefficients.get("a0_m"))
        + float(coefficients.get("au")) * float(u_m)
        + float(coefficients.get("av")) * float(v_m)
    )
    if not math.isclose(aperture, declared_aperture, rel_tol=1e-8, abs_tol=1e-10):
        raise ValueError("serialized aperture map disagrees with contact planes")

    z_zero = insertion_axis - float(insertion_axis @ closing) * closing
    z_zero = _unit(z_zero, label="projected insertion approach")
    approach = (
        math.cos(float(roll_rad)) * z_zero
        + math.sin(float(roll_rad)) * np.cross(closing, z_zero)
    )
    approach = _unit(approach, label="rolled approach")
    x_axis = _unit(np.cross(closing, approach), label="end-effector x axis")
    T_P_E = np.eye(4)
    T_P_E[:3, :3] = np.column_stack((x_axis, closing, approach))
    T_P_E[:3, 3] = np.mean(contacts_P, axis=0)
    T_W_E = T_W_P @ T_P_E
    contacts_W = contacts_P @ T_W_P[:3, :3].T + T_W_P[:3, 3]
    return {
        "family_id": str(family.get("family_id")),
        "parameters": {
            "u_m": float(u_m),
            "v_m": float(v_m),
            "roll_rad": float(roll_rad),
        },
        "T_P_E": T_P_E.tolist(),
        "T_W_E": T_W_E.tolist(),
        "contacts_P_m": contacts_P.tolist(),
        "contacts_W_m": contacts_W.tolist(),
        "aperture_m": aperture,
        "scope": SCOPE,
        "insertion_safe": False,
    }


def select_representative_candidates(
    document: Mapping[str, Any],
    count: int = 6,
    roll_rad: float = 0.0,
) -> dict[str, Any]:
    """Select display samples without changing the underlying continuous map.

    The family with the largest individual convex-domain component is selected.
    Fractions from 10% through 90% are then placed along the longest projection
    of the complete family-domain union.  At each fraction, the midpoint of the
    widest valid cross-section supplies the other parameter.
    """
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 10:
        raise ValueError("count must be an integer in [1, 10]")
    roll = float(roll_rad)
    if not math.isfinite(roll):
        raise ValueError("roll_rad must be finite")
    if document.get("scope") != SCOPE or document.get("insertion_safe") is not False:
        raise ValueError(
            f"document must have scope={SCOPE} and insertion_safe=false"
        )
    raw_families = document.get("families")
    if not isinstance(raw_families, list) or not raw_families:
        raise ValueError("document must contain at least one family")

    parsed: list[tuple[float, str, Mapping[str, Any], list[np.ndarray]]] = []
    for family_index, value in enumerate(raw_families):
        family = _mapping(value, label=f"families[{family_index}]")
        family_id = family.get("family_id")
        if not isinstance(family_id, str) or not family_id:
            raise ValueError(f"families[{family_index}].family_id is invalid")
        parameterization = _mapping(
            family.get("parameterization"),
            label=f"families[{family_index}].parameterization",
        )
        roll_bounds = _finite_array(
            parameterization.get("roll_bounds_rad"),
            (2,),
            label=f"{family_id}.roll_bounds_rad",
        )
        if roll_bounds[1] < roll_bounds[0]:
            raise ValueError(f"{family_id}.roll_bounds_rad must be ordered")
        domains_raw = parameterization.get("domains")
        if not isinstance(domains_raw, list) or not domains_raw:
            raise ValueError(f"{family_id} must contain domains")
        vertices = [
            _domain_vertices(
                _mapping(domain, label=f"{family_id}.domains[{domain_index}]"),
                label=f"{family_id}.domains[{domain_index}]",
            )
            for domain_index, domain in enumerate(domains_raw)
        ]
        largest = max(_polygon_area(polygon) for polygon in vertices)
        parsed.append((largest, family_id, family, vertices))

    # Secondary family ID ordering makes a tie deterministic.
    largest_area, family_id, family, polygons = sorted(
        parsed, key=lambda item: (-item[0], item[1]))[0]
    parameterization = _mapping(
        family.get("parameterization"), label=f"{family_id}.parameterization")
    roll_bounds = np.asarray(parameterization["roll_bounds_rad"], dtype=float)
    if roll < roll_bounds[0] - 1e-12 or roll > roll_bounds[1] + 1e-12:
        raise ValueError(
            f"roll_rad={roll} lies outside selected family bounds "
            f"[{roll_bounds[0]}, {roll_bounds[1]}]"
        )

    minimum = np.min(np.concatenate(polygons, axis=0), axis=0)
    maximum = np.max(np.concatenate(polygons, axis=0), axis=0)
    long_axis = int(np.argmax(maximum - minimum))
    short_axis = 1 - long_axis
    projection_intervals = [
        (float(np.min(vertices[:, long_axis])),
         float(np.max(vertices[:, long_axis])))
        for vertices in polygons
    ]
    fractions = np.array([0.5]) if count == 1 else np.linspace(0.1, 0.9, count)
    candidates: list[dict[str, Any]] = []
    for index, fraction in enumerate(fractions):
        long_value = _quantile_on_intervals(projection_intervals, float(fraction))
        cross_sections = [
            section
            for vertices in polygons
            if (section := _polygon_cross_section(
                vertices, long_axis=long_axis, coordinate=long_value
            )) is not None
        ]
        merged_sections = _merge_intervals(cross_sections)
        if not merged_sections:
            raise ValueError(
                f"selected family has no cross-section at fraction {fraction}"
            )
        short_interval = max(
            merged_sections, key=lambda interval: (interval[1] - interval[0], -interval[0])
        )
        parameters = np.zeros(2)
        parameters[long_axis] = long_value
        parameters[short_axis] = 0.5 * sum(short_interval)
        evaluation = _evaluate_serialized_family(
            document,
            family,
            u_m=float(parameters[0]),
            v_m=float(parameters[1]),
            roll_rad=roll,
        )
        evaluation["candidate_id"] = f"C{index + 1}"
        evaluation["display_fraction"] = float(fraction)
        candidates.append(evaluation)

    return {
        "selection": {
            "mode": "single_family",
            "policy": (
                "largest_single_convex_domain_area_then_even_quantiles_"
                "over_longest_family_union_projection"
            ),
            "ranking_intent": (
                "deterministic display heuristic; not a grasp-quality or "
                "insertion-safety ranking"
            ),
            "family_id": family_id,
            "family_domain_count": len(polygons),
            "family_largest_domain_area_m2": largest_area,
            "long_parameter": "u_m" if long_axis == 0 else "v_m",
            "cross_parameter": "v_m" if long_axis == 0 else "u_m",
            "sample_fractions": fractions.tolist(),
            "roll_rad": roll,
            "sample_count": count,
        },
        "candidates": candidates,
    }


def _family_display_record(
    family: Mapping[str, Any],
    *,
    family_index: int,
) -> dict[str, Any]:
    family_id = family.get("family_id")
    if not isinstance(family_id, str) or not family_id:
        raise ValueError(f"families[{family_index}].family_id is invalid")
    closing_axis = _unit(
        family.get("closing_axis_P"),
        label=f"{family_id}.closing_axis_P",
    )
    parameterization = _mapping(
        family.get("parameterization"),
        label=f"{family_id}.parameterization",
    )
    domains_raw = parameterization.get("domains")
    if not isinstance(domains_raw, list) or not domains_raw:
        raise ValueError(f"{family_id} must contain domains")
    polygons = [
        _domain_vertices(
            _mapping(domain, label=f"{family_id}.domains[{domain_index}]"),
            label=f"{family_id}.domains[{domain_index}]",
        )
        for domain_index, domain in enumerate(domains_raw)
    ]
    aperture_map = _mapping(
        family.get("aperture_map"),
        label=f"{family_id}.aperture_map",
    )
    coefficients_raw = _mapping(
        aperture_map.get("coefficients"),
        label=f"{family_id}.aperture_map.coefficients",
    )
    coefficients = np.array([
        float(coefficients_raw.get("a0_m")),
        float(coefficients_raw.get("au")),
        float(coefficients_raw.get("av")),
    ])
    if not np.all(np.isfinite(coefficients)):
        raise ValueError(f"{family_id} aperture coefficients must be finite")
    vertices = np.concatenate(polygons, axis=0)
    apertures = (
        coefficients[0]
        + coefficients[1] * vertices[:, 0]
        + coefficients[2] * vertices[:, 1]
    )
    return {
        "family_id": family_id,
        "family": family,
        "closing_axis_P": closing_axis,
        "largest_domain_area_m2": max(
            _polygon_area(polygon) for polygon in polygons),
        "maximum_aperture_m": float(np.max(apertures)),
    }


def _top_down_orientation_label(closing_axis_W: np.ndarray) -> str:
    projection = np.asarray(closing_axis_W[:2], dtype=float)
    norm = float(np.linalg.norm(projection))
    if norm <= 1e-8:
        return "out_of_plane"
    projection /= norm
    if abs(float(projection[0])) >= math.cos(math.radians(22.5)):
        return "horizontal"
    if abs(float(projection[1])) >= math.cos(math.radians(22.5)):
        return "vertical"
    return "diagonal"


def select_orientation_diverse_candidates(
    document: Mapping[str, Any],
    *,
    count_per_orientation: int = 3,
    roll_rad: float = 0.0,
    orientation_tolerance_deg: float = 15.0,
    maximum_orientation_groups: int = 3,
) -> dict[str, Any]:
    """Select representatives from distinct jaw-closing-axis groups.

    Closing axes are unoriented lines for a parallel-jaw gripper, so ``+c`` and
    ``-c`` belong to the same group. Ranked families are assigned to the first
    group leader within ``orientation_tolerance_deg``; this deterministic
    leader clustering is intentionally not transitive. The display
    representative for each group is chosen by largest convex domain, then
    largest aperture, then family ID. This is a visualization heuristic—not a
    reachability, contact-material, or grasp-quality ranking.
    """
    if (
        isinstance(count_per_orientation, bool)
        or not isinstance(count_per_orientation, int)
        or not 1 <= count_per_orientation <= 5
    ):
        raise ValueError("count_per_orientation must be an integer in [1, 5]")
    if (
        isinstance(maximum_orientation_groups, bool)
        or not isinstance(maximum_orientation_groups, int)
        or not 1 <= maximum_orientation_groups <= 4
    ):
        raise ValueError("maximum_orientation_groups must be an integer in [1, 4]")
    tolerance_deg = float(orientation_tolerance_deg)
    if not math.isfinite(tolerance_deg) or not 0.0 < tolerance_deg < 90.0:
        raise ValueError("orientation_tolerance_deg must lie in (0, 90)")
    if document.get("scope") != SCOPE or document.get("insertion_safe") is not False:
        raise ValueError(
            f"document must have scope={SCOPE} and insertion_safe=false"
        )
    raw_families = document.get("families")
    if not isinstance(raw_families, list) or not raw_families:
        raise ValueError("document must contain at least one family")
    records = [
        _family_display_record(
            _mapping(family, label=f"families[{family_index}]"),
            family_index=family_index,
        )
        for family_index, family in enumerate(raw_families)
    ]
    records.sort(key=lambda record: (
        -float(record["largest_domain_area_m2"]),
        -float(record["maximum_aperture_m"]),
        str(record["family_id"]),
    ))

    cosine_threshold = math.cos(math.radians(tolerance_deg))
    groups: list[dict[str, Any]] = []
    for record in records:
        axis = np.asarray(record["closing_axis_P"], dtype=float)
        assigned = next(
            (
                group for group in groups
                if abs(float(axis @ group["representative_axis_P"]))
                >= cosine_threshold
            ),
            None,
        )
        if assigned is None:
            groups.append({
                "representative_axis_P": axis,
                "families": [record],
            })
        else:
            assigned["families"].append(record)
    detected_orientation_group_count = len(groups)
    groups = groups[:maximum_orientation_groups]

    inputs = _mapping(document.get("inputs"), label="inputs")
    T_W_P = validate_transform(np.asarray(inputs.get("T_W_P_insert"), dtype=float))
    group_summaries: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    used_prefixes: set[str] = set()
    for group_index, group in enumerate(groups):
        representative = group["families"][0]
        family = representative["family"]
        family_id = str(representative["family_id"])
        subset = dict(document)
        subset["families"] = [family]
        sampled = select_representative_candidates(
            subset,
            count=count_per_orientation,
            roll_rad=roll_rad,
        )
        closing_axis_P = np.asarray(
            representative["closing_axis_P"], dtype=float)
        closing_axis_W = T_W_P[:3, :3] @ closing_axis_P
        orientation_label = _top_down_orientation_label(closing_axis_W)
        base_prefix = {
            "horizontal": "H",
            "vertical": "V",
            "diagonal": "D",
            "out_of_plane": "Z",
        }[orientation_label]
        prefix = base_prefix
        if prefix in used_prefixes:
            prefix = f"O{group_index + 1}"
        used_prefixes.add(prefix)
        orientation_id = f"orientation_{group_index + 1}"
        for candidate_index, candidate in enumerate(sampled["candidates"]):
            candidate["candidate_id"] = f"{prefix}{candidate_index + 1}"
            candidate["orientation_id"] = orientation_id
            candidate["top_down_orientation"] = orientation_label
            candidate["external_visibility_certified"] = False
            candidates.append(candidate)
        group_summaries.append({
            "orientation_id": orientation_id,
            "plot_prefix": prefix,
            "top_down_orientation": orientation_label,
            "family_id": family_id,
            "closing_axis_P": closing_axis_P.tolist(),
            "closing_axis_W": closing_axis_W.tolist(),
            "clustered_family_count": len(group["families"]),
            "family_largest_domain_area_m2": float(
                representative["largest_domain_area_m2"]),
            "display_aperture_m": float(
                sampled["candidates"][0]["aperture_m"]),
            "external_visibility_certified": False,
            "contact_material_semantics": "unknown_from_stl",
            "insertion_safe": False,
        })

    return {
        "selection": {
            "mode": "orientation_diverse",
            "policy": (
                "leader_cluster_unoriented_closing_axes_then_select_largest_"
                "domain_largest_aperture_family_per_group"
            ),
            "ranking_intent": (
                "deterministic display heuristic; not external-visibility, "
                "grasp-quality, or insertion-safety ranking"
            ),
            "orientation_tolerance_deg": tolerance_deg,
            "detected_orientation_group_count": detected_orientation_group_count,
            "orientation_group_count": len(group_summaries),
            "omitted_orientation_group_count": (
                detected_orientation_group_count - len(group_summaries)
            ),
            "maximum_orientation_groups": maximum_orientation_groups,
            "sample_count_per_orientation": count_per_orientation,
            "sample_count": len(candidates),
            "roll_rad": float(roll_rad),
            "orientation_groups": group_summaries,
        },
        "candidates": candidates,
    }


def encode_png(rgb: np.ndarray) -> bytes:
    """Encode an RGB uint8 array as PNG using only the Python standard library."""
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
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
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
    dash_px: float = 9.0,
) -> None:
    start = np.asarray(first, dtype=float)
    end = np.asarray(second, dtype=float)
    distance = float(np.linalg.norm(end - start))
    if distance <= 0.0:
        return
    direction = (end - start) / distance
    for begin in np.arange(0.0, distance, 2.0 * dash_px):
        finish = min(begin + dash_px, distance)
        _draw_line(
            image,
            start + begin * direction,
            start + finish * direction,
            color,
            thickness=thickness,
        )


def _fill_triangle(
    image: np.ndarray,
    triangle: np.ndarray,
    color: Sequence[int],
) -> None:
    points = np.asarray(triangle, dtype=float)
    first_edge = points[1] - points[0]
    second_edge = points[2] - points[0]
    area2 = float(
        first_edge[0] * second_edge[1] - first_edge[1] * second_edge[0]
    )
    if abs(area2) < 0.25:
        return
    height, width = image.shape[:2]
    x0 = max(0, int(math.floor(float(np.min(points[:, 0])))))
    x1 = min(width - 1, int(math.ceil(float(np.max(points[:, 0])))))
    y0 = max(0, int(math.floor(float(np.min(points[:, 1])))))
    y1 = min(height - 1, int(math.ceil(float(np.max(points[:, 1])))))
    if x0 > x1 or y0 > y1:
        return
    yy, xx = np.mgrid[y0:y1 + 1, x0:x1 + 1]
    sample = np.stack((xx + 0.5, yy + 0.5), axis=-1)
    edges = np.roll(points, -1, axis=0) - points
    relative = sample[:, :, None, :] - points[None, None, :, :]
    cross = (
        edges[None, None, :, 0] * relative[:, :, :, 1]
        - edges[None, None, :, 1] * relative[:, :, :, 0]
    )
    mask = np.all(cross >= -1e-7, axis=2) | np.all(cross <= 1e-7, axis=2)
    image[y0:y1 + 1, x0:x1 + 1][mask] = np.asarray(color, dtype=np.uint8)


def _fill_polygon(
    image: np.ndarray,
    vertices: np.ndarray,
    color: Sequence[int],
) -> None:
    for index in range(1, len(vertices) - 1):
        _fill_triangle(
            image,
            np.stack((vertices[0], vertices[index], vertices[index + 1])),
            color,
        )


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


def _projector(
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    rectangle: tuple[int, int, int, int],
):
    left, top, right, bottom = rectangle
    extent = np.maximum(bounds_max - bounds_min, 1e-12)
    scale = min((right - left) / extent[0], (bottom - top) / extent[1])
    center_world = 0.5 * (bounds_min + bounds_max)
    center_pixel = np.array([(left + right) / 2.0, (top + bottom) / 2.0])

    def project(points_xy: np.ndarray) -> np.ndarray:
        points = np.asarray(points_xy, dtype=float)
        result = np.empty_like(points)
        result[..., 0] = center_pixel[0] + scale * (
            points[..., 0] - center_world[0])
        result[..., 1] = center_pixel[1] - scale * (
            points[..., 1] - center_world[1])
        return result

    return project, float(scale)


def _draw_arrow(
    image: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    color: Sequence[int],
    *,
    thickness: int = 2,
) -> None:
    _draw_line(image, start, end, color, thickness=thickness)
    vector = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0:
        return
    direction = vector / norm
    side = np.array([-direction[1], direction[0]])
    for sign in (-1.0, 1.0):
        _draw_line(
            image,
            end,
            end - 9.0 * direction + sign * 4.5 * side,
            color,
            thickness=thickness,
        )


def _render_rgb(
    *,
    mesh,
    T_W_P: np.ndarray,
    selection: Mapping[str, Any],
    width: int,
    height: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if width < 800 or height < 600:
        raise ValueError("render dimensions must be at least 800x600")
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[:] = [18, 23, 31]
    header_bottom = 94
    footer_top = height - 86
    image[:header_bottom] = [25, 33, 44]
    image[footer_top:] = [25, 33, 44]
    plot_right = width - 350
    plot_rectangle = (52, header_bottom + 18, plot_right - 20, footer_top - 20)
    image[
        plot_rectangle[1]:plot_rectangle[3],
        plot_rectangle[0]:plot_rectangle[2],
    ] = [13, 18, 25]
    image[header_bottom:footer_top, plot_right:] = [21, 28, 38]

    triangles_W = (
        mesh.triangles @ T_W_P[:3, :3].T + T_W_P[:3, 3]
    )
    candidates = selection["candidates"]
    contact_points = np.concatenate([
        np.asarray(candidate["contacts_W_m"], dtype=float)
        for candidate in candidates
    ], axis=0)
    for candidate in candidates:
        contacts = np.asarray(candidate["contacts_W_m"], dtype=float)
        projected_separation = float(np.linalg.norm(
            contacts[1, :2] - contacts[0, :2]))
        if projected_separation < 0.05 * float(candidate["aperture_m"]):
            raise ValueError(
                "selected jaw-closing axis is nearly parallel to the top-down "
                "view; its two contacts cannot be represented honestly in "
                "world XY"
            )
    mesh_xy = triangles_W[:, :, :2]
    points = np.concatenate((mesh_xy.reshape(-1, 2), contact_points[:, :2]), axis=0)
    bounds_min = np.min(points, axis=0) - 0.0045
    bounds_max = np.max(points, axis=0) + 0.0045
    project, pixels_per_m = _projector(bounds_min, bounds_max, plot_rectangle)

    # Painter-sort the actual STL triangles by world height.
    order = np.argsort(np.mean(triangles_W[:, :, 2], axis=1))
    normals_W = mesh.normals @ T_W_P[:3, :3].T
    z_min = float(np.min(triangles_W[:, :, 2]))
    z_span = max(
        float(np.max(triangles_W[:, :, 2]) - z_min),
        1e-12,
    )
    for triangle_index in order:
        triangle = triangles_W[int(triangle_index)]
        projected = project(triangle[:, :2])
        height_fraction = (
            float(np.mean(triangle[:, 2])) - z_min
        ) / z_span
        facing = abs(float(normals_W[int(triangle_index), 2]))
        shade = int(round(78 + 62 * height_fraction + 35 * facing))
        shade = max(58, min(185, shade))
        _fill_triangle(image, projected, [shade, shade + 4, shade + 10])

    palette = [
        [0, 210, 255],
        [255, 174, 66],
        [186, 120, 255],
        [74, 222, 128],
        [255, 104, 116],
        [255, 221, 74],
        [80, 145, 255],
        [255, 126, 212],
    ]
    selected = _mapping(selection.get("selection"), label="selection")
    selection_mode = str(selected.get("mode", "single_family"))
    if selection_mode == "orientation_diverse":
        groups_for_layout = selected.get("orientation_groups", [])
        rejected_group_count = sum(
            group.get("full_part_support_span_within_opening_range") is False
            for group in groups_for_layout
            if isinstance(group, Mapping)
        )
        legend_y_after_candidates = (
            258
            + 28 * len(groups_for_layout)
            + 24 * rejected_group_count
            + 26 * len(candidates)
        )
        required_height = max(800, legend_y_after_candidates + 290)
        if width < 1100 or height < required_height:
            raise ValueError(
                "orientation-diverse render requires width >= 1100 and "
                f"height >= {required_height} for the requested groups/samples"
            )
    orientation_color_indices = {
        str(group["orientation_id"]): group_index
        for group_index, group in enumerate(
            selected.get("orientation_groups", [])
        )
        if isinstance(group, Mapping) and "orientation_id" in group
    }

    def candidate_color(
        candidate_index: int,
        candidate: Mapping[str, Any],
    ) -> list[int]:
        orientation_id = candidate.get("orientation_id")
        if selection_mode == "orientation_diverse" and isinstance(
            orientation_id, str
        ):
            return palette[
                orientation_color_indices.get(orientation_id, candidate_index)
                % len(palette)
            ]
        return palette[candidate_index % len(palette)]

    pad_length_m = 0.0024
    pad_width_m = 0.0026
    for index, candidate in enumerate(candidates):
        color = candidate_color(index, candidate)
        contacts_W = np.asarray(candidate["contacts_W_m"], dtype=float)
        contacts_px = project(contacts_W[:, :2])
        midpoint_W = np.mean(contacts_W, axis=0)
        midpoint_px = project(midpoint_W[None, :2])[0]
        closing_xy = contacts_W[1, :2] - contacts_W[0, :2]
        closing_norm = float(np.linalg.norm(closing_xy))
        if closing_norm <= 1e-12:
            continue
        closing_xy /= closing_norm
        tangent_xy = np.array([-closing_xy[1], closing_xy[0]])
        for contact_index, contact in enumerate(contacts_W):
            outward = (-closing_xy if contact_index == 0 else closing_xy)
            inner = contact[:2]
            outer = inner + pad_length_m * outward
            polygon_W = np.stack((
                inner - 0.5 * pad_width_m * tangent_xy,
                inner + 0.5 * pad_width_m * tangent_xy,
                outer + 0.5 * pad_width_m * tangent_xy,
                outer - 0.5 * pad_width_m * tangent_xy,
            ))
            polygon_px = project(polygon_W)
            _fill_polygon(image, polygon_px, [max(0, c - 35) for c in color])
            for edge_index, edge_start in enumerate(polygon_px):
                _draw_line(
                    image,
                    edge_start,
                    polygon_px[(edge_index + 1) % len(polygon_px)],
                    color,
                    thickness=2,
                )
            _draw_disk(image, contacts_px[contact_index], 5, [245, 248, 252])
            arrow_start = project(
                (outer + 0.0005 * (-outward))[None, :])[0]
            arrow_end = project(
                (inner + 0.0004 * outward)[None, :])[0]
            _draw_arrow(image, arrow_start, arrow_end, color, thickness=1)
        _draw_dashed_line(
            image, contacts_px[0], contacts_px[1], color, thickness=1)
        _draw_disk(image, midpoint_px, 8, [15, 20, 27])
        _draw_disk(image, midpoint_px, 5, color)
        approach_W = np.asarray(candidate["T_W_E"], dtype=float)[:3, 2]
        approach_xy_norm = float(np.linalg.norm(approach_W[:2]))
        if approach_xy_norm <= 1e-8:
            if approach_W[2] < 0.0:
                # Circle plus X: direction points into the top-down image.
                arm = 5
                _draw_line(
                    image,
                    midpoint_px + [-arm, -arm],
                    midpoint_px + [arm, arm],
                    [245, 248, 252],
                    thickness=1,
                )
                _draw_line(
                    image,
                    midpoint_px + [-arm, arm],
                    midpoint_px + [arm, -arm],
                    [245, 248, 252],
                    thickness=1,
                )
            else:
                # Circle plus dot: direction points out of the image.
                _draw_disk(image, midpoint_px, 2, [245, 248, 252])
        else:
            # Project the palm-to-contact approach direction into world XY.
            approach_xy = approach_W[:2] / approach_xy_norm
            arrow_start_W = midpoint_W[:2] - 0.004 * approach_xy
            arrow_start_px = project(arrow_start_W[None, :])[0]
            _draw_arrow(
                image,
                arrow_start_px,
                midpoint_px,
                [245, 248, 252],
                thickness=1,
            )
        closing_direction_px = contacts_px[1] - contacts_px[0]
        closing_direction_px /= float(np.linalg.norm(closing_direction_px))
        candidate_label = str(candidate["candidate_id"])
        label_scale = 2
        label_target = contacts_px[1] + 48.0 * closing_direction_px
        label_anchor = label_target - np.array([
            0.5 * len(candidate_label) * 4 * label_scale,
            0.5 * 5 * label_scale,
        ])
        _draw_text(
            image,
            (int(label_anchor[0]), int(label_anchor[1])),
            candidate_label,
            color,
            scale=label_scale,
        )

    white = [236, 241, 247]
    muted = [148, 162, 180]
    warning = [255, 190, 70]
    _draw_text(
        image, (52, 22), "TOP-DOWN REPRESENTATIVE GRASP CANDIDATES",
        white, scale=4)
    subtitle = (
        "MULTIPLE CLOSING ORIENTATIONS - OBJECT ONLY"
        if selection_mode == "orientation_diverse"
        else "ONE CONTINUOUS FAMILY - OBJECT ONLY"
    )
    _draw_text(
        image, (54, 66), subtitle,
        muted, scale=2)

    legend_x = plot_right + 28
    _draw_text(image, (legend_x, 126), "DISPLAY SELECTION", white, scale=3)
    first_approach = np.asarray(candidates[0]["T_W_E"], dtype=float)[:3, 2]
    first_approach_xy_norm = float(np.linalg.norm(first_approach[:2]))
    if first_approach_xy_norm <= 1e-8:
        if first_approach[2] < 0.0:
            approach_summary = "ROLL 0.0 DEG - INTO PAGE"
            approach_key = "CENTER X   APPROACH INTO PAGE"
            approach_glyph = "into_page_cross"
        else:
            approach_summary = "ROLL 0.0 DEG - OUT OF PAGE"
            approach_key = "CENTER DOT APPROACH OUT OF PAGE"
            approach_glyph = "out_of_page_dot"
    else:
        approach_summary = "ROLL 0.0 DEG - ARROW IN VIEW"
        approach_key = "WHITE ARROW  APPROACH PROJECTION"
        approach_glyph = "world_xy_arrow"

    if selection_mode == "orientation_diverse":
        groups = selected.get("orientation_groups")
        if not isinstance(groups, list) or not groups:
            raise ValueError(
                "orientation-diverse selection must contain orientation_groups"
            )
        _draw_text(
            image,
            (legend_x, 164),
            f"{len(groups)} CLOSING ORIENTATIONS",
            warning,
            scale=2,
        )
        y = 192
        for group_index, group in enumerate(groups):
            color = palette[group_index % len(palette)]
            orientation = str(group["top_down_orientation"]).upper()
            family_label = str(group["family_id"]).upper().replace(
                "FAMILY_", "F")
            aperture_mm = float(group["display_aperture_m"]) * 1000.0
            _draw_disk(image, (legend_x + 5, y + 5), 5, color)
            _draw_text(
                image,
                (legend_x + 20, y),
                (
                    f"{group['plot_prefix']} {orientation} LOCAL "
                    f"{family_label} {aperture_mm:.3f}MM"
                ),
                white,
                scale=2,
            )
            y += 28
        opening_range = selected.get("configured_opening_range_m")
        if isinstance(opening_range, list) and len(opening_range) == 2:
            rejected_groups = [
                group for group in groups
                if group.get(
                    "full_part_support_span_within_opening_range"
                ) is False
            ]
            for group in rejected_groups:
                span_mm = float(group["full_part_support_span_m"]) * 1000.0
                maximum_mm = float(opening_range[1]) * 1000.0
                _draw_text(
                    image,
                    (legend_x + 20, y),
                    (
                        f"{group['plot_prefix']} FULL PART {span_mm:.3f} "
                        f"> MAX {maximum_mm:.3f} MM"
                    ),
                    [255, 104, 116],
                    scale=2,
                )
                y += 24
        _draw_text(image, (legend_x, y + 4), approach_summary, muted, scale=2)
        y += 38
        _draw_text(
            image, (legend_x, y), "DISPLAYED CANDIDATES",
            white, scale=2)
        y += 28
        for index, candidate in enumerate(candidates):
            color = candidate_color(index, candidate)
            _draw_disk(image, (legend_x + 5, y + 5), 5, color)
            _draw_text(
                image,
                (legend_x + 20, y),
                (
                    f"{candidate['candidate_id']}  "
                    f"AP {float(candidate['aperture_m']) * 1000.0:.3f} MM"
                ),
                white,
                scale=2,
            )
            y += 26
    else:
        _draw_text(
            image, (legend_x, 164), f"SELECTED {selected['family_id']}",
            warning, scale=2)
        aperture_values_mm = np.array([
            float(candidate["aperture_m"]) * 1000.0
            for candidate in candidates
        ])
        if float(np.ptp(aperture_values_mm)) <= 1e-6:
            aperture_label = f"APERTURE {aperture_values_mm[0]:.3f} MM"
        else:
            aperture_label = (
                f"APERTURE {np.min(aperture_values_mm):.3f}-"
                f"{np.max(aperture_values_mm):.3f} MM"
            )
        _draw_text(
            image, (legend_x, 190), aperture_label,
            muted, scale=2)
        _draw_text(
            image, (legend_x, 216), approach_summary,
            muted, scale=2)
        _draw_text(
            image, (legend_x, 254), "EXACT DISPLAY PARAMETERS",
            white, scale=2)
        y = 282
        for index, candidate in enumerate(candidates):
            color = candidate_color(index, candidate)
            parameters = candidate["parameters"]
            _draw_disk(image, (legend_x + 5, y + 5), 5, color)
            _draw_text(
                image,
                (legend_x + 20, y),
                (
                    f"{candidate['candidate_id']}  "
                    f"U {parameters['u_m'] * 1000.0:+.2f}  "
                    f"V {parameters['v_m'] * 1000.0:+.2f} MM"
                ),
                white,
                scale=2,
            )
            y += 30

    _draw_text(image, (legend_x, y + 18), "GLYPH KEY", white, scale=2)
    _draw_text(image, (legend_x, y + 46), "WHITE DOT  IDEAL CONTACT", muted, scale=2)
    _draw_text(image, (legend_x, y + 70), "DASH LINE  CLOSING AXIS", muted, scale=2)
    _draw_text(image, (legend_x, y + 94), approach_key, muted, scale=2)
    _draw_text(image, (legend_x, y + 108), "WORLD TOP VIEW", white, scale=2)
    origin = np.array([legend_x + 28.0, y + 184.0])
    _draw_arrow(image, origin, origin + [55, 0], palette[0], thickness=2)
    _draw_arrow(image, origin, origin + [0, -50], palette[3], thickness=2)
    _draw_text(image, (legend_x + 90, int(y + 177)), "+X W", palette[0], scale=2)
    _draw_text(image, (legend_x + 42, int(y + 128)), "+Y W", palette[3], scale=2)

    # Metric scale bar in the plot.
    scale_bar_m = 0.010
    scale_bar_px = scale_bar_m * pixels_per_m
    scale_start = np.array([plot_rectangle[0] + 28.0, plot_rectangle[3] - 30.0])
    _draw_line(
        image, scale_start, scale_start + [scale_bar_px, 0],
        white, thickness=2)
    _draw_line(
        image, scale_start + [0, -7], scale_start + [0, 7],
        white, thickness=2)
    _draw_line(
        image,
        scale_start + [scale_bar_px, -7],
        scale_start + [scale_bar_px, 7],
        white,
        thickness=2,
    )
    _draw_text(
        image,
        (int(scale_start[0]), int(scale_start[1] - 25)),
        "10 MM",
        white,
        scale=2,
    )

    footer_family_label = (
        "ORIENTATION GROUPS"
        if selection_mode == "orientation_diverse"
        else "A CONTINUOUS SURFACE FAMILY"
    )
    _draw_text(
        image,
        (52, footer_top + 18),
        f"SCHEMATIC JAWS - REPRESENTATIVE SAMPLES FROM {footer_family_label}",
        warning,
        scale=2,
    )
    _draw_text(
        image,
        (52, footer_top + 50),
        "NOT CERTIFIED: GRIPPER COLLISION / PCB CLEARANCE / VISIBILITY / INSERTION SAFETY",
        white,
        scale=2,
    )
    view = {
        "type": "orthographic_top_down",
        "frame": "world",
        "horizontal_axis": "+X_W",
        "vertical_axis": "+Y_W",
        "view_direction": "-Z_W",
        "approach_glyph": approach_glyph,
        "image_size_px": [width, height],
        "content_bounds_xy_m": [bounds_min.tolist(), bounds_max.tolist()],
        "viewport_bounds_xy_m": [
            (
                0.5 * (bounds_min + bounds_max)
                - np.array([
                    (plot_rectangle[2] - plot_rectangle[0]) / (2.0 * pixels_per_m),
                    (plot_rectangle[3] - plot_rectangle[1]) / (2.0 * pixels_per_m),
                ])
            ).tolist(),
            (
                0.5 * (bounds_min + bounds_max)
                + np.array([
                    (plot_rectangle[2] - plot_rectangle[0]) / (2.0 * pixels_per_m),
                    (plot_rectangle[3] - plot_rectangle[1]) / (2.0 * pixels_per_m),
                ])
            ).tolist(),
        ],
        "pixels_per_m": pixels_per_m,
    }
    return image, view


def render_top_down_candidate_image(
    map_path: str | Path,
    output_png: str | Path,
    *,
    count: int = 6,
    selection_mode: str = "single_family",
    count_per_orientation: int = 3,
    orientation_tolerance_deg: float = 15.0,
    maximum_orientation_groups: int = 3,
    width: int = 1400,
    height: int = 900,
) -> dict[str, Any]:
    """Write a deterministic top-down PNG and return its companion metadata."""
    map_file = Path(map_path).resolve()
    output_file = Path(output_png).resolve()
    if not map_file.is_file():
        raise FileNotFoundError(f"grasp-map JSON was not found: {map_file}")
    if output_file.suffix.lower() != ".png":
        raise ValueError("output_png must name a .png file")
    document = _load_document(map_file)
    provenance = _mapping(document.get("provenance"), label="provenance")
    part = _mapping(provenance.get("part"), label="provenance.part")
    part_relative = part.get("path")
    if not isinstance(part_relative, str) or not part_relative:
        raise ValueError("provenance.part.path must be a non-empty string")
    supplied_part_path = Path(part_relative)
    if supplied_part_path.is_absolute():
        raise ValueError("provenance.part.path must be repository-relative")
    part_path = (ROOT / supplied_part_path).resolve()
    try:
        part_path.relative_to(ROOT)
    except ValueError as error:
        raise ValueError("provenance.part.path escapes the repository") from error
    if not part_path.is_file():
        raise FileNotFoundError(f"part STL was not found: {part_path}")
    expected_part_hash = part.get("sha256")
    if not isinstance(expected_part_hash, str) or _sha256(part_path) != expected_part_hash:
        raise ValueError("part STL does not match grasp-map provenance")
    scale_to_m = float(part.get("scale_to_m"))
    if not math.isfinite(scale_to_m) or scale_to_m <= 0.0:
        raise ValueError("provenance.part.scale_to_m must be positive")

    normalized_mode = {
        "single": "single_family",
        "single_family": "single_family",
        "orientations": "orientation_diverse",
        "orientation_diverse": "orientation_diverse",
    }.get(str(selection_mode))
    if normalized_mode is None:
        raise ValueError(
            "selection_mode must be single_family or orientation_diverse"
        )
    if normalized_mode == "orientation_diverse":
        selection = select_orientation_diverse_candidates(
            document,
            count_per_orientation=count_per_orientation,
            roll_rad=0.0,
            orientation_tolerance_deg=orientation_tolerance_deg,
            maximum_orientation_groups=maximum_orientation_groups,
        )
    else:
        selection = select_representative_candidates(
            document, count=count, roll_rad=0.0)
    inputs = _mapping(document.get("inputs"), label="inputs")
    T_W_P = validate_transform(np.asarray(inputs.get("T_W_P_insert"), dtype=float))
    mesh = load_scaled_binary_stl(part_path, scale_to_m=scale_to_m)
    if normalized_mode == "orientation_diverse":
        opening_range = _finite_array(
            inputs.get("opening_range_m"),
            (2,),
            label="inputs.opening_range_m",
        )
        selection["selection"]["configured_opening_range_m"] = (
            opening_range.tolist())
        vertices_P = mesh.vertices
        for group in selection["selection"]["orientation_groups"]:
            closing_axis_P = _unit(
                group["closing_axis_P"],
                label=f"{group['orientation_id']}.closing_axis_P",
            )
            support = vertices_P @ closing_axis_P
            full_span = float(np.max(support) - np.min(support))
            group["full_part_support_span_m"] = full_span
            group["full_part_support_span_within_opening_range"] = bool(
                opening_range[0] <= full_span <= opening_range[1]
            )
    rgb, view = _render_rgb(
        mesh=mesh,
        T_W_P=T_W_P,
        selection=selection,
        width=int(width),
        height=int(height),
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_bytes(encode_png(rgb))
    return {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "scope": SCOPE,
        "insertion_safe": False,
        "source_map": {
            "path": (
                map_file.relative_to(ROOT).as_posix()
                if map_file.is_relative_to(ROOT)
                else str(map_file)
            ),
            "sha256": _sha256(map_file),
            "artifact_type": document["artifact_type"],
        },
        "part": {
            "path": part_relative,
            "sha256": expected_part_hash,
            "scale_to_m": scale_to_m,
        },
        "view": view,
        "selection": selection["selection"],
        "candidates": selection["candidates"],
        "render_semantics": {
            "part_geometry": "actual STL orthographic projection",
            "gripper_geometry": "schematic ideal parallel-jaw glyph",
            "samples_are_visualization_only": True,
            "samples_define_map": False,
        },
        "limitations": [
            (
                "representative samples from distinct closing-axis groups only"
                if normalized_mode == "orientation_diverse"
                else "representative samples from one continuous family only"
            ),
            "ideal point contacts and schematic jaw pads",
            "complete gripper collision is not checked",
            "PCB and insertion-path clearance are not checked",
            "directional external visibility is not certified",
            "robot reachability and insertion safety are not certified",
        ],
    }


__all__ = [
    "ARTIFACT_TYPE",
    "encode_png",
    "render_top_down_candidate_image",
    "select_orientation_diverse_candidates",
    "select_representative_candidates",
]
