"""Conservative pose-proposal templates for task-and-motion planning.

Template files are hints only.  They are never converted into collision
geometry and they never bypass CAD contact, robot IK, or scene-collision
validation.  The generic data model carries an explicit coordinate frame and
semantic role so a consumer cannot silently reinterpret a world-frame part
pose as a part-frame gripper pose.

Plain XYZ convention
--------------------

Each non-comment row has exactly three or six scalar columns:

``x y z``
    A position proposal with no orientation (three DoF).

``x y z roll pitch yaw``
    An explicit six-DoF pose proposal.  RPY is intrinsic XYZ, equivalently the
    active rotation ``Rz(yaw) @ Ry(pitch) @ Rx(roll)``.  Translation units and
    angle units are declared at load time; defaults are metres and degrees.

ASCII PCD convention
--------------------

PCD columns are selected by their ``FIELDS`` names, never by column count.
``x``, ``y``, and ``z`` are required.  ``normal_x``, ``normal_y``, and
``normal_z`` are optional but must appear together; they form a preferred TCP
``+Z`` approach-direction hint in the declared frame.  Thus a six-field
XYZ+normal cloud remains a three-DoF position template, not XYZ+RPY.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from .geometry_grasps import GraspCandidate
from .offline import fingerprint_content, fingerprint_file
from .se3 import so3_geodesic, transform_from_rpy


RPY_CONVENTION = "intrinsic_xyz_active_Rz_yaw_Ry_pitch_Rx_roll"
PROPOSAL_ONLY_USAGE = "proposal_only_not_collision_geometry"


class TemplateFrame(str, Enum):
    PART = "part"
    WORLD = "world"


class TemplateRole(str, Enum):
    GRASP_TCP = "grasp_tcp"
    PART_POSE = "part_pose"


def _enum_value(value: str | Enum, enum_type: type[Enum], name: str):
    raw = value.value if isinstance(value, Enum) else str(value).strip().lower()
    try:
        return enum_type(raw)
    except ValueError as error:
        choices = ", ".join(item.value for item in enum_type)
        raise ValueError(f"{name} must be one of: {choices}; got {raw!r}") from error


def _readonly_vector(value, name: str, *, unit: bool = False) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite three-vector")
    if unit:
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-12:
            raise ValueError(f"{name} must be nonzero")
        vector = vector / norm
    result = vector.copy()
    result.setflags(write=False)
    return result


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if result <= 0 or result != value:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _length_scale(units: str) -> float:
    normalized = str(units).strip().lower().replace("µ", "u")
    scales = {
        "m": 1.0,
        "meter": 1.0,
        "metre": 1.0,
        "mm": 1e-3,
        "millimeter": 1e-3,
        "millimetre": 1e-3,
        "cm": 1e-2,
        "centimeter": 1e-2,
        "centimetre": 1e-2,
        "um": 1e-6,
        "micrometer": 1e-6,
        "micrometre": 1e-6,
        "in": 0.0254,
        "inch": 0.0254,
    }
    if normalized not in scales:
        raise ValueError(
            "xyz_units must explicitly be m, mm, cm, um, or in; "
            f"got {units!r}")
    return scales[normalized]


def _angle_scale(units: str) -> float:
    normalized = str(units).strip().lower()
    if normalized in {"deg", "degree", "degrees"}:
        return np.pi / 180.0
    if normalized in {"rad", "radian", "radians"}:
        return 1.0
    raise ValueError(f"rpy_units must explicitly be deg or rad; got {units!r}")


@dataclass(frozen=True)
class PoseProposal:
    """One generic three- or six-DoF proposal with explicit semantics."""

    proposal_id: str
    frame: TemplateFrame | str
    role: TemplateRole | str
    position_m: np.ndarray
    rpy_rad: np.ndarray | None = None
    normal_hint: np.ndarray | None = None
    source_line: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.proposal_id, str) or not self.proposal_id.strip():
            raise ValueError("proposal_id must be a non-empty string")
        object.__setattr__(self, "frame", _enum_value(
            self.frame, TemplateFrame, "frame"))
        object.__setattr__(self, "role", _enum_value(
            self.role, TemplateRole, "role"))
        object.__setattr__(self, "position_m", _readonly_vector(
            self.position_m, "position_m"))
        if self.rpy_rad is not None:
            object.__setattr__(self, "rpy_rad", _readonly_vector(
                self.rpy_rad, "rpy_rad"))
        if self.normal_hint is not None:
            object.__setattr__(self, "normal_hint", _readonly_vector(
                self.normal_hint, "normal_hint", unit=True))
        if self.source_line is not None:
            object.__setattr__(self, "source_line", _positive_integer(
                self.source_line, "source_line"))

    @property
    def dof(self) -> int:
        # Named PCD normals remain hints and never add orientation DoF.
        return 6 if self.rpy_rad is not None else 3

    @property
    def T_frame_target(self) -> np.ndarray | None:
        """Return the explicit pose, or ``None`` for a position-only hint."""
        if self.rpy_rad is None:
            return None
        return transform_from_rpy(self.position_m, self.rpy_rad)


@dataclass(frozen=True)
class PoseTemplate:
    """A parsed proposal source; never an exact surface/collision model."""

    template_id: str
    source_path: str
    source_format: str
    frame: TemplateFrame | str
    role: TemplateRole | str
    proposals: tuple[PoseProposal, ...]
    source_fingerprint: str
    xyz_units: str
    rpy_units: str
    usage: str = field(default=PROPOSAL_ONLY_USAGE, init=False)
    rpy_convention: str = field(default=RPY_CONVENTION, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.template_id, str) or not self.template_id.strip():
            raise ValueError("template_id must be a non-empty string")
        path = str(Path(self.source_path).resolve())
        source_format = str(self.source_format).strip().lower()
        if source_format not in {"xyz", "ascii_pcd"}:
            raise ValueError("source_format must be xyz or ascii_pcd")
        frame = _enum_value(self.frame, TemplateFrame, "frame")
        role = _enum_value(self.role, TemplateRole, "role")
        proposals = tuple(self.proposals)
        if not proposals:
            raise ValueError("pose template must contain at least one proposal")
        if any(not isinstance(item, PoseProposal) for item in proposals):
            raise TypeError("proposals must contain PoseProposal values")
        if any(item.frame != frame or item.role != role for item in proposals):
            raise ValueError("all proposals must match their template frame and role")
        if not isinstance(self.source_fingerprint, str) or not self.source_fingerprint:
            raise ValueError("source_fingerprint must be non-empty")
        object.__setattr__(self, "source_path", path)
        object.__setattr__(self, "source_format", source_format)
        object.__setattr__(self, "frame", frame)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "proposals", proposals)

    @property
    def semantic_fingerprint(self) -> str:
        return fingerprint_content({
            "source": self.source_fingerprint,
            "format": self.source_format,
            "frame": self.frame.value,
            "role": self.role.value,
            "xyz_units": self.xyz_units,
            "rpy_units": self.rpy_units,
            "rpy_convention": self.rpy_convention,
            "usage": self.usage,
        })


def _proposal(
    template_id: str,
    index: int,
    line_number: int,
    values: np.ndarray,
    *,
    frame: TemplateFrame,
    role: TemplateRole,
    length_scale: float,
    angle_scale: float,
    normal_hint: np.ndarray | None = None,
) -> PoseProposal:
    position = values[:3] * length_scale
    rpy = None if len(values) == 3 else values[3:6] * angle_scale
    return PoseProposal(
        proposal_id=f"{template_id}:{index:06d}",
        frame=frame,
        role=role,
        position_m=position,
        rpy_rad=rpy,
        normal_hint=normal_hint,
        source_line=line_number,
    )


def _parse_xyz(
    path: Path,
    template_id: str,
    frame: TemplateFrame,
    role: TemplateRole,
    length_scale: float,
    angle_scale: float,
    max_proposals: int,
) -> tuple[PoseProposal, ...]:
    proposals = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError(f"XYZ template must be UTF-8 text: {path}") from error
    for line_number, raw in enumerate(lines, 1):
        content = raw.split("#", 1)[0].strip()
        if not content:
            continue
        tokens = content.replace(",", " ").split()
        if len(tokens) not in (3, 6):
            raise ValueError(
                f"XYZ template row at {path}:{line_number} must have exactly "
                "3 (XYZ) or 6 (XYZ+RPY) columns")
        try:
            values = np.asarray([float(token) for token in tokens], dtype=float)
        except ValueError as error:
            raise ValueError(
                f"XYZ template has a non-numeric value at {path}:{line_number}") from error
        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"XYZ template values must be finite at {path}:{line_number}")
        proposals.append(_proposal(
            template_id, len(proposals), line_number, values,
            frame=frame, role=role, length_scale=length_scale,
            angle_scale=angle_scale))
        if len(proposals) > max_proposals:
            raise ValueError(
                f"XYZ template exceeds max_proposals={max_proposals}: {path}")
    return tuple(proposals)


_PCD_KEYS = {
    "VERSION", "FIELDS", "SIZE", "TYPE", "COUNT", "WIDTH", "HEIGHT",
    "VIEWPOINT", "POINTS", "DATA",
}


def _parse_pcd_header(lines: Sequence[str], path: Path) -> tuple[dict, int]:
    header: dict[str, list[str]] = {}
    for line_index, raw in enumerate(lines):
        content = raw.strip()
        if not content or content.startswith("#"):
            continue
        tokens = content.split()
        key = tokens[0].upper()
        if key not in _PCD_KEYS:
            raise ValueError(f"unknown PCD header key {tokens[0]!r} at {path}:{line_index + 1}")
        if key in header:
            raise ValueError(f"duplicate PCD {key} header at {path}:{line_index + 1}")
        if len(tokens) < 2:
            raise ValueError(f"PCD {key} header has no value at {path}:{line_index + 1}")
        header[key] = tokens[1:]
        if key == "DATA":
            if len(header[key]) != 1 or header[key][0].lower() != "ascii":
                mode = " ".join(header[key])
                raise ValueError(
                    f"only DATA ascii PCD templates are supported; got {mode!r}")
            return header, line_index + 1
    raise ValueError(f"PCD template has no DATA ascii header: {path}")


def _header_ints(header: Mapping[str, list[str]], key: str, path: Path) -> list[int]:
    if key not in header:
        raise ValueError(f"PCD template is missing required {key} header: {path}")
    try:
        values = [int(token) for token in header[key]]
    except ValueError as error:
        raise ValueError(f"PCD {key} values must be integers: {path}") from error
    if any(value <= 0 for value in values):
        raise ValueError(f"PCD {key} values must be positive: {path}")
    return values


def _parse_pcd(
    path: Path,
    template_id: str,
    frame: TemplateFrame,
    role: TemplateRole,
    length_scale: float,
    max_proposals: int,
) -> tuple[PoseProposal, ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError(f"PCD header/data must be UTF-8 ASCII text: {path}") from error
    header, data_start = _parse_pcd_header(lines, path)
    required = {"FIELDS", "SIZE", "TYPE", "WIDTH", "HEIGHT", "POINTS", "DATA"}
    missing = sorted(required - set(header))
    if missing:
        raise ValueError(f"PCD template is missing headers {missing}: {path}")

    fields = [token.lower() for token in header["FIELDS"]]
    if len(fields) != len(set(fields)):
        raise ValueError(f"PCD FIELDS names must be unique: {path}")
    if not {"x", "y", "z"}.issubset(fields):
        raise ValueError(f"PCD FIELDS must include x y z: {path}")
    normal_names = {"normal_x", "normal_y", "normal_z"}
    present_normals = normal_names.intersection(fields)
    if present_normals and present_normals != normal_names:
        raise ValueError(
            f"PCD normals must include normal_x normal_y normal_z together: {path}")

    sizes = _header_ints(header, "SIZE", path)
    counts = (_header_ints(header, "COUNT", path)
              if "COUNT" in header else [1] * len(fields))
    types = [token.upper() for token in header["TYPE"]]
    if not (len(fields) == len(sizes) == len(types) == len(counts)):
        raise ValueError("PCD FIELDS/SIZE/TYPE/COUNT lengths must agree")
    if any(item not in {"F", "I", "U"} for item in types):
        raise ValueError(f"PCD TYPE values must be F, I, or U: {path}")
    selected = {"x", "y", "z"} | present_normals
    for name in selected:
        index = fields.index(name)
        if counts[index] != 1:
            raise ValueError(f"PCD field {name} must have COUNT 1: {path}")
        if types[index] != "F":
            raise ValueError(f"PCD field {name} must have floating TYPE F: {path}")

    width = _header_ints(header, "WIDTH", path)
    height = _header_ints(header, "HEIGHT", path)
    points = _header_ints(header, "POINTS", path)
    if len(width) != 1 or len(height) != 1 or len(points) != 1:
        raise ValueError("PCD WIDTH, HEIGHT, and POINTS must each have one value")
    if width[0] * height[0] != points[0]:
        raise ValueError("PCD WIDTH*HEIGHT must equal POINTS")
    if points[0] > max_proposals:
        raise ValueError(
            f"PCD template exceeds max_proposals={max_proposals}: {path}")

    offsets = {}
    offset = 0
    for name, count in zip(fields, counts):
        offsets[name] = offset
        offset += count
    proposals = []
    for line_index in range(data_start, len(lines)):
        content = lines[line_index].strip()
        if not content or content.startswith("#"):
            continue
        tokens = content.split()
        if len(tokens) != offset:
            raise ValueError(
                f"PCD row at {path}:{line_index + 1} has {len(tokens)} columns; "
                f"expected {offset}")
        try:
            values = np.asarray([float(token) for token in tokens], dtype=float)
        except ValueError as error:
            raise ValueError(f"PCD row is non-numeric at {path}:{line_index + 1}") from error
        xyz = np.array([values[offsets[name]] for name in ("x", "y", "z")])
        if not np.all(np.isfinite(xyz)):
            raise ValueError(f"PCD XYZ must be finite at {path}:{line_index + 1}")
        normal = None
        if present_normals:
            normal = np.array([
                values[offsets[name]]
                for name in ("normal_x", "normal_y", "normal_z")])
            if not np.all(np.isfinite(normal)):
                raise ValueError(
                    f"PCD normal must be finite at {path}:{line_index + 1}")
        proposals.append(_proposal(
            template_id, len(proposals), line_index + 1, xyz,
            frame=frame, role=role, length_scale=length_scale,
            angle_scale=1.0, normal_hint=normal))
    if len(proposals) != points[0]:
        raise ValueError(
            f"PCD POINTS declares {points[0]} but {len(proposals)} rows were read: {path}")
    return tuple(proposals)


def load_pose_template(
    path: str | Path,
    *,
    frame: TemplateFrame | str,
    role: TemplateRole | str,
    xyz_units: str = "m",
    rpy_units: str = "deg",
    source_format: str | None = None,
    template_id: str | None = None,
    max_proposals: int = 10000,
) -> PoseTemplate:
    """Parse an XYZ or ASCII-PCD proposal template with explicit semantics."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"pose template does not exist: {source}")
    parsed_frame = _enum_value(frame, TemplateFrame, "frame")
    parsed_role = _enum_value(role, TemplateRole, "role")
    length_scale = _length_scale(xyz_units)
    angle_scale = _angle_scale(rpy_units)
    limit = _positive_integer(max_proposals, "max_proposals")
    identifier = source.stem if template_id is None else str(template_id).strip()
    if not identifier:
        raise ValueError("template_id must be non-empty")

    if source_format is None:
        parsed_format = "ascii_pcd" if source.suffix.lower() == ".pcd" else "xyz"
    else:
        parsed_format = str(source_format).strip().lower()
        aliases = {"pcd": "ascii_pcd", "ascii-pcd": "ascii_pcd", "txt": "xyz"}
        parsed_format = aliases.get(parsed_format, parsed_format)
    if parsed_format == "xyz":
        proposals = _parse_xyz(
            source, identifier, parsed_frame, parsed_role,
            length_scale, angle_scale, limit)
    elif parsed_format == "ascii_pcd":
        proposals = _parse_pcd(
            source, identifier, parsed_frame, parsed_role,
            length_scale, limit)
    else:
        raise ValueError("source_format must be xyz or ascii_pcd")
    if not proposals:
        raise ValueError(f"pose template contains no proposals: {source}")
    return PoseTemplate(
        template_id=identifier,
        source_path=str(source),
        source_format=parsed_format,
        frame=parsed_frame,
        role=parsed_role,
        proposals=proposals,
        source_fingerprint=fingerprint_file(source),
        xyz_units=str(xyz_units),
        rpy_units=str(rpy_units),
    )


def load_declared_pose_templates(
    specifications: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None,
    *,
    resolve_path: Callable[[str], str] = str,
    max_proposals_per_template: int = 10000,
) -> tuple[PoseTemplate, ...]:
    """Load manifest-like template declarations without owning a manifest schema."""
    if specifications is None:
        return ()
    if isinstance(specifications, Mapping):
        entries = (specifications,)
    elif isinstance(specifications, Sequence) and not isinstance(
            specifications, (str, bytes, bytearray)):
        entries = tuple(specifications)
    else:
        raise TypeError("proposal_templates must be a mapping or sequence of mappings")
    templates = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise TypeError(f"proposal_templates[{index}] must be a mapping")
        missing = [name for name in ("path", "frame", "role") if name not in entry]
        if missing:
            raise ValueError(
                f"proposal_templates[{index}] is missing required fields {missing}")
        units = entry.get("xyz_units", entry.get("units", "m"))
        templates.append(load_pose_template(
            resolve_path(str(entry["path"])),
            frame=entry["frame"],
            role=entry["role"],
            xyz_units=units,
            rpy_units=entry.get("rpy_units", "deg"),
            source_format=entry.get("format"),
            template_id=entry.get("name", f"template_{index:03d}"),
            max_proposals=max_proposals_per_template,
        ))
    return tuple(templates)


@dataclass(frozen=True)
class GraspTemplateMatch:
    """Association between one hint and one CAD-contact-validated grasp."""

    proposal_id: str
    grasp_name: str
    position_error_m: float
    rotation_error_rad: float | None
    normal_error_rad: float | None


def rank_contact_validated_grasps(
    candidates: Mapping[str, GraspCandidate],
    templates: Sequence[PoseTemplate],
    *,
    part_scale: float,
    position_tolerance_fraction: float = 0.12,
    rotation_tolerance_deg: float = 20.0,
    normal_tolerance_deg: float = 35.0,
    max_matches_per_proposal: int = 2,
) -> tuple[list[str], tuple[GraspTemplateMatch, ...]]:
    """Prioritize, but never create, CAD/contact-validated grasp candidates.

    The returned names are a permutation of ``candidates``: unmatched valid
    grasps remain as a complete fallback.  Consequently a template cannot
    inject a raw transform into collision or execution planning.
    """
    if not isinstance(candidates, Mapping):
        raise TypeError("candidates must be a mapping")
    names = list(candidates)
    if any(not isinstance(item, GraspCandidate) for item in candidates.values()):
        raise TypeError("candidates must contain GraspCandidate values")
    scale = float(part_scale)
    position_fraction = float(position_tolerance_fraction)
    rotation_tolerance = np.radians(float(rotation_tolerance_deg))
    normal_tolerance = np.radians(float(normal_tolerance_deg))
    match_limit = _positive_integer(max_matches_per_proposal,
                                    "max_matches_per_proposal")
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("part_scale must be positive and finite")
    if not np.isfinite(position_fraction) or position_fraction <= 0.0:
        raise ValueError("position_tolerance_fraction must be positive and finite")
    if not np.isfinite(rotation_tolerance) or not 0.0 < rotation_tolerance <= np.pi:
        raise ValueError("rotation_tolerance_deg must be in (0, 180]")
    if not np.isfinite(normal_tolerance) or not 0.0 < normal_tolerance <= np.pi:
        raise ValueError("normal_tolerance_deg must be in (0, 180]")
    position_tolerance = position_fraction * scale

    prioritized = []
    matches = []
    for template in templates:
        if not isinstance(template, PoseTemplate):
            raise TypeError("templates must contain PoseTemplate values")
        if template.frame != TemplateFrame.PART or template.role != TemplateRole.GRASP_TCP:
            raise ValueError(
                "receiver grasp planning supports only frame='part', "
                "role='grasp_tcp' proposal templates; got "
                f"frame={template.frame.value!r}, role={template.role.value!r}")
        for proposal in template.proposals:
            target = proposal.T_frame_target
            ranked = []
            for base_index, name in enumerate(names):
                candidate = candidates[name]
                position_error = float(np.linalg.norm(
                    candidate.T_P_E[:3, 3] - proposal.position_m))
                if position_error > position_tolerance:
                    continue
                rotation_error = None
                if target is not None:
                    rotation_error = so3_geodesic(
                        candidate.T_P_E[:3, :3], target[:3, :3])
                    if rotation_error > rotation_tolerance:
                        continue
                normal_error = None
                if proposal.normal_hint is not None:
                    cosine = float(np.clip(
                        candidate.approach_direction @ proposal.normal_hint,
                        -1.0, 1.0))
                    normal_error = float(np.arccos(cosine))
                    if normal_error > normal_tolerance:
                        continue
                objective = position_error / position_tolerance
                if rotation_error is not None:
                    objective += rotation_error / rotation_tolerance
                if normal_error is not None:
                    objective += normal_error / normal_tolerance
                ranked.append((objective, base_index, name, position_error,
                               rotation_error, normal_error))
            for _, _, name, position_error, rotation_error, normal_error in sorted(
                    ranked)[:match_limit]:
                if name not in prioritized:
                    prioritized.append(name)
                matches.append(GraspTemplateMatch(
                    proposal.proposal_id, name, position_error,
                    rotation_error, normal_error))

    ordered = prioritized + [name for name in names if name not in prioritized]
    if len(ordered) != len(names) or set(ordered) != set(names):
        raise AssertionError("template ranking must be a permutation of CAD grasps")
    return ordered, tuple(matches)


__all__ = [
    "GraspTemplateMatch",
    "PROPOSAL_ONLY_USAGE",
    "PoseProposal",
    "PoseTemplate",
    "RPY_CONVENTION",
    "TemplateFrame",
    "TemplateRole",
    "load_declared_pose_templates",
    "load_pose_template",
    "rank_contact_validated_grasps",
]
