"""Vendor-neutral, YAML-driven workcell URDF generator.

The canonical URDF contains geometry and a connected kinematic/frame tree.
Camera intrinsics and operating envelopes are emitted as companion YAML files,
because URDF has no portable representation for those calibration values.
"""
from __future__ import annotations

import copy
import math
import os
import re
import tempfile
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

from ..core.paths import REPOSITORY_ROOT


ROOT = Path(REPOSITORY_ROOT).resolve()
NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
UNRESOLVED_EXPRESSION_RE = re.compile(r"\$\{|\$\(|\$\$")
UNIT_SCALE = {"m": 1.0, "cm": 0.01, "mm": 0.001}
SUPPORTED_JOINT_TYPES = {
    "fixed",
    "revolute",
    "continuous",
    "prismatic",
    "planar",
    "floating",
}
DISTORTION_LENGTHS = {
    "plumb_bob": 5,
    "rational_polynomial": 8,
    "equidistant": 4,
}


@dataclass(frozen=True)
class ImportResult:
    instance: str
    source_path: Path
    source_root: str
    output_root: str
    links: dict[str, str]
    joints: dict[str, str]
    materials: dict[str, str]
    transmissions: dict[str, str]
    joint_types: dict[str, str]
    pruned_links: tuple[str, ...]


@dataclass(frozen=True)
class GenerationResult:
    urdf_path: Path
    camera_info_paths: tuple[Path, ...]
    report_path: Path
    report: dict[str, Any]
    wrote_files: bool


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def _validate_name(value: Any, label: str) -> str:
    if not isinstance(value, str) or not NAME_RE.fullmatch(value):
        raise ValueError(
            f"{label} must match {NAME_RE.pattern!r}; got {value!r}"
        )
    return value


def _reject_unknown_keys(
    value: dict[str, Any], allowed: set[str], label: str
) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"{label} contains unknown keys: {sorted(unknown)}")


def _numbers(values: Iterable[float]) -> str:
    def clean(value: float) -> str:
        result = f"{float(value):.12g}"
        return "0" if result in ("-0", "-0.0") else result

    return " ".join(clean(value) for value in values)


def _rpy_matrix(rpy: Iterable[float]) -> np.ndarray:
    roll, pitch, yaw = (float(value) for value in rpy)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=float,
    )


def _quaternion_matrix(quaternion_xyzw: Iterable[float]) -> np.ndarray:
    quaternion = np.asarray(tuple(quaternion_xyzw), dtype=float)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError("quaternion_xyzw must contain four finite values")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise ValueError("quaternion_xyzw must be nonzero")
    x, y, z, w = quaternion / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _matrix_rpy(rotation: np.ndarray) -> np.ndarray:
    pitch = math.atan2(
        -float(rotation[2, 0]),
        math.hypot(float(rotation[0, 0]), float(rotation[1, 0])),
    )
    if abs(math.cos(pitch)) > 1e-9:
        roll = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
        yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    else:
        roll = math.atan2(-float(rotation[1, 2]), float(rotation[1, 1]))
        yaw = 0.0
    return np.asarray([roll, pitch, yaw], dtype=float)


def _validate_transform(transform: np.ndarray, label: str) -> np.ndarray:
    transform = np.asarray(transform, dtype=float)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError(f"{label} must be a finite 4x4 matrix")
    if not np.allclose(
        transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9, rtol=0.0
    ):
        raise ValueError(f"{label} must end in [0, 0, 0, 1]")
    rotation = transform[:3, :3]
    tolerance = 1e-5
    if not np.allclose(
        rotation.T @ rotation, np.eye(3), atol=tolerance, rtol=0.0
    ):
        raise ValueError(f"{label} rotation must be orthonormal")
    if not np.isclose(
        np.linalg.det(rotation), 1.0, atol=tolerance, rtol=0.0
    ):
        raise ValueError(f"{label} rotation must have determinant +1")
    u, _, vt = np.linalg.svd(rotation)
    projected = u @ vt
    if np.linalg.det(projected) < 0.0:
        u[:, -1] *= -1.0
        projected = u @ vt
    result = transform.copy()
    result[:3, :3] = projected
    return result


def pose_transform(
    value: dict[str, Any] | None,
    label: str,
    *,
    required: bool = True,
) -> np.ndarray:
    """Parse an explicit ``parent_T_child`` pose into a validated SE(3)."""
    if value is None:
        if required:
            raise ValueError(f"{label} is required")
        return np.eye(4)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    if "matrix" in value:
        _reject_unknown_keys(value, {"matrix"}, label)
        return _validate_transform(np.asarray(value["matrix"], dtype=float), label)

    _reject_unknown_keys(
        value,
        {"position_m", "rpy_deg", "rotation_matrix", "quaternion_xyzw"},
        label,
    )

    position = np.asarray(value.get("position_m"), dtype=float)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError(f"{label}.position_m must contain three finite values")
    orientation_keys = [
        key
        for key in ("rpy_deg", "rotation_matrix", "quaternion_xyzw")
        if key in value
    ]
    if len(orientation_keys) != 1:
        raise ValueError(
            f"{label} must define exactly one of rpy_deg, rotation_matrix, "
            "or quaternion_xyzw"
        )
    key = orientation_keys[0]
    if key == "rpy_deg":
        rpy_deg = np.asarray(value[key], dtype=float)
        if rpy_deg.shape != (3,) or not np.all(np.isfinite(rpy_deg)):
            raise ValueError(f"{label}.rpy_deg must contain three finite values")
        rotation = _rpy_matrix(np.radians(rpy_deg))
    elif key == "rotation_matrix":
        rotation = np.asarray(value[key], dtype=float)
        if rotation.shape != (3, 3):
            raise ValueError(f"{label}.rotation_matrix must be 3x3")
    else:
        rotation = _quaternion_matrix(value[key])
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = position
    return _validate_transform(result, label)


def _origin(parent: ET.Element, transform: np.ndarray) -> ET.Element:
    return ET.SubElement(
        parent,
        "origin",
        {
            "xyz": _numbers(transform[:3, 3]),
            "rpy": _numbers(_matrix_rpy(transform[:3, :3])),
        },
    )


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(dir=path.parent, delete=False)
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _yaml_bytes(value: Any) -> bytes:
    return yaml.safe_dump(value, sort_keys=False, allow_unicode=True).encode("utf-8")


class WorkcellUrdfGenerator:
    """Compile a general workcell manifest into URDF and companion metadata."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        output_override: str | Path | None = None,
        camera_info_dir_override: str | Path | None = None,
        report_override: str | Path | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest = _load_yaml(self.manifest_path)
        _reject_unknown_keys(
            self.manifest,
            {
                "schema_version",
                "name",
                "root_link",
                "output",
                "package_roots",
                "frames",
                "static_bodies",
                "robots",
                "grippers",
                "cameras",
                "attached_frames",
            },
            "manifest",
        )
        if self.manifest.get("schema_version") != 1:
            raise ValueError("workcell generator schema_version must be 1")
        self.workcell_name = _validate_name(
            self.manifest.get("name"), "name"
        )
        self.root_link = _validate_name(
            self.manifest.get("root_link", "world"), "root_link"
        )
        output = self.manifest.get("output")
        if not isinstance(output, dict):
            raise ValueError("output must be a mapping")
        _reject_unknown_keys(
            output,
            {
                "urdf",
                "camera_info_dir",
                "report",
                "mesh_uri_mode",
                "extension_policy",
                "mujoco_fusestatic",
            },
            "output",
        )
        self.output_path = self._resolve_output(
            output_override or output.get("urdf"), "output.urdf"
        )
        self.camera_info_dir = self._resolve_output(
            camera_info_dir_override or output.get("camera_info_dir"),
            "output.camera_info_dir",
            directory=True,
        )
        self.report_path = self._resolve_output(
            report_override or output.get("report"), "output.report"
        )
        if self.output_path == self.report_path:
            raise ValueError("output.urdf and output.report must be different paths")
        if self.camera_info_dir in {self.output_path, self.report_path}:
            raise ValueError(
                "output.camera_info_dir cannot be the URDF or report file path"
            )
        self.mesh_uri_mode = output.get("mesh_uri_mode", "relative")
        if self.mesh_uri_mode not in {"relative", "absolute", "preserve_package"}:
            raise ValueError(
                "output.mesh_uri_mode must be relative, absolute, or preserve_package"
            )
        self.extension_policy = output.get("extension_policy", "drop")
        if self.extension_policy not in {"drop", "reject"}:
            raise ValueError("output.extension_policy must be drop or reject")
        self.mujoco_fusestatic = output.get("mujoco_fusestatic")
        if self.mujoco_fusestatic is not None and not isinstance(
            self.mujoco_fusestatic, bool
        ):
            raise ValueError("output.mujoco_fusestatic must be boolean when set")
        self.package_roots = self._package_roots(self.manifest.get("package_roots", {}))

        self.robot = ET.Element("robot", {"name": self.workcell_name})
        self.robot.append(
            ET.Comment(
                "GENERATED from a workcell manifest; do not hand-edit. "
                "Camera intrinsics and operating envelopes are companion YAML."
            )
        )
        if self.mujoco_fusestatic is not None:
            mujoco_extension = ET.SubElement(self.robot, "mujoco")
            ET.SubElement(
                mujoco_extension,
                "compiler",
                {
                    "fusestatic": (
                        "true" if self.mujoco_fusestatic else "false"
                    )
                },
            )
        ET.SubElement(self.robot, "link", {"name": self.root_link})
        self.known_links: set[str] = {self.root_link}
        self.frames: dict[str, str] = {self.root_link: self.root_link}
        self.static_bodies: dict[str, str] = {}
        self.robots: dict[str, ImportResult] = {}
        self.grippers: dict[str, ImportResult] = {}
        self.camera_links: dict[str, dict[str, str]] = {}
        self.warnings: list[str] = []
        self.camera_info_documents: dict[Path, dict[str, Any]] = {}
        self.report: dict[str, Any] = {
            "schema_version": 1,
            "workcell": self.workcell_name,
            "source_manifest": str(self.manifest_path),
            "root_link": self.root_link,
            "robots": {},
            "grippers": {},
            "cameras": {},
            "static_bodies": {},
            "warnings": self.warnings,
        }

    def _resolve_output(
        self, value: str | Path | None, label: str, *, directory: bool = False
    ) -> Path:
        if value is None:
            raise ValueError(f"{label} is required")
        path = Path(value)
        if path.is_absolute():
            return path.resolve()
        try:
            inside_repository = (
                os.path.commonpath((self.manifest_path, ROOT)) == str(ROOT)
            )
        except ValueError:
            inside_repository = False
        base = ROOT if inside_repository else self.manifest_path.parent
        return (base / path).resolve()

    def _resolve_asset(self, value: str, label: str) -> Path:
        if not isinstance(value, str) or not value:
            raise ValueError(f"{label} must be a non-empty path")
        if UNRESOLVED_EXPRESSION_RE.search(value):
            raise ValueError(f"{label} contains an unresolved expression: {value!r}")
        path = Path(value)
        if path.is_absolute():
            candidate = path
        else:
            candidates = [self.manifest_path.parent / path, ROOT / path]
            candidate = next((item for item in candidates if item.is_file()), candidates[0])
        candidate = candidate.resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"{label} was not found: {candidate}")
        return candidate

    def _package_roots(self, value: Any) -> dict[str, Path]:
        if not isinstance(value, dict):
            raise ValueError("package_roots must be a mapping")
        result: dict[str, Path] = {}
        for name, raw_path in value.items():
            _validate_name(name, f"package_roots key {name!r}")
            path = Path(raw_path)
            if not path.is_absolute():
                path = self.manifest_path.parent / path
            path = path.resolve()
            if not path.is_dir():
                raise FileNotFoundError(f"package root {name!r} was not found: {path}")
            result[name] = path
        return result

    def _asset_uri(
        self, raw: str, source_dir: Path, label: str
    ) -> str:
        if not isinstance(raw, str) or not raw:
            raise ValueError(f"{label} must be a non-empty URI")
        if UNRESOLVED_EXPRESSION_RE.search(raw):
            raise ValueError(f"{label} contains an unresolved expression: {raw!r}")
        original_package = raw.startswith("package://")
        if original_package:
            remainder = raw[len("package://") :]
            package, separator, relative = remainder.partition("/")
            if not separator or package not in self.package_roots:
                raise ValueError(
                    f"{label} uses unresolved package {package!r}; add package_roots.{package}"
                )
            path = self.package_roots[package] / relative
        elif raw.startswith("file://"):
            parsed = urllib.parse.urlparse(raw)
            path = Path(urllib.parse.unquote(parsed.path))
        elif "://" in raw or raw.startswith("model://"):
            raise ValueError(f"{label} uses unsupported URI {raw!r}")
        else:
            path = Path(raw)
            if not path.is_absolute():
                path = source_dir / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{label} was not found: {path}")
        if original_package and self.mesh_uri_mode == "preserve_package":
            return raw
        if self.mesh_uri_mode == "absolute":
            return path.as_posix()
        return Path(os.path.relpath(path, self.output_path.parent)).as_posix()

    @staticmethod
    def _prefixed(instance: str, source_name: str) -> str:
        if not source_name or any(character.isspace() for character in source_name):
            raise ValueError(f"invalid URDF source name {source_name!r}")
        return f"{instance}__{source_name}"

    def _resolve_parent(self, reference: Any, label: str) -> str:
        if isinstance(reference, str):
            if reference not in self.known_links:
                raise KeyError(f"{label} references unknown link {reference!r}")
            return reference
        if not isinstance(reference, dict) or len(reference) == 0:
            raise ValueError(f"{label} must be a link string or typed mapping")
        kinds = [key for key in ("frame", "body", "robot", "gripper", "camera") if key in reference]
        if len(kinds) != 1:
            raise ValueError(f"{label} must select exactly one parent kind")
        kind = kinds[0]
        allowed = {kind, "link"} if kind in {"robot", "gripper", "camera"} else {kind}
        _reject_unknown_keys(reference, allowed, label)
        name = reference[kind]
        if not isinstance(name, str) or not name:
            raise ValueError(f"{label}.{kind} must be a non-empty name")
        if kind == "frame":
            if name not in self.frames:
                raise KeyError(f"{label} references unknown frame {name!r}")
            return self.frames[name]
        if kind == "body":
            if name not in self.static_bodies:
                raise KeyError(f"{label} references unknown body {name!r}")
            return self.static_bodies[name]
        if kind == "camera":
            if name not in self.camera_links:
                raise KeyError(f"{label} references unknown camera {name!r}")
            frame = reference.get("link", "link")
            if frame not in self.camera_links[name]:
                raise KeyError(f"{label} camera {name!r} has no frame {frame!r}")
            return self.camera_links[name][frame]
        imports = self.robots if kind == "robot" else self.grippers
        if name not in imports:
            raise KeyError(f"{label} references unknown {kind} {name!r}")
        source_link = reference.get("link")
        if not source_link:
            raise ValueError(f"{label} {kind} parent requires link")
        try:
            return imports[name].links[source_link]
        except KeyError as error:
            raise KeyError(
                f"{label} {kind} {name!r} has no imported link {source_link!r}"
            ) from error

    def _add_fixed_link(
        self,
        child: str,
        parent: str,
        transform: np.ndarray,
        *,
        link: ET.Element | None = None,
    ) -> ET.Element:
        if child in self.known_links:
            raise ValueError(f"duplicate generated link {child!r}")
        if parent not in self.known_links:
            raise ValueError(f"parent link {parent!r} does not exist")
        link = link if link is not None else ET.Element("link", {"name": child})
        link.set("name", child)
        self.robot.append(link)
        joint = ET.SubElement(
            self.robot,
            "joint",
            {"name": f"fixed__{child}", "type": "fixed"},
        )
        ET.SubElement(joint, "parent", {"link": parent})
        ET.SubElement(joint, "child", {"link": child})
        _origin(joint, transform)
        self.known_links.add(child)
        return link

    def _add_inertial(self, link: ET.Element, value: dict[str, Any], label: str) -> None:
        if not isinstance(value, dict):
            raise ValueError(f"{label} must be a mapping")
        _reject_unknown_keys(value, {"mass_kg", "inertia", "origin"}, label)
        mass = float(value["mass_kg"])
        if not math.isfinite(mass) or mass <= 0.0:
            raise ValueError(f"{label}.mass_kg must be positive")
        inertia_value = value.get("inertia")
        if not isinstance(inertia_value, dict):
            raise ValueError(f"{label}.inertia must be a mapping")
        required = ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")
        _reject_unknown_keys(inertia_value, set(required), f"{label}.inertia")
        inertia_numbers = {key: float(inertia_value[key]) for key in required}
        if not all(math.isfinite(number) for number in inertia_numbers.values()):
            raise ValueError(f"{label}.inertia values must be finite")
        inertial = ET.SubElement(link, "inertial")
        _origin(
            inertial,
            pose_transform(value.get("origin"), f"{label}.origin", required=False),
        )
        ET.SubElement(inertial, "mass", {"value": _numbers((mass,))})
        ET.SubElement(
            inertial,
            "inertia",
            {key: _numbers((inertia_numbers[key],)) for key in required},
        )

    def _mesh_geometry(
        self, geometry: ET.Element, value: Any, label: str
    ) -> None:
        if isinstance(value, str):
            value = {"path": value, "units": "m"}
        if not isinstance(value, dict):
            raise ValueError(f"{label}.mesh must be a mapping")
        _reject_unknown_keys(value, {"path", "units", "scale"}, f"{label}.mesh")
        path = self._resolve_asset(value.get("path"), f"{label}.mesh.path")
        units = value.get("units")
        if units not in UNIT_SCALE:
            raise ValueError(f"{label}.mesh.units must be one of {sorted(UNIT_SCALE)}")
        raw_scale = value.get("scale", [1.0, 1.0, 1.0])
        scale = np.asarray(raw_scale, dtype=float)
        if scale.shape != (3,) or not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
            raise ValueError(f"{label}.mesh.scale must contain three positive values")
        scale *= UNIT_SCALE[units]
        filename = (
            path.as_posix()
            if self.mesh_uri_mode == "absolute"
            else Path(os.path.relpath(path, self.output_path.parent)).as_posix()
        )
        ET.SubElement(
            geometry,
            "mesh",
            {"filename": filename, "scale": _numbers(scale)},
        )

    def _geometry(self, parent: ET.Element, value: Any, label: str) -> None:
        if not isinstance(value, dict):
            raise ValueError(f"{label} must be a geometry mapping")
        _reject_unknown_keys(
            value, {"mesh", "box", "cylinder", "sphere"}, label
        )
        kinds = [key for key in ("mesh", "box", "cylinder", "sphere") if key in value]
        if len(kinds) != 1:
            raise ValueError(
                f"{label} must define exactly one of mesh, box, cylinder, or sphere"
            )
        geometry = ET.SubElement(parent, "geometry")
        kind = kinds[0]
        raw = value[kind]
        if kind == "mesh":
            self._mesh_geometry(geometry, raw, label)
        elif kind == "box":
            if isinstance(raw, dict):
                _reject_unknown_keys(raw, {"size_m"}, f"{label}.box")
            size = raw.get("size_m") if isinstance(raw, dict) else raw
            size = np.asarray(size, dtype=float)
            if size.shape != (3,) or not np.all(np.isfinite(size)) or np.any(size <= 0.0):
                raise ValueError(f"{label}.box size must contain three positive metres")
            ET.SubElement(geometry, "box", {"size": _numbers(size)})
        elif kind == "cylinder":
            if not isinstance(raw, dict):
                raise ValueError(f"{label}.cylinder must be a mapping")
            _reject_unknown_keys(
                raw, {"radius_m", "length_m"}, f"{label}.cylinder"
            )
            radius, length = float(raw["radius_m"]), float(raw["length_m"])
            if not all(math.isfinite(value) and value > 0.0 for value in (radius, length)):
                raise ValueError(f"{label}.cylinder dimensions must be positive")
            ET.SubElement(
                geometry,
                "cylinder",
                {"radius": _numbers((radius,)), "length": _numbers((length,))},
            )
        else:
            if isinstance(raw, dict):
                _reject_unknown_keys(raw, {"radius_m"}, f"{label}.sphere")
            radius = float(raw.get("radius_m") if isinstance(raw, dict) else raw)
            if not math.isfinite(radius) or radius <= 0.0:
                raise ValueError(f"{label}.sphere radius must be positive")
            ET.SubElement(geometry, "sphere", {"radius": _numbers((radius,))})

    def _add_shape(
        self,
        link: ET.Element,
        tag: str,
        value: dict[str, Any],
        label: str,
        index: int,
    ) -> None:
        if not isinstance(value, dict) or "geometry" not in value:
            raise ValueError(f"{label} must contain geometry")
        allowed = {"name", "origin", "geometry"}
        if tag == "visual":
            allowed.add("material")
        _reject_unknown_keys(value, allowed, label)
        source_name = value.get("name", f"{tag}_{index:02d}")
        if not isinstance(source_name, str) or not source_name or any(
            character.isspace() for character in source_name
        ):
            raise ValueError(f"{label}.name must be a non-empty name without spaces")
        link_name = str(link.get("name"))
        element = ET.SubElement(
            link,
            tag,
            {"name": f"{link_name}__{tag}__{source_name}__{index:02d}"},
        )
        _origin(
            element,
            pose_transform(value.get("origin"), f"{label}.origin", required=False),
        )
        self._geometry(element, value["geometry"], f"{label}.geometry")
        if tag == "visual" and "material" in value:
            material_value = value["material"]
            if not isinstance(material_value, dict):
                raise ValueError(f"{label}.material must be a mapping")
            _reject_unknown_keys(
                material_value, {"name", "rgba"}, f"{label}.material"
            )
            material_source_name = material_value.get(
                "name", f"material_{index:02d}"
            )
            if (
                not isinstance(material_source_name, str)
                or not material_source_name
                or any(
                    character.isspace() for character in material_source_name
                )
            ):
                raise ValueError(
                    f"{label}.material.name must be a non-empty name without spaces"
                )
            material = ET.SubElement(
                element,
                "material",
                {
                    "name": (
                        f"{link_name}__material__{material_source_name}__"
                        f"{index:02d}"
                    )
                },
            )
            rgba = np.asarray(material_value["rgba"], dtype=float)
            if rgba.shape != (4,) or not np.all(np.isfinite(rgba)):
                raise ValueError(f"{label}.material.rgba must have four values")
            ET.SubElement(material, "color", {"rgba": _numbers(rgba)})

    def _expand_box_set(
        self,
        link: ET.Element,
        value: dict[str, Any],
        label: str,
        start_index: int,
        *,
        outer_transform: np.ndarray | None = None,
        source_name: str = "box_set",
    ) -> int:
        if not isinstance(value, dict):
            raise ValueError(f"{label} must be a mapping")
        _reject_unknown_keys(value, {"path", "units", "groups", "origin"}, label)
        path = self._resolve_asset(value.get("path"), f"{label}.path")
        data = _load_yaml(path)
        units = value.get("units", data.get("units"))
        if units == "from_file":
            units = data.get("units")
        if units not in UNIT_SCALE:
            raise ValueError(f"{label}.units must resolve to one of {sorted(UNIT_SCALE)}")
        groups = value.get("groups")
        if groups is None:
            groups = [
                key
                for key, entries in data.items()
                if key != "units" and isinstance(entries, list)
            ]
        if not isinstance(groups, list) or not groups:
            raise ValueError(f"{label}.groups must be a non-empty list")
        scale = UNIT_SCALE[units]
        base_transform = (
            np.eye(4) if outer_transform is None else outer_transform
        ) @ pose_transform(value.get("origin"), f"{label}.origin", required=False)
        index = start_index
        for group in groups:
            entries = data.get(group)
            if not isinstance(entries, list):
                raise ValueError(f"{label} group {group!r} is not a list")
            for item in entries:
                center = np.asarray(item["center"], dtype=float) * scale
                half_extents = np.asarray(item["half_extents"], dtype=float) * scale
                if (
                    center.shape != (3,)
                    or half_extents.shape != (3,)
                    or not np.all(np.isfinite(center))
                    or not np.all(np.isfinite(half_extents))
                    or np.any(half_extents <= 0.0)
                ):
                    raise ValueError(f"{label} contains an invalid box in group {group!r}")
                local = np.eye(4)
                local[:3, 3] = center
                collision = ET.SubElement(
                    link,
                    "collision",
                    {
                        "name": (
                            f"{link.get('name')}__collision__{source_name}_"
                            f"{index:03d}"
                        )
                    },
                )
                _origin(collision, base_transform @ local)
                geometry = ET.SubElement(collision, "geometry")
                ET.SubElement(
                    geometry,
                    "box",
                    {"size": _numbers(2.0 * half_extents)},
                )
                index += 1
        return index

    @staticmethod
    def _items(value: Any, label: str) -> list[dict[str, Any]]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError(f"{label} must be a list")
        if not all(isinstance(item, dict) for item in value):
            raise ValueError(f"every entry in {label} must be a mapping")
        return value

    def _populate_link(
        self,
        link: ET.Element,
        value: dict[str, Any],
        label: str,
    ) -> dict[str, int]:
        """Add inertial, visual, and collision content to a generated link."""
        if "inertial" in value:
            self._add_inertial(link, value["inertial"], f"{label}.inertial")
        visuals = self._items(value.get("visuals"), f"{label}.visuals")
        collisions = self._items(value.get("collisions"), f"{label}.collisions")
        for index, visual in enumerate(visuals):
            self._add_shape(
                link, "visual", visual, f"{label}.visuals[{index}]", index
            )
        collision_index = 0
        for index, collision in enumerate(collisions):
            geometry = collision.get("geometry")
            if isinstance(geometry, dict) and "box_set" in geometry:
                _reject_unknown_keys(
                    collision,
                    {"name", "origin", "geometry"},
                    f"{label}.collisions[{index}]",
                )
                if len(geometry) != 1:
                    raise ValueError(
                        f"{label}.collisions[{index}].geometry.box_set cannot "
                        "be combined with another geometry"
                    )
                source_name = collision.get("name", "box_set")
                if not isinstance(source_name, str) or not source_name or any(
                    character.isspace() for character in source_name
                ):
                    raise ValueError(
                        f"{label}.collisions[{index}].name must be a name "
                        "without spaces"
                    )
                collision_index = self._expand_box_set(
                    link,
                    geometry["box_set"],
                    f"{label}.collisions[{index}].geometry.box_set",
                    collision_index,
                    outer_transform=pose_transform(
                        collision.get("origin"),
                        f"{label}.collisions[{index}].origin",
                        required=False,
                    ),
                    source_name=source_name,
                )
            else:
                self._add_shape(
                    link,
                    "collision",
                    collision,
                    f"{label}.collisions[{index}]",
                    collision_index,
                )
                collision_index += 1
        return {"visuals": len(visuals), "collisions": collision_index}

    def _add_frames(self, key: str = "frames") -> None:
        pending = list(enumerate(self._items(self.manifest.get(key), key)))
        declared: set[str] = set()
        for index, item in pending:
            name = _validate_name(item.get("name"), f"{key}[{index}].name")
            if name in self.frames or name in declared:
                raise ValueError(f"duplicate frame name {name!r}")
            declared.add(name)

        while pending:
            progress = False
            deferred: list[tuple[int, dict[str, Any]]] = []
            failures: list[str] = []
            for index, item in pending:
                label = f"{key}[{index}]"
                _reject_unknown_keys(
                    item, {"name", "parent", "parent_T_child"}, label
                )
                name = item["name"]
                try:
                    parent = self._resolve_parent(item.get("parent"), f"{label}.parent")
                except KeyError as error:
                    deferred.append((index, item))
                    failures.append(str(error))
                    continue
                link_name = f"frame__{name}"
                transform = pose_transform(
                    item.get("parent_T_child"), f"{label}.parent_T_child"
                )
                self._add_fixed_link(link_name, parent, transform)
                self.frames[name] = link_name
                progress = True
            if not progress:
                detail = "; ".join(failures)
                raise ValueError(
                    f"could not resolve {key} parents (cycle or missing parent): {detail}"
                )
            pending = deferred

    def _add_static_bodies(self) -> None:
        entries = self._items(self.manifest.get("static_bodies"), "static_bodies")
        for index, item in enumerate(entries):
            label = f"static_bodies[{index}]"
            _reject_unknown_keys(
                item,
                {
                    "name",
                    "parent",
                    "parent_T_body",
                    "visuals",
                    "collisions",
                    "inertial",
                },
                label,
            )
            name = _validate_name(item.get("name"), f"{label}.name")
            if name in self.static_bodies:
                raise ValueError(f"duplicate static body name {name!r}")
            parent = self._resolve_parent(item.get("parent"), f"{label}.parent")
            transform = pose_transform(
                item.get("parent_T_body"), f"{label}.parent_T_body"
            )
            link_name = f"body__{name}"
            link = ET.Element("link", {"name": link_name})
            counts = self._populate_link(link, item, label)
            self._add_fixed_link(link_name, parent, transform, link=link)
            self.static_bodies[name] = link_name
            self.report["static_bodies"][name] = {
                "link": link_name,
                "parent_link": parent,
                **counts,
            }

    def _warn_or_reject_extension(self, source: Path, tag: str) -> None:
        message = f"dropped unsupported <{tag}> extension from {source}"
        if self.extension_policy == "reject":
            raise ValueError(message.replace("dropped", "found"))
        self.warnings.append(message)

    def _filter_children(
        self,
        element: ET.Element,
        allowed: set[str],
        source: Path,
        context: str,
    ) -> None:
        for child in list(element):
            if child.tag not in allowed:
                self._warn_or_reject_extension(
                    source, f"{context}/{child.tag}"
                )
                element.remove(child)

    @staticmethod
    def _unique_named(
        elements: Iterable[ET.Element], label: str
    ) -> dict[str, ET.Element]:
        result: dict[str, ET.Element] = {}
        for element in elements:
            name = element.get("name")
            if not name or any(character.isspace() for character in name):
                raise ValueError(f"{label} has an invalid or missing name {name!r}")
            if name in result:
                raise ValueError(f"{label} contains duplicate name {name!r}")
            result[name] = element
        return result

    def _parse_source_urdf(
        self, source_path: Path
    ) -> tuple[
        ET.Element,
        dict[str, ET.Element],
        dict[str, ET.Element],
        str,
        dict[str, list[tuple[str, str]]],
    ]:
        if source_path.suffix.lower() == ".xacro":
            raise ValueError(
                f"{source_path} is Xacro; expand it to a concrete URDF first"
            )
        text = source_path.read_text(encoding="utf-8")
        if UNRESOLVED_EXPRESSION_RE.search(text):
            raise ValueError(f"{source_path} contains unresolved Xacro expressions")
        try:
            source_root = ET.fromstring(text)
        except ET.ParseError as error:
            raise ValueError(f"invalid URDF XML in {source_path}: {error}") from error
        if source_root.tag != "robot":
            raise ValueError(f"{source_path} root element must be <robot>")

        allowed = {"material", "link", "joint", "transmission"}
        for child in source_root:
            if child.tag not in allowed:
                self._warn_or_reject_extension(source_path, child.tag)

        links = self._unique_named(source_root.findall("./link"), f"{source_path} links")
        joints = self._unique_named(source_root.findall("./joint"), f"{source_path} joints")
        self._unique_named(
            source_root.findall("./material"), f"{source_path} root materials"
        )
        self._unique_named(
            source_root.findall("./transmission"),
            f"{source_path} transmissions",
        )
        if not links:
            raise ValueError(f"{source_path} must contain at least one link")

        children: set[str] = set()
        adjacency: dict[str, list[tuple[str, str]]] = {name: [] for name in links}
        for name, joint in joints.items():
            joint_type = joint.get("type")
            if joint_type not in SUPPORTED_JOINT_TYPES:
                raise ValueError(
                    f"{source_path} joint {name!r} has unsupported type {joint_type!r}"
                )
            parents = joint.findall("./parent")
            child_nodes = joint.findall("./child")
            if len(parents) != 1 or len(child_nodes) != 1:
                raise ValueError(
                    f"{source_path} joint {name!r} needs exactly one parent and child"
                )
            parent_name = parents[0].get("link")
            child_name = child_nodes[0].get("link")
            if parent_name not in links or child_name not in links:
                raise ValueError(
                    f"{source_path} joint {name!r} references an unknown link"
                )
            if child_name in children:
                raise ValueError(
                    f"{source_path} link {child_name!r} has more than one parent"
                )
            children.add(child_name)
            adjacency[parent_name].append((name, child_name))

        roots = set(links) - children
        if len(roots) != 1:
            raise ValueError(
                f"{source_path} must be a single-root tree; detected roots {sorted(roots)}"
            )
        detected_root = next(iter(roots))
        visited: set[str] = set()
        active: set[str] = set()

        def visit(link_name: str) -> None:
            if link_name in active:
                raise ValueError(f"{source_path} contains a kinematic cycle")
            if link_name in visited:
                return
            active.add(link_name)
            for _, child_name in adjacency[link_name]:
                visit(child_name)
            active.remove(link_name)
            visited.add(link_name)

        visit(detected_root)
        if visited != set(links):
            missing = sorted(set(links) - visited)
            raise ValueError(
                f"{source_path} contains disconnected or cyclic links: {missing}"
            )
        return source_root, links, joints, detected_root, adjacency

    @staticmethod
    def _pruned_links(
        links: dict[str, ET.Element],
        joints: dict[str, ET.Element],
        adjacency: dict[str, list[tuple[str, str]]],
        prune_joints: list[str],
        exclude_subtrees: list[str],
        label: str,
    ) -> set[str]:
        removed: set[str] = set()
        starts: list[str] = []
        for joint_name in prune_joints:
            if joint_name not in joints:
                raise ValueError(f"{label} references unknown prune joint {joint_name!r}")
            child = joints[joint_name].find("./child")
            assert child is not None
            starts.append(str(child.get("link")))
        for link_name in exclude_subtrees:
            if link_name not in links:
                raise ValueError(
                    f"{label} references unknown excluded link {link_name!r}"
                )
            starts.append(link_name)
        stack = list(starts)
        while stack:
            link_name = stack.pop()
            if link_name in removed:
                continue
            removed.add(link_name)
            stack.extend(child for _, child in adjacency[link_name])
        return removed

    def _rewrite_imported_link(
        self,
        link: ET.Element,
        instance: str,
        link_map: dict[str, str],
        material_map: dict[str, str],
        source_path: Path,
        label: str,
    ) -> ET.Element:
        result = copy.deepcopy(link)
        source_name = str(link.get("name"))
        result.set("name", link_map[source_name])
        allowed_children = {"inertial", "visual", "collision"}
        for child in list(result):
            if child.tag not in allowed_children:
                self._warn_or_reject_extension(source_path, f"link/{child.tag}")
                result.remove(child)
        for inertial in result.findall("./inertial"):
            self._filter_children(
                inertial,
                {"origin", "mass", "inertia"},
                source_path,
                "link/inertial",
            )
        for element in result.findall("./visual") + result.findall("./collision"):
            standard = {"origin", "geometry"}
            if element.tag == "visual":
                standard.add("material")
            self._filter_children(
                element, standard, source_path, f"link/{element.tag}"
            )
            geometries = element.findall("./geometry")
            if len(geometries) != 1:
                raise ValueError(
                    f"{label} {element.tag} must contain exactly one geometry"
                )
            geometry = geometries[0]
            self._filter_children(
                geometry,
                {"box", "cylinder", "sphere", "mesh"},
                source_path,
                f"link/{element.tag}/geometry",
            )
            if len(list(geometry)) != 1:
                raise ValueError(
                    f"{label} {element.tag} geometry must contain exactly one "
                    "supported shape"
                )
            material = element.find("./material")
            if material is not None:
                self._filter_children(
                    material,
                    {"color", "texture"},
                    source_path,
                    "link/visual/material",
                )
        for shape_index, element in enumerate(
            result.findall(".//visual") + result.findall(".//collision")
        ):
            shape_name = str(
                element.get("name", f"{element.tag}_{shape_index:02d}")
            )
            if not shape_name or any(
                character.isspace() for character in shape_name
            ):
                raise ValueError(
                    f"{label} shape name {shape_name!r} cannot be empty or "
                    "contain spaces"
                )
            element.set(
                "name",
                f"{link_map[source_name]}__{element.tag}__"
                f"{shape_name}__{shape_index:02d}",
            )
        for material in result.findall(".//material"):
            name = material.get("name")
            if name:
                material.set("name", material_map[name])
        for mesh_index, mesh in enumerate(result.findall(".//mesh")):
            raw = mesh.get("filename")
            mesh.set(
                "filename",
                self._asset_uri(
                    str(raw), source_path.parent, f"{label} mesh[{mesh_index}]"
                ),
            )
        for texture_index, texture in enumerate(result.findall(".//texture")):
            raw = texture.get("filename")
            texture.set(
                "filename",
                self._asset_uri(
                    str(raw), source_path.parent, f"{label} texture[{texture_index}]"
                ),
            )
        return result

    def _import_urdf(
        self,
        item: dict[str, Any],
        parent: str,
        parent_T_root: np.ndarray,
        label: str,
    ) -> ImportResult:
        instance = _validate_name(item.get("name"), f"{label}.name")
        path_keys = [key for key in ("path", "urdf") if key in item]
        if len(path_keys) != 1:
            raise ValueError(f"{label} must define exactly one of path or urdf")
        raw_path = item[path_keys[0]]
        source_path = self._resolve_asset(raw_path, f"{label}.path")
        source_root, links, joints, detected_root, adjacency = self._parse_source_urdf(
            source_path
        )
        asserted_root = item.get("root_link")
        if asserted_root is not None and asserted_root != detected_root:
            raise ValueError(
                f"{label}.root_link asserted {asserted_root!r}, but source root is "
                f"{detected_root!r}"
            )
        prune_joints = item.get("prune_subtrees_at_joints", [])
        exclude_subtrees = item.get("exclude_subtrees", [])
        if not isinstance(prune_joints, list) or not all(
            isinstance(value, str) for value in prune_joints
        ):
            raise ValueError(f"{label}.prune_subtrees_at_joints must be a string list")
        if not isinstance(exclude_subtrees, list) or not all(
            isinstance(value, str) for value in exclude_subtrees
        ):
            raise ValueError(f"{label}.exclude_subtrees must be a string list")
        removed_links = self._pruned_links(
            links,
            joints,
            adjacency,
            prune_joints,
            exclude_subtrees,
            label,
        )
        if detected_root in removed_links:
            raise ValueError(f"{label} cannot prune its source root {detected_root!r}")
        kept_links = {name: link for name, link in links.items() if name not in removed_links}
        kept_joints: dict[str, ET.Element] = {}
        removed_joints: set[str] = set()
        for name, joint in joints.items():
            parent_name = str(joint.find("./parent").get("link"))  # type: ignore[union-attr]
            child_name = str(joint.find("./child").get("link"))  # type: ignore[union-attr]
            if parent_name in kept_links and child_name in kept_links:
                kept_joints[name] = joint
            else:
                removed_joints.add(name)

        link_map = {
            name: self._prefixed(instance, name) for name in kept_links
        }
        joint_map = {
            name: self._prefixed(instance, name) for name in kept_joints
        }
        root_materials = source_root.findall("./material")
        kept_materials = [
            material
            for link in kept_links.values()
            for material in link.findall(".//material")
        ]
        defined_materials = {
            str(material.get("name"))
            for material in root_materials + kept_materials
            if material.get("name")
            and (material.find("./color") is not None or material.find("./texture") is not None)
        }
        referenced_materials = {
            str(material.get("name"))
            for material in kept_materials
            if material.get("name") and len(list(material)) == 0
        }
        unresolved_materials = referenced_materials - defined_materials
        if unresolved_materials:
            raise ValueError(
                f"{label} contains unresolved material references "
                f"{sorted(unresolved_materials)}"
            )
        material_names = {
            material.get("name")
            for material in root_materials + kept_materials
            if material.get("name")
        }
        material_map = {
            str(name): self._prefixed(instance, str(name))
            for name in material_names
        }

        for material in source_root.findall("./material"):
            copied = copy.deepcopy(material)
            self._filter_children(
                copied, {"color", "texture"}, source_path, "material"
            )
            name = copied.get("name")
            if name:
                copied.set("name", material_map[name])
            for texture_index, texture in enumerate(copied.findall(".//texture")):
                texture.set(
                    "filename",
                    self._asset_uri(
                        str(texture.get("filename")),
                        source_path.parent,
                        f"{label} material texture[{texture_index}]",
                    ),
                )
            self.robot.append(copied)

        output_root = link_map[detected_root]
        root_copy = self._rewrite_imported_link(
            kept_links[detected_root],
            instance,
            link_map,
            material_map,
            source_path,
            f"{label}.{detected_root}",
        )
        self._add_fixed_link(
            output_root, parent, parent_T_root, link=root_copy
        )
        for source_name, link in kept_links.items():
            if source_name == detected_root:
                continue
            output_name = link_map[source_name]
            if output_name in self.known_links:
                raise ValueError(f"duplicate imported link {output_name!r}")
            self.robot.append(
                self._rewrite_imported_link(
                    link,
                    instance,
                    link_map,
                    material_map,
                    source_path,
                    f"{label}.{source_name}",
                )
            )
            self.known_links.add(output_name)

        joint_types: dict[str, str] = {}
        for source_name, joint in kept_joints.items():
            copied = copy.deepcopy(joint)
            self._filter_children(
                copied,
                {
                    "origin",
                    "parent",
                    "child",
                    "axis",
                    "calibration",
                    "dynamics",
                    "limit",
                    "mimic",
                    "safety_controller",
                },
                source_path,
                "joint",
            )
            copied.set("name", joint_map[source_name])
            parent_element = copied.find("./parent")
            child_element = copied.find("./child")
            assert parent_element is not None and child_element is not None
            parent_element.set("link", link_map[str(parent_element.get("link"))])
            child_element.set("link", link_map[str(child_element.get("link"))])
            mimic = copied.find("./mimic")
            if mimic is not None:
                target = mimic.get("joint")
                if target in removed_joints:
                    raise ValueError(
                        f"{label} kept joint {source_name!r} mimics pruned joint {target!r}"
                    )
                if target not in joint_map:
                    raise ValueError(
                        f"{label} joint {source_name!r} mimics unknown joint {target!r}"
                    )
                mimic.set("joint", joint_map[str(target)])
            self.robot.append(copied)
            joint_types[joint_map[source_name]] = str(joint.get("type"))

        transmission_map: dict[str, str] = {}
        for transmission in source_root.findall("./transmission"):
            name = transmission.get("name")
            if not name:
                raise ValueError(f"{label} contains a transmission without a name")
            references = [node.get("name") for node in transmission.findall("./joint")]
            if any(reference in removed_joints for reference in references):
                self.warnings.append(
                    f"dropped transmission {name!r} from {source_path}; it references "
                    "a pruned joint"
                )
                continue
            unknown = [reference for reference in references if reference not in joint_map]
            if unknown:
                raise ValueError(
                    f"{label} transmission {name!r} references unknown joints {unknown}"
                )
            copied = copy.deepcopy(transmission)
            self._filter_children(
                copied,
                {"type", "joint", "actuator"},
                source_path,
                "transmission",
            )
            for node in copied.findall("./joint") + copied.findall("./actuator"):
                self._filter_children(
                    node,
                    {
                        "hardwareInterface",
                        "mechanicalReduction",
                        "offset",
                        "role",
                    },
                    source_path,
                    f"transmission/{node.tag}",
                )
            output_name = self._prefixed(instance, name)
            copied.set("name", output_name)
            for node in copied.findall("./joint"):
                node.set("name", joint_map[str(node.get("name"))])
            for node in copied.findall("./actuator"):
                actuator = node.get("name")
                if not actuator:
                    raise ValueError(
                        f"{label} transmission {name!r} has an unnamed actuator"
                    )
                node.set("name", self._prefixed(instance, actuator))
            self.robot.append(copied)
            transmission_map[name] = output_name

        return ImportResult(
            instance=instance,
            source_path=source_path,
            source_root=detected_root,
            output_root=output_root,
            links=link_map,
            joints=joint_map,
            materials=material_map,
            transmissions=transmission_map,
            joint_types=joint_types,
            pruned_links=tuple(sorted(removed_links)),
        )

    def _add_robots(self) -> None:
        entries = self._items(self.manifest.get("robots"), "robots")
        for index, item in enumerate(entries):
            label = f"robots[{index}]"
            _reject_unknown_keys(
                item,
                {
                    "name",
                    "path",
                    "urdf",
                    "parent",
                    "parent_T_root",
                    "root_link",
                    "flange_link",
                    "prune_subtrees_at_joints",
                    "exclude_subtrees",
                },
                label,
            )
            name = _validate_name(item.get("name"), f"{label}.name")
            if name in self.robots or name in self.grippers:
                raise ValueError(f"duplicate component name {name!r}")
            parent = self._resolve_parent(item.get("parent"), f"{label}.parent")
            transform = pose_transform(
                item.get("parent_T_root"), f"{label}.parent_T_root"
            )
            imported = self._import_urdf(item, parent, transform, label)
            flange_link = item.get("flange_link")
            if flange_link is not None and flange_link not in imported.links:
                raise ValueError(
                    f"{label}.flange_link {flange_link!r} was not imported"
                )
            self.robots[name] = imported
            joint_type_counts: dict[str, int] = {}
            for joint_type in imported.joint_types.values():
                joint_type_counts[joint_type] = joint_type_counts.get(joint_type, 0) + 1
            self.report["robots"][name] = {
                "source": str(imported.source_path),
                "source_root": imported.source_root,
                "root_link": imported.output_root,
                "parent_link": parent,
                "flange_source_link": flange_link,
                "flange_link": imported.links.get(flange_link) if flange_link else None,
                "link_count": len(imported.links),
                "joint_count": len(imported.joints),
                "joint_types": joint_type_counts,
                "pruned_source_links": list(imported.pruned_links),
            }

    def _add_grippers(self) -> None:
        entries = self._items(self.manifest.get("grippers"), "grippers")
        for index, item in enumerate(entries):
            label = f"grippers[{index}]"
            _reject_unknown_keys(
                item,
                {
                    "name",
                    "type",
                    "path",
                    "urdf",
                    "parent",
                    "parent_T_root",
                    "parent_T_mount",
                    "root_link",
                    "tcp_link",
                    "prune_subtrees_at_joints",
                    "exclude_subtrees",
                    "visuals",
                    "collisions",
                    "inertial",
                    "tcp",
                },
                label,
            )
            name = _validate_name(item.get("name"), f"{label}.name")
            if name in self.robots or name in self.grippers:
                raise ValueError(f"duplicate component name {name!r}")
            parent = self._resolve_parent(item.get("parent"), f"{label}.parent")
            gripper_type = item.get("type")
            if gripper_type in {"static", "static_mesh"}:
                gripper_type = "mesh"
            if gripper_type not in {"mesh", "urdf"}:
                raise ValueError(f"{label}.type must be mesh or urdf")

            if gripper_type == "urdf":
                transform = pose_transform(
                    item.get("parent_T_root", item.get("parent_T_mount")),
                    f"{label}.parent_T_root",
                )
                imported = self._import_urdf(item, parent, transform, label)
                tcp_link = item.get("tcp_link")
                if tcp_link is not None and tcp_link not in imported.links:
                    raise ValueError(
                        f"{label}.tcp_link {tcp_link!r} was not imported"
                    )
                self.grippers[name] = imported
                self.report["grippers"][name] = {
                    "type": "urdf",
                    "source": str(imported.source_path),
                    "source_root": imported.source_root,
                    "root_link": imported.output_root,
                    "parent_link": parent,
                    "tcp_source_link": tcp_link,
                    "tcp_link": imported.links.get(tcp_link) if tcp_link else None,
                    "link_count": len(imported.links),
                    "joint_count": len(imported.joints),
                }
                continue

            mount_transform = pose_transform(
                item.get("parent_T_mount"), f"{label}.parent_T_mount"
            )
            mount_link = f"gripper__{name}__mount"
            link = ET.Element("link", {"name": mount_link})
            counts = self._populate_link(link, item, label)
            if counts["visuals"] == 0 and counts["collisions"] == 0:
                raise ValueError(
                    f"{label} mesh gripper needs at least one visual or collision"
                )
            self._add_fixed_link(mount_link, parent, mount_transform, link=link)

            semantic_links = {"mount": mount_link}
            tcp_link: str | None = None
            tcp = item.get("tcp")
            if tcp is not None:
                if not isinstance(tcp, dict):
                    raise ValueError(f"{label}.tcp must be a mapping")
                _reject_unknown_keys(
                    tcp, {"name", "mount_T_tcp"}, f"{label}.tcp"
                )
                tcp_name = _validate_name(tcp.get("name", "tcp"), f"{label}.tcp.name")
                tcp_link = f"gripper__{name}__{tcp_name}"
                tcp_transform = pose_transform(
                    tcp.get("mount_T_tcp"), f"{label}.tcp.mount_T_tcp"
                )
                self._add_fixed_link(tcp_link, mount_link, tcp_transform)
                semantic_links[tcp_name] = tcp_link

            imported = ImportResult(
                instance=name,
                source_path=self.manifest_path,
                source_root="mount",
                output_root=mount_link,
                links=semantic_links,
                joints={},
                materials={},
                transmissions={},
                joint_types={},
                pruned_links=(),
            )
            self.grippers[name] = imported
            self.report["grippers"][name] = {
                "type": "mesh",
                "root_link": mount_link,
                "parent_link": parent,
                "tcp_link": tcp_link,
                "link_count": len(semantic_links),
                "joint_count": 0,
                **counts,
            }

    @staticmethod
    def _finite_float(
        value: Any, label: str, *, positive: bool = False
    ) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{label} must be numeric") from error
        if not math.isfinite(result) or (positive and result <= 0.0):
            qualifier = "positive and finite" if positive else "finite"
            raise ValueError(f"{label} must be {qualifier}")
        return result

    def _ordered_range(
        self,
        value: Any,
        label: str,
        first_key: str,
        second_key: str,
    ) -> dict[str, float]:
        if not isinstance(value, dict):
            raise ValueError(f"{label} must be a mapping")
        _reject_unknown_keys(value, {first_key, second_key}, label)
        first = self._finite_float(value.get(first_key), f"{label}.{first_key}", positive=True)
        second = self._finite_float(
            value.get(second_key), f"{label}.{second_key}", positive=True
        )
        if first >= second:
            raise ValueError(
                f"{label} must satisfy {first_key} < {second_key}"
            )
        return {first_key: first, second_key: second}

    def _parse_operating_envelope(
        self, value: Any, label: str
    ) -> dict[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError(f"{label} must be a mapping")
        _reject_unknown_keys(
            value,
            {
                "working_distance_m",
                "view_depth_m",
                "depth_measurement_range_m",
                "focus",
                "depth_of_field_m",
            },
            label,
        )
        result: dict[str, Any] = {}
        working_raw = value.get("working_distance_m")
        if working_raw is not None:
            if isinstance(working_raw, (int, float)) and not isinstance(working_raw, bool):
                nominal = self._finite_float(
                    working_raw, f"{label}.working_distance_m", positive=True
                )
                working = {"min": nominal, "nominal": nominal, "max": nominal}
            elif isinstance(working_raw, dict):
                _reject_unknown_keys(
                    working_raw, {"min", "nominal", "max"},
                    f"{label}.working_distance_m"
                )
                nominal = self._finite_float(
                    working_raw.get("nominal"),
                    f"{label}.working_distance_m.nominal",
                    positive=True,
                )
                minimum = self._finite_float(
                    working_raw.get("min", nominal),
                    f"{label}.working_distance_m.min",
                    positive=True,
                )
                maximum = self._finite_float(
                    working_raw.get("max", nominal),
                    f"{label}.working_distance_m.max",
                    positive=True,
                )
                if not minimum <= nominal <= maximum:
                    raise ValueError(
                        f"{label}.working_distance_m must satisfy min <= nominal <= max"
                    )
                working = {"min": minimum, "nominal": nominal, "max": maximum}
            else:
                raise ValueError(f"{label}.working_distance_m must be numeric or a mapping")
            result["working_distance_m"] = working

        if "view_depth_m" in value:
            result["view_depth_m"] = self._ordered_range(
                value["view_depth_m"], f"{label}.view_depth_m", "near", "far"
            )
        if "depth_measurement_range_m" in value:
            result["depth_measurement_range_m"] = self._ordered_range(
                value["depth_measurement_range_m"],
                f"{label}.depth_measurement_range_m",
                "min",
                "max",
            )

        focus_raw = value.get("focus")
        if "depth_of_field_m" in value:
            if focus_raw is not None:
                raise ValueError(
                    f"{label} cannot define both focus and top-level depth_of_field_m"
                )
            focus_raw = {"depth_of_field_m": value["depth_of_field_m"]}
        if focus_raw is not None:
            if not isinstance(focus_raw, dict):
                raise ValueError(f"{label}.focus must be a mapping")
            _reject_unknown_keys(
                focus_raw,
                {"focus_distance_m", "depth_of_field_m", "source"},
                f"{label}.focus",
            )
            focus: dict[str, Any] = {}
            if "focus_distance_m" in focus_raw:
                focus["focus_distance_m"] = self._finite_float(
                    focus_raw["focus_distance_m"],
                    f"{label}.focus.focus_distance_m",
                    positive=True,
                )
            if "depth_of_field_m" in focus_raw:
                focus["depth_of_field_m"] = self._ordered_range(
                    focus_raw["depth_of_field_m"],
                    f"{label}.focus.depth_of_field_m",
                    "near",
                    "far",
                )
            if "source" in focus_raw:
                if focus_raw["source"] not in {"measured", "manufacturer", "nominal"}:
                    raise ValueError(
                        f"{label}.focus.source must be measured, manufacturer, or nominal"
                    )
                focus["source"] = focus_raw["source"]
            result["focus"] = focus

        working = result.get("working_distance_m")
        view = result.get("view_depth_m")
        if working and view and (
            working["min"] < view["near"] or working["max"] > view["far"]
        ):
            raise ValueError(
                f"{label}.working_distance_m must lie inside view_depth_m"
            )
        depth_range = result.get("depth_measurement_range_m")
        if working and depth_range and (
            working["max"] < depth_range["min"]
            or working["min"] > depth_range["max"]
        ):
            raise ValueError(
                f"{label}.working_distance_m does not intersect "
                "depth_measurement_range_m"
            )
        return result

    @staticmethod
    def _matrix_value(
        value: Any, shape: tuple[int, int], label: str
    ) -> np.ndarray:
        if isinstance(value, dict):
            _reject_unknown_keys(value, {"rows", "cols", "data"}, label)
            if value.get("rows") != shape[0] or value.get("cols") != shape[1]:
                raise ValueError(
                    f"{label} declares {value.get('rows')}x{value.get('cols')}; "
                    f"expected {shape[0]}x{shape[1]}"
                )
            value = value.get("data")
        matrix = np.asarray(value, dtype=float)
        if matrix.shape == (shape[0] * shape[1],):
            matrix = matrix.reshape(shape)
        if matrix.shape != shape or not np.all(np.isfinite(matrix)):
            raise ValueError(f"{label} must be a finite {shape[0]}x{shape[1]} matrix")
        return matrix

    @staticmethod
    def _vector_value(value: Any, label: str) -> np.ndarray:
        if isinstance(value, dict):
            _reject_unknown_keys(value, {"rows", "cols", "data"}, label)
            rows, cols = value.get("rows"), value.get("cols")
            if not isinstance(rows, int) or not isinstance(cols, int):
                raise ValueError(f"{label}.rows and cols must be integers")
            value = value.get("data")
            vector = np.asarray(value, dtype=float)
            if rows * cols != vector.size or min(rows, cols) != 1:
                raise ValueError(f"{label} must describe a row or column vector")
        else:
            vector = np.asarray(value, dtype=float)
        if vector.ndim != 1 or not np.all(np.isfinite(vector)):
            raise ValueError(f"{label} must be a finite list")
        return vector

    def _camera_info(
        self,
        camera_name: str,
        stream_name: str,
        calibration: dict[str, Any],
        label: str,
        *,
        implicit_stream: bool = False,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        calibration_keys = {
            "status",
            "camera_name",
            "image_width",
            "image_height",
            "width_px",
            "height_px",
            "image_size_px",
            "intrinsics",
            "fx",
            "fy",
            "cx",
            "cy",
            "skew",
            "fx_px",
            "fy_px",
            "cx_px",
            "cy_px",
            "skew_px",
            "camera_matrix",
            "K",
            "distortion",
            "distortion_model",
            "distortion_coefficients",
            "D",
            "rectification_matrix",
            "R",
            "projection_matrix",
            "P",
        }
        _reject_unknown_keys(calibration, calibration_keys, label)
        nested = calibration.get("intrinsics")
        if nested is None:
            intrinsics = calibration
        elif isinstance(nested, dict):
            intrinsics = nested
            _reject_unknown_keys(
                intrinsics,
                calibration_keys - {"intrinsics"},
                f"{label}.intrinsics",
            )
        else:
            raise ValueError(f"{label}.intrinsics must be a mapping")

        image_size = calibration.get(
            "image_size_px", intrinsics.get("image_size_px")
        )
        if image_size is not None:
            if not isinstance(image_size, dict):
                raise ValueError(f"{label}.image_size_px must be a mapping")
            _reject_unknown_keys(
                image_size, {"width", "height"}, f"{label}.image_size_px"
            )
            width_raw = image_size.get("width")
            height_raw = image_size.get("height")
        else:
            width_raw = calibration.get(
                "image_width", intrinsics.get("image_width", intrinsics.get("width_px"))
            )
            height_raw = calibration.get(
                "image_height", intrinsics.get("image_height", intrinsics.get("height_px"))
            )
        if (
            isinstance(width_raw, bool)
            or isinstance(height_raw, bool)
            or not isinstance(width_raw, int)
            or not isinstance(height_raw, int)
            or width_raw <= 0
            or height_raw <= 0
        ):
            raise ValueError(f"{label} image_width and image_height must be positive integers")
        width, height = width_raw, height_raw

        def camera_number(*keys: str, default: Any = None, positive: bool = False) -> float:
            raw = default
            for key in keys:
                if key in intrinsics:
                    raw = intrinsics[key]
                    break
            return self._finite_float(raw, f"{label}.{keys[0]}", positive=positive)

        matrix_sources = [calibration]
        if intrinsics is not calibration:
            matrix_sources.append(intrinsics)
        matrix_values = [
            value
            for source in matrix_sources
            for value in (source.get("camera_matrix"), source.get("K"))
            if value is not None
        ]
        component_keys = {
            "fx",
            "fy",
            "cx",
            "cy",
            "skew",
            "fx_px",
            "fy_px",
            "cx_px",
            "cy_px",
            "skew_px",
        }
        if len(matrix_values) > 1:
            raise ValueError(f"{label} defines more than one camera matrix")
        if matrix_values:
            if component_keys & (set(calibration) | set(intrinsics)):
                raise ValueError(
                    f"{label} camera_matrix/K cannot be combined with fx/fy/cx/cy"
                )
            k = self._matrix_value(matrix_values[0], (3, 3), f"{label}.camera_matrix")
            if not np.allclose(k[2], [0.0, 0.0, 1.0], atol=1e-9, rtol=0.0):
                raise ValueError(f"{label}.camera_matrix third row must be [0, 0, 1]")
            if not np.isclose(k[1, 0], 0.0, atol=1e-9, rtol=0.0):
                raise ValueError(f"{label}.camera_matrix K[1,0] must be zero")
            fx, fy, cx, cy, skew = (
                float(k[0, 0]),
                float(k[1, 1]),
                float(k[0, 2]),
                float(k[1, 2]),
                float(k[0, 1]),
            )
            if fx <= 0.0 or fy <= 0.0:
                raise ValueError(f"{label}.camera_matrix focal lengths must be positive")
        else:
            fx = camera_number("fx", "fx_px", positive=True)
            fy = camera_number("fy", "fy_px", positive=True)
            cx = camera_number("cx", "cx_px")
            cy = camera_number("cy", "cy_px")
            skew = camera_number("skew", "skew_px", default=0.0)
            k = np.asarray(
                [[fx, skew, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]
            )

        distortion = calibration.get("distortion", intrinsics.get("distortion"))
        if distortion is not None:
            if not isinstance(distortion, dict):
                raise ValueError(f"{label}.distortion must be a mapping")
            _reject_unknown_keys(
                distortion, {"model", "coefficients"}, f"{label}.distortion"
            )
            distortion_model = distortion.get("model")
            coefficients = distortion.get("coefficients")
        else:
            distortion_model = calibration.get(
                "distortion_model", intrinsics.get("distortion_model", "plumb_bob")
            )
            coefficients = calibration.get(
                "distortion_coefficients",
                calibration.get(
                    "D",
                    intrinsics.get(
                        "distortion_coefficients",
                        intrinsics.get("D", [0.0] * 5),
                    ),
                ),
            )
        if distortion_model not in DISTORTION_LENGTHS:
            raise ValueError(
                f"{label}.distortion_model must be one of "
                f"{sorted(DISTORTION_LENGTHS)}"
            )
        distortion_array = self._vector_value(
            coefficients, f"{label}.distortion_coefficients"
        )
        required_length = DISTORTION_LENGTHS.get(distortion_model)
        if required_length is not None and len(distortion_array) != required_length:
            raise ValueError(
                f"{label} {distortion_model} distortion requires {required_length} "
                f"coefficients, got {len(distortion_array)}"
            )

        r_value = calibration.get(
            "rectification_matrix",
            calibration.get(
                "R", intrinsics.get("rectification_matrix", intrinsics.get("R"))
            ),
        )
        p_value = calibration.get(
            "projection_matrix",
            calibration.get(
                "P", intrinsics.get("projection_matrix", intrinsics.get("P"))
            ),
        )
        rectification = (
            np.eye(3)
            if r_value is None
            else self._matrix_value(r_value, (3, 3), f"{label}.rectification_matrix")
        )
        projection = (
            np.column_stack((k, np.zeros(3)))
            if p_value is None
            else self._matrix_value(p_value, (3, 4), f"{label}.projection_matrix")
        )
        rectification_transform = np.eye(4)
        rectification_transform[:3, :3] = rectification
        _validate_transform(rectification_transform, f"{label}.rectification_matrix")
        if not np.allclose(
            projection[2], [0.0, 0.0, 1.0, 0.0], atol=1e-9, rtol=0.0
        ):
            raise ValueError(
                f"{label}.projection_matrix third row must be [0, 0, 1, 0]"
            )
        if not 0.0 <= cx <= float(width) or not 0.0 <= cy <= float(height):
            self.warnings.append(
                f"{label} principal point ({cx}, {cy}) lies outside {width}x{height}"
            )

        document = {
            "image_width": width,
            "image_height": height,
            "camera_name": (
                camera_name
                if implicit_stream
                else f"{camera_name}_{stream_name}"
            ),
            "camera_matrix": {"rows": 3, "cols": 3, "data": k.reshape(-1).tolist()},
            "distortion_model": distortion_model,
            "distortion_coefficients": {
                "rows": 1,
                "cols": len(distortion_array),
                "data": distortion_array.tolist(),
            },
            "rectification_matrix": {
                "rows": 3,
                "cols": 3,
                "data": rectification.reshape(-1).tolist(),
            },
            "projection_matrix": {
                "rows": 3,
                "cols": 4,
                "data": projection.reshape(-1).tolist(),
            },
        }

        derived: dict[str, Any] = {}
        pinhole_model = distortion_model in {
            "plumb_bob",
            "rational_polynomial",
        }
        zero_skew_projection = abs(float(projection[0, 1])) <= 1e-12 and abs(
            float(projection[1, 0])
        ) <= 1e-12
        if pinhole_model and zero_skew_projection:
            fov_fx = float(projection[0, 0])
            fov_fy = float(projection[1, 1])
            fov_cx = float(projection[0, 2])
            fov_cy = float(projection[1, 2])
            if fov_fx <= 0.0 or fov_fy <= 0.0:
                raise ValueError(f"{label}.projection_matrix focal lengths must be positive")
            left = math.degrees(math.atan(fov_cx / fov_fx))
            right = math.degrees(math.atan((width - fov_cx) / fov_fx))
            top = math.degrees(math.atan(fov_cy / fov_fy))
            bottom = math.degrees(math.atan((height - fov_cy) / fov_fy))
            horizontal, vertical = left + right, top + bottom
            if not 0.0 < horizontal < 180.0 or not 0.0 < vertical < 180.0:
                raise ValueError(f"{label} produces an invalid pinhole field of view")
            derived = {
                "method": "rectified_pinhole_from_P",
                "horizontal_fov_deg": horizontal,
                "vertical_fov_deg": vertical,
                "fov_deg": {
                    "left": left,
                    "right": right,
                    "horizontal": horizontal,
                    "top": top,
                    "bottom": bottom,
                    "vertical": vertical,
                },
                "rectified_focal_length_px": {"fx": fov_fx, "fy": fov_fy},
            }
        elif not pinhole_model:
            self.warnings.append(
                f"{label} uses {distortion_model!r}; raw FOV was not inferred with a "
                "pinhole formula"
            )
        else:
            self.warnings.append(
                f"{label} has a nonzero projection skew; FOV was not inferred "
                "with the zero-skew pinhole formula"
            )
        return document, derived

    @staticmethod
    def _camera_streams(item: dict[str, Any], label: str) -> list[dict[str, Any]]:
        streams = item.get("streams")
        if streams is not None:
            if not isinstance(streams, list) or not all(
                isinstance(stream, dict) for stream in streams
            ):
                raise ValueError(f"{label}.streams must be a list of mappings")
            for index, stream in enumerate(streams):
                _reject_unknown_keys(
                    stream,
                    {"name", "type", "calibration"},
                    f"{label}.streams[{index}]",
                )
            return streams
        intrinsics = item.get("intrinsics")
        if intrinsics is None:
            return []
        if not isinstance(intrinsics, dict):
            raise ValueError(f"{label}.intrinsics must be a mapping")
        return [
            {
                "name": item.get("stream_name", "color"),
                "type": item.get("stream_type", "color"),
                "calibration": intrinsics,
                "_implicit": True,
            }
        ]

    def _add_cameras(self) -> None:
        entries = self._items(self.manifest.get("cameras"), "cameras")
        declared_names: set[str] = set()
        for index, item in enumerate(entries):
            label = f"cameras[{index}]"
            _reject_unknown_keys(
                item,
                {
                    "name",
                    "enabled",
                    "mode",
                    "pose_status",
                    "parent",
                    "parent_T_camera_link",
                    "parent_T_camera_optical",
                    "camera_link_T_optical",
                    "physical_model",
                    "visuals",
                    "collisions",
                    "inertial",
                    "intrinsics",
                    "streams",
                    "stream_name",
                    "stream_type",
                    "operating_envelope",
                },
                label,
            )
            name = _validate_name(item.get("name"), f"{label}.name")
            if name in declared_names:
                raise ValueError(f"duplicate camera name {name!r}")
            declared_names.add(name)
            enabled = item.get("enabled", True)
            if not isinstance(enabled, bool):
                raise ValueError(f"{label}.enabled must be boolean")
            if not enabled:
                self.report["cameras"][name] = {"enabled": False}
                continue
            pose_status = item.get("pose_status")
            if pose_status not in {"nominal", "measured", "calibrated"}:
                raise ValueError(
                    f"{label}.pose_status must be nominal, measured, or calibrated"
                )
            mode = item.get("mode")
            if mode is not None and mode not in {
                "eye_to_hand",
                "eye_in_hand",
                "fixed",
                "wrist",
            }:
                raise ValueError(
                    f"{label}.mode must be eye_to_hand, eye_in_hand, fixed, or wrist"
                )
            parent = self._resolve_parent(item.get("parent"), f"{label}.parent")
            pose_keys = [
                key
                for key in ("parent_T_camera_link", "parent_T_camera_optical")
                if key in item
            ]
            if len(pose_keys) != 1:
                raise ValueError(
                    f"{label} must define exactly one of parent_T_camera_link or "
                    "parent_T_camera_optical"
                )
            physical = item.get("physical_model", {})
            if physical is None:
                physical = {}
            if not isinstance(physical, dict):
                raise ValueError(f"{label}.physical_model must be a mapping")
            _reject_unknown_keys(
                physical,
                {"visuals", "collisions", "inertial", "camera_link_T_optical"},
                f"{label}.physical_model",
            )
            body_definition = {
                key: item[key]
                for key in ("visuals", "collisions", "inertial")
                if key in item
            }
            for key in ("visuals", "collisions", "inertial"):
                if key in physical:
                    if key in body_definition:
                        raise ValueError(
                            f"{label}.{key} is duplicated in physical_model"
                        )
                    body_definition[key] = physical[key]
            camera_link_T_optical = pose_transform(
                item.get(
                    "camera_link_T_optical",
                    physical.get(
                        "camera_link_T_optical",
                        {
                            "position_m": [0.0, 0.0, 0.0],
                            "rpy_deg": [-90.0, 0.0, -90.0],
                        },
                    ),
                ),
                f"{label}.camera_link_T_optical",
            )
            if pose_keys[0] == "parent_T_camera_link":
                parent_T_camera_link = pose_transform(
                    item[pose_keys[0]], f"{label}.{pose_keys[0]}"
                )
                parent_T_camera_optical = parent_T_camera_link @ camera_link_T_optical
            else:
                parent_T_camera_optical = pose_transform(
                    item[pose_keys[0]], f"{label}.{pose_keys[0]}"
                )
                parent_T_camera_link = parent_T_camera_optical @ np.linalg.inv(
                    camera_link_T_optical
                )
                parent_T_camera_link = _validate_transform(
                    parent_T_camera_link, f"{label}.derived_parent_T_camera_link"
                )

            body_link = f"camera__{name}__link"
            optical_link = f"camera__{name}__optical_frame"
            link = ET.Element("link", {"name": body_link})
            body_counts = self._populate_link(link, body_definition, f"{label}.physical_model")
            self._add_fixed_link(
                body_link, parent, parent_T_camera_link, link=link
            )
            self._add_fixed_link(
                optical_link, body_link, camera_link_T_optical
            )
            self.camera_links[name] = {"link": body_link, "optical": optical_link}

            envelope = self._parse_operating_envelope(
                item.get("operating_envelope"), f"{label}.operating_envelope"
            )
            stream_reports: dict[str, Any] = {}
            seen_streams: set[str] = set()
            for stream_index, stream in enumerate(self._camera_streams(item, label)):
                stream_label = f"{label}.streams[{stream_index}]"
                stream_name = _validate_name(
                    stream.get("name", "color"), f"{stream_label}.name"
                )
                if stream_name in seen_streams:
                    raise ValueError(
                        f"{label} contains duplicate stream name {stream_name!r}"
                    )
                seen_streams.add(stream_name)
                stream_type = stream.get("type", "color")
                if stream_type not in {"color", "mono", "depth", "infrared"}:
                    raise ValueError(
                        f"{stream_label}.type must be color, mono, depth, or infrared"
                    )
                calibration = stream.get("calibration")
                if not isinstance(calibration, dict):
                    raise ValueError(f"{stream_label}.calibration must be a mapping")
                calibration_status = calibration.get("status", "unspecified")
                if calibration_status not in {
                    "unspecified",
                    "nominal",
                    "measured",
                    "calibrated",
                    "unavailable",
                }:
                    raise ValueError(
                        f"{stream_label}.calibration.status must be nominal, "
                        "measured, calibrated, or unavailable"
                    )
                implicit_stream = bool(stream.get("_implicit", False))
                camera_info, derived = self._camera_info(
                    name,
                    stream_name,
                    calibration,
                    stream_label,
                    implicit_stream=implicit_stream,
                )
                filename = (
                    f"{name}_camera_info.yaml"
                    if implicit_stream
                    else f"{name}_{stream_name}_camera_info.yaml"
                )
                info_path = self.camera_info_dir / filename
                if info_path in {self.output_path, self.report_path}:
                    raise ValueError(
                        f"camera-info output {info_path} conflicts with another output"
                    )
                if info_path in self.camera_info_documents:
                    raise ValueError(f"duplicate camera-info output path {info_path}")
                self.camera_info_documents[info_path] = camera_info
                working = envelope.get("working_distance_m")
                if working and derived:
                    distance = working["nominal"]
                    focal = derived["rectified_focal_length_px"]
                    derived["footprint_at_nominal_working_distance_m"] = {
                        "distance": distance,
                        "width": distance * camera_info["image_width"] / focal["fx"],
                        "height": distance * camera_info["image_height"] / focal["fy"],
                    }
                stream_reports[stream_name] = {
                    "type": stream_type,
                    "calibration_status": calibration_status,
                    "camera_info": str(info_path),
                    "derived": derived,
                }
            if not stream_reports:
                self.warnings.append(
                    f"camera {name!r} has no intrinsics; its URDF frames were emitted "
                    "without camera_info"
                )
            camera_report = {
                "enabled": True,
                "mode": mode,
                "pose_status": pose_status,
                "parent_link": parent,
                "camera_link": body_link,
                "optical_frame": optical_link,
                "parent_T_camera_optical": parent_T_camera_optical.tolist(),
                "body_geometry": body_counts,
                "operating_envelope": envelope,
                "streams": stream_reports,
            }
            if len(stream_reports) == 1:
                only_stream = next(iter(stream_reports.values()))
                camera_report["camera_info"] = only_stream["camera_info"]
                camera_report["derived"] = copy.deepcopy(only_stream["derived"])
            self.report["cameras"][name] = camera_report

    def _add_attached_frames(self) -> None:
        self._add_frames("attached_frames")

    def _validate_output_tree(self) -> dict[str, Any]:
        links = self._unique_named(self.robot.findall("./link"), "generated links")
        joints = self._unique_named(self.robot.findall("./joint"), "generated joints")
        children: set[str] = set()
        adjacency: dict[str, list[str]] = {name: [] for name in links}
        joint_type_counts: dict[str, int] = {}
        mimic_targets: dict[str, str] = {}
        for name, joint in joints.items():
            joint_type = joint.get("type")
            if joint_type not in SUPPORTED_JOINT_TYPES:
                raise ValueError(
                    f"generated joint {name!r} has unsupported type {joint_type!r}"
                )
            joint_type_counts[str(joint_type)] = (
                joint_type_counts.get(str(joint_type), 0) + 1
            )
            parent_nodes = joint.findall("./parent")
            child_nodes = joint.findall("./child")
            if len(parent_nodes) != 1 or len(child_nodes) != 1:
                raise ValueError(
                    f"generated joint {name!r} needs exactly one parent and child"
                )
            parent = parent_nodes[0].get("link")
            child = child_nodes[0].get("link")
            if parent not in links or child not in links:
                raise ValueError(
                    f"generated joint {name!r} references an unknown link"
                )
            if child in children:
                raise ValueError(f"generated link {child!r} has multiple parents")
            children.add(str(child))
            adjacency[str(parent)].append(str(child))
            mimic = joint.find("./mimic")
            if mimic is not None and mimic.get("joint") not in joints:
                raise ValueError(
                    f"generated joint {name!r} mimics unknown joint "
                    f"{mimic.get('joint')!r}"
                )
            if mimic is not None:
                mimic_targets[name] = str(mimic.get("joint"))

        checked_mimics: set[str] = set()
        active_mimics: set[str] = set()

        def visit_mimic(joint_name: str) -> None:
            if joint_name in active_mimics:
                raise ValueError("generated URDF contains a mimic-joint cycle")
            if joint_name in checked_mimics:
                return
            active_mimics.add(joint_name)
            target = mimic_targets.get(joint_name)
            if target is not None:
                visit_mimic(target)
            active_mimics.remove(joint_name)
            checked_mimics.add(joint_name)

        for joint_name in mimic_targets:
            visit_mimic(joint_name)

        roots = set(links) - children
        if roots != {self.root_link}:
            raise ValueError(
                f"generated URDF must have root {self.root_link!r}; detected "
                f"{sorted(roots)}"
            )
        visited: set[str] = set()
        active: set[str] = set()

        def visit(link_name: str) -> None:
            if link_name in active:
                raise ValueError("generated URDF contains a kinematic cycle")
            if link_name in visited:
                return
            active.add(link_name)
            for child in adjacency[link_name]:
                visit(child)
            active.remove(link_name)
            visited.add(link_name)

        visit(self.root_link)
        if visited != set(links):
            raise ValueError(
                f"generated URDF is disconnected; unreachable links "
                f"{sorted(set(links) - visited)}"
            )

        transmissions = self._unique_named(
            self.robot.findall("./transmission"), "generated transmissions"
        )
        for name, transmission in transmissions.items():
            for joint in transmission.findall("./joint"):
                if joint.get("name") not in joints:
                    raise ValueError(
                        f"generated transmission {name!r} references unknown joint "
                        f"{joint.get('name')!r}"
                    )

        for index, asset in enumerate(
            self.robot.findall(".//mesh") + self.robot.findall(".//texture")
        ):
            filename = asset.get("filename")
            if not filename:
                raise ValueError(f"generated asset[{index}] has no filename")
            if filename.startswith("package://"):
                continue
            path = Path(filename)
            if not path.is_absolute():
                path = self.output_path.parent / path
            if not path.resolve().is_file():
                raise FileNotFoundError(
                    f"generated asset[{index}] does not resolve: {filename}"
                )

        return {
            "link_count": len(links),
            "joint_count": len(joints),
            "joint_types": joint_type_counts,
            "transmission_count": len(transmissions),
            "camera_count": len(self.camera_links),
            "robot_count": len(self.robots),
            "gripper_count": len(self.grippers),
            "static_body_count": len(self.static_bodies),
        }

    def _build(self) -> bytes:
        if getattr(self, "_built", False):
            raise RuntimeError("a WorkcellUrdfGenerator instance can only build once")
        self._built = True
        self._add_frames()
        self._add_static_bodies()
        self._add_robots()
        self._add_grippers()
        self._add_cameras()
        self._add_attached_frames()
        self.report["inventory"] = self._validate_output_tree()
        self.report["outputs"] = {
            "urdf": str(self.output_path),
            "camera_info_dir": str(self.camera_info_dir),
            "camera_info": [
                str(path) for path in sorted(self.camera_info_documents)
            ],
            "report": str(self.report_path),
        }
        ET.indent(self.robot, space="  ")
        return ET.tostring(
            self.robot,
            encoding="utf-8",
            xml_declaration=True,
            short_empty_elements=True,
        )

    def generate(self, *, write_files: bool = True) -> GenerationResult:
        """Validate and generate the workcell, optionally without writing files."""
        urdf_bytes = self._build()
        if write_files:
            _atomic_write(self.output_path, urdf_bytes)
            for path, document in sorted(
                self.camera_info_documents.items(), key=lambda item: str(item[0])
            ):
                _atomic_write(path, _yaml_bytes(document))
            _atomic_write(self.report_path, _yaml_bytes(self.report))
        return GenerationResult(
            urdf_path=self.output_path,
            camera_info_paths=(
                tuple(sorted(self.camera_info_documents)) if write_files else ()
            ),
            report_path=self.report_path,
            report=copy.deepcopy(self.report),
            wrote_files=write_files,
        )


def generate_workcell_urdf(
    manifest_path: str | Path,
    *,
    output_override: str | Path | None = None,
    camera_info_dir_override: str | Path | None = None,
    report_override: str | Path | None = None,
    write_files: bool = True,
) -> GenerationResult:
    """Generate a connected workcell URDF and its camera companion files."""
    generator = WorkcellUrdfGenerator(
        manifest_path,
        output_override=output_override,
        camera_info_dir_override=camera_info_dir_override,
        report_override=report_override,
    )
    return generator.generate(write_files=write_files)
