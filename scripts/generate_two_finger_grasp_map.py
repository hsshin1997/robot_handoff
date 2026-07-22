#!/usr/bin/env python3
"""Generate a continuous, object-only two-finger grasp map from YAML.

The resulting candidates use local outward-wound part facets.  They do not
certify directional external visibility, complete gripper collision, PCB
clearance, or insertion-path safety.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mujoco_sim.core.se3 import validate_transform  # noqa: E402
from mujoco_sim.modeling import two_finger_grasp_map as grasp_map_module  # noqa: E402
from mujoco_sim.modeling.two_finger_grasp_map import (  # noqa: E402
    generate_two_finger_grasp_map,
    load_scaled_binary_stl,
)


DEFAULT_CONFIG = (
    ROOT / "projects" / "two_finger_grasp_map" / "config"
    / "connector_header.yaml"
)
CLAIM_LEVEL = "object_only_local_surface_candidate"


class _StrictSafeLoader(yaml.SafeLoader):
    """Safe YAML loader that also rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _StrictSafeLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as error:
            raise ValueError("YAML mapping keys must be hashable") from error
        if duplicate:
            raise ValueError(f"duplicate YAML mapping key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_StrictSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_to_root(path: Path, *, label: str) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError as error:
        raise ValueError(f"{label} must resolve inside repository root {ROOT}") from error


def _strict_mapping(
    value: Any,
    *,
    label: str,
    required: set[str],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a YAML mapping")
    keys = set(value)
    if any(not isinstance(key, str) for key in keys):
        raise ValueError(f"{label} keys must be strings")
    missing = sorted(required - keys)
    unknown = sorted(keys - required)
    if missing:
        raise ValueError(f"{label} is missing keys: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{label} has unknown keys: {', '.join(unknown)}")
    return value


def _number(
    value: Any,
    *,
    label: str,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_inclusive: bool = True,
    maximum_inclusive: bool = True,
) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise ValueError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    if minimum is not None:
        invalid = result < minimum if minimum_inclusive else result <= minimum
        if invalid:
            relation = ">=" if minimum_inclusive else ">"
            raise ValueError(f"{label} must be {relation} {minimum}")
    if maximum is not None:
        invalid = result > maximum if maximum_inclusive else result >= maximum
        if invalid:
            relation = "<=" if maximum_inclusive else "<"
            raise ValueError(f"{label} must be {relation} {maximum}")
    return result


def _number_sequence(
    value: Any,
    *,
    label: str,
    length: int,
) -> tuple[float, ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or len(value) != length
    ):
        raise ValueError(f"{label} must contain exactly {length} numbers")
    return tuple(
        _number(item, label=f"{label}[{index}]")
        for index, item in enumerate(value)
    )


def _nonempty_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _load_config(path: Path) -> Mapping[str, Any]:
    try:
        value = yaml.load(
            path.read_text(encoding="utf-8"),
            Loader=_StrictSafeLoader,
        )
    except (yaml.YAMLError, ValueError) as error:
        raise ValueError(f"invalid YAML in {path}: {error}") from error
    root = _strict_mapping(
        value,
        label="config",
        required={"schema_version", "part", "insertion", "gripper", "map", "output"},
    )
    version = root["schema_version"]
    if not isinstance(version, int) or isinstance(version, bool) or version != 1:
        raise ValueError("schema_version must be integer 1")
    return root


def _resolve_part_path(value: Any) -> Path:
    supplied = Path(_nonempty_string(value, label="part.path"))
    if supplied.is_absolute():
        raise ValueError("part.path must be relative to the repository root")
    resolved = (ROOT / supplied).resolve()
    _relative_to_root(resolved, label="part.path")
    if not resolved.is_file():
        raise FileNotFoundError(f"part STL was not found: {resolved}")
    return resolved


def _resolve_output_path(value: Any, *, config_path: Path) -> Path:
    supplied = Path(_nonempty_string(value, label="output"))
    if supplied.is_absolute():
        raise ValueError("output must be relative to the config directory")
    resolved = (config_path.parent / supplied).resolve()
    _relative_to_root(resolved, label="output")
    if resolved.suffix.lower() != ".json":
        raise ValueError("output must name a .json file")
    return resolved


def generate_from_config(config_path: str | Path) -> tuple[Path, dict[str, Any]]:
    """Generate one deterministic map document and return its output path."""
    supplied_config = Path(config_path)
    path = (
        supplied_config.resolve()
        if supplied_config.is_absolute()
        else (ROOT / supplied_config).resolve()
    )
    _relative_to_root(path, label="config path")
    if not path.is_file():
        raise FileNotFoundError(f"config file was not found: {path}")
    config = _load_config(path)

    part = _strict_mapping(
        config["part"], label="part", required={"path", "scale_to_m"})
    insertion = _strict_mapping(
        config["insertion"],
        label="insertion",
        required={"T_W_P_insert", "insertion_axis_P"},
    )
    gripper = _strict_mapping(
        config["gripper"],
        label="gripper",
        required={"opening_range_m", "friction_coefficient"},
    )
    map_config = _strict_mapping(
        config["map"],
        label="map",
        required={
            "maximum_surface_tilt_from_lateral_deg",
            "maximum_antipodal_normal_error_deg",
            "minimum_surface_area_m2",
            "contact_edge_margin_m",
            "roll_bounds_rad",
        },
    )

    part_path = _resolve_part_path(part["path"])
    output_path = _resolve_output_path(config["output"], config_path=path)
    scale_to_m = _number(
        part["scale_to_m"], label="part.scale_to_m", minimum=0.0,
        minimum_inclusive=False,
    )
    T_W_P_insert = validate_transform(np.asarray(
        insertion["T_W_P_insert"], dtype=float))
    insertion_axis_P = np.asarray(_number_sequence(
        insertion["insertion_axis_P"], label="insertion.insertion_axis_P", length=3,
    ))
    axis_norm = float(np.linalg.norm(insertion_axis_P))
    if axis_norm <= 64.0 * np.finfo(float).eps:
        raise ValueError("insertion.insertion_axis_P must be nonzero")
    insertion_axis_P /= axis_norm

    opening_range = _number_sequence(
        gripper["opening_range_m"], label="gripper.opening_range_m", length=2,
    )
    if opening_range[0] < 0.0 or opening_range[1] <= opening_range[0]:
        raise ValueError(
            "gripper.opening_range_m must satisfy 0 <= minimum < maximum"
        )
    friction = _number(
        gripper["friction_coefficient"],
        label="gripper.friction_coefficient",
        minimum=0.0,
    )
    maximum_tilt_deg = _number(
        map_config["maximum_surface_tilt_from_lateral_deg"],
        label="map.maximum_surface_tilt_from_lateral_deg",
        minimum=0.0,
        maximum=90.0,
    )
    maximum_antipodal_error_deg = _number(
        map_config["maximum_antipodal_normal_error_deg"],
        label="map.maximum_antipodal_normal_error_deg",
        minimum=0.0,
        maximum=180.0,
    )
    minimum_area = _number(
        map_config["minimum_surface_area_m2"],
        label="map.minimum_surface_area_m2",
        minimum=0.0,
    )
    edge_margin = _number(
        map_config["contact_edge_margin_m"],
        label="map.contact_edge_margin_m",
        minimum=0.0,
    )
    roll_bounds = _number_sequence(
        map_config["roll_bounds_rad"], label="map.roll_bounds_rad", length=2,
    )
    if roll_bounds[1] < roll_bounds[0]:
        raise ValueError("map.roll_bounds_rad must be ordered [minimum, maximum]")

    mesh = load_scaled_binary_stl(part_path, scale_to_m=scale_to_m)
    grasp_map = generate_two_finger_grasp_map(
        mesh,
        T_W_P_insert=T_W_P_insert,
        insertion_axis_P=insertion_axis_P,
        opening_range_m=opening_range,
        friction_coefficient=friction,
        roll_bounds_rad=roll_bounds,
        maximum_surface_tilt_from_lateral_rad=np.deg2rad(maximum_tilt_deg),
        maximum_antipodal_normal_error_rad=np.deg2rad(
            maximum_antipodal_error_deg),
        minimum_surface_area_m2=minimum_area,
        contact_edge_margin_m=edge_margin,
    )
    document = grasp_map.to_dict()
    # Replace the machine-specific absolute mesh source stored by the loader
    # with the repository-relative identity used by the config/provenance.
    part_relative = _relative_to_root(part_path, label="part.path")
    document["inputs"]["mesh_source"] = part_relative
    document["claim_level"] = CLAIM_LEVEL
    document["claim_definition"] = (
        "continuous ideal point-contact families on local outward-wound "
        "part facets; directional external visibility is not certified"
    )
    document["provenance"] = {
        "config": {
            "path": _relative_to_root(path, label="config path"),
            "sha256": _sha256(path),
        },
        "part": {
            "path": part_relative,
            "sha256": _sha256(part_path),
            "scale_to_m": scale_to_m,
        },
        "generator": {
            "path": _relative_to_root(Path(__file__), label="generator path"),
            "sha256": _sha256(Path(__file__)),
        },
        "implementation": {
            "path": _relative_to_root(
                Path(grasp_map_module.__file__), label="implementation path"),
            "sha256": _sha256(Path(grasp_map_module.__file__)),
        },
    }
    return output_path, document


def _write_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        document,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    path.write_text(payload, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"strict YAML configuration (default: {DEFAULT_CONFIG.relative_to(ROOT)})",
    )
    arguments = parser.parse_args(argv)
    output, document = generate_from_config(arguments.config)
    _write_json(output, document)
    print(
        f"Wrote {len(document['families'])} continuous grasp families to "
        f"{output.relative_to(ROOT)}"
    )
    print(f"Claim level: {CLAIM_LEVEL}; insertion_safe=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
