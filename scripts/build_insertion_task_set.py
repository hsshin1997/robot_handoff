#!/usr/bin/env python3
"""Build the robot-independent (u, v, roll) insertion task-set artifact.

The phase-1 pose library is consumed only as sampled evidence.  The output is
a continuous parameter-cell cover with explicit SAFE/REJECTED/UNRESOLVED
claims and optional positive penetration witnesses against the supplied finite
PCB mesh.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mujoco_sim.modeling.insertion_grasps import (  # noqa: E402
    load_scaled_binary_stl,
)
from mujoco_sim.modeling.insertion_task_set import (  # noqa: E402
    ContactMode,
    FinitePCBFootprint,
    GripperComponentVertices,
    SampledGripperGeometry,
    apply_whole_cell_task_certificates,
    artifact_sha256,
    build_task_set_document,
    sha256_file,
)
from mujoco_sim.offline_tools.artifacts import atomic_write_json  # noqa: E402


DEFAULT_CONFIG = (
    ROOT / "projects" / "connector_header_insertion"
    / "config" / "task_set.yaml"
)


def _load_yaml(path: Path, *, label: str) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a YAML mapping")
    return value


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _resolve(value: str | Path, owner: Path, *, must_exist: bool = True) -> Path:
    supplied = Path(value).expanduser()
    candidates = (
        [supplied] if supplied.is_absolute()
        else [owner.parent / supplied, ROOT / supplied]
    )
    for candidate in candidates:
        path = candidate.resolve()
        if not must_exist or path.exists():
            return path
    raise FileNotFoundError(f"could not resolve {value!r} from {owner}")


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _verified_asset(value: Mapping[str, Any], owner: Path) -> Path:
    if "path" not in value:
        raise ValueError("asset entry requires path")
    path = _resolve(str(value["path"]), owner)
    actual = sha256_file(path)
    expected = str(value.get("sha256", actual))
    if actual != expected:
        raise ValueError(
            f"asset hash mismatch for {path}: expected {expected}, got {actual}"
        )
    return path


def _strict_bool(value: Any, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be boolean")
    return value


def _sample_unique_vertices(
    triangles: np.ndarray,
    maximum: int,
) -> tuple[np.ndarray, int]:
    if maximum <= 0:
        raise ValueError("maximum_vertices_per_component must be positive")
    unique = np.unique(np.asarray(triangles, dtype=float).reshape(-1, 3), axis=0)
    source_count = len(unique)
    if source_count <= maximum:
        return unique, source_count
    # Lexicographically sorted unique vertices plus uniform indices make this
    # subset deterministic across runs and independent of triangle ordering.
    indices = np.floor(np.linspace(0, source_count, maximum, endpoint=False)).astype(int)
    return unique[indices], source_count


def _sampled_gripper(
    grasp_config: Mapping[str, Any],
    grasp_config_path: Path,
    *,
    maximum_vertices_per_component: int,
) -> tuple[SampledGripperGeometry, dict[str, Any]]:
    try:
        scale = float(grasp_config["assets"]["stl_scale_to_m"])
        gripper = grasp_config["gripper"]
        component_values = gripper["components"]
    except KeyError as error:
        raise ValueError(
            f"grasp-generation config is missing {error.args[0]}") from error
    components = []
    provenance: dict[str, Any] = {}
    for component in component_values:
        path = _verified_asset(component["asset"], grasp_config_path)
        mesh = load_scaled_binary_stl(path, scale_to_m=scale)
        vertices, source_count = _sample_unique_vertices(
            mesh.triangles, maximum_vertices_per_component)
        name = str(component["name"])
        components.append(GripperComponentVertices(
            name=name,
            vertices_C_m=vertices,
            T_G_C_reference=np.asarray(
                component["T_G_C_reference"], dtype=float),
            aperture_multiplier=float(component["aperture_multiplier"]),
            source_unique_vertex_count=source_count,
        ))
        provenance[name] = {
            "path": _display_path(path),
            "sha256": sha256_file(path),
            "triangle_count": int(len(mesh.triangles)),
            "source_unique_vertex_count": source_count,
            "sampled_vertex_count": int(len(vertices)),
        }
    return SampledGripperGeometry(
        T_G_E=np.asarray(gripper["T_G_E"], dtype=float),
        reference_aperture_m=float(gripper["reference_aperture_m"]),
        opening_axis_G=np.asarray(gripper["opening_axis_G"], dtype=float),
        components=tuple(components),
    ), provenance


def _input_record(path: Path) -> dict[str, Any]:
    return {"path": _display_path(path), "sha256": sha256_file(path)}


def build_from_config(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    output_path: str | Path | None = None,
    finite_pcb_override: bool | None = None,
) -> tuple[dict[str, Any], Path]:
    """Build one artifact and return the document and destination."""
    source = Path(config_path).resolve()
    config = _load_yaml(source, label="task-set config")
    if int(config.get("schema_version", 0)) != 1:
        raise ValueError("task-set config schema_version must be 1")
    try:
        input_config = config["inputs"]
        pose_path = _resolve(input_config["pose_library"], source)
        grasp_path = _resolve(input_config["grasp_generation_config"], source)
        socket_path = _resolve(input_config["pcb_socket"], source)
        parameterization = config["parameterization"]
    except KeyError as error:
        raise ValueError(f"task-set config is missing {error.args[0]}") from error
    pose_library = _load_json(pose_path, label="pose library")
    grasp_config = _load_yaml(grasp_path, label="grasp-generation config")
    socket = _load_yaml(socket_path, label="PCB socket contract")
    mode_values = parameterization.get("contact_modes")
    if not isinstance(mode_values, list) or not mode_values:
        raise ValueError("parameterization.contact_modes must be a non-empty list")
    modes = tuple(ContactMode.from_mapping(value) for value in mode_values)

    try:
        capability = grasp_config["gripper"]["capability"]
        raw_opening = np.asarray(capability["opening_range_m"], dtype=float)
        reserve = float(capability["limit_reserve_m"])
        pad_size = np.asarray(capability["pad_size_m"], dtype=float)
    except KeyError as error:
        raise ValueError(
            f"grasp-generation config is missing {error.args[0]}") from error
    usable_opening = np.array([
        raw_opening[0] + reserve,
        raw_opening[1] - reserve,
    ])
    if usable_opening[0] >= usable_opening[1]:
        raise ValueError("gripper limit reserve consumes its opening range")

    finite_config = config.get("finite_pcb_witnesses", {})
    if not isinstance(finite_config, dict):
        raise ValueError("finite_pcb_witnesses must be a mapping")
    configured_finite = _strict_bool(
        finite_config.get("enabled", False),
        label="finite_pcb_witnesses.enabled",
    )
    finite_enabled = (
        configured_finite if finite_pcb_override is None
        else bool(finite_pcb_override)
    )
    finite_bundle = None
    asset_provenance: dict[str, Any] = {}
    if finite_enabled:
        maximum_vertices = int(finite_config["maximum_vertices_per_component"])
        gripper_geometry, gripper_provenance = _sampled_gripper(
            grasp_config,
            grasp_path,
            maximum_vertices_per_component=maximum_vertices,
        )
        pcb_path = _verified_asset(socket["assets"]["pcb"], socket_path)
        pcb_scale = float(socket["assets"]["pcb"]["scale_to_m"])
        pcb_mesh = load_scaled_binary_stl(pcb_path, scale_to_m=pcb_scale)
        footprint = FinitePCBFootprint(
            pcb_mesh.triangles,
            top_surface_tolerance_m=float(
                finite_config["pcb_top_surface_tolerance_m"]),
            spatial_bin_size_m=float(
                finite_config["pcb_spatial_bin_size_m"]),
        )
        finite_bundle = (
            footprint,
            gripper_geometry,
            {
                "maximum_representatives": int(
                    finite_config["maximum_representatives"]),
                "path_samples": int(finite_config["path_samples"]),
                "interior_tolerance_m": float(
                    finite_config["pcb_interior_tolerance_m"]),
            },
        )
        asset_provenance = {
            "pcb": {
                "path": _display_path(pcb_path),
                "sha256": sha256_file(pcb_path),
                "triangle_count": int(len(pcb_mesh.triangles)),
                "top_triangle_count": footprint.top_triangle_count,
            },
            "gripper_components": gripper_provenance,
        }

    module_path = ROOT / "mujoco_sim" / "modeling" / "insertion_task_set.py"
    document = build_task_set_document(
        project_id=str(config["project_id"]),
        modes=modes,
        pose_library=pose_library,
        socket_contract=socket,
        pad_size_m=pad_size,
        usable_opening_range_m=usable_opening,
        minimum_closing_alignment=float(
            parameterization["minimum_seed_closing_alignment"]),
        input_provenance={
            "task_set_config": _input_record(source),
            "pose_library": _input_record(pose_path),
            "grasp_generation_config": _input_record(grasp_path),
            "pcb_socket": _input_record(socket_path),
            "generator": _input_record(Path(__file__).resolve()),
            "modeling_module": _input_record(module_path),
            "assets_used_by_finite_pcb_witnesses": asset_provenance,
        },
        finite_pcb=finite_bundle,
    )
    certificate_config = config.get("whole_cell_task_certificates")
    if not isinstance(certificate_config, dict):
        raise ValueError(
            "task-set config requires whole_cell_task_certificates mapping")
    required_proofs = certificate_config.get("required_proved_constraints")
    imports = certificate_config.get("imports")
    if not isinstance(required_proofs, list):
        raise ValueError(
            "whole_cell_task_certificates.required_proved_constraints "
            "must be a list")
    if not isinstance(imports, list):
        raise ValueError("whole_cell_task_certificates.imports must be a list")
    certificate_imports = []
    for index, import_value in enumerate(imports):
        if not isinstance(import_value, dict):
            raise ValueError(f"certificate import {index} must be a mapping")
        if "path" not in import_value or "sha256" not in import_value:
            raise ValueError(
                f"certificate import {index} requires path and sha256")
        certificate_path = _resolve(import_value["path"], source)
        expected_sha = str(import_value["sha256"])
        actual_sha = sha256_file(certificate_path)
        certificate_imports.append({
            "path": _display_path(certificate_path),
            "expected_sha256": expected_sha,
            "actual_sha256": actual_sha,
            "document": _load_json(
                certificate_path, label=f"whole-cell certificate {index}"),
        })
    apply_whole_cell_task_certificates(
        document,
        certificate_imports,
        required_proved_constraints=required_proofs,
    )
    document["semantic_sha256"] = artifact_sha256(document)

    if output_path is not None:
        destination = Path(output_path).expanduser().resolve()
    else:
        if "output" not in config:
            raise ValueError("task-set config requires output")
        destination = _resolve(config["output"], source, must_exist=False)
    atomic_write_json(destination, document)
    return document, destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG,
        help="task-set YAML (default: project config/task_set.yaml)",
    )
    parser.add_argument("--output", type=Path, help="override output JSON")
    finite = parser.add_mutually_exclusive_group()
    finite.add_argument(
        "--finite-pcb-witnesses", action="store_true", dest="finite_pcb",
        help="force bounded finite-PCB representative checks on",
    )
    finite.add_argument(
        "--no-finite-pcb-witnesses", action="store_false", dest="finite_pcb",
        help="skip expensive finite-PCB representative checks",
    )
    parser.set_defaults(finite_pcb=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    document, output = build_from_config(
        args.config,
        output_path=args.output,
        finite_pcb_override=args.finite_pcb,
    )
    print(json.dumps({
        "output": str(output),
        "semantic_sha256": document["semantic_sha256"],
        "counts": document["counts"],
        "finite_pcb_witness_evaluation": (
            document["finite_pcb_witness_evaluation"]),
        "certified_safe_set_available": document[
            "certification_boundary"]["certified_safe_set_available"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
