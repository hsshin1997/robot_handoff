#!/usr/bin/env python3
"""Prepare every CAD reference in ``mujoco_sim/project.yaml`` for MuJoCo.

Outputs live under ``mujoco_sim/models/generated_cad/<content-digest>/``.
The index and per-asset metadata are written atomically.  STL units are never
guessed: each manifest entry must provide ``*_units`` or ``*_scale_to_m``.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mujoco_sim.cad_preprocess import (  # noqa: E402
    COLLISION_WARNING,
    DEFAULT_MAX_FACES_PER_CHUNK,
    CADPreparation,
    prepare_cad,
)
from mujoco_sim.offline import (  # noqa: E402
    atomic_write_json,
    canonical_json_bytes,
    fingerprint_content,
    fingerprint_file,
)


DEFAULT_PROJECT = ROOT / "mujoco_sim" / "project.yaml"
DEFAULT_GENERATED = ROOT / "mujoco_sim" / "models" / "generated_cad"
_CAD_SUFFIXES = {".stl", ".obj", ".step", ".stp"}


@dataclass(frozen=True)
class ProjectCADReference:
    name: str
    path: str
    units: str | None
    scale_to_m: float | tuple[float, float, float] | None
    role: str
    static_assembly: bool


def _manifest(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise TypeError("project manifest root must be a mapping")
    return value


def _reference(
    name: str,
    value: Any,
    *,
    units: Any,
    scale_to_m: Any,
    role: str,
    static_assembly: bool,
) -> ProjectCADReference | None:
    if isinstance(value, dict):
        if "path" not in value:
            raise ValueError(f"CAD mapping {name!r} must contain a path")
        path = value["path"]
        units = value.get("units", units)
        scale_to_m = value.get("scale_to_m", scale_to_m)
        role = value.get("role", role)
        static_assembly = bool(value.get("static_assembly", static_assembly))
    else:
        path = value
    if not isinstance(path, str):
        raise TypeError(f"CAD path {name!r} must be a string")
    if Path(path).suffix.lower() not in _CAD_SUFFIXES:
        return None
    if units is None and scale_to_m is None:
        raise ValueError(
            f"CAD entry {name!r} ({path}) has no explicit units. Add units: mm/m "
            f"inside a path mapping, or add the sibling field for this entry "
            f"(for example cad_units/model_units/visual_cad_units)."
        )
    if isinstance(scale_to_m, list):
        scale_to_m = tuple(float(item) for item in scale_to_m)
    return ProjectCADReference(name, path, units, scale_to_m, role, static_assembly)


def discover_project_cad(manifest: dict[str, Any]) -> tuple[ProjectCADReference, ...]:
    """Extract supported CAD entries while preserving explicit unit policy."""
    result: list[ProjectCADReference] = []

    workstation = manifest.get("workstation", {})
    if "visual_cad" in workstation:
        item = _reference(
            "workstation.visual_cad", workstation["visual_cad"],
            units=workstation.get("visual_cad_units"),
            scale_to_m=workstation.get("visual_cad_scale_to_m"),
            role="workstation-visual", static_assembly=bool(
                workstation.get("visual_cad_static_assembly", True)
            ),
        )
        if item:
            result.append(item)

    if "collision_cad" in workstation:
        item = _reference(
            "workstation.collision_cad", workstation["collision_cad"],
            units=workstation.get("collision_cad_units"),
            scale_to_m=workstation.get("collision_cad_scale_to_m"),
            role="collision-source", static_assembly=bool(
                workstation.get("collision_cad_static_assembly", True)
            ),
        )
        if item:
            result.append(item)

    additional = workstation.get("additional_collision_cad", [])
    if not isinstance(additional, list):
        raise TypeError("workstation.additional_collision_cad must be a list")
    default_additional_units = workstation.get("additional_collision_cad_units")
    for index, value in enumerate(additional):
        units = (default_additional_units[index]
                 if isinstance(default_additional_units, list) else default_additional_units)
        item = _reference(
            f"workstation.additional_collision_cad[{index}]", value,
            units=units, scale_to_m=None, role="collision-source", static_assembly=True,
        )
        if item:
            result.append(item)

    for name, gripper in sorted(manifest.get("grippers", {}).items()):
        item = _reference(
            f"grippers.{name}.model", gripper["model"],
            units=gripper.get("model_units"), scale_to_m=gripper.get("model_scale_to_m"),
            role="gripper-visual",
            static_assembly=bool(gripper.get(
                "model_static_assembly",
                gripper.get("kind") in ("parallel_jaw_static_fallback", "fixed"),
            )),
        )
        if item:
            result.append(item)

    for name, part in sorted(manifest.get("parts", {}).items()):
        item = _reference(
            f"parts.{name}.cad", part["cad"],
            units=part.get("cad_units"), scale_to_m=part.get("cad_scale_to_m"),
            role="part-visual", static_assembly=bool(part.get("cad_static_assembly", False)),
        )
        if item:
            result.append(item)

    insertion = manifest.get("insertion", {})
    if insertion.get("collision_cad"):
        item = _reference(
            "insertion.collision_cad", insertion["collision_cad"],
            units=insertion.get("collision_cad_units"),
            scale_to_m=insertion.get("collision_cad_scale_to_m"),
            role="insertion-collision", static_assembly=True,
        )
        if item:
            result.append(item)

    # A project may use a direct CAD robot model instead of URDF/MJCF.
    for name, robot in sorted(manifest.get("robots", {}).items()):
        item = _reference(
            f"robots.{name}.model", robot["model"],
            units=robot.get("model_units"), scale_to_m=robot.get("model_scale_to_m"),
            role="robot-visual", static_assembly=bool(robot.get("model_static_assembly", False)),
        )
        if item:
            result.append(item)

    return tuple(sorted(result, key=lambda item: item.name))


def prepare_project(
    project_path: str | os.PathLike[str] = DEFAULT_PROJECT,
    generated_dir: str | os.PathLike[str] = DEFAULT_GENERATED,
    *,
    project_root: str | os.PathLike[str] = ROOT,
    max_faces: int = DEFAULT_MAX_FACES_PER_CHUNK,
    freecad_executable: str | os.PathLike[str] | None = None,
    linear_deflection_mm: float = 0.05,
    angular_deflection_deg: float = 5.0,
) -> dict[str, Any]:
    project_path = Path(project_path).resolve()
    project_root = Path(project_root).resolve()
    generated_dir = Path(generated_dir).resolve()
    manifest = _manifest(project_path)
    references = discover_project_cad(manifest)

    prepared: list[tuple[ProjectCADReference, CADPreparation]] = []
    for reference in references:
        source = Path(reference.path)
        if not source.is_absolute():
            source = project_root / source
        output = prepare_cad(
            source,
            generated_dir,
            units=reference.units,
            scale_to_m=reference.scale_to_m,
            role=reference.role,
            static_assembly=reference.static_assembly,
            max_faces=max_faces,
            freecad_executable=freecad_executable,
            linear_deflection_mm=linear_deflection_mm,
            angular_deflection_deg=angular_deflection_deg,
        )
        prepared.append((reference, output))

    entries = []
    warnings = set()
    for reference, output in prepared:
        metadata = output.metadata
        warnings.update(metadata["warnings"])
        entries.append({
            "name": reference.name,
            "source": reference.path,
            "role": reference.role,
            "static_assembly": reference.static_assembly,
            "artifact_fingerprint": metadata["artifact_fingerprint"],
            "metadata": output.metadata_path.relative_to(generated_dir).as_posix(),
            "visual_chunks": [
                f"{metadata['artifact_fingerprint']}/{chunk['path']}"
                for chunk in metadata["visual"]["chunks"]
            ],
            "mesh_scale_to_m": metadata["source"]["scale_to_m"],
        })
    entries.sort(key=lambda item: item["name"])
    project_identity = fingerprint_content({
        "manifest": manifest,
        "artifacts": [item["artifact_fingerprint"] for item in entries],
    })
    index = {
        "schema_version": 1,
        "project_file": project_path.name,
        "project_source_sha256": fingerprint_file(project_path),
        "project_fingerprint": project_identity,
        "exact_visual_preservation": True,
        "visual_downsampling": False,
        "collision_warning": COLLISION_WARNING,
        "entries": entries,
        "warnings": sorted(warnings),
    }
    index["index_content_sha256"] = fingerprint_content(index)
    atomic_write_json(generated_dir / "index.json", index)
    return index


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=str(DEFAULT_PROJECT))
    parser.add_argument("--project-root", default=str(ROOT))
    parser.add_argument("--generated-dir", default=str(DEFAULT_GENERATED))
    parser.add_argument("--max-faces", type=int, default=DEFAULT_MAX_FACES_PER_CHUNK)
    parser.add_argument("--freecad", default=None,
                        help="absolute FreeCADCmd/freecadcmd path (or set FREECADCMD)")
    parser.add_argument("--linear-deflection-mm", type=float, default=0.05)
    parser.add_argument("--angular-deflection-deg", type=float, default=5.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    index = prepare_project(
        args.project,
        args.generated_dir,
        project_root=args.project_root,
        max_faces=args.max_faces,
        freecad_executable=args.freecad,
        linear_deflection_mm=args.linear_deflection_mm,
        angular_deflection_deg=args.angular_deflection_deg,
    )
    if args.json:
        sys.stdout.buffer.write(canonical_json_bytes(index) + b"\n")
    else:
        print(f"project CAD fingerprint: {index['project_fingerprint']}")
        print(f"assets prepared: {len(index['entries'])}")
        for item in index["entries"]:
            print(f"  {item['name']}: {item['artifact_fingerprint']} "
                  f"({len(item['visual_chunks'])} visual chunk(s))")
        print(f"index: {Path(args.generated_dir) / 'index.json'}")
        print(f"warning: {COLLISION_WARNING}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
