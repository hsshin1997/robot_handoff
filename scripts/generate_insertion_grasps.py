#!/usr/bin/env python3
"""Generate the connector-header phase-1 insertion grasp pose library."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mujoco_sim.core.se3 import validate_transform  # noqa: E402
from mujoco_sim.modeling.grasps import ParallelJawGripper  # noqa: E402
from mujoco_sim.modeling.insertion_grasps import (  # noqa: E402
    AxisAlignedRegion,
    FreeSpacePlane,
    GripperMeshComponent,
    GripperMeshModel,
    InsertionGraspEvaluation,
    InsertionTaskGeometry,
    generate_insertion_grasps,
    load_scaled_binary_stl,
)


DEFAULT_CONFIG = (
    ROOT / "projects" / "connector_header_insertion"
    / "config" / "grasp_generation.yaml"
)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def _resolve_asset(value: str, owner: Path) -> Path:
    supplied = Path(value)
    candidates = [supplied] if supplied.is_absolute() else [ROOT / supplied, owner.parent / supplied]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"asset {value!r} was not found")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_asset(value: dict[str, Any], owner: Path) -> Path:
    path = _resolve_asset(str(value["path"]), owner)
    actual = _sha256(path)
    expected = str(value.get("sha256", actual))
    if actual != expected:
        raise ValueError(
            f"asset hash mismatch for {path}: expected {expected}, got {actual}"
        )
    return path


def _matrix(value: Any, label: str) -> np.ndarray:
    try:
        return validate_transform(np.asarray(value, dtype=float))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is not a valid 4x4 SE(3) matrix: {error}") from error


def _rounded(value: float) -> float:
    result = round(float(value), 12)
    return 0.0 if result == -0.0 else result


def _array(value: np.ndarray) -> list:
    array = np.asarray(value, dtype=float)
    return np.vectorize(_rounded, otypes=[float])(array).tolist()


def _mesh_stats(mesh, path: Path, scale_to_m: float) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": _sha256(path),
        "scale_to_m": scale_to_m,
        "triangle_count": int(len(mesh.triangles)),
        "bounds_min_m": _array(mesh.bounds_min),
        "bounds_max_m": _array(mesh.bounds_max),
        "extent_m": _array(mesh.extent),
    }


def _candidate_record(
    index: int,
    evaluation: InsertionGraspEvaluation,
    T_I_P: np.ndarray,
) -> dict[str, Any]:
    candidate = evaluation.candidate
    T_I_E = T_I_P @ candidate.T_P_E
    identity_payload = json.dumps(
        {
            "T_P_E": np.round(candidate.T_P_E, 9).tolist(),
            "contacts_P_m": np.round(candidate.contact_points, 9).tolist(),
            "required_aperture_m": round(candidate.required_opening, 9),
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    pose_id = "g_" + hashlib.sha256(identity_payload).hexdigest()[:16]
    return {
        "id": pose_id,
        "library_index": index,
        "status": evaluation.status,
        "claim_level": "phase1_geometric_candidate"
        if evaluation.preinsert_compatible else "rejected",
        "family": evaluation.family,
        "preinsert_compatible": evaluation.preinsert_compatible,
        "seated_compatible": evaluation.seated_compatible,
        "contacts_in_graspable_region": evaluation.contacts_in_graspable_region,
        "T_P_E": _array(candidate.T_P_E),
        "T_I_E_at_seated_part_pose": _array(T_I_E),
        "contact_points_P_m": _array(candidate.contact_points),
        "contact_normals_P": _array(candidate.contact_normals),
        "closing_direction_P": _array(candidate.closing_direction),
        "approach_direction_P": _array(candidate.approach_direction),
        "required_aperture_m": _rounded(candidate.required_opening),
        "quality": _rounded(candidate.quality),
        "antipodal_quality": _rounded(candidate.antipodal_quality),
        "support_quality": _rounded(candidate.support_quality),
        "opening_margin": _rounded(candidate.opening_margin),
        "idealized_palm_clearance_m": _rounded(candidate.palm_clearance),
        "seated_pcb_clearance_m": None if evaluation.seated_pcb_clearance_m is None
        else _rounded(evaluation.seated_pcb_clearance_m),
        "preinsert_pcb_clearance_m": None if evaluation.preinsert_pcb_clearance_m is None
        else _rounded(evaluation.preinsert_pcb_clearance_m),
        "collision_free_insertion_travel_m": None
        if evaluation.collision_free_insertion_travel_m is None
        else _rounded(evaluation.collision_free_insertion_travel_m),
        "remaining_to_seat_at_collision_m": None
        if evaluation.remaining_to_seat_at_collision_m is None
        else _rounded(evaluation.remaining_to_seat_at_collision_m),
        "limiting_gripper_component": evaluation.limiting_component,
        "component_seated_clearances_m": {
            key: _rounded(value)
            for key, value in sorted(evaluation.component_clearances_m.items())
        },
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _write_pose_table(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "id", "library_index", "status", "family", "preinsert_task_rank", "seated_task_rank",
        "preinsert_compatible", "seated_compatible",
        "required_aperture_m", "quality", "seated_pcb_clearance_m",
        "preinsert_pcb_clearance_m", "collision_free_insertion_travel_m",
        "remaining_to_seat_at_collision_m", "limiting_gripper_component",
        *[f"T_P_E_{row}{column}" for row in range(4) for column in range(4)],
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = {name: record.get(name) for name in fields}
            transform = record["T_P_E"]
            for r_index in range(4):
                for c_index in range(4):
                    row[f"T_P_E_{r_index}{c_index}"] = transform[r_index][c_index]
            writer.writerow(row)


def _write_insertion_svg(
    path: Path,
    part_mesh,
    *,
    seating_y_m: float,
) -> None:
    """Write the user's Y-Z side view with insertion direction downward."""
    width, height = 900, 560
    margin = 80.0
    triangles = part_mesh.triangles
    horizontal = triangles[:, :, 2]  # +Z_P is right on the page.
    vertical = -triangles[:, :, 1]   # -Y_P is down on the page.
    x_min, x_max = float(np.min(horizontal)), float(np.max(horizontal))
    y_min, y_max = float(np.min(vertical)), float(np.max(vertical))
    span_x = max(x_max - x_min, 1e-9)
    span_y = max(y_max - y_min, 1e-9)
    scale = min((width - 2 * margin) / span_x, (height - 2 * margin) / span_y)

    def project(x: float, y: float) -> tuple[float, float]:
        return (
            margin + (x - x_min) * scale,
            margin + (y - y_min) * scale,
        )

    depth = np.mean(triangles[:, :, 0], axis=1)
    order = np.argsort(depth)
    polygons = []
    depth_span = max(float(np.ptp(depth)), 1e-12)
    for triangle_index in order:
        points = [
            project(horizontal[triangle_index, vertex], vertical[triangle_index, vertex])
            for vertex in range(3)
        ]
        shade = int(100 + 85 * (depth[triangle_index] - float(np.min(depth))) / depth_span)
        point_text = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        polygons.append(
            f'<polygon points="{point_text}" fill="rgb({shade},{shade + 15},{shade + 20})" '
            'fill-opacity="0.42" stroke="none"/>'
        )

    _, board_y = project(x_min, -seating_y_m)
    arrow_x = width - 150
    arrow_top = max(65.0, board_y - 145.0)
    arrow_bottom = board_y - 25.0
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f7f8fa"/>',
        '<text x="40" y="38" font-family="sans-serif" font-size="24" '
        'font-weight="700" fill="#17202a">Connector insertion definition — Y-Z side view</text>',
        *polygons,
        f'<line x1="35" y1="{board_y:.2f}" x2="865" y2="{board_y:.2f}" '
        'stroke="#1b7f4b" stroke-width="5"/>',
        f'<text x="42" y="{board_y - 12:.2f}" font-family="sans-serif" '
        'font-size="18" fill="#14643b">PCB seating plane (Y_P = 3.2526 mm)</text>',
        f'<line x1="{arrow_x}" y1="{arrow_top:.2f}" x2="{arrow_x}" '
        f'y2="{arrow_bottom:.2f}" stroke="#cf3f35" stroke-width="6"/>',
        f'<polygon points="{arrow_x - 12},{arrow_bottom - 18:.2f} '
        f'{arrow_x + 12},{arrow_bottom - 18:.2f} {arrow_x},{arrow_bottom + 4:.2f}" '
        'fill="#cf3f35"/>',
        f'<text x="{arrow_x - 95}" y="{arrow_top - 12:.2f}" font-family="sans-serif" '
        'font-size="18" fill="#a52f28">insertion = −Y_P</text>',
        '<text x="40" y="530" font-family="sans-serif" font-size="16" fill="#46505a">'
        'Page right = +Z_P (long mating-post direction); page down = −Y_P (short PCB-tail direction).</text>',
        '</svg>',
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def _write_summary_svg(path: Path, counts: Counter, family_counts: Counter) -> None:
    statuses = [
        ("phase1_seated_geometric_candidate", "#26734d"),
        ("phase1_preinsert_only_candidate", "#d88919"),
        ("rejected_preinsert_pcb_collision", "#c65345"),
        ("rejected_contact_region", "#7d8790"),
    ]
    width = 940
    rows = len(statuses) + len(family_counts) + 5
    height = max(460, 55 + rows * 34)
    maximum = max([counts[key] for key, _ in statuses] + [1])
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f7f8fa"/>',
        '<text x="34" y="38" font-family="sans-serif" font-size="24" '
        'font-weight="700" fill="#17202a">Phase-1 insertion grasp results</text>',
    ]
    y = 78
    for label, color in statuses:
        count = counts[label]
        bar = 460.0 * count / maximum
        lines.extend([
            f'<text x="35" y="{y + 20}" font-family="monospace" font-size="15" '
            f'fill="#26323c">{label}</text>',
            f'<rect x="405" y="{y}" width="{bar:.2f}" height="24" rx="3" fill="{color}"/>',
            f'<text x="{415 + bar:.2f}" y="{y + 19}" font-family="sans-serif" '
            f'font-size="16" fill="#26323c">{count}</text>',
        ])
        y += 38
    y += 22
    lines.append(
        f'<text x="35" y="{y}" font-family="sans-serif" font-size="20" '
        'font-weight="700" fill="#17202a">Pre-insert-compatible orientation families</text>'
    )
    y += 24
    for family, count in sorted(family_counts.items()):
        lines.append(
            f'<text x="52" y="{y}" font-family="monospace" font-size="15" '
            f'fill="#26323c">{family}: {count}</text>'
        )
        y += 27
    lines.extend([
        f'<text x="35" y="{height - 34}" font-family="sans-serif" font-size="15" '
        'fill="#59636d">Geometric screen only; robot IK, complete collision, pad capture, and insertion-force checks are pending.</text>',
        '</svg>',
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = _load_yaml(config_path)
    if int(config.get("schema_version", 0)) != 1:
        raise ValueError("schema_version must be 1")

    scale_to_m = float(config["assets"]["stl_scale_to_m"])
    part_path = _verified_asset(config["assets"]["part"], config_path)
    part_mesh = load_scaled_binary_stl(part_path, scale_to_m=scale_to_m)

    gripper_config = config["gripper"]
    capability_config = gripper_config["capability"]
    opening_min, opening_max = (
        float(value) for value in capability_config["opening_range_m"]
    )
    limit_reserve = float(capability_config.get("limit_reserve_m", 0.0))
    if opening_max - opening_min <= 2.0 * limit_reserve:
        raise ValueError("gripper limit reserve consumes the opening range")
    capability = ParallelJawGripper(
        min_opening=opening_min + limit_reserve,
        max_opening=opening_max - limit_reserve,
        pad_size=tuple(float(value) for value in capability_config["pad_size_m"]),
        pad_depth=float(capability_config["pad_depth_m"]),
        friction_coefficient=float(capability_config["friction_coefficient"]),
    )

    components = []
    asset_stats: dict[str, Any] = {
        "part": _mesh_stats(part_mesh, part_path, scale_to_m),
    }
    for component_config in gripper_config["components"]:
        path = _verified_asset(component_config["asset"], config_path)
        mesh = load_scaled_binary_stl(path, scale_to_m=scale_to_m)
        component = GripperMeshComponent(
            name=str(component_config["name"]),
            mesh_C=mesh,
            T_G_C_reference=_matrix(
                component_config["T_G_C_reference"],
                f"gripper component {component_config['name']} T_G_C_reference",
            ),
            aperture_multiplier=float(component_config["aperture_multiplier"]),
        )
        components.append(component)
        stats = _mesh_stats(mesh, path, scale_to_m)
        stats["unique_vertex_count"] = component.unique_vertex_count
        asset_stats[component.name] = stats

    full_assembly_path = _verified_asset(
        gripper_config["reference_full_assembly"], config_path,
    )
    full_assembly = load_scaled_binary_stl(
        full_assembly_path, scale_to_m=scale_to_m,
    )
    asset_stats["reference_full_assembly"] = _mesh_stats(
        full_assembly, full_assembly_path, scale_to_m,
    )

    mesh_model = GripperMeshModel(
        T_G_E=_matrix(gripper_config["T_G_E"], "gripper T_G_E"),
        reference_aperture_m=float(gripper_config["reference_aperture_m"]),
        opening_axis_G=np.asarray(gripper_config["opening_axis_G"], dtype=float),
        components=tuple(components),
    )

    task_config = config["task"]
    plane_config = task_config["pcb_plane_P"]
    regions = tuple(
        AxisAlignedRegion(
            name=str(region["name"]),
            minimum_P_m=np.asarray(region["minimum_P_m"], dtype=float),
            maximum_P_m=np.asarray(region["maximum_P_m"], dtype=float),
        )
        for region in task_config["graspable_contact_regions"]
    )
    task = InsertionTaskGeometry(
        insertion_axis_P=np.asarray(task_config["insertion_axis_P"], dtype=float),
        pcb_plane_P=FreeSpacePlane(
            normal_P=np.asarray(plane_config["free_space_normal_P"], dtype=float),
            offset_P_m=float(plane_config["offset_P_m"]),
        ),
        graspable_regions=regions,
        contact_region_tolerance_m=float(task_config["contact_region_tolerance_m"]),
        preinsert_distance_m=float(task_config["preinsert_distance_m"]),
        minimum_pcb_clearance_m=float(task_config["minimum_pcb_clearance_m"]),
    )
    sampling = config["sampling"]
    evaluations = generate_insertion_grasps(
        part_mesh,
        capability,
        task=task,
        gripper_mesh=mesh_model,
        surface_samples=int(sampling["surface_samples"]),
        approaches_per_pair=int(sampling["approaches_per_pair"]),
        max_candidates=int(sampling["max_candidates"]),
    )

    T_I_P = _matrix(task_config["T_I_P"], "task T_I_P")
    records = [
        _candidate_record(index, evaluation, T_I_P)
        for index, evaluation in enumerate(evaluations)
    ]
    if len({record["id"] for record in records}) != len(records):
        raise RuntimeError("content-addressed grasp ID collision")
    for record in records:
        record["preinsert_task_rank"] = None
        record["seated_task_rank"] = None
    counts = Counter(record["status"] for record in records)
    preinsert_records = [record for record in records if record["preinsert_compatible"]]
    seated_records = [record for record in records if record["seated_compatible"]]
    preinsert_ranked = sorted(
        preinsert_records,
        key=lambda record: (
            float(record["remaining_to_seat_at_collision_m"]),
            -float(record["quality"]),
            record["id"],
        ),
    )
    seated_ranked = sorted(
        seated_records,
        key=lambda record: (
            -float(record["seated_pcb_clearance_m"]),
            -float(record["quality"]),
            record["id"],
        ),
    )
    for rank, record in enumerate(preinsert_ranked, start=1):
        record["preinsert_task_rank"] = rank
    for rank, record in enumerate(seated_ranked, start=1):
        record["seated_task_rank"] = rank
    family_counts = Counter(record["family"] for record in preinsert_records)

    output_dir = _resolve_output(config["outputs"]["directory"], config_path)
    pose_path = output_dir / "grasps" / "phase1_pose_library.json"
    table_path = output_dir / "grasps" / "phase1_pose_table.csv"
    summary_path = output_dir / "reports" / "grasp_summary.json"
    insertion_svg_path = output_dir / "renders" / "insertion_definition.svg"
    summary_svg_path = output_dir / "renders" / "grasp_summary.svg"

    config_hash = _sha256(config_path)
    generator_hash = _sha256(Path(__file__).resolve())
    library = {
        "schema_version": 1,
        "project_id": str(config["project_id"]),
        "generator": "scripts/generate_insertion_grasps.py",
        "config_path": str(config_path.relative_to(ROOT)),
        "config_sha256": config_hash,
        "generator_sha256": generator_hash,
        "finite_sampling_contract": {
            "surface_samples": int(sampling["surface_samples"]),
            "approaches_per_pair": int(sampling["approaches_per_pair"]),
            "max_candidates": int(sampling["max_candidates"]),
            "claim": "resolution-qualified deterministic library; not exhaustive over continuous SE(3)",
        },
        "frame_contract": {
            "pose": "T_P_E maps ideal gripper contact frame E into connector STL frame P",
            "composition": "T_W_E = T_W_P @ T_P_E",
            "insertion": "T_I_P maps P into insertion frame I; +Z_I is insertion/down",
        },
        "task_geometry": {
            "insertion_axis_P": _array(task.insertion_axis_P),
            "T_I_P": _array(T_I_P),
            "pcb_free_space_normal_P": _array(task.pcb_plane_P.normal_P),
            "pcb_plane_offset_P_m": _rounded(task.pcb_plane_P.offset_P_m),
            "preinsert_distance_m": _rounded(task.preinsert_distance_m),
            "minimum_pcb_clearance_m": _rounded(task.minimum_pcb_clearance_m),
            "graspable_contact_regions": [
                {
                    "name": region.name,
                    "minimum_P_m": _array(region.minimum_P_m),
                    "maximum_P_m": _array(region.maximum_P_m),
                }
                for region in task.graspable_regions
            ],
        },
        "effective_gripper_capability": {
            "opening_range_m": [
                _rounded(capability.min_opening),
                _rounded(capability.max_opening),
            ],
            "pad_size_m": list(capability.pad_size),
            "pad_depth_m": _rounded(capability.pad_depth),
            "friction_coefficient": _rounded(capability.friction_coefficient),
            "reference_aperture_m": _rounded(mesh_model.reference_aperture_m),
        },
        "certification_boundary": config["certification_boundary"],
        "asset_stats": asset_stats,
        "counts": dict(sorted(counts.items())),
        "candidate_count": len(records),
        "housing_contact_candidate_count": sum(
            bool(record["contacts_in_graspable_region"]) for record in records
        ),
        "preinsert_compatible_count": len(preinsert_records),
        "seated_compatible_count": len(seated_records),
        "candidates": records,
    }
    _write_json(pose_path, library)
    _write_pose_table(table_path, records)

    clearances = [
        float(record["seated_pcb_clearance_m"])
        for record in records
        if record["seated_pcb_clearance_m"] is not None
    ]
    top_witness = None
    if preinsert_ranked:
        top_record = preinsert_ranked[0]
        raw_witness = mesh_model.plane_violation_witness(
            evaluations[int(top_record["library_index"])].candidate,
            task.pcb_plane_P,
            required_clearance_m=task.minimum_pcb_clearance_m,
        )
        top_witness = {
            "candidate_id": top_record["id"],
            "interpretation": (
                "Bounds are in connector frame P and identify where a finite "
                "PCB edge or cutout would be needed; this is not a clearance certificate."
            ),
            "components": {
                component_name: {
                    key: _array(value) if isinstance(value, np.ndarray) else value
                    for key, value in component_record.items()
                }
                for component_name, component_record in raw_witness.items()
            },
        }
    summary = {
        "schema_version": 1,
        "project_id": str(config["project_id"]),
        "config_sha256": config_hash,
        "generator_sha256": generator_hash,
        "counts_by_status": dict(sorted(counts.items())),
        "housing_contact_candidate_count": sum(
            bool(record["contacts_in_graspable_region"]) for record in records
        ),
        "preinsert_compatible_count": len(preinsert_records),
        "seated_compatible_count": len(seated_records),
        "preinsert_families": dict(sorted(family_counts.items())),
        "seated_clearance_range_m": None if not clearances else [
            _rounded(min(clearances)), _rounded(max(clearances)),
        ],
        "preinsert_candidate_ids_by_task_rank": [
            record["id"] for record in preinsert_ranked
        ],
        "seated_candidate_ids_by_task_rank": [
            record["id"] for record in seated_ranked
        ],
        "top_preinsert_candidates": [
            {
                "id": record["id"],
                "rank": record["preinsert_task_rank"],
                "family": record["family"],
                "required_aperture_m": record["required_aperture_m"],
                "remaining_to_seat_at_collision_m": record[
                    "remaining_to_seat_at_collision_m"
                ],
                "quality": record["quality"],
            }
            for record in preinsert_ranked[:20]
        ],
        "top_candidate_seated_plane_witness": top_witness,
        "provisional_assumptions": config["provisional_assumptions"],
        "next_required_gates": config["certification_boundary"]["not_checked"],
    }
    _write_json(summary_path, summary)
    _write_insertion_svg(
        insertion_svg_path,
        part_mesh,
        seating_y_m=float(plane_config["offset_P_m"]),
    )
    _write_summary_svg(summary_svg_path, counts, family_counts)

    return {
        "candidate_count": len(records),
        "preinsert_compatible_count": len(preinsert_records),
        "seated_compatible_count": len(seated_records),
        "outputs": [pose_path, table_path, summary_path, insertion_svg_path, summary_svg_path],
    }


def _resolve_output(value: str, owner: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = owner.parent / path
    return path.resolve()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG,
        help=f"generation contract (default: {DEFAULT_CONFIG.relative_to(ROOT)})",
    )
    return parser


def main() -> int:
    result = generate(_parser().parse_args().config)
    print(
        "generated "
        f"{result['candidate_count']} sampled antipodal poses; "
        f"{result['preinsert_compatible_count']} pass the pre-insert screen and "
        f"{result['seated_compatible_count']} clear the PCB at seating"
    )
    for path in result["outputs"]:
        print(path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
