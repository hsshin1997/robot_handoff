#!/usr/bin/env python3
"""Generate repeatable parallel-jaw grasp candidates directly from CAD.

The output is complete only with respect to the declared finite surface,
friction-cone closing-direction, and roll samples. It is not an exhaustive
proof over continuous SE(3), and it does not certify a physical gripper,
robot, environment, or task.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mujoco_sim.modeling.cad_preprocess import prepare_cad  # noqa: E402
from mujoco_sim.modeling.grasps import (  # noqa: E402
    GraspCandidate,
    ParallelJawGripper,
    TriangleMesh,
    generate_antipodal_grasps,
)
from mujoco_sim.modeling.part_mesh import load_prepared_triangle_mesh  # noqa: E402


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "sampled_parallel_jaw_grasp_candidates"
CLAIM_LEVEL = "resolution_qualified_object_geometry_candidate"
UNRELIABLE_MESH_CLAIM_LEVEL = "unreliable_mesh_sampled_candidate"
DEFAULT_GENERATED_ROOT = ROOT / "build" / "parallel_jaw_grasps" / "cad"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rounded(value: float) -> float:
    result = round(float(value), 12)
    return 0.0 if result == -0.0 else result


def _array(value: np.ndarray) -> list[Any]:
    array = np.asarray(value, dtype=float)
    return np.vectorize(_rounded, otypes=[float])(array).tolist()


def _candidate_id(candidate: GraspCandidate) -> str:
    # Hash the same 12-decimal values that are serialized below so a consumer
    # can authenticate the stable ID from the JSON alone.
    serialized_transform = np.asarray(_array(candidate.T_P_E), dtype=float)
    serialized_contacts = np.asarray(
        _array(candidate.contact_points), dtype=float)
    serialized_opening = _rounded(candidate.required_opening)
    identity = {
        "T_P_E": np.round(serialized_transform, 10).tolist(),
        "contacts_P_m": np.round(serialized_contacts, 10).tolist(),
        "required_opening_m": round(serialized_opening, 10),
    }
    payload = json.dumps(
        identity,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "grasp_" + hashlib.sha256(payload).hexdigest()[:16]


def _candidate_record(index: int, candidate: GraspCandidate) -> dict[str, Any]:
    return {
        "id": _candidate_id(candidate),
        "index": index,
        "T_P_E": _array(candidate.T_P_E),
        "contact_points_P_m": _array(candidate.contact_points),
        "contact_normals_P": _array(candidate.contact_normals),
        "closing_direction_P": _array(candidate.closing_direction),
        "approach_direction_P": _array(candidate.approach_direction),
        "required_opening_m": _rounded(candidate.required_opening),
        "quality": _rounded(candidate.quality),
        "antipodal_quality": _rounded(candidate.antipodal_quality),
        "support_quality": _rounded(candidate.support_quality),
        "opening_margin": _rounded(candidate.opening_margin),
        "idealized_palm_clearance_m": _rounded(candidate.palm_clearance),
    }


def _mesh_topology_audit(mesh: TriangleMesh) -> dict[str, Any]:
    """Audit exact mesh connectivity required by inward normal-ray sampling.

    STL/OBJ preparation preserves coordinates, so exact-coordinate vertex
    welding is intentional here.  Near-coincident seams remain boundary edges
    instead of being silently repaired.
    """
    triangles = mesh.triangles
    vertices, inverse_vertices = np.unique(
        triangles.reshape(-1, 3),
        axis=0,
        return_inverse=True,
    )
    faces = inverse_vertices.reshape(-1, 3)
    edge_vertices = np.stack(
        (
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ),
        axis=1,
    ).reshape(-1, 2)
    edge_faces = np.repeat(np.arange(len(faces), dtype=int), 3)
    canonical_edges = np.sort(edge_vertices, axis=1)
    unique_edges, inverse_edges, edge_counts = np.unique(
        canonical_edges,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    edge_orientation = np.where(
        edge_vertices[:, 0] < edge_vertices[:, 1],
        1.0,
        np.where(edge_vertices[:, 0] > edge_vertices[:, 1], -1.0, 0.0),
    )
    orientation_sum = np.bincount(
        inverse_edges,
        weights=edge_orientation,
        minlength=len(unique_edges),
    )

    extent = max(float(np.linalg.norm(mesh.extent)), np.finfo(float).tiny)
    area_tolerance = 64.0 * np.finfo(float).eps * extent**2
    repeated_vertex_face = (
        (faces[:, 0] == faces[:, 1])
        | (faces[:, 1] == faces[:, 2])
        | (faces[:, 2] == faces[:, 0])
    )
    degenerate = repeated_vertex_face | (mesh.areas <= area_tolerance)
    boundary_edge_count = int(np.sum(edge_counts == 1))
    nonmanifold_edge_count = int(np.sum(edge_counts > 2))
    inconsistent_edge_orientation_count = int(np.sum(
        (edge_counts == 2) & (np.abs(orientation_sum) > 0.5)
    ))

    parents = np.arange(len(faces), dtype=int)

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = int(parents[index])
        return index

    def union(first: int, second: int) -> None:
        first_root, second_root = find(first), find(second)
        if first_root == second_root:
            return
        if first_root > second_root:
            first_root, second_root = second_root, first_root
        parents[second_root] = first_root

    first_face_by_edge: dict[int, int] = {}
    for incidence, edge_index in enumerate(inverse_edges):
        face_index = int(edge_faces[incidence])
        owner = first_face_by_edge.setdefault(int(edge_index), face_index)
        union(owner, face_index)

    components: dict[int, list[int]] = {}
    for face_index in range(len(faces)):
        components.setdefault(find(face_index), []).append(face_index)
    component_records = []
    zero_volume_components = 0
    winding_signs: set[str] = set()
    volume_tolerance = 128.0 * np.finfo(float).eps * extent**3
    for component_index, face_indices in enumerate(
        sorted(components.values(), key=lambda indices: indices[0])
    ):
        component_triangles = triangles[np.asarray(face_indices, dtype=int)]
        reference = np.mean(component_triangles.reshape(-1, 3), axis=0)
        relative = component_triangles - reference
        signed_volume = float(np.sum(np.einsum(
            "ij,ij->i",
            relative[:, 0],
            np.cross(relative[:, 1], relative[:, 2]),
        )) / 6.0)
        if abs(signed_volume) <= volume_tolerance:
            orientation = "zero_or_unresolved"
            zero_volume_components += 1
        else:
            orientation = "positive" if signed_volume > 0.0 else "negative"
            winding_signs.add(orientation)
        component_records.append({
            "index": component_index,
            "face_count": len(face_indices),
            "signed_volume_m3": _rounded(signed_volume),
            "winding_volume_sign": orientation,
        })

    mixed_component_winding_signs = len(winding_signs) > 1
    reliable = (
        int(np.sum(degenerate)) == 0
        and boundary_edge_count == 0
        and nonmanifold_edge_count == 0
        and inconsistent_edge_orientation_count == 0
        and zero_volume_components == 0
        and not mixed_component_winding_signs
    )
    return {
        "method": "exact_coordinate_edge_incidence_and_component_signed_volume",
        "exact_vertex_count": int(len(vertices)),
        "face_count": int(len(faces)),
        "edge_count": int(len(unique_edges)),
        "degenerate_face_count": int(np.sum(degenerate)),
        "boundary_edge_count": boundary_edge_count,
        "nonmanifold_edge_count": nonmanifold_edge_count,
        "inconsistent_paired_edge_orientation_count": (
            inconsistent_edge_orientation_count
        ),
        "component_count": len(component_records),
        "zero_or_unresolved_volume_component_count": zero_volume_components,
        "mixed_component_winding_signs": mixed_component_winding_signs,
        "components": component_records,
        "closed_consistently_wound_two_manifold": reliable,
        "normal_ray_assumptions_accepted": reliable,
        "global_outward_orientation_note": (
            "Per-component signed-volume signs are recorded. Mixed signs are "
            "downgraded because nested-cavity semantics and global material "
            "inside/outside are not inferred or repaired."
        ),
    }


def generate_document(
    cad_path: str | Path,
    *,
    units: str | None,
    scale_to_m: float | None,
    gripper: ParallelJawGripper,
    surface_samples: int = 2048,
    closing_directions_per_surface: int = 5,
    approaches_per_pair: int = 24,
    max_candidates: int | None = None,
    generated_root: str | Path = DEFAULT_GENERATED_ROOT,
    freecad_executable: str | Path | None = None,
    linear_deflection_mm: float = 0.05,
    angular_deflection_deg: float = 5.0,
    allow_unreliable_mesh: bool = False,
) -> dict[str, Any]:
    """Prepare CAD and return a resolution-qualified sampled grasp document.

    ``max_candidates=None`` retains every deduplicated candidate found at the
    requested finite sampling resolution.  Any integer cap intentionally
    turns the output into a ranked/coverage-preserving subset.
    """
    source = Path(cad_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if not isinstance(gripper, ParallelJawGripper):
        raise TypeError("gripper must be a ParallelJawGripper")
    if not isinstance(surface_samples, int) or isinstance(surface_samples, bool):
        raise TypeError("surface_samples must be an integer")
    if not isinstance(approaches_per_pair, int) or isinstance(
        approaches_per_pair, bool
    ):
        raise TypeError("approaches_per_pair must be an integer")
    if not isinstance(closing_directions_per_surface, int) or isinstance(
        closing_directions_per_surface, bool
    ):
        raise TypeError("closing_directions_per_surface must be an integer")
    if (
        surface_samples <= 0
        or closing_directions_per_surface <= 0
        or approaches_per_pair <= 0
    ):
        raise ValueError("sampling budgets must be positive")
    if max_candidates is not None and (
        not isinstance(max_candidates, int)
        or isinstance(max_candidates, bool)
        or max_candidates <= 0
    ):
        raise ValueError("max_candidates must be a positive integer or None")
    if not isinstance(allow_unreliable_mesh, bool):
        raise TypeError("allow_unreliable_mesh must be a boolean")

    preparation = prepare_cad(
        source,
        generated_root,
        units=units,
        scale_to_m=scale_to_m,
        role="parallel-jaw-grasp-source",
        static_assembly=False,
        freecad_executable=freecad_executable,
        linear_deflection_mm=linear_deflection_mm,
        angular_deflection_deg=angular_deflection_deg,
    )
    mesh = load_prepared_triangle_mesh(preparation)
    topology = _mesh_topology_audit(mesh)
    if (
        not topology["normal_ray_assumptions_accepted"]
        and not allow_unreliable_mesh
    ):
        raise ValueError(
            "CAD mesh is not a closed, consistently wound two-manifold under "
            "exact-coordinate topology audit "
            f"(degenerate_faces={topology['degenerate_face_count']}, "
            f"boundary_edges={topology['boundary_edge_count']}, "
            f"nonmanifold_edges={topology['nonmanifold_edge_count']}, "
            "inconsistent_oriented_edges="
            f"{topology['inconsistent_paired_edge_orientation_count']}, "
            "mixed_component_winding_signs="
            f"{topology['mixed_component_winding_signs']}). "
            "Repair the CAD mesh or explicitly set allow_unreliable_mesh=True "
            "(CLI: --allow-unreliable-mesh) to generate downgraded hypotheses."
        )
    candidates = generate_antipodal_grasps(
        mesh,
        gripper,
        surface_samples=surface_samples,
        closing_directions_per_surface=closing_directions_per_surface,
        approaches_per_pair=approaches_per_pair,
        max_candidates=max_candidates,
    )
    dedup_position_tolerance_m = max(
        1e-6 * max(float(np.linalg.norm(mesh.extent)), 1e-9),
        0.15 * min(gripper.pad_size),
        1e-7,
    )

    metadata = preparation.metadata
    candidate_records = [
        _candidate_record(index, candidate)
        for index, candidate in enumerate(candidates)
    ]
    topology_reliable = bool(topology["normal_ray_assumptions_accepted"])
    claim_level = (
        CLAIM_LEVEL if topology_reliable else UNRELIABLE_MESH_CLAIM_LEVEL
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "claim_level": claim_level,
        "continuous_exhaustive": False,
        "all_deduplicated_accepted_candidates_returned": (
            max_candidates is None
        ),
        "candidate_cap_applied": max_candidates is not None,
        "frame_convention": {
            "transform": (
                "T_P_E maps ideal gripper contact-frame E coordinates into "
                "the input CAD/part frame P"
            ),
            "origin_E": "midpoint of the two ideal contacts",
            "positive_x_E": "pad-width direction",
            "positive_y_E": "jaw-closing direction from contact 0 to contact 1",
            "positive_z_E": "approach direction from palm towards contact plane",
            "physical_tcp_note": (
                "E is an ideal contact frame, not necessarily the physical TCP. "
                "Given a calibrated fixed T_G_E, use T_P_G = T_P_E @ inverse(T_G_E)."
            ),
            "units": "metres and radians",
        },
        "cad": {
            "path": str(source),
            "sha256": _sha256(source),
            "format": metadata["source"]["format"],
            "declared_units": metadata["source"]["units"],
            "scale_to_m": metadata["source"]["scale_to_m"],
            "artifact_fingerprint": metadata["artifact_fingerprint"],
            "triangle_count": int(len(mesh.triangles)),
            "bounds_min_P_m": _array(mesh.bounds_min),
            "bounds_max_P_m": _array(mesh.bounds_max),
            "extent_P_m": _array(mesh.extent),
            "topology_audit": topology,
        },
        "gripper_model": {
            "type": "ideal_symmetric_parallel_jaw",
            "opening_range_m": [
                gripper.min_opening,
                gripper.max_opening,
            ],
            "pad_size_m": list(gripper.pad_size),
            "finger_tip_to_palm_depth_m": gripper.pad_depth,
            "friction_coefficient": gripper.friction_coefficient,
        },
        "sampling": {
            "determinism": {
                "repeatable_for_fixed_input_settings_and_numerical_backend": True,
                "cross_platform_bitwise_identity_certified": False,
                "note": (
                    "roll seeding uses a mesh covariance eigendecomposition; "
                    "symmetric or nearly symmetric CAD can have non-unique "
                    "principal eigenvectors across numerical backends"
                ),
            },
            "surface_method": "area_stratified_low_discrepancy_triangle_samples",
            "surface_samples": surface_samples,
            "opposing_contact_method": (
                "first inward ray hit for deterministic directions sampled "
                "inside the source contact friction cone"
            ),
            "closing_directions_per_surface": closing_directions_per_surface,
            "maximum_closing_direction_tilt_rad": _rounded(np.arctan(
                gripper.friction_coefficient
            )),
            "approaches_per_contact_pair": approaches_per_pair,
            "roll_method": (
                "uniform full circle when approaches_per_contact_pair > 4; "
                "otherwise mesh-principal directions and their opposites"
            ),
            "deduplication": {
                "method": "position plus closing/approach angular similarity",
                "position_tolerance_m": _rounded(
                    dedup_position_tolerance_m
                ),
                "angle_tolerance_deg": 7.5,
                "opposite_closing_axes_equivalent": True,
                "opposite_approach_axes_equivalent": False,
            },
            "max_candidates": max_candidates,
        },
        "feasibility_contract": {
            "hard_checks": ([] if not topology_reliable else [
                "exact-coordinate mesh topology passes the closed oriented "
                "two-manifold audit",
            ]) + [
                "first opposing mesh intersection exists",
                "contact separation lies inside the gripper opening range",
                "both contact normals satisfy the Coulomb friction-cone test",
                "the part triangle surface clipped to the jaw slab fits before "
                "an idealized palm that is infinite along the pad-width axis",
            ],
            "scored_but_not_hard_gated": [
                "approximate coplanar surface support under each rectangular pad",
                "opening-range margin",
                "idealized palm-clearance margin",
            ],
            "not_checked": [
                "continuous poses between surface or roll samples",
                "closing directions between the finite friction-cone samples",
                "full finger, palm, or gripper-body collision against the part",
                "collision along the pregrasp-to-grasp approach sweep",
                "environment, fixture, or other-object collision",
                "robot inverse kinematics, joint limits, or motion planning",
                "task wrench closure, material compliance, dynamics, or uncertainty",
                "global material inside/outside semantics for nested mesh components",
                "automatic mesh repair or near-coincident seam welding",
            ] + ([] if topology_reliable else [
                "reliable inward normals and first-opposing-surface semantics "
                "because generation was explicitly allowed on an unreliable mesh",
            ]),
        },
        "candidate_count": len(candidate_records),
        "candidates": candidate_records,
    }


def _write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        document,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    path.write_text(payload, encoding="utf-8")


def _default_output(cad_path: Path) -> Path:
    return ROOT / "build" / "parallel_jaw_grasps" / f"{cad_path.stem}.json"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cad", type=Path, help="STL, OBJ, STEP, or STP part CAD")
    units = parser.add_mutually_exclusive_group(required=True)
    units.add_argument(
        "--units",
        choices=("m", "mm", "cm", "in"),
        help="source CAD linear units",
    )
    units.add_argument(
        "--scale-to-m",
        type=float,
        help="explicit source-coordinate to metre scale",
    )
    parser.add_argument("--min-opening-m", type=float, required=True)
    parser.add_argument("--max-opening-m", type=float, required=True)
    parser.add_argument("--pad-width-m", type=float, required=True)
    parser.add_argument("--pad-height-m", type=float, required=True)
    parser.add_argument("--finger-depth-m", type=float, required=True)
    parser.add_argument("--friction-coefficient", type=float, default=0.5)
    parser.add_argument("--surface-samples", type=int, default=2048)
    parser.add_argument("--closing-directions-per-surface", type=int, default=5)
    parser.add_argument("--approaches-per-pair", type=int, default=24)
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=0,
        help=(
            "0 keeps every deduplicated sampled candidate (default); a "
            "positive output cap is applied after candidate construction"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output JSON (default: build/parallel_jaw_grasps/<cad-stem>.json)",
    )
    parser.add_argument(
        "--generated-root",
        type=Path,
        default=DEFAULT_GENERATED_ROOT,
        help="content-addressed prepared-CAD cache",
    )
    parser.add_argument(
        "--freecad",
        type=Path,
        default=None,
        help="FreeCADCmd/freecadcmd path for STEP/STP input",
    )
    parser.add_argument("--linear-deflection-mm", type=float, default=0.05)
    parser.add_argument("--angular-deflection-deg", type=float, default=5.0)
    parser.add_argument(
        "--allow-unreliable-mesh",
        action="store_true",
        help=(
            "continue with a downgraded claim when the exact topology audit "
            "finds an open, nonmanifold, degenerate, or inconsistently wound mesh"
        ),
    )
    arguments = parser.parse_args(argv)
    if arguments.max_candidates < 0:
        parser.error("--max-candidates must be non-negative")

    gripper = ParallelJawGripper(
        min_opening=arguments.min_opening_m,
        max_opening=arguments.max_opening_m,
        pad_size=(arguments.pad_width_m, arguments.pad_height_m),
        pad_depth=arguments.finger_depth_m,
        friction_coefficient=arguments.friction_coefficient,
    )
    document = generate_document(
        arguments.cad,
        units=arguments.units,
        scale_to_m=arguments.scale_to_m,
        gripper=gripper,
        surface_samples=arguments.surface_samples,
        closing_directions_per_surface=(
            arguments.closing_directions_per_surface
        ),
        approaches_per_pair=arguments.approaches_per_pair,
        max_candidates=(
            None if arguments.max_candidates == 0 else arguments.max_candidates
        ),
        generated_root=arguments.generated_root,
        freecad_executable=arguments.freecad,
        linear_deflection_mm=arguments.linear_deflection_mm,
        angular_deflection_deg=arguments.angular_deflection_deg,
        allow_unreliable_mesh=arguments.allow_unreliable_mesh,
    )
    output = (
        arguments.output.resolve()
        if arguments.output is not None
        else _default_output(arguments.cad).resolve()
    )
    _write_json(output, document)
    print(f"Wrote {document['candidate_count']} grasp candidates to {output}")
    print(
        f"Claim level: {document['claim_level']}; continuous_exhaustive=false; "
        f"all_deduplicated_accepted_candidates_returned="
        f"{str(document['all_deduplicated_accepted_candidates_returned']).lower()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
