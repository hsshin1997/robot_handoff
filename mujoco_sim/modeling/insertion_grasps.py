"""Task-aware filtering for insertion grasps.

The generic antipodal generator in :mod:`mujoco_sim.modeling.grasps` answers
"can an ideal parallel-jaw capability pinch this mesh?"  This module adds the
task geometry needed for insertion:

* contacts must lie on explicitly authored graspable regions;
* the supplied gripper component meshes are placed at the candidate aperture;
* their exact vertex support is tested against an infinite PCB half-space at
  both the pre-insert and seated poses; and
* every result remains explicitly labelled as a phase-1 geometric candidate.

This is deliberately not a motion or force certificate.  Part/gripper
collision away from the sampled contacts, complete pad capture, robot IK,
finite-board-edge collision, calibration uncertainty, and insertion wrench
capacity remain downstream gates.

Frame convention
----------------

``T_P_E`` maps the ideal contact frame ``E`` into the part STL frame ``P``.
``T_G_E`` maps that same contact frame into the reference full-assembly frame
``G``.  Therefore the placed gripper transform is
``T_P_G = T_P_E @ inverse(T_G_E)``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from ..core.se3 import inverse, validate_transform
from .grasps import (
    GraspCandidate,
    ParallelJawGripper,
    TriangleMesh,
    generate_antipodal_grasps,
    load_binary_stl,
)


_EPS = np.finfo(float).eps


def _unit(vector: np.ndarray, *, name: str) -> np.ndarray:
    value = np.asarray(vector, dtype=float)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be a finite three-vector")
    norm = float(np.linalg.norm(value))
    if norm <= 64.0 * _EPS:
        raise ValueError(f"{name} must be nonzero")
    return value / norm


def load_scaled_binary_stl(
    path: str | Path,
    *,
    scale_to_m: float,
) -> TriangleMesh:
    """Load a binary STL and apply its declared linear scale to metres."""
    scale = float(scale_to_m)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale_to_m must be positive and finite")
    source = Path(path)
    native = load_binary_stl(source)
    return TriangleMesh.from_triangles(
        native.triangles * scale,
        source=str(source),
    )


@dataclass(frozen=True)
class AxisAlignedRegion:
    """A conservative graspable-contact region in the part frame."""

    name: str
    minimum_P_m: np.ndarray
    maximum_P_m: np.ndarray

    def __post_init__(self) -> None:
        minimum = np.asarray(self.minimum_P_m, dtype=float)
        maximum = np.asarray(self.maximum_P_m, dtype=float)
        if minimum.shape != (3,) or maximum.shape != (3,):
            raise ValueError("region limits must be three-vectors")
        if not np.all(np.isfinite((minimum, maximum))):
            raise ValueError("region limits must be finite")
        if np.any(maximum <= minimum):
            raise ValueError("each region maximum must exceed its minimum")
        if not self.name:
            raise ValueError("region name must be non-empty")
        object.__setattr__(self, "minimum_P_m", minimum.copy())
        object.__setattr__(self, "maximum_P_m", maximum.copy())

    def contains(self, points_P_m: np.ndarray, *, tolerance_m: float = 0.0) -> np.ndarray:
        points = np.asarray(points_P_m, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must have shape (N, 3)")
        tolerance = float(tolerance_m)
        if tolerance < 0.0 or not np.isfinite(tolerance):
            raise ValueError("tolerance_m must be finite and non-negative")
        return np.all(
            (points >= self.minimum_P_m - tolerance)
            & (points <= self.maximum_P_m + tolerance),
            axis=1,
        )


@dataclass(frozen=True)
class FreeSpacePlane:
    """PCB boundary expressed as ``normal_P @ point >= offset_P_m``.

    ``normal_P`` points from the board into the gripper's permitted free
    half-space.  A positive geometric clearance is therefore safe.
    """

    normal_P: np.ndarray
    offset_P_m: float

    def __post_init__(self) -> None:
        normal = _unit(self.normal_P, name="plane normal_P")
        offset = float(self.offset_P_m)
        if not np.isfinite(offset):
            raise ValueError("plane offset must be finite")
        object.__setattr__(self, "normal_P", normal)
        object.__setattr__(self, "offset_P_m", offset)

    def signed_clearance(self, points_P_m: np.ndarray) -> np.ndarray:
        points = np.asarray(points_P_m, dtype=float)
        if points.shape[-1] != 3:
            raise ValueError("points must end in dimension three")
        return points @ self.normal_P - self.offset_P_m


@dataclass
class GripperMeshComponent:
    """One body/finger STL registered in the full-assembly frame ``G``.

    ``aperture_multiplier`` is total-aperture motion: ``+0.5`` and ``-0.5``
    model symmetric opposing fingers, while zero keeps the body fixed.  The
    recovered placement and the motion rule can be marked provisional by the
    owning project contract; this value type only performs the geometry.
    """

    name: str
    mesh_C: TriangleMesh
    T_G_C_reference: np.ndarray
    aperture_multiplier: float = 0.0
    _vertices_C: np.ndarray = field(init=False, repr=False)
    _support_cache: dict[tuple[float, float, float], float] = field(
        init=False, repr=False, default_factory=dict,
    )

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("component name must be non-empty")
        self.T_G_C_reference = validate_transform(self.T_G_C_reference)
        multiplier = float(self.aperture_multiplier)
        if not np.isfinite(multiplier):
            raise ValueError("aperture_multiplier must be finite")
        self.aperture_multiplier = multiplier
        # STL exporters repeat every triangle vertex.  Deduplication makes
        # exact support queries practical without changing their result.
        vertices = np.unique(self.mesh_C.triangles.reshape(-1, 3), axis=0)
        self._vertices_C = vertices

    @property
    def unique_vertex_count(self) -> int:
        return int(len(self._vertices_C))

    def minimum_support_G(
        self,
        direction_G: np.ndarray,
        *,
        aperture_delta_m: float,
        opening_axis_G: np.ndarray,
    ) -> float:
        """Return ``min(direction_G @ point_G)`` for this placed mesh."""
        direction = _unit(direction_G, name="support direction_G")
        rotation = self.T_G_C_reference[:3, :3]
        direction_C = rotation.T @ direction
        # Rounding only coalesces floating-point copies of the same generated
        # orientation.  At this CAD scale its support error is sub-nanometre.
        key = tuple(np.round(direction_C, 10).tolist())
        local = self._support_cache.get(key)
        if local is None:
            local = float(np.min(self._vertices_C @ direction_C))
            self._support_cache[key] = local
        translation = self.T_G_C_reference[:3, 3] + (
            opening_axis_G
            * self.aperture_multiplier
            * float(aperture_delta_m)
        )
        return local + float(direction @ translation)

    def placed_vertices_G(
        self,
        *,
        aperture_delta_m: float,
        opening_axis_G: np.ndarray,
    ) -> np.ndarray:
        """Return the component's unique vertices at the requested aperture."""
        translation = self.T_G_C_reference[:3, 3] + (
            opening_axis_G
            * self.aperture_multiplier
            * float(aperture_delta_m)
        )
        return (
            self._vertices_C @ self.T_G_C_reference[:3, :3].T
            + translation
        )


@dataclass(frozen=True)
class GripperMeshModel:
    """Registered body/finger collision geometry for PCB-plane checks."""

    T_G_E: np.ndarray
    reference_aperture_m: float
    opening_axis_G: np.ndarray
    components: tuple[GripperMeshComponent, ...]

    def __post_init__(self) -> None:
        transform = validate_transform(self.T_G_E)
        aperture = float(self.reference_aperture_m)
        if aperture <= 0.0 or not np.isfinite(aperture):
            raise ValueError("reference aperture must be positive and finite")
        axis = _unit(self.opening_axis_G, name="opening_axis_G")
        if not self.components:
            raise ValueError("gripper mesh model needs at least one component")
        if len({component.name for component in self.components}) != len(self.components):
            raise ValueError("gripper component names must be unique")
        object.__setattr__(self, "T_G_E", transform)
        object.__setattr__(self, "reference_aperture_m", aperture)
        object.__setattr__(self, "opening_axis_G", axis)

    def plane_clearance(
        self,
        candidate: GraspCandidate,
        plane_P: FreeSpacePlane,
    ) -> tuple[float, str, dict[str, float]]:
        """Return minimum gripper-mesh clearance to the free-space plane."""
        T_P_G = candidate.T_P_E @ inverse(self.T_G_E)
        rotation_P_G = T_P_G[:3, :3]
        direction_G = rotation_P_G.T @ plane_P.normal_P
        translation_term = (
            float(plane_P.normal_P @ T_P_G[:3, 3])
            - plane_P.offset_P_m
        )
        aperture_delta = candidate.required_opening - self.reference_aperture_m
        clearances: dict[str, float] = {}
        for component in self.components:
            support = component.minimum_support_G(
                direction_G,
                aperture_delta_m=aperture_delta,
                opening_axis_G=self.opening_axis_G,
            )
            clearances[component.name] = translation_term + support
        limiting = min(clearances, key=clearances.__getitem__)
        return float(clearances[limiting]), limiting, clearances

    def plane_violation_witness(
        self,
        candidate: GraspCandidate,
        plane_P: FreeSpacePlane,
        *,
        required_clearance_m: float,
    ) -> dict[str, dict[str, object]]:
        """Locate component vertices that violate a planar clearance.

        This diagnostic is intentionally more expensive than the support-only
        gate and is meant for a small number of ranked poses.  Its projected
        bounds tell a later finite-PCB check where an edge or cutout would be
        needed; they do not themselves certify finite-board clearance.
        """
        threshold = float(required_clearance_m)
        if threshold < 0.0 or not np.isfinite(threshold):
            raise ValueError("required_clearance_m must be finite and non-negative")
        T_P_G = candidate.T_P_E @ inverse(self.T_G_E)
        aperture_delta = candidate.required_opening - self.reference_aperture_m
        result: dict[str, dict[str, object]] = {}
        for component in self.components:
            vertices_G = component.placed_vertices_G(
                aperture_delta_m=aperture_delta,
                opening_axis_G=self.opening_axis_G,
            )
            vertices_P = (
                vertices_G @ T_P_G[:3, :3].T + T_P_G[:3, 3]
            )
            clearances = plane_P.signed_clearance(vertices_P)
            minimum_index = int(np.argmin(clearances))
            violating = clearances < threshold
            record: dict[str, object] = {
                "minimum_clearance_m": float(clearances[minimum_index]),
                "minimum_point_P_m": vertices_P[minimum_index].copy(),
                "violating_vertex_count": int(np.count_nonzero(violating)),
            }
            if np.any(violating):
                points = vertices_P[violating]
                record["violating_bounds_min_P_m"] = np.min(points, axis=0)
                record["violating_bounds_max_P_m"] = np.max(points, axis=0)
            else:
                record["violating_bounds_min_P_m"] = None
                record["violating_bounds_max_P_m"] = None
            result[component.name] = record
        return result


@dataclass(frozen=True)
class InsertionTaskGeometry:
    """Authored semantic and planar geometry for one insertion task."""

    insertion_axis_P: np.ndarray
    pcb_plane_P: FreeSpacePlane
    graspable_regions: tuple[AxisAlignedRegion, ...]
    contact_region_tolerance_m: float
    preinsert_distance_m: float
    minimum_pcb_clearance_m: float

    def __post_init__(self) -> None:
        insertion = _unit(self.insertion_axis_P, name="insertion_axis_P")
        if not self.graspable_regions:
            raise ValueError("at least one graspable region is required")
        tolerance = float(self.contact_region_tolerance_m)
        preinsert = float(self.preinsert_distance_m)
        clearance = float(self.minimum_pcb_clearance_m)
        if tolerance < 0.0 or preinsert <= 0.0 or clearance < 0.0:
            raise ValueError("task distances are invalid")
        if not np.all(np.isfinite((tolerance, preinsert, clearance))):
            raise ValueError("task distances must be finite")
        rate = float(self.pcb_plane_P.normal_P @ insertion)
        if rate >= -1e-6:
            raise ValueError(
                "insertion_axis_P must point into the PCB half-space "
                "opposite pcb_plane_P.normal_P"
            )
        object.__setattr__(self, "insertion_axis_P", insertion)
        object.__setattr__(self, "contact_region_tolerance_m", tolerance)
        object.__setattr__(self, "preinsert_distance_m", preinsert)
        object.__setattr__(self, "minimum_pcb_clearance_m", clearance)

    def contacts_are_graspable(self, contacts_P_m: np.ndarray) -> bool:
        contacts = np.asarray(contacts_P_m, dtype=float)
        if contacts.shape != (2, 3):
            raise ValueError("contacts must have shape (2, 3)")
        allowed = np.zeros(2, dtype=bool)
        for region in self.graspable_regions:
            allowed |= region.contains(
                contacts, tolerance_m=self.contact_region_tolerance_m,
            )
        return bool(np.all(allowed))


@dataclass(frozen=True)
class InsertionGraspEvaluation:
    """Task-filter result for one generic antipodal candidate."""

    candidate: GraspCandidate
    status: str
    family: str
    contacts_in_graspable_region: bool
    seated_pcb_clearance_m: float | None
    preinsert_pcb_clearance_m: float | None
    collision_free_insertion_travel_m: float | None
    remaining_to_seat_at_collision_m: float | None
    limiting_component: str | None
    component_clearances_m: dict[str, float]

    @property
    def preinsert_compatible(self) -> bool:
        return self.status in {
            "phase1_preinsert_only_candidate",
            "phase1_seated_geometric_candidate",
        }

    @property
    def seated_compatible(self) -> bool:
        return self.status == "phase1_seated_geometric_candidate"


def grasp_family(candidate: GraspCandidate) -> str:
    """Return a stable principal-axis label for reports and stratification."""
    labels = ("x", "y", "z")
    closing = candidate.closing_direction
    approach = candidate.approach_direction
    closing_axis = int(np.argmax(np.abs(closing)))
    approach_axis = int(np.argmax(np.abs(approach)))
    approach_sign = "+" if approach[approach_axis] >= 0.0 else "-"
    return (
        f"close_{labels[closing_axis]}_"
        f"approach_{approach_sign}{labels[approach_axis]}"
    )


def evaluate_insertion_grasps(
    candidates: Iterable[GraspCandidate],
    *,
    task: InsertionTaskGeometry,
    gripper_mesh: GripperMeshModel,
) -> list[InsertionGraspEvaluation]:
    """Apply semantic-contact and PCB-plane gates to generic candidates."""
    evaluations: list[InsertionGraspEvaluation] = []
    plane_rate = float(task.pcb_plane_P.normal_P @ task.insertion_axis_P)
    preinsert_gain = -plane_rate * task.preinsert_distance_m
    for candidate in candidates:
        family = grasp_family(candidate)
        region_ok = task.contacts_are_graspable(candidate.contact_points)
        if not region_ok:
            evaluations.append(InsertionGraspEvaluation(
                candidate=candidate,
                status="rejected_contact_region",
                family=family,
                contacts_in_graspable_region=False,
                seated_pcb_clearance_m=None,
                preinsert_pcb_clearance_m=None,
                collision_free_insertion_travel_m=None,
                remaining_to_seat_at_collision_m=None,
                limiting_component=None,
                component_clearances_m={},
            ))
            continue

        seated, limiting, component_clearances = gripper_mesh.plane_clearance(
            candidate, task.pcb_plane_P,
        )
        preinsert = seated + preinsert_gain
        threshold = task.minimum_pcb_clearance_m
        if preinsert < threshold:
            status = "rejected_preinsert_pcb_collision"
            travel = 0.0
        else:
            travel = min(
                task.preinsert_distance_m,
                max(0.0, (preinsert - threshold) / (-plane_rate)),
            )
            status = (
                "phase1_seated_geometric_candidate"
                if seated >= threshold
                else "phase1_preinsert_only_candidate"
            )
        remaining = max(0.0, task.preinsert_distance_m - travel)
        evaluations.append(InsertionGraspEvaluation(
            candidate=candidate,
            status=status,
            family=family,
            contacts_in_graspable_region=True,
            seated_pcb_clearance_m=seated,
            preinsert_pcb_clearance_m=preinsert,
            collision_free_insertion_travel_m=travel,
            remaining_to_seat_at_collision_m=remaining,
            limiting_component=limiting,
            component_clearances_m=component_clearances,
        ))
    return evaluations


def generate_insertion_grasps(
    part_mesh_P: TriangleMesh,
    capability: ParallelJawGripper,
    *,
    task: InsertionTaskGeometry,
    gripper_mesh: GripperMeshModel,
    surface_samples: int,
    approaches_per_pair: int,
    max_candidates: int,
) -> list[InsertionGraspEvaluation]:
    """Generate generic grasps and apply the phase-1 insertion task gates."""
    candidates = generate_antipodal_grasps(
        part_mesh_P,
        capability,
        surface_samples=surface_samples,
        approaches_per_pair=approaches_per_pair,
        max_candidates=max_candidates,
    )
    return evaluate_insertion_grasps(
        candidates,
        task=task,
        gripper_mesh=gripper_mesh,
    )


__all__ = [
    "AxisAlignedRegion",
    "FreeSpacePlane",
    "GripperMeshComponent",
    "GripperMeshModel",
    "InsertionGraspEvaluation",
    "InsertionTaskGeometry",
    "evaluate_insertion_grasps",
    "generate_insertion_grasps",
    "grasp_family",
    "load_scaled_binary_stl",
]
