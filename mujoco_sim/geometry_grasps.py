"""Geometry-driven parallel-jaw grasp generation.

This module deliberately has no dependency on MuJoCo or on part-specific
configuration. It turns triangular surface geometry in the part frame into
ranked parallel-jaw grasp hypotheses. Project CAD ingestion, chunk combination,
and SI scaling are handled by :mod:`mujoco_sim.part_mesh`; the only
manipulation-specific input here is a reusable gripper capability description.

Frame convention
----------------

Each :class:`GraspCandidate` stores ``T_P_E == ^P T_E``.  The end-effector
axes are

``+Y``
    direction from contact 0 to contact 1 (the jaw-closing line),
``+Z``
    approach direction, from the palm towards the TCP/contact plane, and
``+X``
    pad-width direction, chosen so that ``[X Y Z]`` is right handed.

The transform origin is the midpoint of the two contacts.  Consequently the
required jaw opening is geometric contact separation, and a downstream
planner can evaluate ``^W T_E = ^W T_P @ ^P T_E`` without another convention
conversion.

Method
------

Surface samples are deterministic and stratified by triangle area.  From each
sample an inward ray is cast through the solid; the first opposing surface is
the second contact.  A pair is retained only when both surface normals lie in
the Coulomb friction cones and the separation fits the gripper opening range.
Roll about the closing line is derived from the mesh surface covariance, not
from hard-coded part axes.  Candidates are non-max suppressed in SE(3) and a
    small farthest-point term preserves spatial coverage on elongated parts.

The generator produces *grasp hypotheses*, not complete motion plans.  Exact
finger/palm collision, robot IK, approach-path collision, task wrench closure,
and co-grasp compatibility remain mandatory downstream gates.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct
from typing import Iterable, Sequence

import numpy as np


_EPS = np.finfo(float).eps


def _finite_scalar(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _unit(vector: np.ndarray, *, name: str) -> np.ndarray:
    value = np.asarray(vector, dtype=float)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be a finite three-vector")
    norm = float(np.linalg.norm(value))
    if norm <= 64.0 * _EPS:
        raise ValueError(f"{name} must be nonzero")
    return value / norm


def _canonical_sign(vector: np.ndarray) -> np.ndarray:
    """Choose a deterministic sign without assuming a semantic part axis."""
    result = _unit(vector, name="direction")
    pivot = int(np.argmax(np.abs(result)))
    return -result if result[pivot] < 0.0 else result


@dataclass(frozen=True)
class TriangleMesh:
    """Triangular surface geometry expressed in its supplied part frame.

    Degenerate facets are retained in ``triangles`` so the object remains a
    faithful parse of the source, but their normal and area are zero and they
    are never selected as surface patches.
    """

    triangles: np.ndarray
    normals: np.ndarray
    areas: np.ndarray
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    source: str | None = None

    def __post_init__(self) -> None:
        triangles = np.asarray(self.triangles, dtype=float)
        normals = np.asarray(self.normals, dtype=float)
        areas = np.asarray(self.areas, dtype=float)
        bounds_min = np.asarray(self.bounds_min, dtype=float)
        bounds_max = np.asarray(self.bounds_max, dtype=float)
        if triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
            raise ValueError("triangles must have shape (N, 3, 3)")
        count = triangles.shape[0]
        if count == 0:
            raise ValueError("mesh must contain at least one triangle")
        if normals.shape != (count, 3):
            raise ValueError("normals must have shape (N, 3)")
        if areas.shape != (count,):
            raise ValueError("areas must have shape (N,)")
        if bounds_min.shape != (3,) or bounds_max.shape != (3,):
            raise ValueError("mesh bounds must be three-vectors")
        if not all(np.all(np.isfinite(item)) for item in
                   (triangles, normals, areas, bounds_min, bounds_max)):
            raise ValueError("mesh arrays must be finite")
        if np.any(areas < 0.0) or np.any(bounds_max < bounds_min):
            raise ValueError("invalid mesh areas or bounds")
        object.__setattr__(self, "triangles", triangles.copy())
        object.__setattr__(self, "normals", normals.copy())
        object.__setattr__(self, "areas", areas.copy())
        object.__setattr__(self, "bounds_min", bounds_min.copy())
        object.__setattr__(self, "bounds_max", bounds_max.copy())

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.bounds_min.copy(), self.bounds_max.copy()

    @property
    def extent(self) -> np.ndarray:
        return self.bounds_max - self.bounds_min

    @property
    def vertices(self) -> np.ndarray:
        """Return vertices without recentering or deduplication."""
        return self.triangles.reshape(-1, 3).copy()

    @property
    def surface_area(self) -> float:
        return float(np.sum(self.areas))

    @classmethod
    def from_binary_stl(cls, path: str | Path) -> "TriangleMesh":
        return load_binary_stl(path)

    @classmethod
    def from_triangles(
        cls,
        triangles: np.ndarray,
        *,
        source: str | None = None,
        stored_normals: np.ndarray | None = None,
    ) -> "TriangleMesh":
        """Build a mesh from native-frame triangles and recompute geometry.

        This is the authoritative constructor when prepared STL chunks or OBJ
        polygons have been combined.  In particular, normals must be computed
        *after* chunk concatenation and unit scaling: treating arbitrary chunks
        as independent closed solids can flip their normals inconsistently,
        and anisotropic scale changes the normal directions.
        """
        values = np.asarray(triangles, dtype=float)
        if values.ndim != 3 or values.shape[1:] != (3, 3) or len(values) == 0:
            raise ValueError("triangles must have non-empty shape (N, 3, 3)")
        if not np.all(np.isfinite(values)):
            raise ValueError("triangles must contain only finite coordinates")
        edges_1 = values[:, 1] - values[:, 0]
        edges_2 = values[:, 2] - values[:, 0]
        cross = np.cross(edges_1, edges_2)
        double_area = np.linalg.norm(cross, axis=1)
        normals = np.zeros_like(cross)
        valid = double_area > 64.0 * _EPS
        normals[valid] = cross[valid] / double_area[valid, None]
        areas = 0.5 * double_area

        vertices = values.reshape(-1, 3)
        bounds_min = np.min(vertices, axis=0)
        bounds_max = np.max(vertices, axis=0)
        reference = 0.5 * (bounds_min + bounds_max)
        relative = values - reference
        volume6 = float(np.sum(np.einsum(
            "ij,ij->i", relative[:, 0],
            np.cross(relative[:, 1], relative[:, 2]))))
        scale = max(float(np.linalg.norm(bounds_max - bounds_min)), 1.0)
        volume_tolerance = 1e-12 * scale**3
        if volume6 < -volume_tolerance:
            normals = -normals
        elif abs(volume6) <= volume_tolerance and stored_normals is not None:
            stored = np.asarray(stored_normals, dtype=float)
            if stored.shape != normals.shape or not np.all(np.isfinite(stored)):
                raise ValueError("stored_normals must have finite shape (N, 3)")
            stored_norm = np.linalg.norm(stored, axis=1)
            usable = valid & (stored_norm > 1e-12)
            flip = usable & (np.einsum("ij,ij->i", normals, stored) < 0.0)
            normals[flip] *= -1.0
        return cls(
            triangles=values,
            normals=normals,
            areas=areas,
            bounds_min=bounds_min,
            bounds_max=bounds_max,
            source=source,
        )


def load_binary_stl(path: str | Path) -> TriangleMesh:
    """Load a binary STL, preserving its coordinate frame and native units.

    Facet normals are recomputed from vertex winding because exported STL
    normal records are frequently stale.  For consistently wound closed
    meshes, a negative signed volume flips all normals to point outward.
    Geometry is never centered, rotated, scaled, repaired, or convexified.
    """
    source = Path(path)
    payload = source.read_bytes()
    if len(payload) < 84:
        raise ValueError(f"{source} is too short to be a binary STL")
    count = struct.unpack_from("<I", payload, 80)[0]
    expected = 84 + 50 * count
    if count == 0 or len(payload) < expected:
        raise ValueError(
            f"{source} is not a complete binary STL "
            f"(declares {count} facets, needs {expected} bytes)"
        )
    dtype = np.dtype([
        ("stored_normal", "<f4", (3,)),
        ("vertices", "<f4", (3, 3)),
        ("attribute", "<u2"),
    ])
    records = np.frombuffer(payload, dtype=dtype, count=count, offset=84)
    return TriangleMesh.from_triangles(
        records["vertices"].astype(float, copy=True),
        source=str(source),
        stored_normals=records["stored_normal"].astype(float),
    )


@dataclass(frozen=True)
class SurfacePatch:
    """A deterministic area-stratified point sample on one mesh facet."""

    point: np.ndarray
    normal: np.ndarray
    represented_area: float
    triangle_index: int

    def __post_init__(self) -> None:
        point = np.asarray(self.point, dtype=float)
        normal = _unit(self.normal, name="patch normal")
        area = _finite_scalar(self.represented_area, "represented_area")
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise ValueError("patch point must be a finite three-vector")
        if area <= 0.0:
            raise ValueError("represented_area must be positive")
        if int(self.triangle_index) < 0:
            raise ValueError("triangle_index must be non-negative")
        object.__setattr__(self, "point", point.copy())
        object.__setattr__(self, "normal", normal.copy())
        object.__setattr__(self, "represented_area", area)
        object.__setattr__(self, "triangle_index", int(self.triangle_index))


def _radical_inverse(index: int, base: int) -> float:
    result = 0.0
    denominator = 1.0
    value = int(index)
    while value:
        value, digit = divmod(value, base)
        denominator *= base
        result += digit / denominator
    return result


def sample_surface_patches(
    mesh: TriangleMesh,
    count: int = 192,
) -> tuple[SurfacePatch, ...]:
    """Return deterministic, approximately area-uniform surface patches.

    One sample is drawn from each equal-area stratum of the flattened STL
    surface.  Low-discrepancy barycentric coordinates avoid a dependence on
    triangle centroids, which is important when a large planar face is stored
    as only two triangles: the face still receives many spatially distinct
    samples.
    """
    if not isinstance(count, (int, np.integer)) or int(count) <= 0:
        raise ValueError("count must be a positive integer")
    count = int(count)
    valid_indices = np.flatnonzero(mesh.areas > 64.0 * _EPS)
    if valid_indices.size == 0:
        raise ValueError("mesh has no non-degenerate surface facets")
    valid_areas = mesh.areas[valid_indices]
    cumulative = np.cumsum(valid_areas)
    total = float(cumulative[-1])
    represented = total / count
    patches: list[SurfacePatch] = []
    for sample_index in range(count):
        target = (sample_index + 0.5) * represented
        local_index = min(int(np.searchsorted(cumulative, target, side="right")),
                          len(valid_indices) - 1)
        triangle_index = int(valid_indices[local_index])
        triangle = mesh.triangles[triangle_index]
        # Uniform triangle sampling: sqrt(u) is the area-correct radial term.
        u = _radical_inverse(sample_index + 1, 2)
        v = _radical_inverse(sample_index + 1, 3)
        root = float(np.sqrt(u))
        barycentric = np.array([1.0 - root, root * (1.0 - v), root * v])
        point = barycentric @ triangle
        patches.append(SurfacePatch(
            point=point,
            normal=mesh.normals[triangle_index],
            represented_area=represented,
            triangle_index=triangle_index,
        ))
    return tuple(patches)


@dataclass(frozen=True)
class ParallelJawGripper:
    """Part-independent geometric capability of a symmetric two-jaw gripper.

    ``pad_size`` is ``(width_along_E_x, height_along_E_z)``.  ``pad_depth`` is
    the usable fingertip-to-palm distance along ``-E_z``.  All dimensions use
    the same unit as the input mesh (metres in the MuJoCo pipeline).
    """

    min_opening: float
    max_opening: float
    pad_size: tuple[float, float]
    pad_depth: float
    friction_coefficient: float = 0.5

    def __post_init__(self) -> None:
        minimum = _finite_scalar(self.min_opening, "min_opening")
        maximum = _finite_scalar(self.max_opening, "max_opening")
        pad = np.asarray(self.pad_size, dtype=float)
        depth = _finite_scalar(self.pad_depth, "pad_depth")
        friction = _finite_scalar(self.friction_coefficient, "friction_coefficient")
        if minimum < 0.0 or maximum <= minimum:
            raise ValueError("opening range must satisfy 0 <= min < max")
        if pad.shape != (2,) or not np.all(np.isfinite(pad)) or np.any(pad <= 0.0):
            raise ValueError("pad_size must contain two positive finite dimensions")
        if depth <= 0.0:
            raise ValueError("pad_depth must be positive")
        if friction < 0.0:
            raise ValueError("friction_coefficient must be non-negative")
        object.__setattr__(self, "min_opening", minimum)
        object.__setattr__(self, "max_opening", maximum)
        object.__setattr__(self, "pad_size", (float(pad[0]), float(pad[1])))
        object.__setattr__(self, "pad_depth", depth)
        object.__setattr__(self, "friction_coefficient", friction)

    @property
    def opening_range(self) -> tuple[float, float]:
        return self.min_opening, self.max_opening


# More explicit name for callers that prefer to distinguish geometry from an
# instantiated gripper model.  It intentionally aliases the same immutable
# value type rather than introducing parallel APIs.
ParallelJawGripperCapability = ParallelJawGripper


@dataclass(frozen=True)
class GraspCandidate:
    """A ranked antipodal contact-pair grasp in the part frame."""

    T_P_E: np.ndarray
    contact_points: np.ndarray
    contact_normals: np.ndarray
    required_opening: float
    approach_direction: np.ndarray
    closing_direction: np.ndarray
    quality: float
    antipodal_quality: float
    support_quality: float
    opening_margin: float
    palm_clearance: float

    def __post_init__(self) -> None:
        transform = np.asarray(self.T_P_E, dtype=float)
        points = np.asarray(self.contact_points, dtype=float)
        normals = np.asarray(self.contact_normals, dtype=float)
        if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
            raise ValueError("T_P_E must be a finite 4x4 transform")
        if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-10):
            raise ValueError("T_P_E has an invalid homogeneous row")
        rotation = transform[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8):
            raise ValueError("T_P_E rotation must be orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-8):
            raise ValueError("T_P_E rotation must be right handed")
        if points.shape != (2, 3) or normals.shape != (2, 3):
            raise ValueError("contacts and normals must have shape (2, 3)")
        if not np.all(np.isfinite(points)) or not np.all(np.isfinite(normals)):
            raise ValueError("contacts and normals must be finite")
        unit_normals = np.vstack([_unit(row, name="contact normal") for row in normals])
        approach = _unit(self.approach_direction, name="approach_direction")
        closing = _unit(self.closing_direction, name="closing_direction")
        if abs(float(approach @ closing)) > 1e-8:
            raise ValueError("approach and closing directions must be orthogonal")
        opening = _finite_scalar(self.required_opening, "required_opening")
        quality_values = [
            _finite_scalar(self.quality, "quality"),
            _finite_scalar(self.antipodal_quality, "antipodal_quality"),
            _finite_scalar(self.support_quality, "support_quality"),
            _finite_scalar(self.opening_margin, "opening_margin"),
            _finite_scalar(self.palm_clearance, "palm_clearance"),
        ]
        if opening <= 0.0 or any(value < 0.0 for value in quality_values[:4]):
            raise ValueError("opening must be positive and quality values non-negative")
        object.__setattr__(self, "T_P_E", transform.copy())
        object.__setattr__(self, "contact_points", points.copy())
        object.__setattr__(self, "contact_normals", unit_normals)
        object.__setattr__(self, "required_opening", opening)
        object.__setattr__(self, "approach_direction", approach)
        object.__setattr__(self, "closing_direction", closing)
        object.__setattr__(self, "quality", quality_values[0])
        object.__setattr__(self, "antipodal_quality", quality_values[1])
        object.__setattr__(self, "support_quality", quality_values[2])
        object.__setattr__(self, "opening_margin", quality_values[3])
        object.__setattr__(self, "palm_clearance", quality_values[4])

    @property
    def transform(self) -> np.ndarray:
        return self.T_P_E.copy()

    @property
    def midpoint(self) -> np.ndarray:
        return self.T_P_E[:3, 3].copy()

    @property
    def contacts(self) -> tuple[np.ndarray, np.ndarray]:
        return self.contact_points[0].copy(), self.contact_points[1].copy()

    @property
    def normals(self) -> tuple[np.ndarray, np.ndarray]:
        return self.contact_normals[0].copy(), self.contact_normals[1].copy()


def _ray_first_hit(
    mesh: TriangleMesh,
    origin: np.ndarray,
    direction: np.ndarray,
    minimum_distance: float,
    *,
    chunk_size: int = 32768,
) -> tuple[np.ndarray, np.ndarray, int] | None:
    """Return the nearest positive Moller--Trumbore ray/triangle hit."""
    direction = _unit(direction, name="ray direction")
    best_t = np.inf
    best_index = -1
    triangle_count = len(mesh.triangles)
    determinant_epsilon = max(1e-12, minimum_distance * 1e-8)
    for begin in range(0, triangle_count, chunk_size):
        end = min(begin + chunk_size, triangle_count)
        triangles = mesh.triangles[begin:end]
        edge_1 = triangles[:, 1] - triangles[:, 0]
        edge_2 = triangles[:, 2] - triangles[:, 0]
        h = np.cross(np.broadcast_to(direction, edge_2.shape), edge_2)
        determinant = np.einsum("ij,ij->i", edge_1, h)
        valid = np.abs(determinant) > determinant_epsilon
        inverse_determinant = np.zeros_like(determinant)
        inverse_determinant[valid] = 1.0 / determinant[valid]
        s = origin[None, :] - triangles[:, 0]
        u = inverse_determinant * np.einsum("ij,ij->i", s, h)
        valid &= (u >= -1e-10) & (u <= 1.0 + 1e-10)
        q = np.cross(s, edge_1)
        v = inverse_determinant * np.einsum(
            "ij,ij->i", np.broadcast_to(direction, q.shape), q)
        valid &= (v >= -1e-10) & ((u + v) <= 1.0 + 1e-10)
        t = inverse_determinant * np.einsum("ij,ij->i", edge_2, q)
        valid &= (t > minimum_distance) & (t < best_t)
        indices = np.flatnonzero(valid)
        if indices.size:
            local = int(indices[np.argmin(t[indices])])
            best_t = float(t[local])
            best_index = begin + local
    if best_index < 0:
        return None
    hit_point = origin + best_t * direction
    return hit_point, mesh.normals[best_index].copy(), best_index


def _surface_principal_directions(mesh: TriangleMesh) -> tuple[np.ndarray, ...]:
    """Area-weighted geometry directions used only to resolve grasp roll."""
    centroids = np.mean(mesh.triangles, axis=1)
    valid = mesh.areas > 64.0 * _EPS
    weights = mesh.areas[valid]
    points = centroids[valid]
    center = np.average(points, axis=0, weights=weights)
    offsets = points - center
    covariance = (offsets * weights[:, None]).T @ offsets / float(np.sum(weights))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    return tuple(_canonical_sign(eigenvectors[:, index]) for index in order)


def _approach_directions(
    closing: np.ndarray,
    principal: Sequence[np.ndarray],
    count: int,
) -> tuple[np.ndarray, ...]:
    projected: list[np.ndarray] = []
    closing = _unit(closing, name="closing direction")
    for vector in principal:
        tangent = np.asarray(vector, dtype=float) - closing * float(vector @ closing)
        if np.linalg.norm(tangent) <= 1e-8:
            continue
        tangent = _canonical_sign(tangent)
        if not any(abs(float(tangent @ old)) > 1.0 - 1e-7 for old in projected):
            projected.append(tangent)
    if not projected:
        reference = np.eye(3)[int(np.argmin(np.abs(closing)))]
        projected.append(_canonical_sign(reference - closing * float(reference @ closing)))
    if len(projected) == 1:
        projected.append(_canonical_sign(np.cross(closing, projected[0])))
    approaches: list[np.ndarray] = []
    for vector in projected:
        approaches.extend((vector, -vector))
    return tuple(approaches[:max(1, int(count))])


def _local_support_quality(
    mesh: TriangleMesh,
    triangle_index: int,
    point: np.ndarray,
    normal: np.ndarray,
    x_axis: np.ndarray,
    z_axis: np.ndarray,
    gripper: ParallelJawGripper,
) -> float:
    """Estimate coplanar surface area available under one rectangular pad."""
    centroids = np.mean(mesh.triangles, axis=1)
    relative = centroids - point
    normal_alignment = mesh.normals @ normal
    plane_distance = np.abs(relative @ normal)
    half_width = 0.5 * gripper.pad_size[0]
    half_height = 0.5 * gripper.pad_size[1]
    scale = max(float(np.linalg.norm(mesh.extent)), 1e-9)
    mask = (
        (normal_alignment >= np.cos(np.radians(12.0)))
        & (plane_distance <= max(1e-7 * scale, 1e-10))
        & (np.abs(relative @ x_axis) <= half_width)
        & (np.abs(relative @ z_axis) <= half_height)
    )
    local_area = float(np.sum(mesh.areas[mask]))
    # A large facet can cover the contact even when its centroid lies outside
    # the pad rectangle.  Always credit the intersected source facet, capped
    # by the physical pad area.
    local_area = max(local_area, min(float(mesh.areas[triangle_index]),
                                    gripper.pad_size[0] * gripper.pad_size[1]))
    pad_area = gripper.pad_size[0] * gripper.pad_size[1]
    return float(np.clip(local_area / pad_area, 0.0, 1.0))


def _opening_margin(opening: float, gripper: ParallelJawGripper) -> float:
    span = gripper.max_opening - gripper.min_opening
    distance = min(opening - gripper.min_opening,
                   gripper.max_opening - opening)
    return float(np.clip(2.0 * distance / span, 0.0, 1.0))


def _candidate_sort_key(candidate: GraspCandidate) -> tuple[float, ...]:
    values = np.concatenate((candidate.midpoint,
                             candidate.closing_direction,
                             candidate.approach_direction))
    return (-candidate.quality, *np.round(values, 12).tolist())


def rank_and_deduplicate(
    candidates: Iterable[GraspCandidate],
    *,
    position_tolerance: float,
    angle_tolerance_deg: float = 7.5,
    max_candidates: int | None = None,
    coverage_scale: float | None = None,
) -> list[GraspCandidate]:
    """Rank candidates and suppress equivalent contact poses.

    Opposite closing-axis signs are considered equivalent because the gripper
    is symmetric; opposite approach directions are intentionally distinct.
    If a result limit is supplied, a deterministic farthest-point term keeps
    good grasps from different portions of a long part.
    """
    tolerance = _finite_scalar(position_tolerance, "position_tolerance")
    angle = _finite_scalar(angle_tolerance_deg, "angle_tolerance_deg")
    if tolerance < 0.0 or not 0.0 <= angle < 90.0:
        raise ValueError("invalid deduplication tolerances")
    if max_candidates is not None and max_candidates <= 0:
        raise ValueError("max_candidates must be positive or None")
    cosine = float(np.cos(np.radians(angle)))
    ordered = sorted(list(candidates), key=_candidate_sort_key)
    unique: list[GraspCandidate] = []
    for candidate in ordered:
        duplicate = any(
            np.linalg.norm(candidate.midpoint - retained.midpoint) <= tolerance
            and abs(float(candidate.closing_direction @ retained.closing_direction)) >= cosine
            and float(candidate.approach_direction @ retained.approach_direction) >= cosine
            for retained in unique
        )
        if not duplicate:
            unique.append(candidate)
    if max_candidates is None or len(unique) <= max_candidates:
        return unique

    scale = float(coverage_scale) if coverage_scale is not None else 0.0
    if not np.isfinite(scale) or scale <= 0.0:
        positions = np.vstack([candidate.midpoint for candidate in unique])
        scale = max(float(np.linalg.norm(np.ptp(positions, axis=0))), tolerance, 1e-12)
    qualities = np.array([candidate.quality for candidate in unique])
    q_min = float(np.min(qualities))
    q_span = max(float(np.max(qualities) - q_min), 1e-12)
    selected_indices = [0]
    remaining = set(range(1, len(unique)))
    while remaining and len(selected_indices) < max_candidates:
        best_index = -1
        best_objective = -np.inf
        for index in sorted(remaining):
            distance = min(np.linalg.norm(unique[index].midpoint - unique[old].midpoint)
                           for old in selected_indices)
            normalized_quality = (unique[index].quality - q_min) / q_span
            objective = 0.82 * normalized_quality + 0.18 * min(distance / scale, 1.0)
            if objective > best_objective + 1e-15:
                best_objective = objective
                best_index = index
        selected_indices.append(best_index)
        remaining.remove(best_index)
    selected = [unique[index] for index in selected_indices]
    return sorted(selected, key=_candidate_sort_key)


def generate_antipodal_grasps(
    mesh: TriangleMesh,
    gripper: ParallelJawGripper,
    *,
    surface_samples: int = 256,
    approaches_per_pair: int = 4,
    max_candidates: int = 128,
) -> list[GraspCandidate]:
    """Generate ranked antipodal parallel-jaw grasps for arbitrary geometry.

    The numerical arguments are computation budgets, not per-part tuning
    parameters.  Feasibility thresholds come from the gripper opening,
    friction cone, pad geometry, and finger depth.
    """
    if not isinstance(mesh, TriangleMesh):
        raise TypeError("mesh must be a TriangleMesh")
    if not isinstance(gripper, ParallelJawGripper):
        raise TypeError("gripper must be a ParallelJawGripper")
    if surface_samples <= 0 or approaches_per_pair <= 0 or max_candidates <= 0:
        raise ValueError("sampling budgets must be positive")
    patches = sample_surface_patches(mesh, int(surface_samples))
    principal = _surface_principal_directions(mesh)
    extent_norm = max(float(np.linalg.norm(mesh.extent)), 1e-9)
    ray_offset = max(1e-8 * extent_norm, 1e-10)
    friction_cosine = 1.0 / np.sqrt(1.0 + gripper.friction_coefficient**2)
    all_vertices = mesh.triangles.reshape(-1, 3)
    candidates: list[GraspCandidate] = []

    for patch in patches:
        inward = -patch.normal
        ray_origin = patch.point + ray_offset * inward
        hit = _ray_first_hit(mesh, ray_origin, inward, ray_offset)
        if hit is None:
            continue
        hit_point, hit_normal, hit_triangle = hit
        separation_vector = hit_point - patch.point
        opening = float(np.linalg.norm(separation_vector))
        if not (gripper.min_opening <= opening <= gripper.max_opening):
            continue
        closing = separation_vector / opening
        contact_points = np.vstack((patch.point, hit_point))
        contact_normals = np.vstack((patch.normal, hit_normal))

        # Normalize finger order for a symmetric gripper.  This removes the
        # duplicate obtained by casting the reciprocal ray.
        canonical = _canonical_sign(closing)
        if float(canonical @ closing) < 0.0:
            closing = -closing
            contact_points = contact_points[::-1].copy()
            contact_normals = contact_normals[::-1].copy()
            source_triangle, target_triangle = hit_triangle, patch.triangle_index
        else:
            source_triangle, target_triangle = patch.triangle_index, hit_triangle

        alignment_0 = float(contact_normals[0] @ (-closing))
        alignment_1 = float(contact_normals[1] @ closing)
        antipodal = min(alignment_0, alignment_1)
        if antipodal + 1e-10 < friction_cosine:
            continue
        midpoint = np.mean(contact_points, axis=0)

        for approach in _approach_directions(
                closing, principal, int(approaches_per_pair)):
            x_axis = _unit(np.cross(closing, approach), name="pad width axis")
            z_axis = _unit(np.cross(x_axis, closing), name="approach axis")
            # The part must fit between the contact plane and palm.  Mesh
            # vertices outside the jaw slab cannot strike this idealized palm,
            # so only inspect points between the two jaw planes plus one pad
            # width of lateral allowance.
            relative = all_vertices - midpoint
            jaw_coordinate = relative @ closing
            slab = np.abs(jaw_coordinate) <= 0.5 * opening + 1e-9
            rear_extent = max(0.0, -float(np.min(relative[slab] @ z_axis)))
            palm_clearance = gripper.pad_depth - rear_extent
            if palm_clearance < -1e-9:
                continue

            support_0 = _local_support_quality(
                mesh, source_triangle, contact_points[0], contact_normals[0],
                x_axis, z_axis, gripper)
            support_1 = _local_support_quality(
                mesh, target_triangle, contact_points[1], contact_normals[1],
                x_axis, z_axis, gripper)
            support = min(support_0, support_1)
            margin = _opening_margin(opening, gripper)
            clearance_score = float(np.clip(
                palm_clearance / gripper.pad_depth, 0.0, 1.0))
            normalized_antipodal = float(np.clip(
                (antipodal - friction_cosine) / max(1.0 - friction_cosine, 1e-12),
                0.0, 1.0))
            quality = (
                0.45 * normalized_antipodal
                + 0.25 * support
                + 0.15 * margin
                + 0.15 * clearance_score
            )
            transform = np.eye(4)
            transform[:3, :3] = np.column_stack((x_axis, closing, z_axis))
            transform[:3, 3] = midpoint
            candidates.append(GraspCandidate(
                T_P_E=transform,
                contact_points=contact_points,
                contact_normals=contact_normals,
                required_opening=opening,
                approach_direction=z_axis,
                closing_direction=closing,
                quality=quality,
                antipodal_quality=antipodal,
                support_quality=support,
                opening_margin=margin,
                palm_clearance=max(0.0, palm_clearance),
            ))

    position_tolerance = max(
        1e-6 * extent_norm,
        0.15 * min(gripper.pad_size),
        1e-7,
    )
    return rank_and_deduplicate(
        candidates,
        position_tolerance=position_tolerance,
        max_candidates=int(max_candidates),
        coverage_scale=extent_norm,
    )


__all__ = [
    "GraspCandidate",
    "ParallelJawGripper",
    "ParallelJawGripperCapability",
    "SurfacePatch",
    "TriangleMesh",
    "generate_antipodal_grasps",
    "load_binary_stl",
    "rank_and_deduplicate",
    "sample_surface_patches",
]
