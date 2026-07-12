"""Geometry-derived stable placements and rectangular-stage instantiation.

The implementation depends only on NumPy and :class:`TriangleMesh`; it does
not require SciPy, trimesh, MuJoCo, or part-specific orientation rules.

Frame convention
----------------

Each :class:`StablePlacement` stores ``T_N_P == ^N T_P``.  It maps points from
the native part frame ``P`` into a canonical horizontal support frame ``N``:

* the support plane is ``N.z = 0``;
* ``+N.z`` points away from the supporting surface; and
* the selected outward part normal maps to ``-N.z``.

The in-plane origin is the centroid of the convex support polygon.  A stage
instance composes this transform as
``T_W_P = T_W_S @ T_S_N @ T_N_P``, where ``S.z = 0`` is the stage surface and
``T_S_N`` contains the selected yaw and feasible in-plane translation.

Method
------

Approximately coplanar triangles with aligned outward normals are grouped into
support facets.  Only groups on the extreme supporting plane are retained, so
a recessed parallel face cannot pass through the stage.  The part COM is
orthogonally projected into the facet's two-dimensional convex hull.  A pose
is stable when the signed distance to every hull edge is at least the requested
support margin.  This is the standard quasistatic gravity criterion; friction
and dynamic tipping are deliberately left to downstream validation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .geometry_grasps import TriangleMesh


_EPS = np.finfo(float).eps


def _readonly(value: np.ndarray | Sequence[float], shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite array with shape {shape}")
    result = array.copy()
    result.setflags(write=False)
    return result


def _finite_scalar(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _unit(value: np.ndarray | Sequence[float], name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite three-vector")
    norm = float(np.linalg.norm(vector))
    if norm <= 64.0 * _EPS:
        raise ValueError(f"{name} must be nonzero")
    return vector / norm


def _validate_transform(value: np.ndarray, name: str) -> np.ndarray:
    transform = _readonly(value, (4, 4), name)
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-10):
        raise ValueError(f"{name} has an invalid homogeneous row")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8):
        raise ValueError(f"{name} rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-8):
        raise ValueError(f"{name} rotation must be right handed")
    return transform


def _mesh_scale(mesh: TriangleMesh) -> float:
    return max(float(np.linalg.norm(mesh.extent)), 1e-12)


@dataclass(frozen=True)
class _MeshComponent:
    triangle_indices: tuple[int, ...]
    closed: bool


def _mesh_components(mesh: TriangleMesh, tolerance: float) -> tuple[_MeshComponent, ...]:
    """Find edge-connected components and classify their closedness.

    CAD assemblies commonly store several disconnected watertight solids in a
    single STL.  Treating one damaged pin as evidence that the *whole* mesh is
    open discards the reliable mass properties of every other component, so
    topology and winding are evaluated per edge-connected component.
    """

    origin = mesh.bounds_min
    vertex_ids: dict[tuple[int, ...], int] = {}
    triangle_vertices = np.empty((len(mesh.triangles), 3), dtype=np.int64)
    for triangle_index, triangle in enumerate(mesh.triangles):
        for local_index, vertex in enumerate(triangle):
            key = tuple(np.rint((vertex - origin) / tolerance).astype(np.int64))
            triangle_vertices[triangle_index, local_index] = vertex_ids.setdefault(
                key, len(vertex_ids)
            )

    parent = np.arange(len(mesh.triangles), dtype=np.int64)

    def find(index: int) -> int:
        root = index
        while int(parent[root]) != root:
            root = int(parent[root])
        while int(parent[index]) != index:
            next_index = int(parent[index])
            parent[index] = root
            index = next_index
        return root

    def union(first: int, second: int) -> None:
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    edge_triangles: dict[tuple[int, int], list[int]] = {}
    triangle_edges: list[tuple[tuple[int, int], ...]] = []
    for triangle_index, vertices in enumerate(triangle_vertices):
        edges: list[tuple[int, int]] = []
        for first, second in ((0, 1), (1, 2), (2, 0)):
            edge = tuple(sorted((int(vertices[first]), int(vertices[second]))))
            if edge[0] == edge[1]:
                continue
            edges.append(edge)
            edge_triangles.setdefault(edge, []).append(triangle_index)
        triangle_edges.append(tuple(edges))
    for incident in edge_triangles.values():
        for triangle_index in incident[1:]:
            union(incident[0], triangle_index)

    grouped: dict[int, list[int]] = {}
    for triangle_index in range(len(mesh.triangles)):
        grouped.setdefault(find(triangle_index), []).append(triangle_index)
    components: list[_MeshComponent] = []
    for indices in grouped.values():
        edge_counts: dict[tuple[int, int], int] = {}
        for triangle_index in indices:
            for edge in triangle_edges[triangle_index]:
                edge_counts[edge] = edge_counts.get(edge, 0) + 1
        closed = bool(edge_counts) and all(count == 2 for count in edge_counts.values())
        components.append(_MeshComponent(tuple(indices), closed))
    components.sort(key=lambda item: item.triangle_indices[0])
    return tuple(components)


def _mesh_is_closed(mesh: TriangleMesh, tolerance: float) -> bool:
    components = _mesh_components(mesh, tolerance)
    return bool(components) and all(component.closed for component in components)


def _component_volume_properties(
    mesh: TriangleMesh,
    component: _MeshComponent,
) -> tuple[float, np.ndarray] | None:
    triangles = mesh.triangles[list(component.triangle_indices)]
    vertices = triangles.reshape(-1, 3)
    reference = 0.5 * (np.min(vertices, axis=0) + np.max(vertices, axis=0))
    relative = triangles - reference
    volume6 = np.einsum(
        "ij,ij->i",
        relative[:, 0],
        np.cross(relative[:, 1], relative[:, 2]),
    )
    total = float(np.sum(volume6))
    extent = np.ptp(vertices, axis=0)
    scale = max(float(np.linalg.norm(extent)), 1e-12)
    volume_tolerance6 = max(6e-12 * float(np.prod(extent)), 1e-15 * scale**3)
    if not np.isfinite(total) or abs(total) <= volume_tolerance6:
        return None
    centroid = np.sum(
        volume6[:, None] * (reference + np.sum(triangles, axis=1)), axis=0
    ) / (4.0 * total)
    tolerance = max(1e-7 * scale, 1e-10)
    if (
        not np.all(np.isfinite(centroid))
        or np.any(centroid < np.min(vertices, axis=0) - tolerance)
        or np.any(centroid > np.max(vertices, axis=0) + tolerance)
    ):
        return None
    return abs(total), centroid


def _outward_component_normals(
    mesh: TriangleMesh,
    components: Sequence[_MeshComponent],
) -> np.ndarray:
    """Correct global winding reversals independently for each closed solid."""

    triangles = mesh.triangles
    cross = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    lengths = np.linalg.norm(cross, axis=1)
    raw = np.zeros_like(cross)
    valid = lengths > 64.0 * _EPS
    raw[valid] = cross[valid] / lengths[valid, None]
    corrected = np.asarray(mesh.normals, dtype=float).copy()
    for component in components:
        if not component.closed:
            continue
        indices = np.asarray(component.triangle_indices, dtype=int)
        triangles_component = triangles[indices]
        vertices = triangles_component.reshape(-1, 3)
        reference = 0.5 * (np.min(vertices, axis=0) + np.max(vertices, axis=0))
        relative = triangles_component - reference
        signed_volume6 = float(np.sum(np.einsum(
            "ij,ij->i",
            relative[:, 0],
            np.cross(relative[:, 1], relative[:, 2]),
        )))
        if abs(signed_volume6) > 1e-18:
            corrected[indices] = raw[indices] * np.sign(signed_volume6)
    return corrected


def estimate_center_of_mass(mesh: TriangleMesh) -> np.ndarray:
    """Return the closed-mesh volume centroid, with a bbox-center fallback.

    Uniform density is assumed because an STL carries no material data.  The
    tetrahedral signed-volume formula is invariant to a global winding
    reversal.  Open, nonmanifold, near-zero-volume, or numerically implausible
    meshes fall back to the native-frame bounding-box center.
    """

    if not isinstance(mesh, TriangleMesh):
        raise TypeError("mesh must be a TriangleMesh")
    fallback = 0.5 * (mesh.bounds_min + mesh.bounds_max)
    scale = _mesh_scale(mesh)
    edge_tolerance = max(1e-8 * scale, 1e-12)
    components = _mesh_components(mesh, edge_tolerance)
    closed_area = sum(
        float(np.sum(mesh.areas[list(component.triangle_indices)]))
        for component in components if component.closed
    )
    # A small watertight fastener inside a mostly open shell is not a useful
    # mass proxy.  Require closed solids to represent most of the CAD surface.
    if closed_area < 0.5 * mesh.surface_area:
        return fallback
    properties = [
        value
        for component in components if component.closed
        for value in [_component_volume_properties(mesh, component)]
        if value is not None
    ]
    if not properties:
        return fallback
    weights = np.array([item[0] for item in properties])
    centers = np.vstack([item[1] for item in properties])
    centroid = np.average(centers, axis=0, weights=weights)
    tolerance = max(1e-7 * scale, 1e-10)
    if (
        not np.all(np.isfinite(centroid))
        or np.any(centroid < mesh.bounds_min - tolerance)
        or np.any(centroid > mesh.bounds_max + tolerance)
    ):
        return fallback
    return centroid


def _cross_2d(origin: np.ndarray, first: np.ndarray, second: np.ndarray) -> float:
    a = first - origin
    b = second - origin
    return float(a[0] * b[1] - a[1] * b[0])


def convex_hull_2d(points: np.ndarray, tolerance: float | None = None) -> np.ndarray:
    """Return the counter-clockwise convex hull of finite 2-D points."""

    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or not np.all(np.isfinite(values)):
        raise ValueError("points must have shape (N, 2) and be finite")
    if len(values) < 3:
        return values.copy()
    scale = max(float(np.linalg.norm(np.ptp(values, axis=0))), 1.0)
    distance_tolerance = (
        max(float(tolerance), 0.0)
        if tolerance is not None
        else 1e-12 * scale
    )
    order = np.lexsort((values[:, 1], values[:, 0]))
    unique: list[np.ndarray] = []
    for index in order:
        point = values[index]
        if not unique or np.linalg.norm(point - unique[-1]) > distance_tolerance:
            unique.append(point.copy())
    if len(unique) < 3:
        return np.asarray(unique, dtype=float)

    area_tolerance = distance_tolerance * scale

    def half(sequence: Iterable[np.ndarray]) -> list[np.ndarray]:
        result: list[np.ndarray] = []
        for point in sequence:
            while (
                len(result) >= 2
                and _cross_2d(result[-2], result[-1], point) <= area_tolerance
            ):
                result.pop()
            result.append(point)
        return result

    lower = half(unique)
    upper = half(reversed(unique))
    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def polygon_area(polygon: np.ndarray) -> float:
    values = np.asarray(polygon, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or len(values) < 3:
        return 0.0
    return 0.5 * abs(float(np.sum(
        values[:, 0] * np.roll(values[:, 1], -1)
        - values[:, 1] * np.roll(values[:, 0], -1)
    )))


def _polygon_centroid(polygon: np.ndarray) -> np.ndarray:
    cross = (
        polygon[:, 0] * np.roll(polygon[:, 1], -1)
        - np.roll(polygon[:, 0], -1) * polygon[:, 1]
    )
    signed_double_area = float(np.sum(cross))
    if abs(signed_double_area) <= 64.0 * _EPS:
        return np.mean(polygon, axis=0)
    x = np.sum((polygon[:, 0] + np.roll(polygon[:, 0], -1)) * cross)
    y = np.sum((polygon[:, 1] + np.roll(polygon[:, 1], -1)) * cross)
    return np.array([x, y], dtype=float) / (3.0 * signed_double_area)


def signed_support_margin(point: Sequence[float], polygon: np.ndarray) -> float:
    """Signed minimum distance from a point to a CCW convex polygon edge.

    Positive values are inside, zero is on the boundary, and negative values
    are outside.  ``polygon`` should normally come from :func:`convex_hull_2d`.
    """

    query = np.asarray(point, dtype=float)
    hull = np.asarray(polygon, dtype=float)
    if query.shape != (2,) or not np.all(np.isfinite(query)):
        raise ValueError("point must be a finite two-vector")
    if hull.ndim != 2 or hull.shape[1] != 2 or len(hull) < 3:
        raise ValueError("polygon must contain at least three 2-D vertices")
    if not np.all(np.isfinite(hull)):
        raise ValueError("polygon must be finite")
    edges = np.roll(hull, -1, axis=0) - hull
    lengths = np.linalg.norm(edges, axis=1)
    if np.any(lengths <= 64.0 * _EPS):
        raise ValueError("polygon contains a zero-length edge")
    offsets = query - hull
    cross = edges[:, 0] * offsets[:, 1] - edges[:, 1] * offsets[:, 0]
    orientation = np.sign(np.sum(
        hull[:, 0] * np.roll(hull[:, 1], -1)
        - hull[:, 1] * np.roll(hull[:, 0], -1)
    ))
    if orientation == 0.0:
        raise ValueError("polygon has zero area")
    return float(np.min(orientation * cross / lengths))


@dataclass(frozen=True)
class StablePlacement:
    """One quasistatically stable native-part pose on ``N.z = 0``."""

    T_N_P: np.ndarray
    support_polygon_N: np.ndarray
    support_normal_P: np.ndarray
    center_of_mass_N: np.ndarray
    support_margin: float
    support_area: float
    probability_proxy: float
    triangle_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        transform = _validate_transform(self.T_N_P, "T_N_P")
        polygon = np.asarray(self.support_polygon_N, dtype=float)
        if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
            raise ValueError("support_polygon_N must have shape (N>=3, 2)")
        if not np.all(np.isfinite(polygon)):
            raise ValueError("support_polygon_N must be finite")
        polygon = polygon.copy()
        polygon.setflags(write=False)
        normal = _unit(self.support_normal_P, "support_normal_P")
        normal.setflags(write=False)
        center = _readonly(self.center_of_mass_N, (3,), "center_of_mass_N")
        margin = _finite_scalar(self.support_margin, "support_margin")
        area = _finite_scalar(self.support_area, "support_area")
        probability = _finite_scalar(self.probability_proxy, "probability_proxy")
        indices = tuple(sorted({int(index) for index in self.triangle_indices}))
        if margin < -1e-12:
            raise ValueError("a StablePlacement cannot have a negative support margin")
        if area <= 0.0:
            raise ValueError("support_area must be positive")
        if not 0.0 <= probability <= 1.0 + 1e-12:
            raise ValueError("probability_proxy must be in [0, 1]")
        if not indices or indices[0] < 0:
            raise ValueError("triangle_indices must be non-empty and non-negative")
        object.__setattr__(self, "T_N_P", transform)
        object.__setattr__(self, "support_polygon_N", polygon)
        object.__setattr__(self, "support_normal_P", normal)
        object.__setattr__(self, "center_of_mass_N", center)
        object.__setattr__(self, "support_margin", max(0.0, margin))
        object.__setattr__(self, "support_area", area)
        object.__setattr__(self, "probability_proxy", min(1.0, probability))
        object.__setattr__(self, "triangle_indices", indices)

    @property
    def rotation_N_P(self) -> np.ndarray:
        return self.T_N_P[:3, :3].copy()


@dataclass
class _FacetGroup:
    indices: list[int]
    normal: np.ndarray
    point: np.ndarray
    total_area: float


@dataclass
class _PlacementGeometry:
    transform: np.ndarray
    polygon: np.ndarray
    normal: np.ndarray
    center: np.ndarray
    margin: float
    area: float
    indices: tuple[int, ...]


def _facet_groups(
    mesh: TriangleMesh,
    outward_normals: np.ndarray,
    normal_tolerance_deg: float,
    plane_tolerance: float,
) -> list[_FacetGroup]:
    cosine = float(np.cos(np.radians(normal_tolerance_deg)))
    centroids = np.mean(mesh.triangles, axis=1)
    groups: list[_FacetGroup] = []
    valid = np.flatnonzero(mesh.areas > 64.0 * _EPS)
    for index in valid:
        index = int(index)
        normal = outward_normals[index]
        area = float(mesh.areas[index])
        triangle = mesh.triangles[index]
        assigned = False
        for group in groups:
            if float(normal @ group.normal) < cosine:
                continue
            if np.max(np.abs((triangle - group.point) @ group.normal)) > plane_tolerance:
                continue
            combined = group.total_area + area
            averaged = group.total_area * group.normal + area * normal
            group.normal = _unit(averaged, "facet normal")
            group.point = (
                group.total_area * group.point + area * centroids[index]
            ) / combined
            group.total_area = combined
            group.indices.append(index)
            assigned = True
            break
        if not assigned:
            groups.append(_FacetGroup(
                indices=[index],
                normal=normal.copy(),
                point=centroids[index].copy(),
                total_area=area,
            ))
    return groups


def _support_rotation(normal_P: np.ndarray) -> np.ndarray:
    """Return ``R_N_P`` whose +N.z axis opposes the support normal."""

    normal = _unit(normal_P, "support normal")
    z_axis_P = -normal
    reference_index = int(np.argmin(np.abs(z_axis_P)))
    reference = np.eye(3)[reference_index]
    x_axis_P = reference - z_axis_P * float(reference @ z_axis_P)
    x_axis_P /= np.linalg.norm(x_axis_P)
    pivot = int(np.argmax(np.abs(x_axis_P)))
    if x_axis_P[pivot] < 0.0:
        x_axis_P = -x_axis_P
    y_axis_P = np.cross(z_axis_P, x_axis_P)
    return np.vstack((x_axis_P, y_axis_P, z_axis_P))


def _rotation_distance(first: np.ndarray, second: np.ndarray) -> float:
    cosine = np.clip((np.trace(first @ second.T) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.arccos(cosine))


def generate_stable_placements(
    mesh: TriangleMesh,
    *,
    center_of_mass_P: Sequence[float] | np.ndarray | None = None,
    normal_tolerance_deg: float = 3.0,
    plane_tolerance: float | None = None,
    minimum_support_margin: float = 0.0,
    minimum_support_area: float = 0.0,
    rotation_tolerance_deg: float = 1.0,
) -> tuple[StablePlacement, ...]:
    """Derive stable support poses from an arbitrary triangular mesh.

    Tolerances are numerical/model-resolution settings, not part-orientation
    rules.  ``center_of_mass_P`` may include known payload/material effects;
    otherwise a uniform-density closed-mesh volume centroid is used.
    """

    if not isinstance(mesh, TriangleMesh):
        raise TypeError("mesh must be a TriangleMesh")
    normal_tolerance = _finite_scalar(normal_tolerance_deg, "normal_tolerance_deg")
    rotation_tolerance = _finite_scalar(rotation_tolerance_deg, "rotation_tolerance_deg")
    minimum_margin = _finite_scalar(minimum_support_margin, "minimum_support_margin")
    minimum_area = _finite_scalar(minimum_support_area, "minimum_support_area")
    if not 0.0 <= normal_tolerance < 90.0:
        raise ValueError("normal_tolerance_deg must be in [0, 90)")
    if not 0.0 <= rotation_tolerance < 180.0:
        raise ValueError("rotation_tolerance_deg must be in [0, 180)")
    if minimum_margin < 0.0 or minimum_area < 0.0:
        raise ValueError("minimum support margin and area must be non-negative")

    scale = _mesh_scale(mesh)
    plane_tol = (
        # Binary STL float quantization and independently exported assembly
        # components routinely perturb nominally common planes by sub-micron
        # amounts.  This remains five orders below the part scale.
        max(1e-5 * scale, 1e-9)
        if plane_tolerance is None
        else _finite_scalar(plane_tolerance, "plane_tolerance")
    )
    if plane_tol <= 0.0:
        raise ValueError("plane_tolerance must be positive")
    topology_tolerance = max(1e-8 * scale, 1e-12)
    components = _mesh_components(mesh, topology_tolerance)
    outward_normals = _outward_component_normals(mesh, components)
    center_P = (
        estimate_center_of_mass(mesh)
        if center_of_mass_P is None
        else np.asarray(center_of_mass_P, dtype=float)
    )
    if center_P.shape != (3,) or not np.all(np.isfinite(center_P)):
        raise ValueError("center_of_mass_P must be a finite three-vector")

    all_vertices = mesh.triangles.reshape(-1, 3)
    candidates: list[_PlacementGeometry] = []
    area_tolerance = max(1e-12 * scale**2, 64.0 * _EPS)
    for group in _facet_groups(mesh, outward_normals, normal_tolerance, plane_tol):
        normal = _unit(group.normal, "facet normal")
        contact_vertices = mesh.triangles[group.indices].reshape(-1, 3)
        facet_offset = float(np.average(contact_vertices @ normal))

        # With the normal placed downward, z_N = d - n_P dot p_P.  Any
        # vertex with n dot p > d would penetrate the support plane, so only
        # an extreme outward patch can define a physical placement.
        extreme_offset = float(np.max(all_vertices @ normal))
        if extreme_offset > facet_offset + plane_tol:
            continue

        rotation = _support_rotation(normal)
        transform = np.eye(4)
        transform[:3, :3] = rotation
        # Use the global extreme rather than the facet's average plane so
        # small tessellation noise cannot place any mesh vertex below N.z=0.
        transform[2, 3] = extreme_offset
        contact_N = contact_vertices @ rotation.T + transform[:3, 3]
        hull = convex_hull_2d(contact_N[:, :2], tolerance=1e-10 * scale)
        area = polygon_area(hull)
        if len(hull) < 3 or area <= max(minimum_area, area_tolerance):
            continue

        # Canonicalize in-plane translation at the support polygon centroid.
        polygon_center = _polygon_centroid(hull)
        transform[:2, 3] = -polygon_center
        hull = hull - polygon_center
        center_N = rotation @ center_P + transform[:3, 3]
        margin = signed_support_margin(center_N[:2], hull)
        if margin + max(1e-12 * scale, 1e-12) < minimum_margin:
            continue
        candidates.append(_PlacementGeometry(
            transform=transform,
            polygon=hull,
            normal=normal,
            center=center_N,
            margin=max(0.0, margin),
            area=area,
            indices=tuple(group.indices),
        ))

    # Equivalent support rotations can arise from duplicate, overlapping, or
    # numerically split CAD facets.  Retain the most stable/largest witness.
    candidates.sort(key=lambda item: (
        -item.margin,
        -item.area,
        *np.round(item.normal, 12).tolist(),
    ))
    unique: list[_PlacementGeometry] = []
    rotation_tol = np.radians(rotation_tolerance)
    for candidate in candidates:
        if any(_rotation_distance(candidate.transform[:3, :3], old.transform[:3, :3])
               <= rotation_tol for old in unique):
            continue
        unique.append(candidate)

    if not unique:
        return ()
    length_floor = max(1e-9 * scale, 1e-12)
    weights = np.array([
        item.area * max(item.margin, length_floor) for item in unique
    ])
    if not np.all(np.isfinite(weights)) or float(np.sum(weights)) <= 0.0:
        weights = np.ones(len(unique))
    probabilities = weights / float(np.sum(weights))
    placements = [
        StablePlacement(
            T_N_P=item.transform,
            support_polygon_N=item.polygon,
            support_normal_P=item.normal,
            center_of_mass_N=item.center,
            support_margin=item.margin,
            support_area=item.area,
            probability_proxy=float(probability),
            triangle_indices=item.indices,
        )
        for item, probability in zip(unique, probabilities)
    ]
    placements.sort(key=lambda item: (
        -item.probability_proxy,
        -item.support_margin,
        -item.support_area,
        *np.round(item.support_normal_P, 12).tolist(),
    ))
    return tuple(placements)


@dataclass(frozen=True)
class RectangularStage:
    """A rectangular support region centered on frame ``S`` at ``S.z=0``."""

    T_W_S: np.ndarray
    size_xy: tuple[float, float]
    edge_margin: float = 0.0

    def __post_init__(self) -> None:
        transform = _validate_transform(self.T_W_S, "T_W_S")
        size = np.asarray(self.size_xy, dtype=float)
        margin = _finite_scalar(self.edge_margin, "edge_margin")
        if size.shape != (2,) or not np.all(np.isfinite(size)) or np.any(size <= 0.0):
            raise ValueError("size_xy must contain two positive finite dimensions")
        if margin < 0.0 or margin >= 0.5 * float(np.min(size)):
            raise ValueError("edge_margin must be non-negative and leave a nonempty stage")
        object.__setattr__(self, "T_W_S", transform)
        object.__setattr__(self, "size_xy", (float(size[0]), float(size[1])))
        object.__setattr__(self, "edge_margin", margin)


@dataclass(frozen=True)
class PlacementInstance:
    """One stable placement centered in its feasible stage translation set."""

    placement: StablePlacement
    yaw_rad: float
    T_W_P: np.ndarray
    footprint_polygon_S: np.ndarray
    support_polygon_S: np.ndarray
    translation_bounds_S: np.ndarray
    edge_clearance: float

    def __post_init__(self) -> None:
        if not isinstance(self.placement, StablePlacement):
            raise TypeError("placement must be a StablePlacement")
        transform = _validate_transform(self.T_W_P, "T_W_P")
        yaw = _finite_scalar(self.yaw_rad, "yaw_rad")
        footprint = np.asarray(self.footprint_polygon_S, dtype=float)
        support = np.asarray(self.support_polygon_S, dtype=float)
        bounds = np.asarray(self.translation_bounds_S, dtype=float)
        if footprint.ndim != 2 or footprint.shape[1] != 2 or len(footprint) < 3:
            raise ValueError("footprint_polygon_S must have shape (N>=3, 2)")
        if support.ndim != 2 or support.shape[1] != 2 or len(support) < 3:
            raise ValueError("support_polygon_S must have shape (N>=3, 2)")
        if bounds.shape != (2, 2) or np.any(bounds[:, 0] > bounds[:, 1]):
            raise ValueError("translation_bounds_S must have shape (2, 2)")
        if not all(np.all(np.isfinite(item)) for item in (footprint, support, bounds)):
            raise ValueError("placement polygons and translation bounds must be finite")
        footprint = footprint.copy(); footprint.setflags(write=False)
        support = support.copy(); support.setflags(write=False)
        bounds = bounds.copy(); bounds.setflags(write=False)
        clearance = _finite_scalar(self.edge_clearance, "edge_clearance")
        if clearance < -1e-12:
            raise ValueError("edge_clearance must be non-negative")
        object.__setattr__(self, "yaw_rad", yaw)
        object.__setattr__(self, "T_W_P", transform)
        object.__setattr__(self, "footprint_polygon_S", footprint)
        object.__setattr__(self, "support_polygon_S", support)
        object.__setattr__(self, "translation_bounds_S", bounds)
        object.__setattr__(self, "edge_clearance", max(0.0, clearance))


def _yaw_values(yaw_samples_deg: Iterable[float]) -> tuple[float, ...]:
    values: list[float] = []
    for raw in yaw_samples_deg:
        degrees = _finite_scalar(raw, "yaw sample")
        radians = (np.radians(degrees) + np.pi) % (2.0 * np.pi) - np.pi
        if not any(abs(float(np.arctan2(np.sin(radians - old),
                                        np.cos(radians - old)))) <= 1e-12
                   for old in values):
            values.append(float(radians))
    if not values:
        raise ValueError("yaw_samples_deg must contain at least one angle")
    return tuple(sorted(values))


def instantiate_on_rectangular_stage(
    mesh: TriangleMesh,
    placements: Sequence[StablePlacement],
    stage: RectangularStage,
    *,
    yaw_samples_deg: Iterable[float] = (0.0,),
    additional_edge_margin: float = 0.0,
) -> tuple[PlacementInstance, ...]:
    """Instantiate stable poses whose complete mesh footprint fits a stage.

    For each support/yaw pair, this computes the full projected part footprint
    (not merely the contact polygon), its complete feasible XY translation
    intervals, and selects their midpoint.  Candidates that cannot satisfy the
    rectangular containment constraint are omitted.
    """

    if not isinstance(mesh, TriangleMesh):
        raise TypeError("mesh must be a TriangleMesh")
    if not isinstance(stage, RectangularStage):
        raise TypeError("stage must be a RectangularStage")
    margin = _finite_scalar(additional_edge_margin, "additional_edge_margin")
    if margin < 0.0:
        raise ValueError("additional_edge_margin must be non-negative")
    half_stage = 0.5 * np.asarray(stage.size_xy) - stage.edge_margin - margin
    if np.any(half_stage <= 0.0):
        raise ValueError("combined edge margins leave no usable stage")
    yaws = _yaw_values(yaw_samples_deg)
    vertices_P = mesh.triangles.reshape(-1, 3)
    instances: list[PlacementInstance] = []

    for placement in placements:
        if not isinstance(placement, StablePlacement):
            raise TypeError("placements must contain only StablePlacement values")
        vertices_N = (
            vertices_P @ placement.T_N_P[:3, :3].T
            + placement.T_N_P[:3, 3]
        )
        footprint_N = convex_hull_2d(vertices_N[:, :2])
        if len(footprint_N) < 3:
            continue
        for yaw in yaws:
            cosine, sine = float(np.cos(yaw)), float(np.sin(yaw))
            rotation_2d = np.array([[cosine, -sine], [sine, cosine]])
            footprint_rotated = footprint_N @ rotation_2d.T
            support_rotated = placement.support_polygon_N @ rotation_2d.T
            minimum = np.min(footprint_rotated, axis=0)
            maximum = np.max(footprint_rotated, axis=0)
            translation_lower = -half_stage - minimum
            translation_upper = half_stage - maximum
            if np.any(translation_lower > translation_upper + 1e-12):
                continue
            translation = 0.5 * (translation_lower + translation_upper)
            footprint_S = footprint_rotated + translation
            support_S = support_rotated + translation
            clearance = float(np.min(np.concatenate((
                half_stage - np.max(footprint_S, axis=0),
                np.min(footprint_S, axis=0) + half_stage,
            ))))

            T_S_N = np.eye(4)
            T_S_N[:2, :2] = rotation_2d
            T_S_N[:2, 3] = translation
            transform = stage.T_W_S @ T_S_N @ placement.T_N_P
            instances.append(PlacementInstance(
                placement=placement,
                yaw_rad=yaw,
                T_W_P=transform,
                footprint_polygon_S=footprint_S,
                support_polygon_S=support_S,
                translation_bounds_S=np.column_stack((
                    translation_lower, translation_upper
                )),
                edge_clearance=max(0.0, clearance),
            ))
    return tuple(instances)


# Shorter semantic aliases for downstream task-and-motion planners.
derive_stable_placements = generate_stable_placements
instantiate_placements = instantiate_on_rectangular_stage


__all__ = [
    "PlacementInstance",
    "RectangularStage",
    "StablePlacement",
    "convex_hull_2d",
    "derive_stable_placements",
    "estimate_center_of_mass",
    "generate_stable_placements",
    "instantiate_on_rectangular_stage",
    "instantiate_placements",
    "polygon_area",
    "signed_support_margin",
]
