"""Continuous, object-only grasp map for an ideal parallel-jaw gripper.

The map produced here is deliberately smaller in scope than a grasp planner.
It answers one geometric question: for a part held at its insertion pose, which
pairs of locally outward-facing lateral surface points can an ideal two-finger
gripper contact?  It does not sample end-effector poses.  Instead, it returns a
finite union of continuous families parameterized by ``(u, v, roll)``.

For one family, a ray parallel to the jaw-closing axis intersects two planar
surface patches.  Projecting the patches along that axis turns their common
contact domain into a union of convex polygons.  The two ray/plane
intersections and their separation are affine functions of ``(u, v)``.
Consequently ``contains`` and ``evaluate`` work at arbitrary continuous
parameters without a lookup table.

The facets belong to the outward-wound boundary of the supplied part mesh and
their normals face the corresponding finger.  We do not require them to be
global convex-hull support planes; that would incorrectly remove useful
housing faces whenever a distant pin protrudes farther.  Directional external
ray visibility and full gripper/part collision remain downstream checks.

Scope warning
-------------

The result is ``OBJECT_ONLY_LOCAL_SURFACE_CANDIDATE``.  It does not certify
finger/palm collision, finite pad contact, force closure under a task wrench,
insertion path clearance, PCB clearance, robot IK, calibration uncertainty, or
dynamics.
Those checks must be applied downstream before calling a grasp insertion-safe.

Frame convention
----------------

``T_X_Y`` maps coordinates in frame Y into frame X.  ``P`` is the part STL
frame at the authored insertion pose and ``W`` is world.  In each evaluated
end-effector frame ``E``, ``+Y_E`` points from the negative contact to the
positive contact (jaw closing line), ``+Z_E`` is the roll-dependent approach
axis, and the origin is the contact midpoint.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..core.se3 import compose, make_transform, validate_transform
from .grasps import TriangleMesh, load_binary_stl


_EPS = np.finfo(float).eps
SCOPE = "OBJECT_ONLY_LOCAL_SURFACE_CANDIDATE"


def _finite_scalar(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def load_scaled_binary_stl(
    path: str | Path,
    *,
    scale_to_m: float,
) -> TriangleMesh:
    """Load a binary STL and apply its declared linear scale to metres."""
    scale = _finite_scalar(scale_to_m, "scale_to_m")
    if scale <= 0.0:
        raise ValueError("scale_to_m must be positive")
    source = Path(path)
    native = load_binary_stl(source)
    return TriangleMesh.from_triangles(
        native.triangles * scale,
        source=str(source),
    )


def _unit(value: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite three-vector")
    norm = float(np.linalg.norm(vector))
    if norm <= 64.0 * _EPS:
        raise ValueError(f"{name} must be nonzero")
    return vector / norm


def _canonical_sign(vector: np.ndarray) -> np.ndarray:
    result = _unit(vector, "direction")
    pivot = int(np.argmax(np.abs(result)))
    return -result if result[pivot] < 0.0 else result


def _polygon_signed_area(vertices: np.ndarray) -> float:
    return 0.5 * float(
        np.sum(
            vertices[:, 0] * np.roll(vertices[:, 1], -1)
            - vertices[:, 1] * np.roll(vertices[:, 0], -1)
        )
    )


def _clean_polygon(vertices: np.ndarray, tolerance: float) -> np.ndarray:
    """Return a deterministic CCW convex polygon, or an empty array."""
    values = np.asarray(vertices, dtype=float)
    if values.ndim != 2 or values.shape[1:] != (2,):
        raise ValueError("polygon vertices must have shape (N, 2)")
    if len(values) < 3:
        return np.empty((0, 2), dtype=float)

    kept: list[np.ndarray] = []
    for point in values:
        if not kept or float(np.linalg.norm(point - kept[-1])) > tolerance:
            kept.append(point)
    if len(kept) > 1 and float(np.linalg.norm(kept[0] - kept[-1])) <= tolerance:
        kept.pop()
    if len(kept) < 3:
        return np.empty((0, 2), dtype=float)

    # Remove numerically collinear vertices.  Repeat because removing one point
    # can expose another collinear triple.
    changed = True
    while changed and len(kept) >= 3:
        changed = False
        reduced: list[np.ndarray] = []
        count = len(kept)
        for index, point in enumerate(kept):
            previous = kept[(index - 1) % count]
            following = kept[(index + 1) % count]
            edge_0 = point - previous
            edge_1 = following - point
            cross = float(edge_0[0] * edge_1[1] - edge_0[1] * edge_1[0])
            scale = max(float(np.linalg.norm(edge_0) * np.linalg.norm(edge_1)), 1.0)
            if abs(cross) <= tolerance * scale and float(edge_0 @ edge_1) >= 0.0:
                changed = True
                continue
            reduced.append(point)
        kept = reduced

    if len(kept) < 3:
        return np.empty((0, 2), dtype=float)
    result = np.asarray(kept, dtype=float)
    area = _polygon_signed_area(result)
    if abs(area) <= tolerance * tolerance:
        return np.empty((0, 2), dtype=float)
    if area < 0.0:
        result = result[::-1].copy()

    # Rotate the list so serialization does not depend on clipping start edge.
    start = min(
        range(len(result)),
        key=lambda index: (
            round(float(result[index, 0]), 14),
            round(float(result[index, 1]), 14),
        ),
    )
    return np.roll(result, -start, axis=0)


def _clip_half_plane(
    polygon: np.ndarray,
    normal: np.ndarray,
    offset: float,
    tolerance: float,
) -> np.ndarray:
    """Clip a convex polygon to ``normal @ p >= offset``."""
    if len(polygon) < 3:
        return np.empty((0, 2), dtype=float)
    output: list[np.ndarray] = []
    previous = polygon[-1]
    previous_value = float(normal @ previous - offset)
    previous_inside = previous_value >= -tolerance
    for current in polygon:
        current_value = float(normal @ current - offset)
        current_inside = current_value >= -tolerance
        if current_inside != previous_inside:
            denominator = previous_value - current_value
            if abs(denominator) > 64.0 * _EPS:
                fraction = previous_value / denominator
                output.append(previous + fraction * (current - previous))
        if current_inside:
            output.append(current)
        previous = current
        previous_value = current_value
        previous_inside = current_inside
    if len(output) < 3:
        return np.empty((0, 2), dtype=float)
    return _clean_polygon(np.asarray(output), tolerance)


def _intersect_convex_polygons(
    first: np.ndarray,
    second: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    result = _clean_polygon(first, tolerance)
    clipper = _clean_polygon(second, tolerance)
    if len(result) < 3 or len(clipper) < 3:
        return np.empty((0, 2), dtype=float)
    for index, start in enumerate(clipper):
        end = clipper[(index + 1) % len(clipper)]
        edge = end - start
        inward = np.array([-edge[1], edge[0]], dtype=float)
        result = _clip_half_plane(result, inward, float(inward @ start), tolerance)
        if len(result) < 3:
            break
    return result


def _triangle_overlap_mask(
    first: np.ndarray,
    seconds: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    """Vectorized separating-axis rejection for one triangle vs many."""
    if first.shape != (3, 2) or seconds.ndim != 3 or seconds.shape[1:] != (3, 2):
        raise ValueError("triangle overlap inputs must have shape (3,2) and (N,3,2)")
    separated = np.zeros(len(seconds), dtype=bool)
    for edge_index, start in enumerate(first):
        edge = first[(edge_index + 1) % 3] - start
        relative = seconds - start
        cross = edge[0] * relative[:, :, 1] - edge[1] * relative[:, :, 0]
        separated |= np.max(cross, axis=1) < -tolerance * max(
            float(np.linalg.norm(edge)), 1.0)
    for edge_index in range(3):
        starts = seconds[:, edge_index, :]
        edges = seconds[:, (edge_index + 1) % 3, :] - starts
        relative = first[None, :, :] - starts[:, None, :]
        cross = (
            edges[:, None, 0] * relative[:, :, 1]
            - edges[:, None, 1] * relative[:, :, 0]
        )
        separated |= np.max(cross, axis=1) < -tolerance * np.maximum(
            np.linalg.norm(edges, axis=1), 1.0)
    return ~separated


def _inset_convex_polygon(
    polygon: np.ndarray,
    margin: float,
    tolerance: float,
) -> np.ndarray:
    """Inset one convex component by an exact Euclidean edge distance."""
    result = _clean_polygon(polygon, tolerance)
    if margin <= 0.0 or len(result) < 3:
        return result
    original = result.copy()
    for index, start in enumerate(original):
        edge = original[(index + 1) % len(original)] - start
        length = float(np.linalg.norm(edge))
        if length <= tolerance:
            continue
        inward = np.array([-edge[1], edge[0]], dtype=float) / length
        result = _clip_half_plane(
            result,
            inward,
            float(inward @ start) + margin,
            tolerance,
        )
        if len(result) < 3:
            break
    return result


def _point_in_convex_polygon(
    vertices: np.ndarray,
    point: np.ndarray,
    tolerance: float,
) -> bool:
    for index, start in enumerate(vertices):
        edge = vertices[(index + 1) % len(vertices)] - start
        relative = point - start
        cross = float(edge[0] * relative[1] - edge[1] * relative[0])
        if cross < -tolerance * max(float(np.linalg.norm(edge)), 1.0):
            return False
    return True


@dataclass(frozen=True)
class ConvexParameterDomain:
    """One convex component of a family's exact ``(u, v)`` domain."""

    vertices_uv_m: np.ndarray

    def __post_init__(self) -> None:
        vertices = np.asarray(self.vertices_uv_m, dtype=float)
        if vertices.ndim != 2 or vertices.shape[1:] != (2,) or len(vertices) < 3:
            raise ValueError("vertices_uv_m must have shape (N>=3, 2)")
        if not np.all(np.isfinite(vertices)):
            raise ValueError("domain vertices must be finite")
        cleaned = _clean_polygon(vertices, 1e-13)
        if len(cleaned) < 3:
            raise ValueError("domain polygon must have positive area")
        object.__setattr__(self, "vertices_uv_m", cleaned)

    @property
    def area_m2(self) -> float:
        return _polygon_signed_area(self.vertices_uv_m)

    @property
    def bounds_uv_m(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.min(self.vertices_uv_m, axis=0),
            np.max(self.vertices_uv_m, axis=0),
        )

    def contains(self, u_m: float, v_m: float, *, tolerance_m: float = 1e-10) -> bool:
        point = np.array([
            _finite_scalar(u_m, "u_m"),
            _finite_scalar(v_m, "v_m"),
        ])
        tolerance = _finite_scalar(tolerance_m, "tolerance_m")
        if tolerance < 0.0:
            raise ValueError("tolerance_m must be non-negative")
        return _point_in_convex_polygon(self.vertices_uv_m, point, tolerance)

    def to_dict(self) -> dict[str, Any]:
        minimum, maximum = self.bounds_uv_m
        return {
            "type": "convex_polygon",
            "vertices_uv_m": self.vertices_uv_m.tolist(),
            "bounds_uv_m": [minimum.tolist(), maximum.tolist()],
            "area_m2": self.area_m2,
        }


@dataclass(frozen=True)
class GraspMapEvaluation:
    """An arbitrary continuous evaluation of one grasp family."""

    family_id: str
    parameters: np.ndarray
    T_P_E: np.ndarray
    T_W_E: np.ndarray
    contacts_P_m: np.ndarray
    contacts_W_m: np.ndarray
    aperture_m: float
    scope: str = SCOPE
    insertion_safe: bool = False

    def __post_init__(self) -> None:
        parameters = np.asarray(self.parameters, dtype=float)
        contacts_P = np.asarray(self.contacts_P_m, dtype=float)
        contacts_W = np.asarray(self.contacts_W_m, dtype=float)
        if parameters.shape != (3,) or not np.all(np.isfinite(parameters)):
            raise ValueError("parameters must be a finite (u, v, roll) vector")
        if contacts_P.shape != (2, 3) or contacts_W.shape != (2, 3):
            raise ValueError("contacts must have shape (2, 3)")
        if not np.all(np.isfinite((contacts_P, contacts_W))):
            raise ValueError("contacts must be finite")
        aperture = _finite_scalar(self.aperture_m, "aperture_m")
        if aperture < 0.0:
            raise ValueError("aperture_m must be non-negative")
        object.__setattr__(self, "parameters", parameters.copy())
        object.__setattr__(self, "T_P_E", validate_transform(self.T_P_E))
        object.__setattr__(self, "T_W_E", validate_transform(self.T_W_E))
        object.__setattr__(self, "contacts_P_m", contacts_P.copy())
        object.__setattr__(self, "contacts_W_m", contacts_W.copy())
        object.__setattr__(self, "aperture_m", aperture)
        if self.insertion_safe:
            raise ValueError("object-only evaluations cannot be insertion-safe")

    def to_dict(self) -> dict[str, Any]:
        return {
            "family_id": self.family_id,
            "parameters": {
                "u_m": float(self.parameters[0]),
                "v_m": float(self.parameters[1]),
                "roll_rad": float(self.parameters[2]),
            },
            "T_P_E": self.T_P_E.tolist(),
            "T_W_E": self.T_W_E.tolist(),
            "contacts_P_m": self.contacts_P_m.tolist(),
            "contacts_W_m": self.contacts_W_m.tolist(),
            "aperture_m": self.aperture_m,
            "scope": self.scope,
            "insertion_safe": False,
        }


@dataclass(frozen=True)
class TwoFingerGraspFamily:
    """One continuous ideal point-contact parallel-jaw family."""

    family_id: str
    T_W_P_insert: np.ndarray
    insertion_axis_P: np.ndarray
    closing_axis_P: np.ndarray
    u_axis_P: np.ndarray
    v_axis_P: np.ndarray
    parameter_origin_P_m: np.ndarray
    negative_plane_normal_P: np.ndarray
    negative_plane_offset_m: float
    positive_plane_normal_P: np.ndarray
    positive_plane_offset_m: float
    aperture_coefficients_m: np.ndarray
    domains: tuple[ConvexParameterDomain, ...]
    roll_bounds_rad: tuple[float, float]
    friction_coefficient: float
    negative_triangle_indices: tuple[int, ...]
    positive_triangle_indices: tuple[int, ...]
    contact_edge_margin_m: float

    def __post_init__(self) -> None:
        if not self.family_id:
            raise ValueError("family_id must be non-empty")
        if not self.domains:
            raise ValueError("a family needs at least one parameter domain")
        transform = validate_transform(self.T_W_P_insert)
        insertion = _unit(self.insertion_axis_P, "insertion_axis_P")
        closing = _unit(self.closing_axis_P, "closing_axis_P")
        u_axis = _unit(self.u_axis_P, "u_axis_P")
        v_axis = _unit(self.v_axis_P, "v_axis_P")
        basis = np.column_stack((u_axis, v_axis, closing))
        if not np.allclose(basis.T @ basis, np.eye(3), atol=2e-8, rtol=0.0):
            raise ValueError("u, v, and closing axes must be orthonormal")
        if float(np.linalg.det(basis)) < 0.999999:
            raise ValueError("[u, v, closing] axes must be right handed")
        origin = np.asarray(self.parameter_origin_P_m, dtype=float)
        coefficients = np.asarray(self.aperture_coefficients_m, dtype=float)
        if origin.shape != (3,) or not np.all(np.isfinite(origin)):
            raise ValueError("parameter_origin_P_m must be a finite three-vector")
        if coefficients.shape != (3,) or not np.all(np.isfinite(coefficients)):
            raise ValueError("aperture_coefficients_m must be finite [a0, au, av]")
        negative_normal = _unit(
            self.negative_plane_normal_P, "negative_plane_normal_P")
        positive_normal = _unit(
            self.positive_plane_normal_P, "positive_plane_normal_P")
        roll = tuple(float(value) for value in self.roll_bounds_rad)
        if len(roll) != 2 or not np.all(np.isfinite(roll)) or roll[1] < roll[0]:
            raise ValueError("roll_bounds_rad must be finite [minimum, maximum]")
        friction = _finite_scalar(self.friction_coefficient, "friction_coefficient")
        margin = _finite_scalar(self.contact_edge_margin_m, "contact_edge_margin_m")
        if friction < 0.0 or margin < 0.0:
            raise ValueError("friction and contact edge margin must be non-negative")
        object.__setattr__(self, "T_W_P_insert", transform)
        object.__setattr__(self, "insertion_axis_P", insertion)
        object.__setattr__(self, "closing_axis_P", closing)
        object.__setattr__(self, "u_axis_P", u_axis)
        object.__setattr__(self, "v_axis_P", v_axis)
        object.__setattr__(self, "parameter_origin_P_m", origin.copy())
        object.__setattr__(self, "negative_plane_normal_P", negative_normal)
        object.__setattr__(self, "positive_plane_normal_P", positive_normal)
        object.__setattr__(self, "aperture_coefficients_m", coefficients.copy())
        object.__setattr__(self, "roll_bounds_rad", roll)
        object.__setattr__(self, "friction_coefficient", friction)
        object.__setattr__(self, "contact_edge_margin_m", margin)

    def aperture(self, u_m: float, v_m: float) -> float:
        u = _finite_scalar(u_m, "u_m")
        v = _finite_scalar(v_m, "v_m")
        return float(
            self.aperture_coefficients_m
            @ np.array([1.0, u, v], dtype=float)
        )

    def contains(
        self,
        u_m: float,
        v_m: float,
        roll_rad: float,
        *,
        tolerance: float = 1e-10,
    ) -> bool:
        roll = _finite_scalar(roll_rad, "roll_rad")
        tol = _finite_scalar(tolerance, "tolerance")
        if tol < 0.0:
            raise ValueError("tolerance must be non-negative")
        if roll < self.roll_bounds_rad[0] - tol or roll > self.roll_bounds_rad[1] + tol:
            return False
        return any(domain.contains(u_m, v_m, tolerance_m=tol) for domain in self.domains)

    def _contact(self, u_m: float, v_m: float, *, positive: bool) -> np.ndarray:
        q = (
            self.parameter_origin_P_m
            + float(u_m) * self.u_axis_P
            + float(v_m) * self.v_axis_P
        )
        if positive:
            normal = self.positive_plane_normal_P
            offset = self.positive_plane_offset_m
        else:
            normal = self.negative_plane_normal_P
            offset = self.negative_plane_offset_m
        denominator = float(normal @ self.closing_axis_P)
        t = (float(offset) - float(normal @ q)) / denominator
        return q + t * self.closing_axis_P

    def evaluate(self, u_m: float, v_m: float, roll_rad: float) -> GraspMapEvaluation:
        if not self.contains(u_m, v_m, roll_rad):
            raise ValueError(
                f"({u_m}, {v_m}, {roll_rad}) is outside family {self.family_id}"
            )
        negative = self._contact(u_m, v_m, positive=False)
        positive = self._contact(u_m, v_m, positive=True)
        contacts_P = np.stack((negative, positive))
        aperture = float((positive - negative) @ self.closing_axis_P)

        # Roll zero follows the insertion direction projected orthogonal to the
        # closing line.  Rodrigues' formula about +Y_E gives a right-hand roll.
        z_zero = self.insertion_axis_P - (
            float(self.insertion_axis_P @ self.closing_axis_P)
            * self.closing_axis_P
        )
        z_zero = _unit(z_zero, "projected insertion approach")
        roll = float(roll_rad)
        approach = (
            np.cos(roll) * z_zero
            + np.sin(roll) * np.cross(self.closing_axis_P, z_zero)
        )
        approach = _unit(approach, "rolled approach")
        x_axis = _unit(np.cross(self.closing_axis_P, approach), "E x axis")
        rotation_P_E = np.column_stack((x_axis, self.closing_axis_P, approach))
        T_P_E = make_transform(rotation_P_E, 0.5 * (negative + positive))
        T_W_E = compose(self.T_W_P_insert, T_P_E)
        contacts_W = (
            contacts_P @ self.T_W_P_insert[:3, :3].T
            + self.T_W_P_insert[:3, 3]
        )
        return GraspMapEvaluation(
            family_id=self.family_id,
            parameters=np.array([u_m, v_m, roll_rad], dtype=float),
            T_P_E=T_P_E,
            T_W_E=T_W_E,
            contacts_P_m=contacts_P,
            contacts_W_m=contacts_W,
            aperture_m=aperture,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "family_id": self.family_id,
            "type": "continuous_ideal_point_contact_parallel_jaw_family",
            "scope": SCOPE,
            "insertion_safe": False,
            "parameterization": {
                "parameters": ["u_m", "v_m", "roll_rad"],
                "parameter_origin_P_m": self.parameter_origin_P_m.tolist(),
                "u_axis_P": self.u_axis_P.tolist(),
                "v_axis_P": self.v_axis_P.tolist(),
                "roll_bounds_rad": list(self.roll_bounds_rad),
                "domains": [domain.to_dict() for domain in self.domains],
            },
            "closing_axis_P": self.closing_axis_P.tolist(),
            "planes": {
                "negative": {
                    "normal_P": self.negative_plane_normal_P.tolist(),
                    "offset_m": self.negative_plane_offset_m,
                    "triangle_indices": list(self.negative_triangle_indices),
                },
                "positive": {
                    "normal_P": self.positive_plane_normal_P.tolist(),
                    "offset_m": self.positive_plane_offset_m,
                    "triangle_indices": list(self.positive_triangle_indices),
                },
            },
            "aperture_map": {
                "type": "affine",
                "formula": "a0_m + au*u_m + av*v_m",
                "coefficients": {
                    "a0_m": float(self.aperture_coefficients_m[0]),
                    "au": float(self.aperture_coefficients_m[1]),
                    "av": float(self.aperture_coefficients_m[2]),
                },
            },
            "friction_coefficient": self.friction_coefficient,
            "contact_edge_margin_m": self.contact_edge_margin_m,
        }


@dataclass(frozen=True)
class TwoFingerGraspMap:
    """Finite union of continuous ideal parallel-jaw grasp families."""

    T_W_P_insert: np.ndarray
    insertion_axis_P: np.ndarray
    opening_range_m: tuple[float, float]
    friction_coefficient: float
    lateral_normal_threshold: float
    maximum_antipodal_normal_error_rad: float
    roll_bounds_rad: tuple[float, float]
    minimum_surface_area_m2: float
    contact_edge_margin_m: float
    plane_tolerance_m: float
    normal_tolerance: float
    families: tuple[TwoFingerGraspFamily, ...]
    mesh_source: str | None = None
    scope: str = SCOPE
    insertion_safe: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "T_W_P_insert", validate_transform(self.T_W_P_insert))
        object.__setattr__(
            self, "insertion_axis_P", _unit(self.insertion_axis_P, "insertion_axis_P"))
        if self.insertion_safe:
            raise ValueError("the object-only map cannot be insertion-safe")

    def matching_families(
        self, u_m: float, v_m: float, roll_rad: float
    ) -> tuple[TwoFingerGraspFamily, ...]:
        return tuple(
            family for family in self.families
            if family.contains(u_m, v_m, roll_rad)
        )

    def contains(self, u_m: float, v_m: float, roll_rad: float) -> bool:
        return bool(self.matching_families(u_m, v_m, roll_rad))

    def evaluate(
        self, u_m: float, v_m: float, roll_rad: float
    ) -> tuple[GraspMapEvaluation, ...]:
        """Return the set of all family evaluations at ``(u, v, roll)``."""
        return tuple(
            family.evaluate(u_m, v_m, roll_rad)
            for family in self.matching_families(u_m, v_m, roll_rad)
        )

    def to_dict(self) -> dict[str, Any]:
        insertion_axis_W = (
            self.T_W_P_insert[:3, :3] @ self.insertion_axis_P
        )
        return {
            "schema_version": 1,
            "artifact_type": "two_finger_continuous_grasp_map",
            "scope": self.scope,
            "insertion_safe": False,
            "limitations": [
                "ideal point contacts only",
                "directional external finger-ray visibility is not certified",
                "closed consistently outward-wound mesh is assumed, not certified",
                "no gripper-body or finger collision geometry",
                "no PCB or insertion-path clearance",
                "no robot IK, calibration uncertainty, force, or dynamics certificate",
            ],
            "inputs": {
                "mesh_source": self.mesh_source,
                "T_W_P_insert": self.T_W_P_insert.tolist(),
                "insertion_axis_P": self.insertion_axis_P.tolist(),
                "insertion_axis_W": insertion_axis_W.tolist(),
                "opening_range_m": list(self.opening_range_m),
                "friction_coefficient": self.friction_coefficient,
                "lateral_normal_threshold": self.lateral_normal_threshold,
                "maximum_antipodal_normal_error_rad": (
                    self.maximum_antipodal_normal_error_rad
                ),
                "roll_bounds_rad": list(self.roll_bounds_rad),
                "minimum_surface_area_m2": self.minimum_surface_area_m2,
                "contact_edge_margin_m": self.contact_edge_margin_m,
                "plane_tolerance_m": self.plane_tolerance_m,
                "normal_tolerance": self.normal_tolerance,
            },
            "set_representation": {
                "type": "finite_union_of_continuous_families",
                "family_count": len(self.families),
                "domain": "union of exact convex projected-triangle intersections",
                "edge_margin_policy": (
                    "zero: exact under the planar point-contact model"
                    if self.contact_edge_margin_m == 0.0 else
                    "positive: conservative per-convex-component inset"
                ),
            },
            "families": [family.to_dict() for family in self.families],
        }


@dataclass
class _SupportPlaneGroup:
    normal_P: np.ndarray
    offset_m: float
    triangle_indices: list[int]
    surface_area_m2: float


def _extract_coplanar_surface_groups(
    mesh: TriangleMesh,
    *,
    insertion_axis_P: np.ndarray,
    lateral_normal_threshold: float,
    minimum_surface_area_m2: float,
    plane_tolerance_m: float,
    normal_tolerance: float,
) -> list[_SupportPlaneGroup]:
    valid = np.flatnonzero(mesh.areas > max(plane_tolerance_m**2, 64.0 * _EPS))
    records: list[tuple[tuple[float, ...], int, np.ndarray, float]] = []
    for index in valid:
        normal = _unit(mesh.normals[index], "triangle normal")
        if abs(float(normal @ insertion_axis_P)) > lateral_normal_threshold + normal_tolerance:
            continue
        triangle = mesh.triangles[index]
        offset = float(np.mean(triangle @ normal))
        centroid = np.mean(triangle, axis=0)
        key = tuple(np.round(np.concatenate((normal, [offset], centroid)), 13))
        records.append((key, int(index), normal, offset))
    records.sort(key=lambda record: record[0])

    groups: list[_SupportPlaneGroup] = []
    for _, index, normal, offset in records:
        assigned: _SupportPlaneGroup | None = None
        for group in groups:
            if (
                float(normal @ group.normal_P) >= 1.0 - normal_tolerance
                and abs(offset - group.offset_m) <= plane_tolerance_m
                and np.max(np.abs(
                    mesh.triangles[index] @ group.normal_P - group.offset_m
                )) <= plane_tolerance_m
            ):
                assigned = group
                break
        if assigned is None:
            groups.append(_SupportPlaneGroup(
                normal_P=normal.copy(),
                offset_m=offset,
                triangle_indices=[index],
                surface_area_m2=float(mesh.areas[index]),
            ))
        else:
            assigned.triangle_indices.append(index)
            assigned.surface_area_m2 += float(mesh.areas[index])

    retained: list[_SupportPlaneGroup] = []
    for group in groups:
        if group.surface_area_m2 + plane_tolerance_m**2 < minimum_surface_area_m2:
            continue
        group.triangle_indices.sort()
        retained.append(group)
    retained.sort(key=lambda group: tuple(np.round(
        np.concatenate((group.normal_P, [group.offset_m])), 13)))
    return retained


def _project_triangle(
    triangle_P: np.ndarray,
    origin_P: np.ndarray,
    u_axis_P: np.ndarray,
    v_axis_P: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    relative = triangle_P - origin_P
    projected = np.column_stack((relative @ u_axis_P, relative @ v_axis_P))
    return _clean_polygon(projected, tolerance)


def _clip_aperture(
    polygon: np.ndarray,
    coefficients: np.ndarray,
    opening_range_m: tuple[float, float],
    tolerance: float,
) -> np.ndarray:
    # aperture = a0 + au*u + av*v
    gradient = coefficients[1:]
    result = _clip_half_plane(
        polygon,
        gradient,
        opening_range_m[0] - coefficients[0],
        tolerance,
    )
    if len(result) < 3:
        return result
    return _clip_half_plane(
        result,
        -gradient,
        coefficients[0] - opening_range_m[1],
        tolerance,
    )


def _domain_key(domain: ConvexParameterDomain) -> tuple[float, ...]:
    return tuple(np.round(domain.vertices_uv_m.reshape(-1), 12))


def generate_two_finger_grasp_map(
    mesh: TriangleMesh,
    *,
    T_W_P_insert: np.ndarray,
    insertion_axis_P: Sequence[float] | np.ndarray,
    opening_range_m: Sequence[float],
    friction_coefficient: float,
    roll_bounds_rad: Sequence[float] = (-np.pi, np.pi),
    lateral_normal_threshold: float | None = None,
    maximum_surface_tilt_from_lateral_rad: float | None = None,
    maximum_antipodal_normal_error_rad: float = np.deg2rad(10.0),
    minimum_surface_area_m2: float = 0.0,
    contact_edge_margin_m: float = 0.0,
    plane_tolerance_m: float | None = None,
    normal_tolerance: float = 1e-8,
) -> TwoFingerGraspMap:
    """Construct the continuous ideal two-finger grasp map.

    Exactly one of ``lateral_normal_threshold`` and
    ``maximum_surface_tilt_from_lateral_rad`` may be supplied.  A maximum
    lateral tilt ``delta`` is equivalent to
    ``abs(surface_normal @ insertion_axis) <= sin(delta)``.

    Positive ``contact_edge_margin_m`` produces a conservative subset by
    insetting every convex projected-triangle overlap component.  At zero
    margin the union is exact under this module's planar point-contact model.
    """
    if not isinstance(mesh, TriangleMesh):
        raise TypeError("mesh must be a TriangleMesh")
    transform = validate_transform(T_W_P_insert)
    insertion = _unit(insertion_axis_P, "insertion_axis_P")
    opening = tuple(float(value) for value in opening_range_m)
    if (
        len(opening) != 2
        or not np.all(np.isfinite(opening))
        or opening[0] < 0.0
        or opening[1] < opening[0]
    ):
        raise ValueError("opening_range_m must be finite [0 <= minimum <= maximum]")
    friction = _finite_scalar(friction_coefficient, "friction_coefficient")
    if friction < 0.0:
        raise ValueError("friction_coefficient must be non-negative")
    roll = tuple(float(value) for value in roll_bounds_rad)
    if len(roll) != 2 or not np.all(np.isfinite(roll)) or roll[1] < roll[0]:
        raise ValueError("roll_bounds_rad must be finite [minimum, maximum]")
    antipodal_error = _finite_scalar(
        maximum_antipodal_normal_error_rad,
        "maximum_antipodal_normal_error_rad",
    )
    if not 0.0 <= antipodal_error <= np.pi:
        raise ValueError("maximum_antipodal_normal_error_rad must lie in [0, pi]")
    minimum_area = _finite_scalar(minimum_surface_area_m2, "minimum_surface_area_m2")
    edge_margin = _finite_scalar(contact_edge_margin_m, "contact_edge_margin_m")
    normal_tol = _finite_scalar(normal_tolerance, "normal_tolerance")
    if minimum_area < 0.0 or edge_margin < 0.0 or normal_tol < 0.0:
        raise ValueError("area, edge margin, and tolerances must be non-negative")

    if lateral_normal_threshold is not None and maximum_surface_tilt_from_lateral_rad is not None:
        raise ValueError(
            "supply lateral_normal_threshold or maximum_surface_tilt_from_lateral_rad, not both"
        )
    if maximum_surface_tilt_from_lateral_rad is not None:
        tilt = _finite_scalar(
            maximum_surface_tilt_from_lateral_rad,
            "maximum_surface_tilt_from_lateral_rad",
        )
        if not 0.0 <= tilt <= np.pi / 2.0:
            raise ValueError("maximum lateral tilt must lie in [0, pi/2]")
        lateral_threshold = float(np.sin(tilt))
    elif lateral_normal_threshold is None:
        lateral_threshold = float(np.sin(np.deg2rad(10.0)))
    else:
        lateral_threshold = _finite_scalar(
            lateral_normal_threshold, "lateral_normal_threshold")
        if not 0.0 <= lateral_threshold <= 1.0:
            raise ValueError("lateral_normal_threshold must lie in [0, 1]")

    extent = float(np.linalg.norm(mesh.extent))
    if plane_tolerance_m is None:
        plane_tolerance = max(1e-10 * max(extent, 1.0), 1e-12)
    else:
        plane_tolerance = _finite_scalar(plane_tolerance_m, "plane_tolerance_m")
        if plane_tolerance <= 0.0:
            raise ValueError("plane_tolerance_m must be positive")
    polygon_tolerance = max(plane_tolerance, 1e-12)

    groups = _extract_coplanar_surface_groups(
        mesh,
        insertion_axis_P=insertion,
        lateral_normal_threshold=lateral_threshold,
        minimum_surface_area_m2=minimum_area,
        plane_tolerance_m=plane_tolerance,
        normal_tolerance=normal_tol,
    )
    center_P = 0.5 * (mesh.bounds_min + mesh.bounds_max)
    friction_cosine = 1.0 / float(np.sqrt(1.0 + friction * friction))
    opposition_cosine = float(np.cos(antipodal_error))
    families: list[TwoFingerGraspFamily] = []

    for first_index, first in enumerate(groups):
        for second in groups[first_index + 1:]:
            if float(first.normal_P @ (-second.normal_P)) < opposition_cosine - normal_tol:
                continue
            difference = second.normal_P - first.normal_P
            if float(np.linalg.norm(difference)) <= normal_tol:
                continue
            closing = _canonical_sign(difference)
            if float(first.normal_P @ closing) > float(second.normal_P @ closing):
                negative, positive = second, first
            else:
                negative, positive = first, second
            if (
                float(positive.normal_P @ closing) < friction_cosine - normal_tol
                or float((-negative.normal_P) @ closing) < friction_cosine - normal_tol
            ):
                continue
            if abs(float(closing @ insertion)) > lateral_threshold + normal_tol:
                continue

            v_axis = insertion - float(insertion @ closing) * closing
            if float(np.linalg.norm(v_axis)) <= 64.0 * _EPS:
                continue
            v_axis = _unit(v_axis, "projected insertion axis")
            u_axis = _unit(np.cross(v_axis, closing), "surface u axis")
            # [u, v, closing] is right handed by construction.
            origin = center_P.copy()
            negative_denominator = float(negative.normal_P @ closing)
            positive_denominator = float(positive.normal_P @ closing)
            if negative_denominator >= -normal_tol or positive_denominator <= normal_tol:
                continue

            # t_i(u,v) = (d_i - n_i @ (origin + u U + v V))/(n_i @ C)
            negative_t = np.array([
                (negative.offset_m - float(negative.normal_P @ origin))
                / negative_denominator,
                -float(negative.normal_P @ u_axis) / negative_denominator,
                -float(negative.normal_P @ v_axis) / negative_denominator,
            ])
            positive_t = np.array([
                (positive.offset_m - float(positive.normal_P @ origin))
                / positive_denominator,
                -float(positive.normal_P @ u_axis) / positive_denominator,
                -float(positive.normal_P @ v_axis) / positive_denominator,
            ])
            aperture_coefficients = positive_t - negative_t

            negative_projected = [
                _project_triangle(
                    mesh.triangles[index], origin, u_axis, v_axis, polygon_tolerance)
                for index in negative.triangle_indices
            ]
            positive_projected = [
                _project_triangle(
                    mesh.triangles[index], origin, u_axis, v_axis, polygon_tolerance)
                for index in positive.triangle_indices
            ]
            positive_mins = np.array([
                np.min(polygon, axis=0) if len(polygon) >= 3 else [np.inf, np.inf]
                for polygon in positive_projected
            ])
            positive_maxs = np.array([
                np.max(polygon, axis=0) if len(polygon) >= 3 else [-np.inf, -np.inf]
                for polygon in positive_projected
            ])
            raw_components: list[np.ndarray] = []
            for first_polygon in negative_projected:
                if len(first_polygon) < 3:
                    continue
                first_min = np.min(first_polygon, axis=0)
                first_max = np.max(first_polygon, axis=0)
                overlaps = np.all(
                    (positive_mins <= first_max + polygon_tolerance)
                    & (positive_maxs >= first_min - polygon_tolerance),
                    axis=1,
                )
                second_indices = np.flatnonzero(overlaps)
                if len(second_indices) == 0:
                    continue
                second_triangles = np.stack([
                    positive_projected[int(index)] for index in second_indices
                ])
                second_indices = second_indices[_triangle_overlap_mask(
                    first_polygon, second_triangles, polygon_tolerance)]
                for second_index in second_indices:
                    second_polygon = positive_projected[int(second_index)]
                    polygon = _intersect_convex_polygons(
                        first_polygon, second_polygon, polygon_tolerance)
                    if len(polygon) < 3:
                        continue
                    raw_components.append(polygon)
            if not raw_components:
                continue

            components: list[ConvexParameterDomain] = []
            seen: set[tuple[float, ...]] = set()
            for polygon in raw_components:
                polygon = _inset_convex_polygon(
                    polygon, edge_margin, polygon_tolerance)
                if len(polygon) < 3:
                    continue
                polygon = _clip_aperture(
                    polygon, aperture_coefficients, opening, polygon_tolerance)
                if len(polygon) < 3:
                    continue
                domain = ConvexParameterDomain(polygon)
                key = _domain_key(domain)
                if key not in seen:
                    seen.add(key)
                    components.append(domain)
            if not components:
                continue
            components.sort(key=_domain_key)
            family_id = f"family_{len(families):04d}"
            families.append(TwoFingerGraspFamily(
                family_id=family_id,
                T_W_P_insert=transform,
                insertion_axis_P=insertion,
                closing_axis_P=closing,
                u_axis_P=u_axis,
                v_axis_P=v_axis,
                parameter_origin_P_m=origin,
                negative_plane_normal_P=negative.normal_P,
                negative_plane_offset_m=negative.offset_m,
                positive_plane_normal_P=positive.normal_P,
                positive_plane_offset_m=positive.offset_m,
                aperture_coefficients_m=aperture_coefficients,
                domains=tuple(components),
                roll_bounds_rad=roll,
                friction_coefficient=friction,
                negative_triangle_indices=tuple(negative.triangle_indices),
                positive_triangle_indices=tuple(positive.triangle_indices),
                contact_edge_margin_m=edge_margin,
            ))

    return TwoFingerGraspMap(
        T_W_P_insert=transform,
        insertion_axis_P=insertion,
        opening_range_m=opening,
        friction_coefficient=friction,
        lateral_normal_threshold=lateral_threshold,
        maximum_antipodal_normal_error_rad=antipodal_error,
        roll_bounds_rad=roll,
        minimum_surface_area_m2=minimum_area,
        contact_edge_margin_m=edge_margin,
        plane_tolerance_m=plane_tolerance,
        normal_tolerance=normal_tol,
        families=tuple(families),
        mesh_source=mesh.source,
    )


__all__ = [
    "ConvexParameterDomain",
    "GraspMapEvaluation",
    "SCOPE",
    "TwoFingerGraspFamily",
    "TwoFingerGraspMap",
    "generate_two_finger_grasp_map",
    "load_scaled_binary_stl",
]
