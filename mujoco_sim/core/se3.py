"""Numerically robust rigid-transform helpers for the MuJoCo handoff stack.

Conventions follow :mod:`docs/handoff_pipeline_detailed.md`:

* ``T_X_Y`` maps coordinates expressed in frame Y into frame X.
* transforms compose from left to right, e.g. ``T_W_P @ T_P_E``.
* spatial vectors are ordered ``(v, omega)`` (linear, then angular).
* angles are radians and translations are metres.

The functions in this module deliberately validate rotations instead of
silently projecting arbitrary matrices onto SO(3).  A reflected or otherwise
invalid CAD/calibration transform should fail close to its source.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np


DEFAULT_ATOL = 1e-7


def _vector3(value: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    """Return *value* as a finite three-vector."""
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector


def validate_rotation(rotation: np.ndarray, *, atol: float = DEFAULT_ATOL) -> np.ndarray:
    """Validate and return a copy of a rotation matrix in SO(3).

    ``atol`` applies to both orthogonality and determinant checks.  No
    orthogonalization is performed: callers must opt into any desired repair
    of noisy input explicitly.
    """
    if atol < 0 or not np.isfinite(atol):
        raise ValueError("atol must be a finite non-negative scalar")
    matrix = np.asarray(rotation, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError(f"rotation must have shape (3, 3), got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("rotation must contain only finite values")
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=atol, rtol=0.0):
        raise ValueError("rotation is not orthonormal")
    determinant = float(np.linalg.det(matrix))
    if not np.isclose(determinant, 1.0, atol=atol, rtol=0.0):
        raise ValueError(f"rotation determinant must be +1, got {determinant:.16g}")
    return matrix.copy()


def validate_transform(transform: np.ndarray, *, atol: float = DEFAULT_ATOL) -> np.ndarray:
    """Validate and return a copy of a homogeneous transform in SE(3)."""
    matrix = np.asarray(transform, dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError(f"transform must have shape (4, 4), got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("transform must contain only finite values")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=atol, rtol=0.0):
        raise ValueError("transform bottom row must be [0, 0, 0, 1]")
    validate_rotation(matrix[:3, :3], atol=atol)
    return matrix.copy()


def make_transform(
    rotation: np.ndarray,
    translation: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Construct ``[[rotation, translation], [0, 0, 0, 1]]``."""
    matrix = np.eye(4)
    matrix[:3, :3] = validate_rotation(rotation)
    matrix[:3, 3] = _vector3(translation, "translation")
    return matrix


def compose(*transforms: np.ndarray) -> np.ndarray:
    """Compose transforms in the supplied order.

    ``compose(T_X_Y, T_Y_Z)`` returns ``T_X_Z``.  The empty composition is
    the identity transform.
    """
    rotation = np.eye(3)
    translation = np.zeros(3)
    for transform in transforms:
        matrix = validate_transform(transform)
        translation = translation + rotation @ matrix[:3, 3]
        rotation = rotation @ matrix[:3, :3]
    return make_transform(rotation, translation)


def inverse(transform: np.ndarray) -> np.ndarray:
    """Return the analytic rigid inverse of an SE(3) transform."""
    matrix = validate_transform(transform)
    rotation_t = matrix[:3, :3].T
    return make_transform(rotation_t, -rotation_t @ matrix[:3, 3])


def skew(vector: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return ``[vector]_x``, satisfying ``skew(a) @ b == cross(a, b)``."""
    x, y, z = _vector3(vector, "vector")
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ]
    )


def _vee(skew_matrix: np.ndarray) -> np.ndarray:
    """Inverse of :func:`skew` for an already skew-symmetric matrix."""
    return np.array([skew_matrix[2, 1], skew_matrix[0, 2], skew_matrix[1, 0]])


def so3_exp(rotation_vector: Sequence[float] | np.ndarray) -> np.ndarray:
    """SO(3) exponential map from an axis-angle rotation vector.

    A Taylor branch avoids cancellation in both Rodrigues coefficients near
    zero.  Rotation vectors outside the principal ball are accepted, as the
    exponential map is periodic.
    """
    phi = _vector3(rotation_vector, "rotation_vector")
    theta_squared = float(phi @ phi)
    phi_hat = skew(phi)

    if theta_squared < 1e-8:
        theta_fourth = theta_squared * theta_squared
        theta_sixth = theta_fourth * theta_squared
        sin_over_theta = (
            1.0 - theta_squared / 6.0 + theta_fourth / 120.0 - theta_sixth / 5040.0
        )
        one_minus_cos_over_theta_squared = (
            0.5 - theta_squared / 24.0 + theta_fourth / 720.0 - theta_sixth / 40320.0
        )
    else:
        theta = float(np.sqrt(theta_squared))
        sin_over_theta = float(np.sin(theta) / theta)
        one_minus_cos_over_theta_squared = float((1.0 - np.cos(theta)) / theta_squared)

    return (
        np.eye(3)
        + sin_over_theta * phi_hat
        + one_minus_cos_over_theta_squared * (phi_hat @ phi_hat)
    )


def so3_log(rotation: np.ndarray) -> np.ndarray:
    """Principal SO(3) logarithm as an axis-angle vector.

    The returned norm lies in ``[0, pi]``.  Near pi the antisymmetric part of
    a rotation vanishes, so the axis is recovered from ``(R + I) / 2`` and
    its sign is chosen from the residual antisymmetric part when possible.
    """
    matrix = validate_rotation(rotation)
    cosine = float(np.clip((np.trace(matrix) - 1.0) * 0.5, -1.0, 1.0))
    theta = float(np.arccos(cosine))
    antisymmetric_vector = _vee(matrix - matrix.T)

    if theta < 1e-7:
        theta_squared = theta * theta
        # theta / (2 sin(theta)) through O(theta^4).
        coefficient = 0.5 + theta_squared / 12.0 + 7.0 * theta_squared**2 / 720.0
        return coefficient * antisymmetric_vector

    if np.pi - theta < 1e-5:
        axis_outer = 0.5 * (matrix + np.eye(3))
        axis_outer = 0.5 * (axis_outer + axis_outer.T)
        eigenvalues, eigenvectors = np.linalg.eigh(axis_outer)
        axis = eigenvectors[:, int(np.argmax(eigenvalues))]
        axis /= np.linalg.norm(axis)

        # For theta just below pi, R - R.T still fixes the sign.  At exactly
        # pi the sign is intrinsically ambiguous; use a deterministic choice.
        if np.linalg.norm(antisymmetric_vector) > 1e-12:
            if float(axis @ antisymmetric_vector) < 0.0:
                axis = -axis
        else:
            largest = int(np.argmax(np.abs(axis)))
            if axis[largest] < 0.0:
                axis = -axis
        return theta * axis

    return (theta / (2.0 * np.sin(theta))) * antisymmetric_vector


def so3_geodesic(first: np.ndarray, second: np.ndarray) -> float:
    """Return the geodesic distance between two SO(3) rotations in radians."""
    first_matrix = validate_rotation(first)
    second_matrix = validate_rotation(second)
    return float(np.linalg.norm(so3_log(first_matrix.T @ second_matrix)))


def adjoint(transform: np.ndarray) -> np.ndarray:
    """Return the SE(3) adjoint for twists ordered ``(v, omega)``.

    For ``T = (R, t)``, this is exactly the ordering used by the detailed
    handoff document::

        Ad_T = [[R, skew(t) @ R],
                [0,               R]]
    """
    matrix = validate_transform(transform)
    rotation = matrix[:3, :3]
    result = np.zeros((6, 6))
    result[:3, :3] = rotation
    result[:3, 3:] = skew(matrix[:3, 3]) @ rotation
    result[3:, 3:] = rotation
    return result


def rpy_matrix(rpy_rad: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return a rotation from roll, pitch, yaw angles in radians.

    The convention is intrinsic XYZ (equivalently the active matrix
    ``Rz(yaw) @ Ry(pitch) @ Rx(roll)``), matching the transform convention in
    ``project.yaml``.
    """
    roll, pitch, yaw = _vector3(rpy_rad, "rpy_rad")
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    rotation_x = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    rotation_y = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rotation_z = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rotation_z @ rotation_y @ rotation_x


def transform_from_rpy(
    translation: Sequence[float] | np.ndarray,
    rpy_rad: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Construct an SE(3) transform from translation and intrinsic XYZ RPY."""
    return make_transform(rpy_matrix(rpy_rad), translation)


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a transform to one point or an array whose last dimension is 3.

    A ``(3,)`` input produces ``(3,)`` output; inputs of shape ``(..., 3)``
    retain all leading dimensions.
    """
    matrix = validate_transform(transform)
    point_array = np.asarray(points, dtype=float)
    if point_array.ndim == 0 or point_array.shape[-1] != 3:
        raise ValueError(f"points must have shape (3,) or (..., 3), got {point_array.shape}")
    if not np.all(np.isfinite(point_array)):
        raise ValueError("points must contain only finite values")
    return point_array @ matrix[:3, :3].T + matrix[:3, 3]


__all__ = [
    "DEFAULT_ATOL",
    "adjoint",
    "compose",
    "inverse",
    "make_transform",
    "rpy_matrix",
    "skew",
    "so3_exp",
    "so3_geodesic",
    "so3_log",
    "transform_from_rpy",
    "transform_points",
    "validate_rotation",
    "validate_transform",
]
