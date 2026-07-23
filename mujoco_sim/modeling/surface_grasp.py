"""Map a surface-relative parallel-jaw grasp into an SE(3) transform.

The repository uses ``T_X_Y`` for the transform that maps coordinates from
frame ``Y`` into frame ``X``.  An evaluated surface frame ``S`` has:

``+X_S``
    the local ``u`` tangent,
``+Y_S``
    the local ``v`` tangent, and
``+Z_S``
    the outward surface normal.

The parameterized gripper-pose frame ``G`` follows the existing gripper-axis
convention:

``+X_G``
    pad-width direction,
``+Y_G``
    jaw opening/closing direction, and
``+Z_G``
    approach direction from the palm towards the surface anchor.

At zero angles, ``+Y_G`` follows ``+X_S`` and ``+Z_G`` is ``-Z_S``.
``psi`` rotates about the outward surface normal, ``alpha`` then tilts about
the current jaw-closing axis, and ``beta`` finally tilts about the current
pad-width axis.  The intrinsic rotation order is therefore fixed as
``Rz(psi) -> Ry(alpha) -> Rx(beta)`` after the zero-angle frame alignment.

The opening ``w`` is gripper configuration, not part of SE(3).  The
high-level helper returns it beside the homogeneous transform.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from ..core.se3 import make_transform, so3_exp, validate_transform


# Columns are the zero-angle G axes expressed in S:
#   +X_G = +Y_S, +Y_G = +X_S, +Z_G = -Z_S.
_R_S_G_ZERO = np.array(
    [
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
    ]
)


def _finite_scalar(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def surface_grasp_to_transform(
    T_X_S_uv: np.ndarray,
    *,
    standoff_m: float,
    normal_rotation_rad: float,
    closing_axis_tilt_rad: float,
    pad_axis_tilt_rad: float,
) -> np.ndarray:
    """Return ``T_X_G`` for one already-evaluated surface frame.

    ``T_X_S_uv`` is the local surface frame evaluated at the desired
    ``(u, v)``.  Positive ``standoff_m`` places the gripper origin behind the
    surface anchor along ``-Z_G``.  Thus, if ``q_X`` is the surface point and
    ``a_X`` is the final approach direction, the invariant is

    ``q_X = p_X_G + standoff_m * a_X``.

    ``G`` is the abstract gripper-pose frame defined by this parameterization;
    it is deliberately distinct from the contact-midpoint frame ``E`` used by
    :mod:`mujoco_sim.modeling.grasps`.  If ``G`` is not the real controller
    TCP, apply the calibrated fixed transform between them downstream.
    """
    surface = validate_transform(T_X_S_uv)
    standoff = _finite_scalar(standoff_m, "standoff_m")
    psi = _finite_scalar(normal_rotation_rad, "normal_rotation_rad")
    alpha = _finite_scalar(closing_axis_tilt_rad, "closing_axis_tilt_rad")
    beta = _finite_scalar(pad_axis_tilt_rad, "pad_axis_tilt_rad")
    if standoff < 0.0:
        raise ValueError("standoff_m must be non-negative")

    rotation_X_G = (
        surface[:3, :3]
        @ so3_exp([0.0, 0.0, psi])
        @ _R_S_G_ZERO
        @ so3_exp([0.0, alpha, 0.0])
        @ so3_exp([beta, 0.0, 0.0])
    )
    approach_X = rotation_X_G[:, 2]
    translation_X_G = surface[:3, 3] - standoff * approach_X
    return make_transform(rotation_X_G, translation_X_G)


def surface_grasp_to_se3(
    parameters: Sequence[float] | np.ndarray,
    surface_frame_at: Callable[[float, float], np.ndarray],
) -> tuple[np.ndarray, float]:
    """Map ``(u, v, d, psi, alpha, beta, w)`` to ``(T_X_G, w)``.

    ``surface_frame_at(u, v)`` must return ``T_X_S_uv`` with the surface-frame
    convention documented by this module.  This callback makes the helper
    usable with planar charts, analytic surfaces, or CAD-specific surface
    evaluators without pretending that every triangle mesh has a global UV
    chart.

    The returned matrix is in SE(3); the returned opening is separate because
    an internal jaw coordinate is not a rigid-body degree of freedom.
    """
    values = np.asarray(parameters, dtype=float)
    if values.shape != (7,):
        raise ValueError(
            "parameters must have shape (7,) ordered "
            "(u, v, d, psi, alpha, beta, w)"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("surface-grasp parameters must be finite")
    u_m, v_m, d_m, psi_rad, alpha_rad, beta_rad, opening_m = values
    if d_m < 0.0:
        raise ValueError("d must be non-negative")
    if opening_m < 0.0:
        raise ValueError("w must be non-negative")
    if not callable(surface_frame_at):
        raise TypeError("surface_frame_at must be callable")

    T_X_S_uv = surface_frame_at(float(u_m), float(v_m))
    transform = surface_grasp_to_transform(
        T_X_S_uv,
        standoff_m=float(d_m),
        normal_rotation_rad=float(psi_rad),
        closing_axis_tilt_rad=float(alpha_rad),
        pad_axis_tilt_rad=float(beta_rad),
    )
    return transform, float(opening_m)


__all__ = [
    "surface_grasp_to_se3",
    "surface_grasp_to_transform",
]
