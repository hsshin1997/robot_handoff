"""Numerical acceptance tests for the MuJoCo SE(3) convention layer."""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from mujoco_sim.core.se3 import (  # noqa: E402
    adjoint,
    compose,
    inverse,
    make_transform,
    rpy_matrix,
    skew,
    so3_exp,
    so3_geodesic,
    so3_log,
    transform_from_rpy,
    transform_points,
    validate_transform,
)


def _twist_hat(twist: np.ndarray) -> np.ndarray:
    """se(3) matrix for this project's (v, omega) ordering."""
    result = np.zeros((4, 4))
    result[:3, :3] = skew(twist[3:])
    result[:3, 3] = twist[:3]
    return result


def test_identity_inverse_composition_and_validation():
    identity = compose()
    assert np.array_equal(identity, np.eye(4))

    transform = transform_from_rpy([0.31, -0.27, 0.82], [0.4, -0.2, 1.1])
    assert np.allclose(compose(transform, inverse(transform)), identity, atol=1e-14)
    assert np.allclose(compose(inverse(transform), transform), identity, atol=1e-14)
    assert np.array_equal(validate_transform(transform), transform)

    invalid_bottom_row = transform.copy()
    invalid_bottom_row[3, 0] = 0.01
    try:
        validate_transform(invalid_bottom_row)
    except ValueError:
        pass
    else:
        raise AssertionError("a non-homogeneous bottom row was accepted")

    reflection = np.eye(4)
    reflection[0, 0] = -1.0
    try:
        validate_transform(reflection)
    except ValueError:
        pass
    else:
        raise AssertionError("a reflection was accepted as an SE(3) transform")


def test_skew_and_point_transform_conventions():
    first = np.array([0.2, -0.7, 1.4])
    second = np.array([-0.9, 0.3, 0.6])
    assert np.allclose(skew(first) @ second, np.cross(first, second))

    transform = transform_from_rpy([1.0, 2.0, 3.0], [0.0, 0.0, np.pi / 2.0])
    points = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, -1.0]])
    expected = np.array([[1.0, 3.0, 3.0], [-1.0, 2.0, 2.0]])
    assert np.allclose(transform_points(transform, points), expected, atol=1e-15)
    assert np.allclose(transform_points(transform, points[0]), expected[0], atol=1e-15)


def test_rpy_order_matches_intrinsic_xyz_definition():
    roll, pitch, yaw = 0.37, -0.61, 1.24
    rx = so3_exp([roll, 0.0, 0.0])
    ry = so3_exp([0.0, pitch, 0.0])
    rz = so3_exp([0.0, 0.0, yaw])
    assert np.allclose(rpy_matrix([roll, pitch, yaw]), rz @ ry @ rx, atol=1e-15)


def test_so3_exp_log_round_trips_near_zero_and_pi():
    axis = np.array([0.37, -0.48, 0.795])
    axis /= np.linalg.norm(axis)
    angles = (0.0, 1e-12, 1e-7, 0.73, np.pi - 1e-8, np.pi)

    for angle in angles:
        rotation = so3_exp(axis * angle)
        validate_transform(make_transform(rotation, [0.0, 0.0, 0.0]))
        recovered = so3_log(rotation)
        assert 0.0 <= np.linalg.norm(recovered) <= np.pi + 1e-12
        assert np.allclose(so3_exp(recovered), rotation, atol=2e-8), angle

    # The logarithm is principal, so exp(log(R)) also round-trips rotations
    # produced from rotation vectors outside the principal ball.
    outside_principal_ball = so3_exp(axis * (1.25 * np.pi))
    assert np.allclose(
        so3_exp(so3_log(outside_principal_ball)), outside_principal_ball, atol=1e-12
    )


def test_so3_geodesic_is_frame_invariant_and_bounded():
    first = so3_exp([0.2, -0.3, 0.1])
    delta = so3_exp([0.0, 0.0, 0.42])
    second = first @ delta
    common = so3_exp([-0.6, 0.1, 0.7])

    assert np.isclose(so3_geodesic(first, second), 0.42, atol=1e-14)
    assert np.isclose(
        so3_geodesic(common @ first, common @ second),
        so3_geodesic(first, second),
        atol=1e-14,
    )
    assert np.isclose(so3_geodesic(first, first), 0.0, atol=1e-15)


def test_adjoint_twist_covariance_and_composition():
    first = transform_from_rpy([0.21, -0.14, 0.63], [0.3, -0.4, 0.1])
    second = transform_from_rpy([-0.17, 0.52, 0.08], [-0.2, 0.25, 0.6])

    # Ad(T1 T2) = Ad(T1) Ad(T2).
    assert np.allclose(
        adjoint(compose(first, second)), adjoint(first) @ adjoint(second), atol=1e-14
    )

    # Lie-algebra covariance: hat(Ad_T xi) = T hat(xi) T^-1.  This also
    # verifies the required (v, omega), not (omega, v), block ordering.
    twist = np.array([0.13, -0.09, 0.22, 0.4, -0.31, 0.08])
    transformed_twist = adjoint(first) @ twist
    assert np.allclose(
        _twist_hat(transformed_twist),
        first @ _twist_hat(twist) @ inverse(first),
        atol=1e-14,
    )

    # Covariance propagation must agree whether two frame changes are applied
    # sequentially or as their composed transform.
    seed = np.array(
        [
            [4.0, 1.0, 0.0, 0.3, 0.0, 0.1],
            [1.0, 3.0, 0.2, 0.0, 0.4, 0.0],
            [0.0, 0.2, 2.0, 0.1, 0.0, 0.2],
            [0.3, 0.0, 0.1, 1.5, 0.2, 0.0],
            [0.0, 0.4, 0.0, 0.2, 1.2, 0.1],
            [0.1, 0.0, 0.2, 0.0, 0.1, 1.0],
        ]
    )
    covariance = seed @ seed.T
    sequential = adjoint(first) @ (adjoint(second) @ covariance @ adjoint(second).T) @ adjoint(first).T
    composed_adjoint = adjoint(compose(first, second))
    composed = composed_adjoint @ covariance @ composed_adjoint.T
    assert np.allclose(sequential, composed, atol=2e-14)
    assert np.allclose(composed, composed.T, atol=2e-14)
    assert np.linalg.eigvalsh(composed).min() > -1e-12


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS  {test.__name__}")
    print(f"\n{len(tests)}/{len(tests)} passed")
