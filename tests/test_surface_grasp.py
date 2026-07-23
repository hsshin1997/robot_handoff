"""Focused tests for the surface-relative grasp-to-SE(3) mapping."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mujoco_sim.core.se3 import (  # noqa: E402
    compose,
    make_transform,
    so3_exp,
    transform_from_rpy,
    validate_transform,
)
from mujoco_sim.modeling.surface_grasp import (  # noqa: E402
    surface_grasp_to_se3,
    surface_grasp_to_transform,
)


R_S_G_ZERO = np.array([
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0],
])


def _assert_raises(error_type, text, function):
    try:
        function()
    except error_type as error:
        assert text in str(error)
    else:
        raise AssertionError(f"expected {error_type.__name__}: {text}")


def test_zero_angles_align_gripper_and_place_positive_standoff_outside():
    q_X = np.array([0.25, -0.40, 0.75])
    T_X_S = make_transform(np.eye(3), q_X)
    T_X_G = surface_grasp_to_transform(
        T_X_S,
        standoff_m=0.12,
        normal_rotation_rad=0.0,
        closing_axis_tilt_rad=0.0,
        pad_axis_tilt_rad=0.0,
    )
    validate_transform(T_X_G)
    assert np.array_equal(T_X_G[:3, :3], R_S_G_ZERO)
    assert np.allclose(T_X_G[:3, 1], [1.0, 0.0, 0.0])
    assert np.allclose(T_X_G[:3, 2], [0.0, 0.0, -1.0])
    assert np.allclose(T_X_G[:3, 3], q_X + [0.0, 0.0, 0.12])
    assert np.allclose(
        T_X_G[:3, 3] + 0.12 * T_X_G[:3, 2],
        q_X,
    )


def test_angle_axes_and_intrinsic_order_are_explicit():
    psi, alpha, beta = 0.4, -0.3, 0.2
    actual = surface_grasp_to_transform(
        np.eye(4),
        standoff_m=0.0,
        normal_rotation_rad=psi,
        closing_axis_tilt_rad=alpha,
        pad_axis_tilt_rad=beta,
    )
    expected_rotation = (
        so3_exp([0.0, 0.0, psi])
        @ R_S_G_ZERO
        @ so3_exp([0.0, alpha, 0.0])
        @ so3_exp([beta, 0.0, 0.0])
    )
    assert np.allclose(actual[:3, :3], expected_rotation)

    quarter_turn = surface_grasp_to_transform(
        np.eye(4),
        standoff_m=0.0,
        normal_rotation_rad=np.pi / 2.0,
        closing_axis_tilt_rad=0.0,
        pad_axis_tilt_rad=0.0,
    )
    # Positive psi rotates the closing axis from +X_S towards +Y_S.
    assert np.allclose(quarter_turn[:3, 1], [0.0, 1.0, 0.0], atol=1e-12)


def test_mapping_is_left_equivariant_under_a_parent_frame_change():
    T_X_S = transform_from_rpy([0.1, -0.2, 0.3], [0.2, -0.4, 0.6])
    T_A_X = transform_from_rpy([-0.4, 0.7, 0.2], [-0.3, 0.1, 0.5])
    arguments = {
        "standoff_m": 0.08,
        "normal_rotation_rad": 0.7,
        "closing_axis_tilt_rad": -0.25,
        "pad_axis_tilt_rad": 0.15,
    }
    T_X_G = surface_grasp_to_transform(T_X_S, **arguments)
    T_A_G = surface_grasp_to_transform(compose(T_A_X, T_X_S), **arguments)
    assert np.allclose(T_A_G, compose(T_A_X, T_X_G), atol=1e-12)


def test_seven_parameter_helper_evaluates_uv_and_keeps_width_outside_se3():
    calls = []

    def plane_frame(u_m, v_m):
        calls.append((u_m, v_m))
        return make_transform(np.eye(3), [u_m, v_m, 0.0])

    first, first_width = surface_grasp_to_se3(
        [0.2, -0.3, 0.1, 0.4, -0.2, 0.15, 0.025],
        plane_frame,
    )
    second, second_width = surface_grasp_to_se3(
        [0.2, -0.3, 0.1, 0.4, -0.2, 0.15, 0.030],
        plane_frame,
    )
    assert calls == [(0.2, -0.3), (0.2, -0.3)]
    assert np.array_equal(first, second)
    assert first_width == 0.025
    assert second_width == 0.030
    anchor = np.array([0.2, -0.3, 0.0])
    assert np.allclose(first[:3, 3] + 0.1 * first[:3, 2], anchor)


def test_invalid_surface_grasp_inputs_fail_closed():
    _assert_raises(
        ValueError,
        "non-negative",
        lambda: surface_grasp_to_transform(
            np.eye(4),
            standoff_m=-0.1,
            normal_rotation_rad=0.0,
            closing_axis_tilt_rad=0.0,
            pad_axis_tilt_rad=0.0,
        ),
    )
    _assert_raises(
        ValueError,
        "shape (7,)",
        lambda: surface_grasp_to_se3([0.0] * 6, lambda _u, _v: np.eye(4)),
    )
    _assert_raises(
        ValueError,
        "w must be non-negative",
        lambda: surface_grasp_to_se3(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.01],
            lambda _u, _v: np.eye(4),
        ),
    )
    invalid_surface = np.eye(4)
    invalid_surface[0, 0] = 2.0
    _assert_raises(
        ValueError,
        "rotation",
        lambda: surface_grasp_to_se3(
            [0.0] * 7,
            lambda _u, _v: invalid_surface,
        ),
    )


if __name__ == "__main__":
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(f"passed {len(tests)} surface-grasp SE(3) tests")
