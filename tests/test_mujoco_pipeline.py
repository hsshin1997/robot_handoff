"""Mathematical and end-to-end checks for the new MuJoCo handoff planner."""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from mujoco_sim.planning import HandoffPlanner
from mujoco_sim.se3 import inverse, make_transform, transform_from_rpy
from mujoco_sim.sim import WorkcellSim
from mujoco_sim.uncertainty import check_axis_aligned_capture, combine_independent


_planner = None


def planner():
    global _planner
    if _planner is None:
        _planner = HandoffPlanner(WorkcellSim())
    return _planner


def test_known_start_pose_recovers_document_grasp_convention():
    p = planner()
    expected = inverse(p.X_start) @ p.kin.fk("A", p.q_start["A"])
    assert np.allclose(p.g_A_start, expected, atol=1e-10)
    # The minimal project manifest stores ^E T_P exactly once at its boundary.
    T_E_P = p.project.T_tcp_part_start
    assert np.allclose(p.g_A_start, inverse(T_E_P), atol=1e-8)


def test_tcp_jacobian_matches_finite_difference_translation():
    p = planner()
    q = p.q_start["A"].copy()
    J = p.kin.jacobian("A", q)
    eps = 1e-6
    base = p.kin.fk("A", q)[:3, 3]
    numerical = np.zeros((3, 6))
    for index in range(6):
        perturbed = q.copy(); perturbed[index] += eps
        numerical[:, index] = (p.kin.fk("A", perturbed)[:3, 3] - base) / eps
    assert np.allclose(J[:3], numerical, atol=2e-5)


def test_geometry_grasps_are_spatially_distinct_and_fit_asset_capability():
    p = planner()
    capability = p.project.gripper("B")
    candidates = list(p.grasp_candidates.values())
    assert len(candidates) >= 8
    assert all(capability.opening_min <= item.required_opening <= capability.opening_max
               for item in candidates)
    centers = np.array([item.T_P_E[:3, 3] for item in candidates])
    assert np.ptp(centers[:, 0]) > 0.020  # both ends of the real connector
    assert any(np.linalg.norm(center - p.part_center) > 0.010 for center in centers)


def test_correction_polytope_uses_configured_hole_frame():
    p = planner()
    X = p.insertion_poses[0][1]
    assert len(list(p._correction_vertices(X))) == 16


def test_uncertainty_capture_gate_uses_combined_3sigma():
    grasp = np.diag([0.0002**2] * 3 + [np.radians(0.1)**2] * 3)
    calibration = np.diag([0.0003**2] * 3 + [np.radians(0.05)**2] * 3)
    result = check_axis_aligned_capture(
        combine_independent(grasp, calibration), [0.002] * 3,
        [np.radians(0.5)] * 3)
    assert result.accepted
    assert np.all(result.translation_3sigma > 0.001)


def test_legacy_center_to_center_cograsp_is_rejected_before_ik():
    p = planner()
    # A deliberately centered legacy candidate, built locally so the runtime
    # planner exposes no axis/roll rule-based grasp API.
    z = np.array([0.0, -1.0, 0.0])
    x = np.cross(np.array([0.0, 0.0, 1.0]), z)
    y = np.cross(z, x)
    centered = make_transform(np.column_stack((x, y, z)), p.part_center)
    compatible, separation = p._gripper_compatibility(p.g_A_start, centered)
    assert not compatible
    assert separation < 0.0


def test_handoff_grid_is_generated_from_user_region():
    p = planner()
    region = p.project.region("handoff")
    poses = p.pose_grid()
    assert poses
    assert all(region.contains(pose[:3, 3]) for pose in poses)


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS  {test.__name__}")
    print(f"\n{len(tests)}/{len(tests)} passed")
