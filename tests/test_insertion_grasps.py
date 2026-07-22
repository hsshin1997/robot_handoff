"""Focused tests for task-aware insertion grasp filtering."""
from __future__ import annotations

import os
import sys

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from mujoco_sim.modeling.grasps import GraspCandidate, TriangleMesh
from mujoco_sim.modeling.insertion_grasps import (
    AxisAlignedRegion,
    FreeSpacePlane,
    GripperMeshComponent,
    GripperMeshModel,
    InsertionTaskGeometry,
    evaluate_insertion_grasps,
)


def _box_triangles(minimum, maximum):
    low = np.asarray(minimum, dtype=float)
    high = np.asarray(maximum, dtype=float)
    vertices = np.array([
        [low[0], low[1], low[2]],
        [high[0], low[1], low[2]],
        [high[0], high[1], low[2]],
        [low[0], high[1], low[2]],
        [low[0], low[1], high[2]],
        [high[0], low[1], high[2]],
        [high[0], high[1], high[2]],
        [low[0], high[1], high[2]],
    ])
    faces = np.array([
        [0, 2, 1], [0, 3, 2],
        [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4],
        [3, 7, 6], [3, 6, 2],
        [0, 4, 7], [0, 7, 3],
        [1, 2, 6], [1, 6, 5],
    ])
    return vertices[faces]


def _candidate(*, translation_y=0.0, contact_x=0.0):
    transform = np.eye(4)
    transform[1, 3] = translation_y
    contacts = np.array([
        [contact_x, translation_y - 0.005, 0.0],
        [contact_x, translation_y + 0.005, 0.0],
    ])
    return GraspCandidate(
        T_P_E=transform,
        contact_points=contacts,
        contact_normals=np.array([[0.0, -1.0, 0.0], [0.0, 1.0, 0.0]]),
        required_opening=0.01,
        approach_direction=np.array([0.0, 0.0, 1.0]),
        closing_direction=np.array([0.0, 1.0, 0.0]),
        quality=0.8,
        antipodal_quality=1.0,
        support_quality=1.0,
        opening_margin=1.0,
        palm_clearance=0.1,
    )


def _model(*, aperture_multiplier=0.0, opening_axis=(0.0, 1.0, 0.0)):
    mesh = TriangleMesh.from_triangles(
        _box_triangles([-0.01, -0.01, -0.01], [0.01, 0.01, 0.01]),
        source="synthetic gripper component",
    )
    component = GripperMeshComponent(
        "test_component", mesh, np.eye(4), aperture_multiplier,
    )
    return GripperMeshModel(
        T_G_E=np.eye(4),
        reference_aperture_m=0.01,
        opening_axis_G=np.asarray(opening_axis, dtype=float),
        components=(component,),
    )


def _task(*, region_max_x=0.1):
    return InsertionTaskGeometry(
        insertion_axis_P=np.array([0.0, -1.0, 0.0]),
        pcb_plane_P=FreeSpacePlane(np.array([0.0, 1.0, 0.0]), 0.0),
        graspable_regions=(AxisAlignedRegion(
            "housing", np.array([-0.1, -0.1, -0.1]),
            np.array([region_max_x, 0.1, 0.1]),
        ),),
        contact_region_tolerance_m=0.0,
        preinsert_distance_m=0.02,
        minimum_pcb_clearance_m=0.001,
    )


def test_plane_support_uses_t_p_e_without_inverting_it():
    evaluations = evaluate_insertion_grasps(
        [_candidate(translation_y=0.02)], task=_task(), gripper_mesh=_model(),
    )
    result = evaluations[0]
    # The component spans y=[-10,+10] mm in E/G and T_P_E shifts it +20 mm.
    assert np.isclose(result.seated_pcb_clearance_m, 0.01)
    assert result.status == "phase1_seated_geometric_candidate"
    assert np.isclose(result.collision_free_insertion_travel_m, 0.02)
    assert np.isclose(result.remaining_to_seat_at_collision_m, 0.0)


def test_preinsert_only_status_and_collision_distance_have_correct_sign():
    result = evaluate_insertion_grasps(
        [_candidate()], task=_task(), gripper_mesh=_model(),
    )[0]
    assert np.isclose(result.seated_pcb_clearance_m, -0.01)
    assert np.isclose(result.preinsert_pcb_clearance_m, 0.01)
    assert result.status == "phase1_preinsert_only_candidate"
    assert np.isclose(result.collision_free_insertion_travel_m, 0.009)
    assert np.isclose(result.remaining_to_seat_at_collision_m, 0.011)


def test_contact_region_rejects_before_running_mesh_clearance():
    result = evaluate_insertion_grasps(
        [_candidate(contact_x=0.05)],
        task=_task(region_max_x=0.02),
        gripper_mesh=_model(),
    )[0]
    assert result.status == "rejected_contact_region"
    assert result.seated_pcb_clearance_m is None
    assert result.limiting_component is None


def test_component_aperture_motion_changes_exact_mesh_support():
    candidate = _candidate()
    # Override the candidate opening to create +10 mm total aperture delta.
    candidate = GraspCandidate(
        T_P_E=candidate.T_P_E,
        contact_points=np.array([[0.0, -0.01, 0.0], [0.0, 0.01, 0.0]]),
        contact_normals=candidate.contact_normals,
        required_opening=0.02,
        approach_direction=candidate.approach_direction,
        closing_direction=candidate.closing_direction,
        quality=candidate.quality,
        antipodal_quality=candidate.antipodal_quality,
        support_quality=candidate.support_quality,
        opening_margin=candidate.opening_margin,
        palm_clearance=candidate.palm_clearance,
    )
    plane = FreeSpacePlane(np.array([0.0, 1.0, 0.0]), 0.0)
    clearance, limiting, per_component = _model(
        aperture_multiplier=0.5,
    ).plane_clearance(candidate, plane)
    assert limiting == "test_component"
    assert np.isclose(clearance, -0.005)
    assert np.isclose(per_component["test_component"], -0.005)


def test_plane_violation_witness_reports_actual_vertex_bounds():
    witness = _model().plane_violation_witness(
        _candidate(),
        FreeSpacePlane(np.array([0.0, 1.0, 0.0]), 0.0),
        required_clearance_m=0.001,
    )["test_component"]
    assert np.isclose(witness["minimum_clearance_m"], -0.01)
    assert witness["violating_vertex_count"] == 4
    assert np.allclose(witness["violating_bounds_min_P_m"], [-0.01, -0.01, -0.01])
    assert np.allclose(witness["violating_bounds_max_P_m"], [0.01, -0.01, 0.01])


def test_insertion_axis_must_point_into_the_opposite_halfspace():
    try:
        InsertionTaskGeometry(
            insertion_axis_P=np.array([0.0, 1.0, 0.0]),
            pcb_plane_P=FreeSpacePlane(np.array([0.0, 1.0, 0.0]), 0.0),
            graspable_regions=(AxisAlignedRegion(
                "housing", np.array([-1.0, -1.0, -1.0]),
                np.array([1.0, 1.0, 1.0]),
            ),),
            contact_region_tolerance_m=0.0,
            preinsert_distance_m=0.02,
            minimum_pcb_clearance_m=0.0,
        )
    except ValueError as error:
        assert "opposite" in str(error)
    else:
        raise AssertionError("invalid insertion/plane signs were accepted")


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS  {test.__name__}")
    print(f"\n{len(tests)}/{len(tests)} passed")
