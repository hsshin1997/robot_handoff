"""Regression tests for semantic collision allowances and swept resolution."""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import mujoco
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from mujoco_sim.simulation.collision import (AllowedContact, CollisionPolicy,
                                  SceneCollisionChecker)  # noqa: E402
from mujoco_sim.simulation.kinematics import GP7Kinematics  # noqa: E402
from mujoco_sim.simulation.workcell import WorkcellSim  # noqa: E402


def checker():
    sim = WorkcellSim()
    return sim, SceneCollisionChecker(sim, GP7Kinematics(sim))


def _contact(sim, first: str, second: str, distance: float):
    return SimpleNamespace(
        geom1=sim.model.geom(first).id,
        geom2=sim.model.geom(second).id,
        dist=distance,
    )


def chunk_checker():
    """Small scene exercising automatically numbered part collision geoms."""
    model = mujoco.MjModel.from_xml_string("""
<mujoco model="part-chunk-policy-test">
  <option gravity="0 0 0"/>
  <worldbody>
    <body name="part_body">
      <geom name="part_collision_7" type="sphere" size="0.01" pos="0 0 0"/>
    </body>
    <body name="gripper_body">
      <geom name="A_gripper_collision_01" type="sphere" size="0.01" pos="0.03 0 0"/>
    </body>
    <body name="support_body">
      <geom name="reorientation_surface" type="sphere" size="0.01" pos="0.025 0 0"/>
    </body>
    <body name="wrist_body">
      <geom name="A_link_6_collision" type="sphere" size="0.01" pos="1 0 0"/>
    </body>
    <body name="unrelated_body">
      <geom name="part_collision_fixture" type="sphere" size="0.01" pos="2 0 0"/>
    </body>
  </worldbody>
</mujoco>
""")
    data = mujoco.MjData(model)
    sim = SimpleNamespace(model=model, data=data)
    return sim, SceneCollisionChecker(sim, kinematics=None)


def test_holder_allowance_never_hides_wrist_part_collision():
    sim, collision = checker()
    contact = _contact(sim, "part_collision", "A_link_6_collision", -1e-5)
    assert not collision._allowed(contact, ("A",))


def test_static_holder_contact_is_gripper_only_and_penetration_bounded():
    sim, collision = checker()
    shallow = _contact(sim, "part_collision", "A_gripper_collision_01", -0.0005)
    deep = _contact(sim, "part_collision", "A_gripper_collision_01", -0.002)
    assert collision._allowed(shallow, ("A",))
    assert not collision._allowed(deep, ("A",))


def test_numbered_part_chunk_uses_holder_semantics_but_not_wrist_allowance():
    sim, collision = chunk_checker()
    shallow = _contact(sim, "part_collision_7", "A_gripper_collision_01", -0.0005)
    deep = _contact(sim, "part_collision_7", "A_gripper_collision_01", -0.002)
    wrist = _contact(sim, "part_collision_7", "A_link_6_collision", -0.0001)
    assert collision._allowed(shallow, ("A",))
    assert not collision._allowed(deep, ("A",))
    assert not collision._allowed(wrist, ("A",))


def test_canonical_phase_allowance_matches_every_numbered_part_chunk():
    sim, collision = chunk_checker()
    policy = CollisionPolicy(allowed_contacts=(
        AllowedContact("part_collision", "reorientation_surface", 0.00075),
    ))
    shallow = _contact(sim, "part_collision_7", "reorientation_surface", -0.0005)
    deep = _contact(sim, "part_collision_7", "reorientation_surface", -0.001)
    assert collision._allowed(shallow, policy=policy)
    assert not collision._allowed(deep, policy=policy)


def test_part_like_nondigit_suffix_does_not_receive_part_allowances():
    sim, collision = chunk_checker()
    holder_contact = _contact(
        sim, "part_collision_fixture", "A_gripper_collision_01", -0.0001)
    support_contact = _contact(
        sim, "part_collision_fixture", "reorientation_surface", -0.0001)
    policy = CollisionPolicy(
        part_holders=("A",),
        allowed_contacts=(AllowedContact(
            "part_collision", "reorientation_surface"),),
    )
    assert not collision._allowed(holder_contact, policy=policy)
    assert not collision._allowed(support_contact, policy=policy)


def test_minimum_clearance_applies_chunk_phase_and_holder_semantics():
    _, collision = chunk_checker()
    part_group = ("part_collision_7",)
    raw = collision.minimum_clearance(
        geom_groups=part_group, distance_cap=0.05,
        policy=CollisionPolicy())
    phase_allowed = collision.minimum_clearance(
        geom_groups=part_group, distance_cap=0.05,
        policy=CollisionPolicy(allowed_contacts=(AllowedContact(
            "part_collision", "reorientation_surface"),)))
    holder_and_phase_allowed = collision.minimum_clearance(
        geom_groups=part_group, distance_cap=0.05,
        policy=CollisionPolicy(
            part_holders=("A",),
            allowed_contacts=(AllowedContact(
                "part_collision", "reorientation_surface"),),
        ))
    assert np.isclose(raw, 0.005, atol=1e-9)
    assert np.isclose(phase_allowed, 0.01, atol=1e-9)
    assert np.isclose(holder_and_phase_allowed, 0.05, atol=1e-9)


def test_edge_replay_resolution_scales_with_motion_length():
    _, collision = checker()
    short = collision._dense_edge(np.zeros(6), np.full(6, 0.01))
    long = collision._dense_edge(np.zeros(6), np.full(6, 0.70))
    assert len(short) == 2
    assert len(long) > 8
    assert np.max(np.abs(np.diff(long, axis=0))) <= collision.edge_max_joint_step + 1e-12


def test_clearance_margin_is_nonforcing_and_does_not_hide_self_penetration():
    sim, collision = chunk_checker()
    collision = SceneCollisionChecker(
        sim, kinematics=None, clearance_margin_m=0.006)
    active = (sim.model.geom_contype != 0) | (sim.model.geom_conaffinity != 0)
    assert np.allclose(sim.model.geom_margin[active], 0.003)
    assert np.allclose(sim.model.geom_gap[active], sim.model.geom_margin[active])

    positive_self_gap = _contact(
        sim, "A_gripper_collision_01", "A_link_6_collision", 0.001)
    real_self_penetration = _contact(
        sim, "A_gripper_collision_01", "A_link_6_collision", -0.0001)
    positive_environment_gap = _contact(
        sim, "A_gripper_collision_01", "reorientation_surface", 0.001)
    assert collision._allowed(positive_self_gap)
    assert not collision._allowed(real_self_penetration)
    assert not collision._allowed(positive_environment_gap)


def test_phase_prefix_allowance_can_permit_proximity_but_not_penetration():
    sim, collision = chunk_checker()
    positive = _contact(
        sim, "A_gripper_collision_01", "reorientation_surface", 0.0002)
    negative = _contact(
        sim, "A_gripper_collision_01", "reorientation_surface", -1e-6)
    contacts = (("A_gripper_collision_*", "reorientation_surface", 0.0),)
    assert collision._allowed(positive, allowed_geom_pairs=contacts)
    assert not collision._allowed(negative, allowed_geom_pairs=contacts)


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS  {test.__name__}")
    print(f"\n{len(tests)}/{len(tests)} passed")
