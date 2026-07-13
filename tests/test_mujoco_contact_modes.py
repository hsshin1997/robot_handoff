"""Phase-contact policy must reflect the fidelity of supplied geometry."""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import mujoco_sim.diagnostics.contact_audit as audit_module
from mujoco_sim.diagnostics.contact_audit import ContactSample, _summarize
from mujoco_sim.simulation.collision import AllowedContact
from mujoco_sim.planner.planner import (EXACT_INSERTION_CONTACTS,
                                 PLACEHOLDER_INSERTION_CONTACTS,
                                 REORIENTATION_CONTACTS,
                                 insertion_contacts)


def _project(insertion):
    return SimpleNamespace(manifest={"insertion": insertion})


def test_placeholder_allows_only_numerical_part_board_overlap():
    contacts = insertion_contacts(_project({}))
    assert contacts == PLACEHOLDER_INSERTION_CONTACTS
    assert contacts == (("part_collision", "pcb_board*", 1e-5),)


def test_exact_fixture_cad_does_not_inherit_solid_board_exception():
    contacts = insertion_contacts(_project({"collision_cad": "fixture.step"}))
    assert contacts == EXACT_INSERTION_CONTACTS
    assert contacts == (("part_collision", "insertion_collision*", 0.0),)
    allowance = AllowedContact(*contacts[0])
    assert allowance.allows(
        "part_collision_07", "insertion_collision_02", 0.0)
    assert not allowance.allows(
        "part_collision_07", "insertion_collision_02", 1e-12)


def test_reorientation_policy_never_allows_gripper_stage_penetration():
    assert REORIENTATION_CONTACTS[0] == (
        "part_collision", "reorientation_surface", 5e-5)
    assert REORIENTATION_CONTACTS[1] == (
        "A_gripper_collision_*", "reorientation_surface", 0.0)


def test_contact_audit_summary_preserves_signed_distance_and_policy():
    samples = [
        ContactSample("insert", 3, ("pcb_board", "part_collision"),
                      -2e-6, 2e-6, True, (0.0, 0.0, 0.0)),
        ContactSample("insert", 4, ("pcb_board", "A_wrist"),
                      -3e-4, 3e-4, False, (0.0, 0.0, 0.0)),
    ]
    report = _summarize(samples)
    assert report["minimum_signed_distance_m"] == -3e-4
    assert report["maximum_penetration_m"] == 3e-4
    assert report["forbidden_contact_samples"] == 1


def test_insertion_audit_uses_the_same_parked_A_state_as_plan_and_executor():
    checked_qA = []

    class Collision:
        @staticmethod
        def check(qA, qB, X_part, **kwargs):
            checked_qA.append(np.asarray(qA).copy())
            return SimpleNamespace(
                free=True, reason="ok", pair=None, penetration=0.0)

    planner = SimpleNamespace(
        q_start={"A": np.full(6, 0.25)},
        insertion_poses=[("pcb", np.eye(4))],
        kin=SimpleNamespace(fk=lambda robot, q: np.eye(4)),
        collision=Collision(),
        project=_project({}),
    )
    direct = SimpleNamespace(
        qA_retreat=np.full(6, -0.75),
        g_B=np.eye(4),
        downstream=SimpleNamespace(
            trajectories={"pcb_insert": [np.zeros(6), np.ones(6)]}),
    )
    original = audit_module._contacts
    audit_module._contacts = lambda *args, **kwargs: []
    try:
        report = audit_module.audit_insertion(object(), planner, direct)
    finally:
        audit_module._contacts = original
    assert report["targets"][0]["gate_failure"] is None
    assert len(checked_qA) == 2
    assert all(np.array_equal(value, planner.q_start["A"])
               for value in checked_qA)


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS  {test.__name__}")
    print(f"\n{len(tests)}/{len(tests)} passed")
