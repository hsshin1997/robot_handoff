"""Tests for the regrasp branch (src/regrasp.py).

Run directly:   python tests/test_regrasp.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np

import kin
from handoff import HandoffPlanner
from regrasp import RegraspPlanner, TABLE_PATH, face_down_rotations
from scene import Scene

_ctx = None


def ctx():
    global _ctx
    if _ctx is None:
        s = Scene()
        pl = HandoffPlanner(s)
        _ctx = (s, pl, RegraspPlanner(s, pl))
    return _ctx


# ---------- placements ----------

def test_face_down_rotations_are_valid():
    rots = face_down_rotations()
    assert len(rots) == 6
    seen = set()
    for name, R, k in rots:
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert abs(np.linalg.det(R) - 1.0) < 1e-12
        seen.add(name)
    assert len(seen) == 6


def test_placement_rests_on_plate():
    s, pl, rg = ctx()
    for name, R, k in rg.placements:
        X = rg.placement_pose(R, k, yaw=0.3)
        # part bottom = center - half extent of the DOWN axis must sit just
        # above the plate top
        bottom = X[2, 3] - rg.sampler.half[k]
        assert 0.001 < bottom - rg.nest_top < 0.01, (name, bottom)
        # the down axis of the part must point straight down in world
        down_world = X[:3, :3] @ (np.eye(3)[k])
        assert abs(abs(down_world[2]) - 1.0) < 1e-9


# ---------- viability table ----------

def test_table_exists_with_viable_entries():
    s, pl, rg = ctx()
    assert os.path.exists(TABLE_PATH), "run RegraspPlanner.build_table() first"
    table = rg.load_table()
    viable = list(rg._viable(table))
    assert viable, "no handoff-viable canonical grasp — regrasp branch is dead"
    # each cached plan must be kinematically consistent with its grasp
    for mode_i, g_new, plan in viable:
        TB = kin.fk(s.robotB, plan.qB_grasp)
        expect = plan.X_h @ plan.g
        assert np.linalg.norm(TB[:3, 3] - expect[:3, 3]) < kin.POS_TOL


# ---------- the regrasp search ----------

def test_regrasp_plan_found_and_consistent():
    s, pl, rg = ctx()
    bad = rg.sampler.canonical(1, roll=0.7)   # mode 1: not handoff-viable
    plan = rg.find_regrasp(bad)
    assert plan is not None, "regrasp search failed on a placeable grasp"

    # place pose: A's flange must put the part exactly at X_place
    T = kin.fk(s.robotA, plan.qA_place)
    assert np.linalg.norm(
        T[:3, 3] - (plan.X_place @ kin.inv_T(bad))[:3, 3]) < kin.POS_TOL
    # re-pick pose: A's flange must match the new grasp at X_place
    T = kin.fk(s.robotA, plan.qA_pick)
    assert np.linalg.norm(
        T[:3, 3] - (plan.X_place @ kin.inv_T(plan.g_new))[:3, 3]) < kin.POS_TOL

    # both states re-verified collision-free through the one checker
    free, why = rg.c.check_state(plan.qA_place, pl.home_qB, holder="A",
                                 T_flange_part=bad)
    assert free, f"place state collides: {why}"
    rg.c.set_part_world(plan.X_place)
    free, why = rg.c.check_state(plan.qA_pick, pl.home_qB, holder=None,
                                 finger_ok=("A",))
    assert free, f"re-pick state collides: {why}"

    # re-pick approach must not come from below the plate (above or side ok)
    a_world = plan.X_place[:3, :3] @ kin.inv_T(plan.g_new)[:3, 0]
    # (tool x in part frame is column 0 of T_part_flange rotation)
    assert a_world[2] <= 0.3, a_world


# ---------- runner ----------

if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
