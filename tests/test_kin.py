"""Tests for the kinematics + collision layer (src/kin.py).

Run directly (no pytest needed):   python tests/test_kin.py
Or with pytest:                    pytest tests/test_kin.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import pybullet as p

import kin
from scene import Scene

_scene = None


def scene() -> Scene:
    global _scene
    if _scene is None:
        _scene = Scene()
    return _scene


def checker() -> kin.CollisionChecker:
    return kin.CollisionChecker(scene())


DOWN = np.array([  # flange orientation: gripper (tool0 +x) pointing straight down
    [0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0],
    [-1.0, 0.0, 0.0],
])


def flange_T(pos, R=None) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = np.eye(3) if R is None else R
    T[:3, 3] = pos
    return T


# ---------- FK ----------

def test_fk_matches_known_home_geometry():
    s = scene()
    T = kin.fk(s.robotA, [0.0] * 6)
    assert np.allclose(T[:3, 3], [0.56, 0.0, 0.815], atol=1e-3), T[:3, 3]
    T = kin.fk(s.robotB, [0.0] * 6)
    assert np.allclose(T[:3, 3], [0.29, 0.0, 0.815], atol=1e-3), T[:3, 3]


def test_fk_respects_base_transform():
    s = scene()
    # same q, bases 0.85 m apart and yawed pi: flange z must match, x mirrored about 0.425
    q = [0.3, -0.2, 0.1, 0.2, -0.5, 0.1]
    TA, TB = kin.fk(s.robotA, q), kin.fk(s.robotB, q)
    assert abs(TA[2, 3] - TB[2, 3]) < 1e-6
    assert abs((TA[0, 3] + TB[0, 3]) / 2.0 - 0.425) < 1e-6
    assert abs(TA[1, 3] + TB[1, 3]) < 1e-6


# ---------- IK ----------

def test_ik_roundtrip_random_reachable_poses():
    s = scene()
    rng = np.random.default_rng(42)
    for robot in (s.robotA, s.robotB):
        ok = 0
        for _ in range(25):
            q_true = rng.uniform(robot.lower * 0.6, robot.upper * 0.6)
            T_target = kin.fk(robot, q_true)
            q_sol = kin.ik(robot, T_target, rng=rng)
            assert q_sol is not None, f"IK failed on FK-generated pose {T_target[:3,3]}"
            T_sol = kin.fk(robot, q_sol)
            assert np.linalg.norm(T_sol[:3, 3] - T_target[:3, 3]) < kin.POS_TOL
            assert kin.rot_angle(T_sol[:3, :3], T_target[:3, :3]) < kin.ROT_TOL
            assert kin.within_limits(robot, q_sol)
            ok += 1
        assert ok == 25


def test_ik_rejects_unreachable():
    s = scene()
    far = flange_T([3.0, 0.0, 0.8])                       # 3 m out: beyond 0.927 reach
    assert kin.ik(s.robotA, far, restarts=3) is None
    below = flange_T([0.3, 0.0, -1.2], DOWN)              # under the cell floor
    assert kin.ik(s.robotA, below, restarts=3) is None


def test_ik_same_path_for_both_robots():
    # the shared-IK contract: one function, works on either robot instance
    s = scene()
    mid = flange_T([0.425, 0.0, 0.60], DOWN)              # between the arms
    qa = kin.ik(s.robotA, mid, seed=s.cfg["home_qA"])
    qb = kin.ik(s.robotB, mid, seed=s.cfg["home_qB"])
    assert qa is not None, "A cannot reach the mid pose"
    assert qb is not None, "B cannot reach the mid pose"
    assert np.allclose(kin.fk(s.robotA, qa)[:3, 3], mid[:3, 3], atol=kin.POS_TOL)
    assert np.allclose(kin.fk(s.robotB, qb)[:3, 3], mid[:3, 3], atol=kin.POS_TOL)


# ---------- joint limits ----------

def test_limits_and_margin():
    s = scene()
    r = s.robotA
    assert kin.within_limits(r, np.zeros(6))
    assert not kin.within_limits(r, r.upper + 0.1)
    near_edge = r.upper - 0.05
    assert kin.within_limits(r, near_edge)
    assert not kin.within_limits(r, near_edge, margin=0.1)
    assert kin.limit_margin(r, (r.lower + r.upper) / 2.0) > 0.49


# ---------- collision ----------

def test_home_state_is_collision_free():
    s = scene()
    c = checker()
    free, why = c.check_state(s.cfg["home_qA"], s.cfg["home_qB"], holder="A")
    assert free, why


def test_zero_pose_detects_arm_arm_collision():
    c = checker()
    free, why = c.check_state([0.0] * 6, [0.0] * 6, holder="A")
    assert not free
    assert why in ("robot_robot_collision", "part_collision"), why


def test_gripper_into_nest_detected():
    """Oracle-style consumption: a pose with the fingers inside the plate must
    collide on EVERY IK branch; a pose hovering above must have at least one
    collision-free branch."""
    s = scene()
    c = checker()
    nest = s.cfg["nest"]
    cx, cy = nest["center_xy"]

    bad = flange_T([cx, cy, nest["top_z"] + 0.200 - 0.04], DOWN)
    sols = kin.ik_solutions(s.robotA, bad, seed=s.cfg["home_qA"])
    assert sols, "IK should reach the nest region"
    for q in sols:
        free, why = c.check_state(q, s.cfg["home_qB"], holder="A")
        assert not free, "fingers 40 mm inside the plate reported collision-free"

    good = flange_T([cx, cy, nest["top_z"] + 0.200 + 0.06], DOWN)
    sols = kin.ik_solutions(s.robotA, good, seed=s.cfg["home_qA"])
    assert sols, "IK should reach above the nest"
    assert any(c.check_state(q, s.cfg["home_qB"], holder="A")[0] for q in sols), \
        "no collision-free branch hovering 60 mm above the nest"


def test_part_placement_follows_holder():
    s = scene()
    c = checker()
    c.check_state(s.cfg["home_qA"], s.cfg["home_qB"], holder="A")
    T_fl = kin.fk(s.robotA, s.cfg["home_qA"])
    expect = T_fl @ np.asarray(s.cfg["T_flangeA_part"], dtype=float)
    got = np.array(p.getBasePositionAndOrientation(s.part_id)[0])
    assert np.linalg.norm(got - expect[:3, 3]) < 1e-6


def test_min_clearance_sane_at_home():
    s = scene()
    c = checker()
    c.check_state(s.cfg["home_qA"], s.cfg["home_qB"], holder="A")
    d = c.min_clearance(holder="A")
    assert 0.01 < d <= 0.05, d


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
