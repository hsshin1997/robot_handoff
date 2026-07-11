"""Tests for the RL environment and policy (src/rl_env.py, src/rl_policy.py).

Run directly:   python tests/test_rl.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np

import kin
from rl_env import HandoffEnv, GraspSampler, TCP_OFFSET
from rl_policy import HandoffPolicy

_env = None


def env() -> HandoffEnv:
    global _env
    if _env is None:
        _env = HandoffEnv(seed=3)
    return _env


# ---------- grasp sampler ----------

def test_sampled_grasps_are_valid_transforms():
    e = env()
    rng = np.random.default_rng(0)
    for _ in range(50):
        T = e.sampler.sample(rng)
        R = T[:3, :3]
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
        assert abs(np.linalg.det(R) - 1.0) < 1e-9
        # part origin must sit near the TCP (0.2 m out along tool x),
        # within the jitter budget
        assert np.linalg.norm(T[:3, 3] - [TCP_OFFSET, 0, 0]) < 0.03, T[:3, 3]


def test_sampler_respects_finger_gap():
    e = env()
    # every mode's closing axis extent must fit the 24 mm gap
    for a, c in e.sampler.modes:
        width = 2 * e.sampler.half[np.argmax(np.abs(c))]
        assert width < 0.024


# ---------- env ----------

def test_env_reset_step_deterministic():
    e = env()
    T = e.sampler.sample(np.random.default_rng(9))
    obs1 = e.reset(T)
    r1, i1 = e.step(np.zeros(e.N_CONT), 0)
    obs2 = e.reset(T)
    r2, i2 = e.step(np.zeros(e.N_CONT), 0)
    assert np.allclose(obs1, obs2)
    assert r1 == r2 and i1["gate"] == i2["gate"]


def test_env_rejects_non_insertable_grasp():
    e = env()
    bad = [i for i, n in enumerate(e.grasp_names) if n not in e.g_star]
    if not bad:
        return  # every grasp is insertable in this config
    e.reset()
    r, info = e.step(np.zeros(e.N_CONT), bad[0])
    assert r == 0.0 and info["gate"] == 0


def test_env_success_reachable_with_nominal_grasp():
    """The action encoding the known-good candidate (pinned in
    test_handoff.py) must score a full success with the nominal grasp —
    otherwise the reward landscape is dead."""
    e = env()
    T_nominal = np.eye(4)
    T_nominal[0, 3] = 0.200      # part COG at the TCP, no roll
    e.reset(T_nominal)
    # action mapping to X_h = (0.5875, -0.10, 0.575, yaw 45°, roll 0), top_r0
    a = np.array([0.5, -1.0, 0.0, 0.5, 0.0])
    gi = e.grasp_names.index("top_r0")
    r, info = e.step(a, gi)
    assert info["gate"] == 3, f"known-good action stopped at gate {info['gate']}"
    assert r >= 1.0
    # verify the returned tuple is real: FK matches the candidate
    X_h, g_name = e.action_to_candidate(a, gi)
    g, _ = e.g_star[g_name]
    TB = kin.fk(e.s.robotB, info["qB_grasp"])
    assert np.linalg.norm(TB[:3, 3] - (X_h @ g)[:3, 3]) < kin.POS_TOL


# ---------- policy ----------

def test_policy_shapes_and_determinism():
    e = env()
    pol = HandoffPolicy(e.obs_dim, e.N_CONT, e.n_grasps, seed=1)
    obs = e.reset()
    a, gi = pol.act(obs, np.random.default_rng(0), deterministic=True)
    a2, gi2 = pol.act(obs, np.random.default_rng(99), deterministic=True)
    assert a.shape == (e.N_CONT,) and np.allclose(a, a2) and gi == gi2
    assert np.all(a >= -1) and np.all(a <= 1)


def test_policy_update_moves_toward_reward():
    """On a synthetic bandit (reward = 1 iff grasp==2 and a[0]>0), a few
    updates must raise the probability of the rewarded action."""
    rng = np.random.default_rng(5)
    pol = HandoffPolicy(4, 2, 4, seed=2)
    obs = np.ones(4) * 0.3

    def p_grasp2():
        _, _, logits = pol.forward(obs)
        p = np.exp(logits - logits.max()); p /= p.sum()
        return p[2]

    before = p_grasp2()
    for _ in range(60):
        batch = []
        for _ in range(32):
            a, gi = pol.act(obs, rng)
            r = float(gi == 2 and a[0] > 0)
            batch.append((obs, a, gi, r))
        pol.update(batch)
    after = p_grasp2()
    mean_after = pol.forward(obs)[0]
    assert after > before + 0.2, (before, after)
    assert mean_after[0] > 0.2, mean_after


def test_policy_save_load_roundtrip(tmp_path=None):
    e = env()
    pol = HandoffPolicy(e.obs_dim, e.N_CONT, e.n_grasps, seed=6)
    path = "/tmp/pol_test.npz"
    pol.save(path)
    pol2 = HandoffPolicy.load(path, e.obs_dim, e.N_CONT, e.n_grasps)
    obs = e.reset()
    a1, g1 = pol.act(obs, np.random.default_rng(0), deterministic=True)
    a2, g2 = pol2.act(obs, np.random.default_rng(0), deterministic=True)
    assert np.allclose(a1, a2) and g1 == g2


# ---------- runner ----------

if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as ex:
            failed += 1
            print(f"FAIL  {t.__name__}: {ex}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
