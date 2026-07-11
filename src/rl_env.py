"""One-step RL environment for learning handoff strategy.

Episode = one handoff attempt:
  reset(): sample a random initial grasp T_flangeA_part from the part CAD
           (which face A grabbed, roll about the approach, pose jitter) —
           mimicking bin-pick variability. Returns the observation.
  step(a): a = handoff parameters (part pose X_h + grasp choice for B).
           Runs the SAME oracle gates as handoff.py (same kin.py IK and
           collision path — nothing learned bypasses verification) and
           returns a shaped reward. done is always True (contextual bandit).

Reward shaping (max 1.0 + margin bonus):
  +0.3  gate 1: A can present the part at X_h
  +0.3  gate 2: B can take it at the co-grasp instant
  +0.4  gate 3: B carries it to the insert pose (waypoints checked)
  +0.2 * margin score on full success
Choosing a grasp outside G* (can never insert) ends the episode at 0.

The learned policy is a PROPOSER. At deployment, its proposal is verified
by the oracle; on failure, fall back to handoff.HandoffPlanner.search().
"""
from __future__ import annotations

import numpy as np

import kin
from handoff import HandoffPlanner
from scene import Scene, _stl_bbox

TCP_OFFSET = 0.200   # flange -> fingertip center (see gp7.urdf)


def _rot_from_axes(x, y):
    x = np.asarray(x, float); x /= np.linalg.norm(x)
    y = np.asarray(y, float); y -= x * (x @ y); y /= np.linalg.norm(y)
    return np.column_stack([x, y, np.cross(x, y)])


def _rotvec_to_R(v):
    v = np.asarray(v, float)
    th = np.linalg.norm(v)
    if th < 1e-12:
        return np.eye(3)
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


class GraspSampler:
    """Random-but-physical initial grasps of the part by A, derived from the
    part CAD bbox: pick a graspable face pair (width fits the finger gap),
    random roll about the approach axis, small translational/rotational
    jitter (pick error)."""

    def __init__(self, scene: Scene, finger_gap: float = 0.024,
                 jitter_t: float = 0.004, jitter_r: float = 0.12):
        mesh = scene.cfg.get("part_mesh")
        if mesh:
            import os
            from scene import ROOT
            lo, hi = _stl_bbox(os.path.join(ROOT, mesh))
            self.half = (hi - lo) / 2.0
        else:
            self.half = np.asarray(scene.cfg["part_half_extents"], float)
        self.jitter_t, self.jitter_r = jitter_t, jitter_r
        # approach axis a (tool axis, fingers advance along a);
        # closing axis c (fingers squeeze along c): need part extent along c
        # to fit in the gap. Enumerate axis pairs.
        axes = np.eye(3)
        self.modes = []
        for ia in range(3):
            for s in (1.0, -1.0):
                a = s * axes[ia]
                for ic in range(3):
                    if ic == ia:
                        continue
                    if 2 * self.half[ic] < finger_gap - 0.002:
                        self.modes.append((a, axes[ic]))
        assert self.modes, "no graspable face pair fits the finger gap"

    def canonical(self, mode_idx: int, roll: float = 0.0) -> np.ndarray:
        """Deterministic, jitter-free grasp for mode `mode_idx` — the grasp A
        ends up with after a clean re-pick from the nest."""
        a, c = self.modes[mode_idx]
        R_part_tool = _rot_from_axes(a, c) @ _rotvec_to_R([roll, 0, 0])
        T_part_flange = np.eye(4)
        T_part_flange[:3, :3] = R_part_tool
        T_part_flange[:3, 3] = -TCP_OFFSET * a
        return kin.inv_T(T_part_flange)

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        """T_flangeA_part: part pose in A's flange (tool0) frame."""
        a, c = self.modes[rng.integers(len(self.modes))]
        roll = rng.uniform(0, 2 * np.pi)
        # part frame in gripper coords: gripper tool axis = part -a is wrong;
        # fingers advance along +a in PART frame means tool axis maps to -a?
        # Convention (matches grasp_set_G): tool axis points along a toward
        # the part interior, TCP lands on the part origin.
        R_tool_x = a                    # tool x (tool axis) in part frame
        R_tool_y = c                    # closing direction in part frame
        R_part_tool = _rot_from_axes(R_tool_x, R_tool_y)
        R_part_tool = R_part_tool @ _rotvec_to_R([roll, 0, 0])
        # T_part_flange: flange sits TCP_OFFSET behind the part origin
        T_part_flange = np.eye(4)
        T_part_flange[:3, :3] = R_part_tool
        T_part_flange[:3, 3] = -TCP_OFFSET * a
        T_flange_part = kin.inv_T(T_part_flange)
        # pick-error jitter in the flange frame
        J = np.eye(4)
        J[:3, :3] = _rotvec_to_R(rng.normal(0, self.jitter_r / 3.0, 3))
        J[:3, 3] = rng.normal(0, self.jitter_t / 3.0, 3)
        return T_flange_part @ J


class HandoffEnv:
    """Contextual-bandit environment over handoff parameters."""

    # action: 5 continuous in [-1, 1] (x, y, z, yaw, roll) + grasp index
    N_CONT = 5

    def __init__(self, scene: Scene | None = None, seed: int = 0,
                 ik_restarts: int = 5):
        self.s = scene or Scene()
        self.pl = HandoffPlanner(self.s)
        self.pl.restarts = ik_restarts
        self.rng = np.random.default_rng(seed)
        self.sampler = GraspSampler(self.s)
        sp = self.s.cfg["handoff_search"]
        self.lo = np.array([sp["x"][0], sp["y"][0], sp["z"][0]])
        self.hi = np.array([sp["x"][1], sp["y"][1], sp["z"][1]])
        # gate 3 is independent of the initial grasp and X_h: cache G* once
        self.g_star = {name: (g, ins)
                       for name, g, ins in self.pl.filter_downstream()}
        self.grasp_names = [n for n, _ in self.pl.G]
        self.n_grasps = len(self.grasp_names)
        self.obs_dim = 12
        self._T_fA_part = None

    # -- observation: initial grasp (9) + part half extents (3) --

    def _obs(self) -> np.ndarray:
        T = self._T_fA_part
        t = (T[:3, 3] - np.array([TCP_OFFSET, 0, 0])) * 20.0   # ~[-1,1]
        r6 = np.concatenate([T[:3, 0], T[:3, 1]])
        return np.concatenate([t, r6, self.sampler.half * 20.0]).astype(np.float64)

    def reset(self, T_flangeA_part: np.ndarray | None = None) -> np.ndarray:
        self._T_fA_part = (np.asarray(T_flangeA_part, float)
                           if T_flangeA_part is not None
                           else self.sampler.sample(self.rng))
        self.pl.T_fA_part = self._T_fA_part
        self.pl.T_part_fA = kin.inv_T(self._T_fA_part)
        return self._obs()

    def action_to_candidate(self, a_cont, grasp_idx):
        a = np.clip(np.asarray(a_cont, float), -1.0, 1.0)
        pos = self.lo + (a[:3] * 0.5 + 0.5) * (self.hi - self.lo)
        yaw = a[3] * (np.pi / 2)
        roll = a[4] * np.pi
        cz, sz = np.cos(yaw), np.sin(yaw)
        cx, sx = np.cos(roll), np.sin(roll)
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        X = np.eye(4)
        X[:3, :3] = Rz @ Rx
        X[:3, 3] = pos
        return X, self.grasp_names[int(grasp_idx)]

    def step(self, a_cont, grasp_idx) -> tuple[float, dict]:
        from collections import Counter
        X_h, name = self.action_to_candidate(a_cont, grasp_idx)
        info = {"gate": 0, "grasp": name, "X_h": X_h}
        if name not in self.g_star:
            return 0.0, info                     # grasp can never insert
        g, ins = self.g_star[name]

        A_cands = self.pl.gate_A_presents(X_h)
        if not A_cands:
            return 0.0, info
        info["gate"] = 1
        reward = 0.3

        stats = Counter()
        for A_cand in A_cands:
            plan = self.pl._try_pair(X_h, A_cand, name, g, ins, stats)
            if plan is None:
                if stats.get("gate2b_A_retreat", 0) or \
                        stats.get("gate3_path_to_insert", 0):
                    info["gate"] = 2
                    reward = 0.6
                continue
            info["gate"] = 3
            jl = min(kin.limit_margin(self.s.robotA, plan.qA),
                     kin.limit_margin(self.s.robotB, plan.qB_grasp),
                     kin.limit_margin(self.s.robotB, plan.qB_insert))
            self.pl.c.check_state(plan.qA, plan.qB_grasp, holder="A",
                                  T_flange_part=self._T_fA_part,
                                  finger_ok=("A", "B"))
            clear = self.pl.c.min_clearance(holder="A")
            reward = 1.0 + 0.2 * (jl + clear / 0.05) / 2.0
            info.update(qA=plan.qA, qB_grasp=plan.qB_grasp,
                        qB_insert=plan.qB_insert, waypoints=plan.waypoints,
                        plan=plan)
            return reward, info
        return reward, info
