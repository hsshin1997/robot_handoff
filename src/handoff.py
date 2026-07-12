"""Handoff feasibility oracle + search over (handoff pose, grasp).

A candidate is (X_h, g): X_h = world pose of the PART at the transfer
instant, g = T_part_flangeB from grasp_set_G. A's grasp is fixed
(T_flangeA_part, known from the pick). Frame algebra:

    A's flange at handoff :  X_h @ inv(T_flangeA_part)
    B's flange at handoff :  X_h @ g
    B's flange at insert  :  T_world_insert @ g          (same g — the crux)

Gate 3 depends only on g, so grasp_set_G is filtered to the
downstream-feasible subset G* ONCE, before the pose loop (doc §3.2).
All IK and collision goes through kin.py — one path, no duplicates.
"""
from __future__ import annotations

import json
import os
import time
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

import kin
from scene import Scene, ROOT

CACHE_PATH = os.path.join(ROOT, "models", "plan_cache.json")

# GP7 rated joint speeds (S L U R B T), rad/s — for cycle-time estimates
GP7_SPEED = np.radians([375.0, 315.0, 410.0, 550.0, 550.0, 1000.0])
COGRASP_DWELL = 0.3          # s, the co-grasp instant (doc: keep < 300 ms)


def phase_time(q_from, q_to, frac: float = 0.7) -> float:
    """Time for a joint move at `frac` of rated speed (accel/decel margin)."""
    dq = np.abs(np.asarray(q_to, dtype=float) - np.asarray(q_from, dtype=float))
    return float(np.max(dq / (GP7_SPEED * frac)))


@dataclass
class HandoffPlan:
    """A fully verified, executable handoff. `segments` holds every checked
    joint-space leg in execution order — this is what a robot controller
    consumes:
      A_approach        : qA_pre -> qA          (A brings the part in)
      B_approach        : qB_pre -> qB_grasp    (B's final linear-ish approach)
      A_retreat         : qA -> qA_out          (A backs off after release)
      B_to_preinsert    : qB_grasp -> qB_preinsert (transit waypoints)
      B_insert_approach : qB_preinsert -> qB_insert (descent onto the hole)
    Every listed configuration passed limits + singularity + collision."""
    X_h: np.ndarray          # world part pose at transfer
    grasp_name: str
    g: np.ndarray            # T_part_flangeB
    qA: np.ndarray           # A presents
    qB_grasp: np.ndarray     # B takes
    qB_insert: np.ndarray    # B at insert pose
    waypoints: list          # checked qB configs handoff -> pre-insert
    qA_pre: np.ndarray | None = None
    qB_pre: np.ndarray | None = None
    qA_out: np.ndarray | None = None
    qB_preinsert: np.ndarray | None = None
    segments: dict = field(default_factory=dict)
    score: float = 0.0
    exec_time: float = 0.0   # estimated physical execution time (s)


@dataclass
class SearchReport:
    plan: HandoffPlan | None
    stats: Counter = field(default_factory=Counter)
    n_candidates: int = 0
    G_star: list = field(default_factory=list)   # names surviving gate 3

    @property
    def feasible(self) -> bool:
        return self.plan is not None

    def dominant_failure(self) -> str:
        fails = {k: v for k, v in self.stats.items() if k != "ok"}
        return max(fails, key=fails.get) if fails else "none"


def _rpy_deg_to_R(rpy) -> np.ndarray:
    r, pch, y = np.radians(rpy)
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(pch), np.sin(pch), np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


class PlanCache:
    """Warm-start memory: every solved handoff is stored as
    (T_flangeA_part -> X_h, grasp_name). Online, the entries nearest to the
    current grasp are re-verified through the oracle first (~0.1 s each) —
    solved cases repeat, so this usually replaces the grid search entirely.
    This is also the 'continuous learning without RL' loop: the cache grows
    with every cycle."""

    def __init__(self, path: str = CACHE_PATH):
        self.path = path
        self.entries: list[dict] = []
        if os.path.exists(path):
            self.entries = json.load(open(path))

    def nearest(self, T_fA_part, k: int = 6) -> list[dict]:
        T = np.asarray(T_fA_part, dtype=float)

        def dist(e):
            E = np.asarray(e["T_fA_part"])
            return (np.linalg.norm(E[:3, 3] - T[:3, 3])
                    + 0.3 * kin.rot_angle(E[:3, :3], T[:3, :3]))
        return sorted(self.entries, key=dist)[:k]

    def add(self, T_fA_part, plan: "HandoffPlan") -> None:
        self.entries.append({"T_fA_part": np.asarray(T_fA_part).tolist(),
                             "X_h": plan.X_h.tolist(),
                             "grasp_name": plan.grasp_name})
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        json.dump(self.entries, open(self.path, "w"))


class HandoffPlanner:
    def __init__(self, scene: Scene, checker: kin.CollisionChecker | None = None):
        self.s = scene
        self.c = checker or kin.CollisionChecker(scene)
        cfg = scene.cfg
        sp = cfg.get("handoff_search", {})
        self.m_jl = float(sp.get("joint_limit_margin", 0.09))
        self.restarts = int(sp.get("ik_restarts", 8))
        self.n_way = int(sp.get("n_path_waypoints", 4))
        self.sp = sp

        self.T_fA_part = np.asarray(cfg["T_flangeA_part"], dtype=float)
        self.T_part_fA = kin.inv_T(self.T_fA_part)
        self.X_ins = np.asarray(cfg["T_world_insert"], dtype=float)
        self.G = self._filter_grasp_width(self._expand_symmetry(
            [(e["name"], np.asarray(e["T"], dtype=float)) for e in cfg["grasp_set_G"]],
            cfg.get("part_symmetry_rpy_deg", [])))
        self.home_qA = np.asarray(cfg["home_qA"], dtype=float)
        self.home_qB = np.asarray(cfg["home_qB"], dtype=float)
        self.rng = np.random.default_rng(7)
        self.baseA = np.asarray(cfg["robotA_base"], dtype=float)[:3, 3]
        self.baseB = np.asarray(cfg["robotB_base"], dtype=float)[:3, 3]
        # GP7 horizontal reach is 0.927 m to the flange; leave headroom
        self.reach_max = 0.90
        self.reach_min = 0.18

        # approach/retreat geometry (doc §4.1 G4/G6, §5)
        ap = sp.get("approach", {})
        self.d_pre = float(ap.get("d_pre", 0.040))
        self.d_retreat = float(ap.get("d_retreat", 0.060))
        self.d_app = float(ap.get("d_app", 0.030))
        self.n_sweep = int(ap.get("n_sweep", 3))
        self.X_app = self.X_ins.copy()
        self.X_app[2, 3] += self.d_app          # pre-insert hover pose

        # singularity gate (doc §4.1 G3): w_min calibrated per robot as a
        # percentile of manipulability over random reachable configs
        pct = float(sp.get("singularity_w_percentile", 5.0))
        self.w_min = {scene.robotA.body: kin.calibrate_w_min(scene.robotA, pct),
                      scene.robotB.body: kin.calibrate_w_min(scene.robotB, pct)}

    @staticmethod
    def _expand_symmetry(G, sym_rpy_deg):
        """Grasp orbit expansion (doc §0): if the part maps onto itself under
        rotation sigma, then sigma @ g is an equally valid grasp."""
        if not sym_rpy_deg:
            return G
        out = list(G)
        for i, rpy in enumerate(sym_rpy_deg):
            S = np.eye(4)
            S[:3, :3] = _rpy_deg_to_R(rpy)
            for name, g in G:
                g2 = S @ g
                if not any(np.allclose(g2, gg, atol=1e-9) for _, gg in out):
                    out.append((f"{name}~s{i+1}", g2))
        return out

    FINGER_GAP = 0.024      # max jaw opening (see gp7.urdf gripper)

    def _filter_grasp_width(self, G):
        """Drop grasps whose closing width exceeds the jaw opening. The
        closing direction is the flange y-axis expressed in the part frame
        (column 1 of T_part_flangeB); the part extent along it must fit.
        Without this filter, physically impossible grasps pass the kinematic
        gates because part-finger contact is whitelisted at the co-grasp."""
        mesh = self.s.cfg.get("part_mesh")
        if mesh:
            from scene import _stl_bbox, ROOT as _R
            lo, hi = _stl_bbox(os.path.join(_R, mesh))
            half = (hi - lo) / 2.0
        else:
            half = np.asarray(self.s.cfg["part_half_extents"], dtype=float)
        out = []
        for name, g in G:
            close_dir = np.abs(g[:3, 1])
            width = 2.0 * float(half @ close_dir)   # extent along closing dir
            if width < self.FINGER_GAP - 0.002:
                out.append((name, g))
        if not out:
            raise ValueError("no grasp in grasp_set_G fits the finger gap "
                             "for this part — add narrower-closing grasps")
        return out

    def _config_ok(self, robot, q) -> bool:
        """Limits with margin + singularity clearance (gates G3)."""
        return (kin.within_limits(robot, q, self.m_jl)
                and kin.manipulability(robot, q) >= self.w_min[robot.body])

    def _backoff(self, robot, q, dist):
        """Branch-continuous config `dist` behind q along the tool axis
        (flange -x). Seeded from q so the same IK branch is kept."""
        T = kin.fk(robot, q)
        T_pre = T @ np.array([[1, 0, 0, -dist], [0, 1, 0, 0],
                              [0, 0, 1, 0], [0, 0, 0, 1.0]], dtype=float)
        sols = kin.ik_solutions(robot, T_pre, seed=q, restarts=0, max_solutions=1)
        if not sols or not self._config_ok(robot, sols[0]):
            return None
        return sols[0]

    def _sweep_ok(self, robot, q_from, q_to, other_robot_q, holder,
                  T_fp, finger_ok) -> list | None:
        """Interpolated collision/limit/singularity checks along one robot's
        joint move, with the other robot held fixed. Returns the checked
        intermediate configs (or None)."""
        qA = q_to if robot.name == "A" else None
        qB = q_to if robot.name == "B" else None
        other = self.s.robotB if robot.name == "A" else self.s.robotA
        other.set_q(other_robot_q)
        ways = []
        for t in np.linspace(0.0, 1.0, self.n_sweep + 2)[1:-1]:
            q = (1 - t) * np.asarray(q_from) + t * np.asarray(q_to)
            if not self._config_ok(robot, q):
                return None
            free, _ = self.c.check_state(q if robot.name == "A" else None,
                                         q if robot.name == "B" else None,
                                         holder=holder, T_flange_part=T_fp,
                                         finger_ok=finger_ok)
            if not free:
                return None
            ways.append(q)
        return ways

    def _reachable(self, base: np.ndarray, T_flange) -> bool:
        """Cheap reach-sphere prefilter — skips hopeless IK calls."""
        d = float(np.linalg.norm(np.asarray(T_flange)[:3, 3] - base))
        return self.reach_min < d < self.reach_max

    # ---------- gate 3 (offline, per grasp only) ----------

    def filter_downstream(self) -> list[tuple[str, np.ndarray, dict]]:
        """G* = grasps B can use to insert: IK at the insert pose AND at the
        pre-insert hover (same branch), limits + singularity + collision at
        both, swept descent between them. Returns [(name, g, ins), ...] with
        ins = {qB_insert, qB_preinsert, descent}."""
        out = []
        for name, g in self.G:
            T_fB_part = kin.inv_T(g)
            for q_ins in kin.ik_solutions(self.s.robotB, self.X_ins @ g,
                                          seed=self.home_qB,
                                          restarts=self.restarts, rng=self.rng):
                if not self._config_ok(self.s.robotB, q_ins):
                    continue
                free, _ = self.c.check_state(self.home_qA, q_ins, holder="B",
                                             T_flange_part=T_fB_part)
                if not free:
                    continue
                # pre-insert hover, branch-continuous with the insert config
                sols = kin.ik_solutions(self.s.robotB, self.X_app @ g,
                                        seed=q_ins, restarts=0, max_solutions=1)
                if not sols or not self._config_ok(self.s.robotB, sols[0]):
                    continue
                q_pre = sols[0]
                free, _ = self.c.check_state(self.home_qA, q_pre, holder="B",
                                             T_flange_part=T_fB_part)
                if not free:
                    continue
                descent = self._sweep_ok(self.s.robotB, q_pre, q_ins,
                                         self.home_qA, "B", T_fB_part, None)
                if descent is None:
                    continue
                out.append((name, g, {"qB_insert": q_ins, "qB_preinsert": q_pre,
                                      "descent": descent}))
                break
        return out

    # ---------- gates 1-2 (online, per candidate) ----------

    def gate_A_presents(self, X_h) -> list[dict]:
        """A's branches presenting the part at X_h, each with a checked
        approach: pre-present config (backed off along A's tool axis, part in
        hand) and swept collision checks pre -> present. B parked home."""
        T_flange = X_h @ self.T_part_fA
        if not self._reachable(self.baseA, T_flange):
            return []
        good = []
        for q in kin.ik_solutions(self.s.robotA, T_flange, seed=self.home_qA,
                                  restarts=self.restarts, rng=self.rng):
            if not self._config_ok(self.s.robotA, q):
                continue
            free, _ = self.c.check_state(q, self.home_qB, holder="A",
                                         T_flange_part=self.T_fA_part)
            if not free:
                continue
            q_pre = self._backoff(self.s.robotA, q, self.d_pre)
            if q_pre is None:
                continue
            free, _ = self.c.check_state(q_pre, self.home_qB, holder="A",
                                         T_flange_part=self.T_fA_part)
            if not free:
                continue
            app = self._sweep_ok(self.s.robotA, q_pre, q, self.home_qB,
                                 "A", self.T_fA_part, None)
            if app is None:
                continue
            good.append({"qA": q, "qA_pre": q_pre, "A_approach": [q_pre] + app + [q]})
        return good

    def gate_B_takes(self, X_h, g, qA) -> dict | None:
        """B takes at the co-grasp instant: grasp config, pre-grasp back-off
        along B's tool axis, and swept approach — all with A at qA holding the
        part (both grippers may touch it during the final approach)."""
        T_flange = X_h @ g
        if not self._reachable(self.baseB, T_flange):
            return None
        for q in kin.ik_solutions(self.s.robotB, T_flange, seed=self.home_qB,
                                  restarts=self.restarts, rng=self.rng):
            if not self._config_ok(self.s.robotB, q):
                continue
            free, _ = self.c.check_state(qA, q, holder="A",
                                         T_flange_part=self.T_fA_part,
                                         finger_ok=("A", "B"))
            if not free:
                continue
            q_pre = self._backoff(self.s.robotB, q, self.d_pre)
            if q_pre is None:
                continue
            free, _ = self.c.check_state(qA, q_pre, holder="A",
                                         T_flange_part=self.T_fA_part,
                                         finger_ok=("A", "B"))
            if not free:
                continue
            app = self._sweep_ok(self.s.robotB, q_pre, q, qA, "A",
                                 self.T_fA_part, ("A", "B"))
            if app is None:
                continue
            return {"qB": q, "qB_pre": q_pre, "B_approach": [q_pre] + app + [q]}
        return None

    def retreat_A(self, qA, qB_grasp, g) -> dict | None:
        """A backs off along its tool -x after releasing (part now in B's
        hand, B holding at the handoff pose). Doc §5 step 5."""
        q_out = self._backoff(self.s.robotA, qA, self.d_retreat)
        if q_out is None:
            return None
        T_fB_part = kin.inv_T(g)
        self.s.robotB.set_q(qB_grasp)
        free, _ = self.c.check_state(q_out, qB_grasp, holder="B",
                                     T_flange_part=T_fB_part, finger_ok=("A", "B"))
        if not free:
            return None
        ways = self._sweep_ok(self.s.robotA, qA, q_out, qB_grasp, "B",
                              T_fB_part, ("A", "B"))
        if ways is None:
            return None
        return {"qA_out": q_out, "A_retreat": [np.asarray(qA)] + ways + [q_out]}

    def path_to_insert(self, qB_grasp, qB_preinsert, g) -> list | None:
        """Joint-interpolated waypoints handoff -> PRE-insert, each within
        limits/singularity and collision-free with the part in B's hand and
        A back home. The final descent was verified in filter_downstream."""
        T_fB_part = kin.inv_T(g)
        ways = []
        for t in np.linspace(0.0, 1.0, self.n_way + 2)[1:-1]:
            q = (1 - t) * np.asarray(qB_grasp) + t * np.asarray(qB_preinsert)
            if not self._config_ok(self.s.robotB, q):
                return None
            free, _ = self.c.check_state(self.home_qA, q, holder="B",
                                         T_flange_part=T_fB_part)
            if not free:
                return None
            ways.append(q)
        return ways

    # ---------- candidate oracle ----------

    def _try_pair(self, X_h, A_cand: dict, name, g, ins: dict,
                  stats: Counter) -> HandoffPlan | None:
        """Gates 2 + retreat + transit for one (A branch, grasp) pair."""
        qA = A_cand["qA"]
        B_cand = self.gate_B_takes(X_h, g, qA)
        if B_cand is None:
            stats["gate2_B_takes"] += 1
            return None
        ret = self.retreat_A(qA, B_cand["qB"], g)
        if ret is None:
            stats["gate2b_A_retreat"] += 1
            return None
        ways = self.path_to_insert(B_cand["qB"], ins["qB_preinsert"], g)
        if ways is None:
            stats["gate3_path_to_insert"] += 1
            return None
        stats["ok"] += 1
        plan = HandoffPlan(
            X_h=np.asarray(X_h, dtype=float), grasp_name=name, g=g,
            qA=qA, qB_grasp=B_cand["qB"], qB_insert=ins["qB_insert"],
            waypoints=ways, qA_pre=A_cand["qA_pre"], qB_pre=B_cand["qB_pre"],
            qA_out=ret["qA_out"], qB_preinsert=ins["qB_preinsert"],
            segments={
                "A_approach": A_cand["A_approach"],
                "B_approach": B_cand["B_approach"],
                "A_retreat": ret["A_retreat"],
                "B_to_preinsert": [B_cand["qB"]] + ways + [ins["qB_preinsert"]],
                "B_insert_approach": [ins["qB_preinsert"]] + ins["descent"]
                                     + [ins["qB_insert"]],
            })
        plan.score = self._score(plan)
        plan.exec_time = self.estimate_exec_time(plan)
        return plan

    def check_candidate(self, X_h, name, g, ins,
                        stats: Counter | None = None) -> HandoffPlan | None:
        """ins: the dict from filter_downstream (or a bare qB_insert array —
        it will be upgraded through the insert gates)."""
        stats = stats if stats is not None else Counter()
        if not isinstance(ins, dict):
            match = [e for n, _, e in self.filter_downstream() if n == name]
            if not match:
                stats["gate3_insert"] += 1
                return None
            ins = match[0]
        A_cands = self.gate_A_presents(X_h)
        if not A_cands:
            stats["gate1_A_presents"] += 1
            return None
        for A_cand in A_cands:
            plan = self._try_pair(X_h, A_cand, name, g, ins, stats)
            if plan is not None:
                return plan
        return None

    def estimate_exec_time(self, plan: HandoffPlan) -> float:
        """Physical handoff duration estimate (s): arms run to their pre
        poses concurrently, sequential final approaches (B closes only after
        A is stable), co-grasp dwell, A retreat, B transit + insert descent."""
        qA_pre = plan.qA_pre if plan.qA_pre is not None else plan.qA
        qB_pre = plan.qB_pre if plan.qB_pre is not None else plan.qB_grasp
        t = max(phase_time(self.home_qA, qA_pre) + phase_time(qA_pre, plan.qA),
                phase_time(self.home_qB, qB_pre))
        t += phase_time(qB_pre, plan.qB_grasp)          # B's final approach
        t += COGRASP_DWELL
        qA_out = plan.qA_out if plan.qA_out is not None else self.home_qA
        t += phase_time(plan.qA, qA_out)                # A clears
        q_pre_ins = (plan.qB_preinsert if plan.qB_preinsert is not None
                     else plan.qB_insert)
        qs = [plan.qB_grasp] + list(plan.waypoints) + [q_pre_ins, plan.qB_insert]
        t += sum(phase_time(a, b) for a, b in zip(qs[:-1], qs[1:]))
        return float(t)

    def _score(self, plan: HandoffPlan) -> float:
        """Margin score: worst normalized joint-limit margin across the chain
        + clearance at the co-grasp instant (doc §4.2, trimmed)."""
        jl = min(kin.limit_margin(self.s.robotA, plan.qA),
                 kin.limit_margin(self.s.robotB, plan.qB_grasp),
                 kin.limit_margin(self.s.robotB, plan.qB_insert))
        self.c.check_state(plan.qA, plan.qB_grasp, holder="A",
                           T_flange_part=self.T_fA_part, finger_ok=("A", "B"))
        clear = self.c.min_clearance(holder="A")
        return jl + 0.3 * (clear / 0.05)

    # ---------- pose grid ----------

    def pose_grid(self) -> list[np.ndarray]:
        sp = self.sp
        axes = []
        for k in ("x", "y", "z"):
            lo, hi, step = sp[k]
            axes.append(np.arange(lo, hi + 1e-9, step))
        Rs = [_rpy_deg_to_R(rpy) for rpy in sp["orientations_rpy_deg"]]
        center = np.array([a.mean() for a in axes])
        poses = []
        for x in axes[0]:
            for y in axes[1]:
                for z in axes[2]:
                    for i, R in enumerate(Rs):
                        X = np.eye(4)
                        X[:3, :3] = R
                        X[:3, 3] = (x, y, z)
                        # try central positions and plain orientations first
                        w = np.linalg.norm([x, y, z] - center) + 0.05 * i
                        poses.append((w, X))
        poses.sort(key=lambda t: t[0])
        return [X for _, X in poses]

    # ---------- search ----------

    def search(self, return_best: bool = False,
               time_budget: float | None = None,
               objective: str = "margin") -> SearchReport:
        """time_budget (s): optional cap; on expiry returns the current best
        (or infeasible) with stats['timeout'] set — useful for batch studies.
        objective (with return_best): 'margin' = safest plan,
        'time' = fastest estimated physical execution."""
        import time as _time
        deadline = (_time.time() + time_budget) if time_budget else None
        report = SearchReport(plan=None)
        g_star = self.filter_downstream()
        report.G_star = [n for n, _, _ in g_star]
        if not g_star:
            report.stats["gate3_insert_no_grasp"] += 1
            return report

        best = None
        for X_h in self.pose_grid():
            if deadline is not None and _time.time() > deadline:
                report.stats["timeout"] += 1
                break
            # gate 1 once per pose, shared across grasps
            A_cands = self.gate_A_presents(X_h)
            if not A_cands:
                report.stats["gate1_A_presents"] += len(g_star)
                report.n_candidates += len(g_star)
                continue
            for name, g, ins in g_star:
                report.n_candidates += 1
                plan = None
                for A_cand in A_cands:
                    plan = self._try_pair(X_h, A_cand, name, g, ins,
                                          report.stats)
                    if plan is not None:
                        break
                if plan is None:
                    continue
                if not return_best:
                    report.plan = plan
                    return report
                if best is None or self._better(plan, best, objective):
                    best = plan
        report.plan = best
        return report

    @staticmethod
    def _better(a: HandoffPlan, b: HandoffPlan, objective: str) -> bool:
        if objective == "time":
            return a.exec_time < b.exec_time
        return a.score > b.score

    # ---------- fast online pipeline ----------

    def plan_fast(self, cache: PlanCache | None = None, regrasp_planner=None,
                  warm_k: int = 6, search_budget: float = 1.5):
        """Production-latency planning:
          1. verify the nearest cached plans (~0.1 s each)
          2. budgeted grid search (search_budget s)
          3. regrasp branch (uses its own precomputed table)
        Returns (kind, plan, timings): kind in {'cache','search','regrasp',None}.
        New search successes are appended to the cache."""
        cache = cache or PlanCache()
        timings = {}
        t0 = time.time()
        if not hasattr(self, "_g_star_memo"):
            # G* depends only on the config (insert pose + grasp set), never
            # on the initial grasp — compute once per planner instance
            self._g_star_memo = {n: (g, ins) for n, g, ins in self.filter_downstream()}
        g_star = self._g_star_memo
        timings["g_star"] = time.time() - t0

        t0 = time.time()
        for e in cache.nearest(self.T_fA_part, warm_k):
            name = e["grasp_name"]
            if name not in g_star:
                continue
            g, ins = g_star[name]
            plan = self.check_candidate(np.asarray(e["X_h"]), name, g, ins)
            if plan is not None:
                timings["cache"] = time.time() - t0
                return "cache", plan, timings
        timings["cache"] = time.time() - t0

        t0 = time.time()
        rep = self.search(time_budget=search_budget)
        timings["search"] = time.time() - t0
        if rep.feasible:
            cache.add(self.T_fA_part, rep.plan)
            return "search", rep.plan, timings

        if regrasp_planner is not None:
            t0 = time.time()
            rplan = regrasp_planner.find_regrasp(self.T_fA_part)
            timings["regrasp"] = time.time() - t0
            if rplan is not None:
                return "regrasp", rplan, timings
        return None, None, timings
