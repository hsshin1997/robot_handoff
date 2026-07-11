"""Regrasp branch: when no direct handoff exists for how A is holding the
part, place it on the reorientation nest, re-pick with a better grasp, then
hand off (pipeline doc §6, one placement hop).

Key factorization (mirrors G* for insertion): whether a re-pick grasp g_A'
admits a direct handoff does NOT depend on the placement — it only depends
on g_A'. So the set of *handoff-viable canonical grasps* (with their full
handoff plans) is computed ONCE and cached to models/regrasp_table.json.
The online regrasp search then only has to find a (placement, yaw, g_A')
where A can place from its current grasp and re-pick with g_A'.

Placements: the part is boxy — its stable placements are the 6 axis-aligned
face-down orientations on the nest plate, free in yaw.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

import kin
from handoff import HandoffPlanner, HandoffPlan
from rl_env import GraspSampler
from scene import Scene, ROOT

TABLE_PATH = os.path.join(ROOT, "models", "regrasp_table.json")
REST_EPS = 0.004          # resting gap above the plate (keeps the collision
                          # checker's 2 mm clearance threshold honest)


def face_down_rotations() -> list[tuple[str, np.ndarray, int]]:
    """(name, R_l, down_axis_index): rotations putting each part axis down."""
    out = []
    ex, ey, ez = np.eye(3)
    for k, (axis, others) in enumerate(
            [(ex, (ey, ez)), (ey, (ez, ex)), (ez, (ex, ey))]):
        for s, tag in ((1.0, "+"), (-1.0, "-")):
            down = s * axis                       # part axis pointing down
            # rotation mapping `down` (part) -> -z (world)
            x_w = others[0]                       # arbitrary consistent frame
            z_w = -down
            y_w = np.cross(z_w, x_w)
            R = np.column_stack([x_w, y_w, z_w]).T   # world <- part
            out.append((f"{tag}{'xyz'[k]}_down", R, k))
    return out


@dataclass
class RegraspPlan:
    placement: str
    X_place: np.ndarray       # world part pose resting on the nest
    qA_place: np.ndarray      # A places the part (still holding with old grasp)
    qA_pick: np.ndarray       # A re-picks with the new grasp
    g_new: np.ndarray         # new T_flangeA_part after the re-pick
    handoff: HandoffPlan      # cached direct-handoff plan for g_new


class RegraspPlanner:
    def __init__(self, scene: Scene, planner: HandoffPlanner | None = None):
        self.s = scene
        self.pl = planner or HandoffPlanner(scene)
        self.c = self.pl.c
        self.sampler = GraspSampler(scene)
        self.placements = face_down_rotations()
        n = scene.cfg["nest"]
        self.nest_xy = np.asarray(n["center_xy"], dtype=float)
        self.nest_top = float(n["top_z"])
        self.yaws = np.radians([0, 45, 90, 135, 180, 225, 270, 315])
        # placement positions on the plate: center first, then near the four
        # edges — edge placements let a HORIZONTAL re-pick approach work,
        # with the palm hanging off the side of the plate.
        d = min(n["size"]) / 2.0 - 0.020
        self.offsets = [(0.0, 0.0), (d, 0.0), (-d, 0.0), (0.0, d), (0.0, -d)]

    # ---------- placement geometry ----------

    def placement_pose(self, R_l: np.ndarray, down_axis: int, yaw: float,
                       offset=(0.0, 0.0)) -> np.ndarray:
        cz, sz = np.cos(yaw), np.sin(yaw)
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        X = np.eye(4)
        X[:3, :3] = Rz @ R_l
        X[:3, 3] = [self.nest_xy[0] + offset[0], self.nest_xy[1] + offset[1],
                    self.nest_top + self.sampler.half[down_axis] + REST_EPS]
        return X

    # ---------- offline: handoff-viability table ----------

    TABLE_VERSION = 2      # bump when HandoffPlan serialization changes

    def build_table(self, path: str = TABLE_PATH, modes: list[int] | None = None,
                    verbose: bool = True, search_budget: float | None = None) -> dict:
        """For each canonical re-pick grasp: does a direct handoff exist?
        Cache the full plan. Resumable: existing entries are kept."""
        table = self.load_table(path) if os.path.exists(path) else {}
        for i in (modes if modes is not None else range(len(self.sampler.modes))):
            key = str(i)
            if key in table:
                continue
            g = self.sampler.canonical(i)
            self.pl.T_fA_part = g
            self.pl.T_part_fA = kin.inv_T(g)
            rep = self.pl.search(time_budget=search_budget)
            if rep.feasible:
                pn = rep.plan
                table[key] = {
                    "g": g.tolist(), "grasp_name": pn.grasp_name,
                    "X_h": pn.X_h.tolist(), "gB": pn.g.tolist(),
                    "qA": pn.qA.tolist(), "qB_grasp": pn.qB_grasp.tolist(),
                    "qB_insert": pn.qB_insert.tolist(),
                    "waypoints": [w.tolist() for w in pn.waypoints],
                    "qA_pre": pn.qA_pre.tolist(), "qB_pre": pn.qB_pre.tolist(),
                    "qA_out": pn.qA_out.tolist(),
                    "qB_preinsert": pn.qB_preinsert.tolist(),
                    "segments": {k: [np.asarray(q).tolist() for q in v]
                                 for k, v in pn.segments.items()},
                }
            else:
                table[key] = None
            if verbose:
                print(f"mode {i}: {'viable' if table[key] else 'not viable'}")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            json.dump({"version": self.TABLE_VERSION, "entries": table},
                      open(path, "w"))
        return table

    @classmethod
    def load_table(cls, path: str = TABLE_PATH) -> dict:
        data = json.load(open(path))
        if not isinstance(data, dict) or data.get("version") != cls.TABLE_VERSION:
            return {}          # stale format -> force rebuild
        return data["entries"]

    def _viable(self, table: dict):
        for key, e in table.items():
            if e is None:
                continue
            plan = HandoffPlan(
                X_h=np.array(e["X_h"]), grasp_name=e["grasp_name"],
                g=np.array(e["gB"]), qA=np.array(e["qA"]),
                qB_grasp=np.array(e["qB_grasp"]), qB_insert=np.array(e["qB_insert"]),
                waypoints=[np.array(w) for w in e["waypoints"]],
                qA_pre=np.array(e["qA_pre"]), qB_pre=np.array(e["qB_pre"]),
                qA_out=np.array(e["qA_out"]),
                qB_preinsert=np.array(e["qB_preinsert"]),
                segments={k: [np.array(q) for q in v]
                          for k, v in e["segments"].items()})
            yield int(key), np.array(e["g"]), plan

    def estimate_exec_time(self, plan: RegraspPlan) -> float:
        """Physical duration of the whole recovery: place, open, move to the
        re-pick, close, then the cached handoff (A starts at qA_pick)."""
        from handoff import phase_time, COGRASP_DWELL
        t = phase_time(self.pl.home_qA, plan.qA_place)
        t += 0.3                                        # open fingers
        t += phase_time(plan.qA_place, plan.qA_pick)
        t += 0.3                                        # close fingers
        h = plan.handoff
        t += max(phase_time(plan.qA_pick, h.qA),
                 phase_time(self.pl.home_qB, h.qB_grasp))
        t += COGRASP_DWELL
        t += phase_time(h.qA, self.pl.home_qA)
        qs = [h.qB_grasp] + list(h.waypoints) + [h.qB_insert]
        t += sum(phase_time(a, b) for a, b in zip(qs[:-1], qs[1:]))
        return float(t)

    # ---------- online: the regrasp search ----------

    def find_regrasp(self, T_fA_part_current: np.ndarray,
                     table: dict | None = None) -> RegraspPlan | None:
        """One placement hop: place from the current grasp, re-pick with a
        handoff-viable canonical grasp, hand off with the cached plan."""
        if table is None:
            table = (self.load_table() if os.path.exists(TABLE_PATH)
                     else self.build_table())
        cur = np.asarray(T_fA_part_current, dtype=float)
        viable = list(self._viable(table))
        if not viable:
            return None
        home_qB = self.pl.home_qB

        for pname, R_l, down in self.placements:
            for yaw in self.yaws:
                for off in self.offsets:
                    X_p = self.placement_pose(R_l, down, yaw, off)
                    # -- A places the part (holding with the current grasp)
                    T_fl = X_p @ kin.inv_T(cur)
                    if not self.pl._reachable(self.pl.baseA, T_fl):
                        continue
                    qA_place = None
                    for q in kin.ik_solutions(self.s.robotA, T_fl,
                                              seed=self.pl.home_qA,
                                              restarts=self.pl.restarts,
                                              rng=self.pl.rng):
                        if not self.pl._config_ok(self.s.robotA, q):
                            continue
                        free, _ = self.c.check_state(q, home_qB, holder="A",
                                                     T_flange_part=cur)
                        if free:
                            qA_place = q
                            break
                    if qA_place is None:
                        continue
                    # -- A re-picks with a viable grasp. Approaches from above
                    # or the side are allowed (side approaches need an edge
                    # placement so the palm clears the plate — the collision
                    # check is the arbiter); only from-below is excluded.
                    for mode_i, g_new, plan in viable:
                        a, _ = self.sampler.modes[mode_i]
                        if (X_p[:3, :3] @ a)[2] > 0.3:
                            continue      # fingers would come from below
                        T_pick = X_p @ kin.inv_T(g_new)
                        if not self.pl._reachable(self.pl.baseA, T_pick):
                            continue
                        for q in kin.ik_solutions(self.s.robotA, T_pick,
                                                  seed=self.pl.home_qA,
                                                  restarts=self.pl.restarts,
                                                  rng=self.pl.rng):
                            if not self.pl._config_ok(self.s.robotA, q):
                                continue
                            self.c.set_part_world(X_p)  # part rests on nest
                            free, _ = self.c.check_state(q, home_qB, holder=None,
                                                         finger_ok=("A",))
                            if free:
                                return RegraspPlan(placement=pname, X_place=X_p,
                                                   qA_place=qA_place, qA_pick=q,
                                                   g_new=g_new, handoff=plan)
        return None
