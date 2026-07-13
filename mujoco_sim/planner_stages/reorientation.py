"""Backward task-graph search for flat-stage reorientation."""
from __future__ import annotations

from dataclasses import replace
from typing import Protocol

import numpy as np

from ..planning_types import RegraspPlan
from ..se3 import inverse
from ..task_graph import (DirectCoGraspEdge, InitialGraspClass,
                          PlacementGraspEdge, TaskGraph)


class ReorientationRuntime(Protocol):
    cfg: dict
    project: object
    g_B_candidates: list[tuple[str, np.ndarray]]
    g_A_start: np.ndarray
    q_start: dict[str, np.ndarray]

    def search_direct(self, *args, **kwargs): ...
    def stable_placement_witnesses(self): ...
    def _solutions(self, *args, **kwargs): ...
    def _config_ok(self, *args, **kwargs): ...
    def _held_path(self, *args, **kwargs): ...
    def _fixed_path(self, *args, **kwargs): ...


class ReorientationSearch:
    """Find a stable place/re-pick that reaches an insertion-valid handoff."""

    def __init__(
        self,
        runtime: ReorientationRuntime,
        support_contact_pairs,
        task_graph_type=TaskGraph,
    ):
        self.runtime = runtime
        self.support_contact_pairs = tuple(support_contact_pairs)
        self.task_graph_type = task_graph_type

    @staticmethod
    def _path_cost(path) -> float:
        values = np.asarray(path, float)
        return (0.0 if len(values) < 2 else
                float(np.linalg.norm(np.diff(values, axis=0), axis=1).sum()))

    def search(self, statistics) -> RegraspPlan | None:
        runtime = self.runtime
        if not runtime.cfg["regrasp"]["enabled"]:
            return None
        planning = runtime.project.solver["planning"]

        # Work backward only from grasps already certified through insertion.
        raw_goals = [("nominal", inverse(runtime.project.T_tcp_part_start))]
        raw_goals.extend(runtime.g_B_candidates[:int(
            planning["reorientation_goal_grasp_limit"])])
        goal_templates = []
        seen = []
        for goal_id, grasp in raw_goals:
            if any(np.allclose(grasp, old, atol=1e-9) for old in seen):
                continue
            seen.append(grasp)
            direct, _, _ = runtime.search_direct(
                grasp, statistics, return_best=False)
            if direct is not None:
                goal_templates.append((str(goal_id), grasp, direct))
            if len(goal_templates) >= int(
                    planning["reorientation_direct_goal_limit"]):
                break
        if not goal_templates:
            statistics["regrasp_no_insertion_valid_goal"] += 1
            return None

        current_id = "current"
        direct_edges = [DirectCoGraspEdge(
            goal_id, direct.grasp_name_B,
            cost=self._path_cost(direct.trajectories["A_approach"])
                 + self._path_cost(direct.trajectories["B_approach"]),
            robustness=max(1e-9, direct.score.clearance),
            edge_id=f"direct:{goal_id}:{direct.grasp_name_B}")
            for goal_id, _, direct in goal_templates]
        placement_edges = []
        options: dict[tuple[str, str], RegraspPlan] = {}
        for placement in runtime.stable_placement_witnesses():
            placement_name = placement.name
            X_place = placement.T_W_P
            place = runtime._solutions(
                "A", X_place @ runtime.g_A_start,
                seed=runtime.q_start["A"])
            if not place or not runtime._config_ok("A", place[0].q):
                continue
            place_ok, place_path, _ = runtime._held_path(
                "A", runtime.q_start["A"], place[0].q,
                runtime.q_start["B"], runtime.g_A_start, ("A",),
                statistics, self.support_contact_pairs)
            if not place_ok:
                continue
            placement_edges.append(PlacementGraspEdge(
                placement_name, current_id, self._path_cost(place_path),
                placement.robustness,
                edge_id=f"place:{placement_name}:current"))
            for goal_id, new_grasp, template in goal_templates:
                repick = runtime._solutions(
                    "A", X_place @ new_grasp, seed=place[0].q)
                if not repick or not runtime._config_ok("A", repick[0].q):
                    continue
                pick_ok, repick_path, _ = runtime._fixed_path(
                    "A", place[0].q, repick[0].q, runtime.q_start["B"],
                    X_place, ("A",), statistics,
                    self.support_contact_pairs)
                if not pick_ok:
                    continue
                goal_ok, repick_to_goal, _ = runtime._held_path(
                    "A", repick[0].q, template.qA_pre,
                    runtime.q_start["B"], new_grasp, ("A",), statistics,
                    self.support_contact_pairs)
                if not goal_ok:
                    continue
                trajectories = dict(template.trajectories)
                trajectories["A_current_to_pre"] = repick_to_goal
                direct = replace(template, trajectories=trajectories)
                placement_edges.append(PlacementGraspEdge(
                    placement_name, goal_id,
                    self._path_cost(repick_path)
                    + self._path_cost(repick_to_goal),
                    placement.robustness,
                    edge_id=f"pick:{placement_name}:{goal_id}"))
                options[(placement_name, goal_id)] = RegraspPlan(
                    placement_name, X_place, runtime.g_A_start, new_grasp,
                    place[0].q, repick[0].q, direct,
                    {"A_to_place": place_path,
                     "A_place_to_repick": repick_path})

        graph = self.task_graph_type(
            [InitialGraspClass.singleton(current_id)],
            sorted({edge.receiver_grasp for edge in direct_edges}),
            direct_edges,
            placement_edges,
        )
        discrete = graph.plan(
            current_id,
            max_reorientation_hops=int(planning["max_reorientation_hops"]))
        if discrete.success:
            placement_id = next(step.target for step in discrete.steps
                                if step.kind == "place")
            goal_id = next(step.target for step in discrete.steps
                           if step.kind == "regrasp")
            statistics["regrasp_graph_edges"] += (
                len(placement_edges) + len(direct_edges))
            return options[(placement_id, goal_id)]
        statistics["regrasp_failed"] += 1
        return None


__all__ = ["ReorientationRuntime", "ReorientationSearch"]
