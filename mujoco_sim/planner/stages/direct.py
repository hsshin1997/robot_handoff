"""Search ordering for the direct handoff stage.

All robot-specific feasibility remains in the injected candidate evaluator.
This module owns only the deterministic warm-first/exhaustive search policy,
which makes ordering and candidate-count behavior cheap to unit test.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from contextlib import nullcontext
import itertools
from typing import Any

import numpy as np

from ...simulation.collision import CollisionPolicy
from ..types import DirectHandoffPlan, DownstreamWitness


class DirectCandidateEvaluator:
    """Apply all hard gates to one handoff-pose/receiver-grasp pair."""

    def __init__(self, runtime):
        self.runtime = runtime

    def evaluate(
        self, X_h, gA, downstream, statistics, *, fast=False,
        warm_only=False,
    ) -> DirectHandoffPlan | None:
        runtime = self.runtime
        gB = downstream.grasp
        compatible, gripper_clearance = runtime._gripper_compatibility(gA, gB)
        if not compatible:
            statistics["grasp_incompatible"] += 1
            return None
        target_A, target_B = X_h @ gA, X_h @ gB
        if (not runtime._reach_lookup("A", target_A)
                or not runtime._reach_lookup("B", target_B)):
            statistics["G1_reach"] += 1
            return None
        dpre = runtime.cfg["handoff_search"]["prehandoff_distance_m"]
        dret = runtime.cfg["handoff_search"]["retreat_distance_m"]
        best = None

        def evaluate_branches(A_solutions, B_solutions, stop_first):
            nonlocal best
            for A, B in itertools.product(A_solutions, B_solutions):
                if (not runtime._config_ok("A", A.q)
                        or not runtime._config_ok("B", B.q)):
                    statistics["G3_margin"] += 1
                    continue
                state = runtime.collision.check(
                    A.q, B.q, X_h, ("A", "B"))
                if not state.free:
                    statistics["G4_cograsp_collision"] += 1
                    continue
                A_pre = runtime._solutions(
                    "A", runtime._backoff_target(target_A, dpre), seed=A.q)
                B_pre = runtime._solutions(
                    "B", runtime._backoff_target(target_B, dpre), seed=B.q)
                A_out = runtime._solutions(
                    "A", runtime._backoff_target(target_A, dret), seed=A.q)
                if not A_pre or not B_pre or not A_out:
                    statistics["G6_approach_ik"] += 1
                    continue
                trajectories = {}
                ok, trajectories["A_current_to_pre"], _ = runtime._held_path(
                    "A", runtime.q_start["A"], A_pre[0].q,
                    runtime.q_start["B"], gA, ("A",), statistics)
                ok2, trajectories["A_approach"], _ = runtime._held_path(
                    "A", A_pre[0].q, A.q, runtime.q_start["B"], gA,
                    ("A",), statistics)
                ok3, trajectories["B_current_to_pre"], _ = runtime._fixed_path(
                    "B", runtime.q_start["B"], B_pre[0].q, A.q, X_h,
                    ("A",), statistics)
                ok4, trajectories["B_approach"], _ = runtime._fixed_path(
                    "B", B_pre[0].q, B.q, A.q, X_h,
                    ("A", "B"), statistics)
                ok5, trajectories["A_retreat"], _ = runtime._fixed_path(
                    "A", A.q, A_out[0].q, B.q, X_h,
                    ("A", "B"), statistics)
                ok6, trajectories["B_to_scanner"], _ = runtime._held_path(
                    "B", B.q, downstream.q_scanner, A_out[0].q, gB,
                    ("B",), statistics)
                # Downstream paths assume A at the declared insertion park.
                ok7, trajectories["A_scanner_clear_to_park"], _ = (
                    runtime._fixed_path(
                        "A", A_out[0].q, runtime.q_start["A"],
                        downstream.q_scanner, runtime.X_scanner,
                        ("B",), statistics))
                if not all((ok, ok2, ok3, ok4, ok5, ok6, ok7)):
                    statistics["G6_path"] += 1
                    continue

                # Path calls mutate shared MjData; restore this exact witness
                # before measuring simultaneous co-grasp clearance.
                co_grasp_state = runtime.collision.check(
                    A.q, B.q, X_h, ("A", "B"))
                if not co_grasp_state.free:
                    statistics["G4_cograsp_collision"] += 1
                    continue
                exact_clearance = runtime.collision.minimum_clearance(
                    policy=CollisionPolicy(part_holders=("A", "B")))
                required_clearance = (
                    runtime.cfg["gates"]["minimum_clearance_m"]
                    + runtime.cfg["gates"][
                        "calibration_translation_3sigma_m"])
                if exact_clearance < required_clearance:
                    statistics["G4_clearance"] += 1
                    continue
                clearance = min(gripper_clearance, exact_clearance)
                score = runtime._score(
                    X_h, A.q, B.q, downstream, clearance)
                plan = DirectHandoffPlan(
                    X_h, gA, downstream.grasp_name, gB, A.q, B.q,
                    A_pre[0].q, B_pre[0].q, A_out[0].q, downstream,
                    trajectories, score)
                if best is None or plan.score.total > best.score.total:
                    best = plan
                if stop_first:
                    return True
            return False

        if fast:
            warm_A = runtime._solutions(
                "A", target_A, seed=runtime.q_start["A"])
            warm_B = runtime._solutions(
                "B", target_B, seed=downstream.q_scanner)
            if (warm_A and warm_B
                    and evaluate_branches(warm_A, warm_B, True)):
                statistics["G2_warm_start_success"] += 1
                return best
            if warm_only:
                statistics["G2_warm_start_rejected"] += 1
                return None

        A_solutions = runtime._solutions("A", target_A)
        B_solutions = runtime._solutions("B", target_B)
        if not A_solutions or not B_solutions:
            statistics["G2_ik"] += 1
            return None
        evaluate_branches(A_solutions, B_solutions, fast)
        return best


class DirectHandoffSearch:
    """Deterministic direct-search orchestration with no MuJoCo dependency."""

    def __init__(
        self,
        pose_grid: Callable[[], Sequence[np.ndarray]],
        evaluate_candidate: Callable[..., DirectHandoffPlan | None],
        profile_span: Callable[[str], Any] | None = None,
    ):
        self._pose_grid = pose_grid
        self._evaluate_candidate = evaluate_candidate
        self._profile_span = profile_span or (lambda _name: nullcontext())

    def search(
        self,
        sender_grasp: np.ndarray,
        downstream: Iterable[DownstreamWitness],
        statistics,
        *,
        return_best: bool,
    ) -> tuple[DirectHandoffPlan | None, int]:
        """Search warm branches first, then retain a completeness fallback."""
        best: DirectHandoffPlan | None = None
        candidates = 0
        poses = tuple(self._pose_grid())
        witnesses = tuple(downstream)

        if not return_best:
            with self._profile_span("warm_candidate_grid"):
                for part_pose in poses:
                    for witness in witnesses:
                        candidates += 1
                        plan = self._evaluate_candidate(
                            part_pose, sender_grasp, witness, statistics,
                            fast=True, warm_only=True)
                        if plan is not None:
                            return plan, candidates

        with self._profile_span("exhaustive_candidate_grid"):
            for part_pose in poses:
                for witness in witnesses:
                    candidates += 1
                    plan = self._evaluate_candidate(
                        part_pose, sender_grasp, witness, statistics,
                        fast=False)
                    if plan is None:
                        continue
                    if not return_best:
                        return plan, candidates
                    if best is None or plan.score.total > best.score.total:
                        best = plan
        return best, candidates


__all__ = ["DirectCandidateEvaluator", "DirectHandoffSearch"]
