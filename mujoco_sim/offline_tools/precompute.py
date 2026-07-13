"""Production offline pass for low-latency handoff decisions."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import time

from .artifacts import PrecomputeContext
from ..planner.planner import HandoffPlanner
from .qualification import physical_prerequisites
from ..simulation.workcell import WorkcellSim


def _timed(callable_):
    started = time.perf_counter()
    value = callable_()
    return value, time.perf_counter() - started


def precompute_runtime(context: PrecomputeContext) -> dict:
    """Build the artifacts used on a production cycle.

    The generated scene must already have been rebuilt from the same project
    manifest. The pass materializes geometry grasps, COM-stable placements,
    downstream B feasibility, and the known-start direct/reorientation policy.
    All artifacts are content-addressed, so repeating this function is a fast
    integrity-checked cache hit.
    """
    project_path = str(Path(context.project_path))
    model_path = None if context.model_path is None else str(context.model_path)
    sim, scene_s = _timed(lambda: WorkcellSim(
        project_path=project_path,
        **({} if model_path is None else {"model_path": model_path})))
    planner, planner_s = _timed(lambda: HandoffPlanner(
        sim, project_path=project_path, cache_dir=str(context.cache.root)))
    placements, placements_s = _timed(lambda: list(planner.stable_placements()))
    stats = Counter()
    downstream, downstream_s = _timed(lambda: planner.filter_downstream(stats))
    direct_result, policy_s = _timed(lambda: planner.search_direct(
        stats=stats, return_best=False))
    direct, candidates, _ = direct_result
    reorientation = None
    if direct is None:
        reorientation, reorientation_s = _timed(
            lambda: planner.search_regrasp(stats))
        policy_s += reorientation_s
    covered = direct is not None or reorientation is not None
    prerequisites = physical_prerequisites(planner.project)
    return {
        "artifact_counts": {
            "grasp_candidates": len(planner.g_B_candidates),
            "stable_placement_instances": len(placements),
            "downstream_receiver_grasps": len(downstream),
            "handoff_candidates_checked": candidates,
        },
        "known_start_policy": (
            "direct" if direct is not None else
            "reorientation" if reorientation is not None else "uncovered"),
        "known_start_coverage": {
            "covered": int(covered), "required": 1,
            "fraction": 1.0 if covered else 0.0,
        },
        "physical_certification": {
            "certified": bool(covered and all(prerequisites.values())),
            "prerequisites": prerequisites,
        },
        "timing_s": {
            "scene_load": scene_s,
            "planner_init": planner_s,
            "stable_placements": placements_s,
            "downstream": downstream_s,
            "online_policy_equivalent": policy_s,
        },
        "statistics": dict(stats),
    }


__all__ = ["precompute_runtime"]
