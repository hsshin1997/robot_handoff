"""Robot-B downstream certification from scanner through insertion.

The stage is intentionally cache-agnostic.  ``HandoffPlanner`` owns artifact
keys and feeds this certifier either during an offline build or a cold query.
"""
from __future__ import annotations

from collections import Counter
from typing import Protocol

import numpy as np

from ..planning_types import DownstreamWitness


class DownstreamRuntime(Protocol):
    """Narrow runtime port required by downstream certification."""

    g_B_candidates: list[tuple[str, np.ndarray]]
    q_start: dict[str, np.ndarray]
    cfg: dict
    limit_margin: float
    kin: object

    @property
    def X_scanner(self) -> np.ndarray: ...

    @property
    def insertion_poses(self) -> list[tuple[str, np.ndarray]]: ...

    def _preinsert_pose(self, X_insert: np.ndarray) -> np.ndarray: ...
    def _solutions(self, robot: str, target: np.ndarray, seed=None): ...
    def _config_ok(self, robot: str, q: np.ndarray) -> bool: ...
    def _correction_ok(self, grasp, q_insert, X_insert): ...
    def _held_path(self, *args, **kwargs): ...


class DownstreamCertifier:
    """Certify receiver grasps for every required downstream target."""

    def __init__(self, runtime: DownstreamRuntime, insertion_contact_pairs):
        self.runtime = runtime
        self.insertion_contact_pairs = tuple(insertion_contact_pairs)

    def certify(self, statistics: Counter | None = None) -> list[DownstreamWitness]:
        runtime = self.runtime
        stats = statistics if statistics is not None else Counter()
        output: list[DownstreamWitness] = []
        qA_park = runtime.q_start["A"]
        dither = np.radians(runtime.cfg["downstream"]["wrist_dither_deg"])

        for name, grasp in runtime.g_B_candidates:
            scanner_solutions = runtime._solutions(
                "B", runtime.X_scanner @ grasp)
            for scanner in scanner_solutions:
                if not runtime._config_ok("B", scanner.q):
                    continue
                q_pre: list[np.ndarray] = []
                q_insert: list[np.ndarray] = []
                corrections: list[list[np.ndarray]] = []
                trajectories: dict[str, list[np.ndarray]] = {}
                quality_values = [
                    runtime.kin.penalized_manipulability("B", scanner.q)]
                sigma_values: list[float] = []
                previous = scanner.q
                feasible = True

                for target_name, X_insert in runtime.insertion_poses:
                    X_pre = runtime._preinsert_pose(X_insert)
                    pre = runtime._solutions(
                        "B", X_pre @ grasp, seed=previous)
                    insert = runtime._solutions(
                        "B", X_insert @ grasp,
                        seed=pre[0].q if pre else previous)
                    if (not pre or not insert
                            or not runtime._config_ok("B", pre[0].q)
                            or not runtime._config_ok("B", insert[0].q)):
                        feasible = False
                        break
                    lower = runtime.kin.lower["B"][5]
                    upper = runtime.kin.upper["B"][5]
                    if not (lower + runtime.limit_margin + dither
                            <= insert[0].q[5]
                            <= upper - runtime.limit_margin - dither):
                        feasible = False
                        break
                    correction_ok, correction, sigma = runtime._correction_ok(
                        grasp, insert[0].q, X_insert)
                    if not correction_ok:
                        feasible = False
                        break
                    transit_ok, transit, _ = runtime._held_path(
                        "B", previous, pre[0].q, qA_park, grasp, ("B",),
                        stats)
                    insert_ok, descent, _ = runtime._held_path(
                        "B", pre[0].q, insert[0].q, qA_park, grasp, ("B",),
                        stats, self.insertion_contact_pairs)
                    if not transit_ok or not insert_ok:
                        feasible = False
                        break
                    q_pre.append(pre[0].q)
                    q_insert.append(insert[0].q)
                    corrections.append(correction)
                    trajectories[f"scanner_to_{target_name}_pre"] = transit
                    trajectories[f"{target_name}_insert"] = descent
                    previous = insert[0].q
                    quality_values.extend((
                        runtime.kin.penalized_manipulability("B", pre[0].q),
                        runtime.kin.penalized_manipulability("B", insert[0].q),
                    ))
                    sigma_values.append(sigma)
                if feasible:
                    output.append(DownstreamWitness(
                        name, grasp, scanner.q, q_pre, q_insert, corrections,
                        trajectories, min(quality_values), min(sigma_values)))
                    break
            if not any(item.grasp_name == name for item in output):
                stats["downstream_rejected"] += 1
        return output


__all__ = ["DownstreamCertifier", "DownstreamRuntime"]
