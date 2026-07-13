"""Transactional execution state and result contracts."""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any

import numpy as np


class UnexpectedCollision(RuntimeError):
    def __init__(self, pair, penetration):
        super().__init__(
            f"unexpected collision {pair}, penetration={penetration:.6f} m")
        self.pair = pair
        self.penetration = float(penetration)


class PipelineState(enum.Enum):
    OWNED_BY_A = "owned_by_A"
    PLACED_FOR_REORIENTATION = "placed_for_reorientation"
    REORIENTED_REPICK = "reoriented_repick_by_A"
    A_AT_HANDOFF = "A_at_handoff"
    B_AT_PREHANDOFF = "B_at_prehandoff"
    B_CAPTURE_VERIFIED = "B_capture_verified_virtual"
    OWNED_BY_B = "owned_by_B"
    A_CLEAR = "A_clear"
    SCANNED = "scanned_virtual_exact"
    AT_PREINSERT = "at_preinsert"
    INSERTED = "inserted_virtual_geometry"
    COMPLETE = "complete"
    ABORTED = "aborted"


@dataclass
class ExecutionEvent:
    state: PipelineState
    timestamp_s: float
    detail: dict = field(default_factory=dict)
    estimated_robot_time_s: float = 0.0


@dataclass
class ExecutionResult:
    success: bool
    outcome: str
    events: list[ExecutionEvent]
    measured_g_B: np.ndarray | None = None
    limitations: tuple[str, ...] = (
        "virtual capture predicate: articulated gripper CAD unavailable",
        "atomic ideal-weld ownership transfer: no physical dual grasp",
        "virtual exact scanner measurement: sensor model unavailable",
        "geometric insertion target: PCB hole/contact model unavailable",
    )
    debug_run_dir: str | None = None
    debug_errors: tuple[str, ...] = ()
    # Deprecated alias retained for existing report consumers.
    estimated_cycle_time_s: float = 0.0
    executed_modeled_time_s: float = 0.0
    planned_modeled_makespan_s: float = 0.0
    timing_estimate_complete: bool = False
    unmodeled_operations: tuple[str, ...] = ()
    wall_elapsed_s: float = 0.0
    stage_timings: tuple[dict[str, Any], ...] = ()
    profile_spans: tuple[dict[str, Any], ...] = ()
    operation_schedule: dict[str, Any] | None = None


__all__ = [
    "ExecutionEvent", "ExecutionResult", "PipelineState",
    "UnexpectedCollision",
]
