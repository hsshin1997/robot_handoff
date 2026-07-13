"""Robot-trajectory timing models, separate from collision discretization.

Collision waypoints are safety samples, not controller commands.  Timing a
trajectory from the number of samples makes the estimate slower whenever a
checker is made more conservative.  This module instead integrates normalized
joint travel along the geometric path.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrajectoryTiming:
    duration_s: float
    joint_path_length_rad: float
    normalized_path_time_s: float
    waypoint_count: int
    timing_model: str


class JointVelocityTimingModel:
    """Cubic-smoothstep timing under per-joint velocity limits.

    This is a simulation timing model, not a hardware minimum-time certificate:
    acceleration, jerk, controller blending, payload derating, and settling
    parameters are not yet available.
    """

    VERSION = "piecewise_cubic_velocity_bound_v2"

    def __init__(self, velocity_limits_rad_s, *, peak_slope: float = 1.5):
        limits = np.asarray(velocity_limits_rad_s, dtype=float)
        if limits.ndim != 1 or not len(limits):
            raise ValueError("velocity limits must be a non-empty vector")
        if not np.all(np.isfinite(limits)) or np.any(limits <= 0.0):
            raise ValueError("velocity limits must be positive and finite")
        peak = float(peak_slope)
        if not np.isfinite(peak) or peak <= 0.0:
            raise ValueError("peak_slope must be positive and finite")
        self.velocity_limits = limits.copy()
        self.peak_slope = peak

    def _path(self, trajectory) -> np.ndarray:
        values = np.asarray(trajectory, dtype=float)
        expected = len(self.velocity_limits)
        if values.ndim != 2 or values.shape[1] != expected:
            raise ValueError(
                f"trajectory must have shape (N, {expected}), got {values.shape}")
        if len(values) == 0:
            raise ValueError("trajectory cannot be empty")
        if not np.all(np.isfinite(values)):
            raise ValueError("trajectory must contain only finite values")
        return values

    @staticmethod
    def _speed(value: float) -> float:
        speed = float(value)
        if not np.isfinite(speed) or speed <= 0.0 or speed > 1.0:
            raise ValueError("speed_fraction must lie in (0, 1]")
        return speed

    def edge_duration(
        self,
        q_from,
        q_to,
        speed_fraction: float,
        *,
        minimum_time_s: float = 0.0,
    ) -> float:
        path = self._path(np.vstack((q_from, q_to)))
        speed = self._speed(speed_fraction)
        minimum = float(minimum_time_s)
        if not np.isfinite(minimum) or minimum < 0.0:
            raise ValueError("minimum_time_s must be non-negative and finite")
        normalized = np.max(
            np.abs(path[1] - path[0]) / (self.velocity_limits * speed))
        return max(self.peak_slope * float(normalized), minimum)

    def analyze(self, trajectory, speed_fraction: float) -> TrajectoryTiming:
        path = self._path(trajectory)
        speed = self._speed(speed_fraction)
        if len(path) < 2:
            normalized = 0.0
            distance = 0.0
        else:
            delta = np.diff(path, axis=0)
            normalized = float(np.sum(np.max(
                np.abs(delta) / (self.velocity_limits * speed), axis=1)))
            distance = float(np.linalg.norm(delta, axis=1).sum())
        return TrajectoryTiming(
            duration_s=self.peak_slope * normalized,
            joint_path_length_rad=distance,
            normalized_path_time_s=normalized,
            waypoint_count=len(path),
            timing_model=self.VERSION,
        )


__all__ = [
    "JointVelocityTimingModel", "TrajectoryTiming",
]
