"""SE(3) uncertainty utilities for the handoff capture-region gate.

Covariances use spatial/left perturbations and twist ordering ``(v, omega)``.
This resolves the right-vs-left perturbation ambiguity in the design document.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .se3 import adjoint


def propagate_left_covariance(deterministic_transform: np.ndarray,
                              covariance: np.ndarray) -> np.ndarray:
    covariance = np.asarray(covariance, dtype=float)
    if covariance.shape != (6, 6):
        raise ValueError("SE(3) covariance must be 6x6")
    Ad = adjoint(deterministic_transform)
    return Ad @ covariance @ Ad.T


def combine_independent(*covariances: np.ndarray) -> np.ndarray:
    if not covariances:
        return np.zeros((6, 6))
    matrices = [np.asarray(item, dtype=float) for item in covariances]
    if any(item.shape != (6, 6) for item in matrices):
        raise ValueError("all SE(3) covariances must be 6x6")
    return np.sum(matrices, axis=0)


@dataclass(frozen=True)
class CaptureRegionResult:
    accepted: bool
    translation_3sigma: np.ndarray
    rotation_3sigma: np.ndarray


def check_axis_aligned_capture(covariance: np.ndarray,
                               translation_half_width,
                               rotation_half_width) -> CaptureRegionResult:
    covariance = np.asarray(covariance, dtype=float)
    sigma = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    translation = 3.0 * sigma[:3]
    rotation = 3.0 * sigma[3:]
    accepted = (np.all(translation <= np.asarray(translation_half_width))
                and np.all(rotation <= np.asarray(rotation_half_width)))
    return CaptureRegionResult(bool(accepted), translation, rotation)
