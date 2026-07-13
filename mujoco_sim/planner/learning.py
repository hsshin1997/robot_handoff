"""Optional learned proposal ordering with non-learned safety gates.

Learning is intentionally outside the validity boundary. A model may reorder
grasp/pose/motion proposals to reduce expected search time, but only candidates
that already passed deterministic geometry, IK, collision, and task gates can
be returned for execution.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np


@dataclass(frozen=True)
class RankedProposal:
    proposal_id: str
    features: np.ndarray
    hard_valid: bool
    deterministic_score: float

    def __post_init__(self):
        features = np.asarray(self.features, dtype=float)
        if features.ndim != 1 or not np.all(np.isfinite(features)):
            raise ValueError("proposal features must be a finite vector")
        if not np.isfinite(self.deterministic_score):
            raise ValueError("deterministic_score must be finite")
        object.__setattr__(self, "features", features.copy())


class SafetyGatedRanker:
    """Rank hard-valid proposals; learned scores are hints, never gates."""

    def __init__(self, predictor: Callable[[np.ndarray], np.ndarray] | None = None):
        self.predictor = predictor

    def rank(self, proposals: Iterable[RankedProposal]) -> list[RankedProposal]:
        valid = [proposal for proposal in proposals if proposal.hard_valid]
        if not valid:
            return []
        if self.predictor is None:
            learned = np.zeros(len(valid))
        else:
            matrix = np.vstack([proposal.features for proposal in valid])
            learned = np.asarray(self.predictor(matrix), dtype=float)
            if learned.shape != (len(valid),) or not np.all(np.isfinite(learned)):
                raise ValueError("predictor must return one finite score per proposal")
        order = sorted(range(len(valid)), key=lambda index: (
            -float(learned[index]),
            -float(valid[index].deterministic_score),
            valid[index].proposal_id,
        ))
        return [valid[index] for index in order]


__all__ = ["RankedProposal", "SafetyGatedRanker"]
