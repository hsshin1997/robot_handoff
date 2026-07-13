"""A learned ranker cannot bypass a deterministic safety rejection."""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from mujoco_sim.planner.learning import RankedProposal, SafetyGatedRanker  # noqa: E402


def test_invalid_high_score_is_never_returned():
    proposals = [
        RankedProposal("unsafe", [100.0], False, 100.0),
        RankedProposal("safe-low", [1.0], True, 0.2),
        RankedProposal("safe-high", [2.0], True, 0.1),
    ]
    ranked = SafetyGatedRanker(lambda features: features[:, 0]).rank(proposals)
    assert [item.proposal_id for item in ranked] == ["safe-high", "safe-low"]
    assert "unsafe" not in {item.proposal_id for item in ranked}


if __name__ == "__main__":
    test_invalid_high_score_is_never_returned()
    print("PASS  test_invalid_high_score_is_never_returned\n\n1/1 passed")
