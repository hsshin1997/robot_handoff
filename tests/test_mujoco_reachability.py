"""Fast indexed reachability-map queries."""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from mujoco_sim.reachability import ReachabilityMap  # noqa: E402


def test_indexed_query_and_round_trip():
    mapping = ReachabilityMap(
        0.1,
        np.array([[1, 2, 3, 4], [2, 2, 3, 4], [9, 9, 9, 1]]),
        np.array([0.4, 0.8, 0.2]),
    )
    T = np.eye(4)
    T[:3, 3] = [0.11, 0.21, 0.31]
    T[:3, 2] = [0.0, 0.0, 1.0]  # direction bin 4
    assert mapping.query(T, neighborhood=0) == 0.4
    assert mapping.query(T, neighborhood=1) == 0.8
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "map.npz")
        mapping.save(path)
        assert ReachabilityMap.load(path).query(T, neighborhood=1) == 0.8


if __name__ == "__main__":
    test_indexed_query_and_round_trip()
    print("PASS  test_indexed_query_and_round_trip\n\n1/1 passed")
