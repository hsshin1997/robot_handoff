"""The visual reorientation scenario must be planner-produced and checked."""
from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from mujoco_sim.visualize_reorientation_demo import build_demo  # noqa: E402


def test_demo_uses_verified_paths_to_an_insertion_feasible_handoff():
    _, planner, plan, bad_grasp = build_demo()
    downstream = planner.filter_downstream()
    assert not any(planner._gripper_compatibility(bad_grasp, item.grasp)[0]
                   for item in downstream)
    assert plan.direct.grasp_name_B in {item.grasp_name for item in downstream}
    assert len(plan.trajectories["A_to_place"]) > 2
    assert len(plan.trajectories["A_place_to_repick"]) > 2
    assert plan.X_place[2, 3] > 0.32


if __name__ == "__main__":
    test_demo_uses_verified_paths_to_an_insertion_feasible_handoff()
    print("PASS  test_demo_uses_verified_paths_to_an_insertion_feasible_handoff"
          "\n\n1/1 passed")
