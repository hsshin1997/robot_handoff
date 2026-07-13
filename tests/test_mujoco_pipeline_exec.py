"""Execution safety regressions independent of a particular cached plan."""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import mujoco
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from mujoco_sim.exec import PipelineExecutor, UnexpectedCollision  # noqa: E402
from mujoco_sim.collision import SceneCollisionChecker  # noqa: E402
from mujoco_sim.planning import HandoffPlanner  # noqa: E402
from mujoco_sim.sim import WorkcellSim  # noqa: E402


def executor():
    sim = WorkcellSim()
    planner = HandoffPlanner(sim)
    return sim, PipelineExecutor(sim, planner)


def guard_result(part_geom_name: str):
    model = mujoco.MjModel.from_xml_string(f"""
<mujoco model="guard-part-name-test">
  <worldbody>
    <body name="candidate_part">
      <freejoint name="candidate_part_free"/>
      <geom name="{part_geom_name}" type="sphere" size="0.01"/>
    </body>
    <body name="obstacle">
      <freejoint name="guard_obstacle_free"/>
      <geom name="guard_obstacle" type="sphere" size="0.01"/>
    </body>
  </worldbody>
</mujoco>
""")
    data = mujoco.MjData(model)
    sim = SimpleNamespace(model=model, data=data)
    collision = SceneCollisionChecker(sim, kinematics=None)
    run = SimpleNamespace(
        sim=sim,
        planner=SimpleNamespace(collision=collision),
    )
    return PipelineExecutor._guard_clear(run)


def test_continuous_monitor_allows_known_initial_held_state():
    sim, run = executor()
    run._move("A", sim.arm_qpos("A"), 0.1, minimum_time=0.003,
              allowed_part_holders=("A",))


def test_continuous_monitor_aborts_part_fixture_penetration():
    sim, run = executor()
    sim.release_part()
    run.owner = None
    run.owner_grasp = np.eye(4)
    X = np.eye(4)
    # Deliberately place the part through the surrounding PCB collision ring,
    # outside the bounded virtual insertion aperture.
    X[:3, 3] = [0.470, -0.455, 0.346]
    run.fixed_part_pose = X
    try:
        run._move("B", sim.arm_qpos("B"), 0.1, minimum_time=0.003,
                  allowed_part_holders=())
    except UnexpectedCollision as error:
        assert "part_collision" in error.pair
        assert any(name.startswith("pcb_board_") for name in error.pair)
        assert error.penetration > 0.0
    else:
        raise AssertionError("execution stepped through an unexpected collision")


def test_force_guard_recognizes_numbered_part_chunks_only():
    chunk_clear, chunk_contacts = guard_result("part_collision_12")
    lookalike_clear, lookalike_contacts = guard_result("part_collision_fixture")
    assert chunk_clear
    assert not chunk_contacts
    assert not lookalike_clear
    assert lookalike_contacts == [
        ("part_collision_fixture", "guard_obstacle")
    ]


def test_executor_requires_one_shared_simulation_state():
    first = WorkcellSim()
    second = WorkcellSim()
    planner = HandoffPlanner(second)
    try:
        PipelineExecutor(first, planner)
    except ValueError as error:
        assert "share one WorkcellSim" in str(error)
    else:
        raise AssertionError("executor accepted a planner from another MjData")


def test_executor_clock_starts_when_execution_starts_not_at_construction():
    sim, run = executor()
    assert run.started is None
    assert run._elapsed() == 0.0


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS  {test.__name__}")
    print(f"\n{len(tests)}/{len(tests)} passed")
