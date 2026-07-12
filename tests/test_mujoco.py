"""Compatibility smoke tests for the current MuJoCo stack.

The former file tested the discarded 16-actuator primitive-gripper scene.
Detailed coverage now lives in test_mujoco_scene/se3/pipeline/exec.py.
"""
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from mujoco_sim.exec import DynamicExecutor, PipelineExecutor
from mujoco_sim.sim import HandoffSim, WorkcellSim


def test_compatibility_aliases_target_current_stack():
    assert HandoffSim is WorkcellSim
    assert DynamicExecutor is PipelineExecutor


def test_model_loads_steps_and_holds_configured_part():
    sim = HandoffSim()
    assert sim.model.nu == 12
    initial = sim.part_pose().copy()
    sim.step(2)
    assert np.linalg.norm(sim.part_pose()[:3, 3] - initial[:3, 3]) < 0.002


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS  {test.__name__}")
    print(f"\n{len(tests)}/{len(tests)} passed")
