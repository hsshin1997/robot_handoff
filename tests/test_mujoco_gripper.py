"""Gripper articulation is asset-derived, never part-tuned."""
from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from mujoco_sim.gripper import inspect_gripper_model  # noqa: E402


def test_static_stl_is_not_misrepresented_as_articulated():
    result = inspect_gripper_model(os.path.join(
        ROOT, "assets", "gp7", "meshes", "gripper.STL"))
    assert not result.articulated
    assert result.actuation is None
    assert result.warnings


def test_mjcf_slide_limits_define_opening_without_part_parameters():
    xml = """<mujoco><worldbody><body>
      <body><joint name="left" type="slide" range="0.001 0.012"/>
        <geom name="left_pad" type="box" size=".001 .002 .003"/></body>
      <body><joint name="right" type="slide" range="0.001 0.012"/>
        <geom name="right_finger" type="box" size=".001 .002 .003"/></body>
    </body></worldbody></mujoco>"""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "gripper.xml"
        path.write_text(xml, encoding="utf-8")
        result = inspect_gripper_model(str(path))
    assert result.articulated
    assert result.contact_geometries == ("left_pad", "right_finger")
    assert np.allclose(result.actuation.aperture_range, (0.002, 0.024))
    positions = result.actuation.joint_positions(0.013)
    assert set(positions) == {"left", "right"}
    assert np.allclose(list(positions.values()), [0.0065, 0.0065])


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS  {test.__name__}")
    print(f"\n{len(tests)}/{len(tests)} passed")
