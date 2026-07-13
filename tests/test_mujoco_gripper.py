"""Gripper articulation is asset-derived, never part-tuned."""
from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from mujoco_sim.gripper import (  # noqa: E402
    bind_gripper_scene,
    command_aperture,
    inspect_gripper_model,
    load_gripper_asset_contract,
)


@contextmanager
def _raises(exception_type, message: str):
    try:
        yield
    except exception_type as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError(f"expected {exception_type.__name__}: {message}")


ARTICULATED_MJCF = """<mujoco model="source_gripper">
  <worldbody>
    <body name="mount">
      <geom name="palm_collision" type="box" size=".02 .02 .01"/>
      <geom name="palm_visual" type="box" size=".02 .02 .01" contype="0" conaffinity="0"/>
      <body name="left_finger">
        <joint name="left_slide" type="slide" axis="0 1 0" range="0.001 0.012"/>
        <geom name="left_pad_collision" type="box" size=".002 .001 .01"/>
        <geom name="left_finger_visual" type="box" size=".002 .001 .01" contype="0" conaffinity="0"/>
      </body>
      <body name="right_finger">
        <joint name="right_slide" type="slide" axis="0 -1 0" range="0.001 0.012"/>
        <geom name="right_pad_collision" type="box" size=".002 .001 .01"/>
        <geom name="right_finger_visual" type="box" size=".002 .001 .01" contype="0" conaffinity="0"/>
      </body>
      <site name="tcp" pos="0 0 .04"/>
    </body>
  </worldbody>
</mujoco>"""


ARTICULATED_URDF = """<robot name="source_gripper">
  <link name="mount">
    <collision name="palm_collision"><geometry><box size=".04 .04 .02"/></geometry></collision>
    <visual name="palm_visual"><geometry><box size=".04 .04 .02"/></geometry></visual>
  </link>
  <link name="left_finger">
    <collision name="left_pad_collision"><geometry><box size=".004 .002 .02"/></geometry></collision>
    <visual name="left_finger_visual"><geometry><box size=".004 .002 .02"/></geometry></visual>
  </link>
  <joint name="left_slide" type="prismatic"><parent link="mount"/><child link="left_finger"/>
    <axis xyz="0 1 0"/><limit lower=".001" upper=".012" effort="10" velocity=".1"/></joint>
  <link name="right_finger">
    <collision name="right_pad_collision"><geometry><box size=".004 .002 .02"/></geometry></collision>
    <visual name="right_finger_visual"><geometry><box size=".004 .002 .02"/></geometry></visual>
  </link>
  <joint name="right_slide" type="prismatic"><parent link="mount"/><child link="right_finger"/>
    <axis xyz="0 -1 0"/><limit lower=".001" upper=".012" effort="10" velocity=".1"/></joint>
  <link name="tcp"/>
  <joint name="mount_tcp" type="fixed"><parent link="mount"/><child link="tcp"/>
    <origin xyz="0 0 .04" rpy="0 0 0"/></joint>
</robot>"""


def _descriptor(model_name="source.xml", *, right_pad="right_pad_collision"):
    return f"""schema_version: 1
model:
  path: {model_name}
  format: mjcf
frames:
  mount: mount
  tcp: tcp
  flange_to_mount:
    position_m: [0, 0, 0]
    rpy_deg: [0, 90, 0]
  mount_to_tcp:
    position_m: [0, 0, 0.04]
    rpy_deg: [0, 0, 0]
actuation:
  type: parallel_jaw
  closed_aperture_m: 0
  joints:
    - name: left_slide
      aperture_multiplier: 1
    - name: right_slide
      aperture_multiplier: 1
geometry:
  pad_collisions: [left_pad_collision, {right_pad}]
  collisions: [palm_collision, left_pad_collision, {right_pad}]
  visuals: [palm_visual, left_finger_visual, right_finger_visual]
scene_name_template: '{{robot}}_{{name}}'
"""


def _write_contract(directory: str, descriptor_text: str | None = None):
    root = Path(directory)
    (root / "source.xml").write_text(ARTICULATED_MJCF, encoding="utf-8")
    descriptor = root / "gripper.yaml"
    descriptor.write_text(descriptor_text or _descriptor(), encoding="utf-8")
    return descriptor


def _write_urdf_contract(directory: str):
    root = Path(directory)
    (root / "source.urdf").write_text(ARTICULATED_URDF, encoding="utf-8")
    descriptor = root / "gripper.yaml"
    descriptor.write_text(
        _descriptor(model_name="source.urdf").replace(
            "format: mjcf", "format: urdf"), encoding="utf-8")
    return descriptor


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


def test_articulated_asset_contract_validates_frames_surface_pads_and_limits():
    with tempfile.TemporaryDirectory() as directory:
        contract = load_gripper_asset_contract(str(_write_contract(directory)))
    assert contract.model_format == "mjcf"
    assert contract.mount_frame == "mount"
    assert contract.tcp_frame == "tcp"
    assert contract.pad_geometries == (
        "left_pad_collision", "right_pad_collision")
    assert np.allclose(contract.actuation.aperture_range, [0.002, 0.024])
    assert np.allclose(contract.T_F_E[:3, 3], [0.04, 0.0, 0.0], atol=1e-12)
    assert contract.scene_name("B", "right_slide") == "B_right_slide"


def test_urdf_contract_validates_named_collision_surfaces_and_link_subtrees():
    with tempfile.TemporaryDirectory() as directory:
        contract = load_gripper_asset_contract(
            str(_write_urdf_contract(directory)))
    assert contract.model_format == "urdf"
    assert contract.pad_geometries == (
        "left_pad_collision", "right_pad_collision")
    assert np.allclose(contract.actuation.aperture_range, [0.002, 0.024])


def test_contract_rejects_named_pad_that_is_not_an_actual_surface_geom():
    with tempfile.TemporaryDirectory() as directory:
        descriptor = _write_contract(
            directory, _descriptor(right_pad="imaginary_convex_hull_pad"))
        with _raises(ValueError, "absent from source model"):
            load_gripper_asset_contract(str(descriptor))


def test_contract_rejects_limits_that_contradict_source_model():
    text = _descriptor().replace(
        "- name: left_slide\n      aperture_multiplier: 1",
        "- name: left_slide\n      range_m: [0.0, 0.02]\n      aperture_multiplier: 1",
    )
    with tempfile.TemporaryDirectory() as directory:
        descriptor = _write_contract(directory, text)
        with _raises(ValueError, "contradicts source limits"):
            load_gripper_asset_contract(str(descriptor))


def test_compiled_scene_binding_is_namespaced_and_aperture_is_commandable():
    import mujoco

    compiled = """<mujoco><worldbody>
      <body name="A_mount">
        <geom name="A_palm_collision" type="box" size=".02 .02 .01"/>
        <geom name="A_palm_visual" type="box" size=".02 .02 .01" contype="0" conaffinity="0"/>
        <body><joint name="A_left_slide" type="slide" range=".001 .012"/>
          <geom name="A_left_pad_collision" type="box" size=".002 .001 .01"/>
          <geom name="A_left_finger_visual" type="box" size=".002 .001 .01" contype="0" conaffinity="0"/></body>
        <body><joint name="A_right_slide" type="slide" range=".001 .012"/>
          <geom name="A_right_pad_collision" type="box" size=".002 .001 .01"/>
          <geom name="A_right_finger_visual" type="box" size=".002 .001 .01" contype="0" conaffinity="0"/></body>
        <site name="A_tcp" pos="0 0 .04"/>
      </body>
    </worldbody></mujoco>"""
    with tempfile.TemporaryDirectory() as directory:
        contract = load_gripper_asset_contract(str(_write_contract(directory)))
        model = mujoco.MjModel.from_xml_string(compiled)
        data = mujoco.MjData(model)
        sim = type("Sim", (), {"model": model, "data": data})()
        binding = bind_gripper_scene(model, "A", contract)
        command_aperture(sim, "", binding.actuation, 0.013)
    assert binding.mount_body == "A_mount"
    assert binding.tcp_site == "A_tcp"
    assert set(binding.pad_geometries) == {
        "A_left_pad_collision", "A_right_pad_collision"}
    assert np.isclose(data.qpos[model.joint("A_left_slide").qposadr[0]], 0.0065)
    assert np.isclose(data.qpos[model.joint("A_right_slide").qposadr[0]], 0.0065)


def test_compiled_scene_binding_fails_closed_on_partial_import():
    import mujoco

    incomplete = """<mujoco><worldbody><body name="A_mount">
      <site name="A_tcp"/><geom name="A_palm_collision" type="box" size=".01 .01 .01"/>
    </body></worldbody></mujoco>"""
    with tempfile.TemporaryDirectory() as directory:
        contract = load_gripper_asset_contract(str(_write_contract(directory)))
        model = mujoco.MjModel.from_xml_string(incomplete)
        with _raises(RuntimeError, "A_left_slide"):
            bind_gripper_scene(model, "A", contract)


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS  {test.__name__}")
    print(f"\n{len(tests)}/{len(tests)} passed")
