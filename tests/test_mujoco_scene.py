"""Acceptance checks for the clean MuJoCo scene foundation."""
from __future__ import annotations

import os
import sys

import mujoco
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
MODEL = os.path.join(ROOT, "mujoco_sim", "models", "scene.xml")


def scene():
    model = mujoco.MjModel.from_xml_path(MODEL)
    data = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "inspection")
    mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)
    return model, data


def test_scene_inventory():
    model, _ = scene()
    assert model.njnt == 13  # 12 robot joints + active part free joint
    assert model.nu == 12
    assert model.nkey == 1
    for name in ("workcell_visual_0", "workcell_visual_1", "workcell_visual_2", "workcell_visual_3",
                 "A_base", "B_base", "A_tool0", "B_tool0", "fixtures",
                 "A_tcp", "B_tcp", "supply_bin_left_center", "supply_bin_right_center",
                 "reorientation_origin", "pcb_origin", "part"):
        obj = mujoco.mjtObj.mjOBJ_GEOM if name.startswith("workcell_visual") else (
            mujoco.mjtObj.mjOBJ_SITE if (name.endswith("tcp") or name.endswith("center") or
                                        name.endswith("origin")) else mujoco.mjtObj.mjOBJ_BODY)
        assert mujoco.mj_name2id(model, obj, name) >= 0, name


def test_robot_mounts_match_workcell_cad():
    _, data = scene()
    assert np.allclose(data.body("A_base").xpos, [0.0, 0.0, 0.0], atol=1e-12)
    assert np.allclose(data.body("B_base").xpos, [0.850, 0.0, 0.0], atol=1e-12)
    # B base +X points toward A after the calibrated 180-degree yaw.
    assert np.allclose(data.body("B_base").xmat.reshape(3, 3)[:, 0], [-1, 0, 0], atol=1e-9)


def test_zero_pose_matches_urdf_fk():
    model = mujoco.MjModel.from_xml_path(MODEL)
    data = mujoco.MjData(model)
    data.qpos[:] = 0
    mujoco.mj_forward(model, data)
    # URDF chain: (0,0,.330)+(.040,0,0)+(0,0,.445)+(.440,0,.040)+(.080,0,0)
    assert np.allclose(data.body("A_tool0").xpos, [0.560, 0.0, 0.815], atol=1e-9)
    assert np.allclose(data.body("B_tool0").xpos, [0.290, 0.0, 0.815], atol=1e-9)
    # TCP is at the gripper CAD's maximum local Z: the finger-tip plane.
    tip = 0.23292807
    assert np.allclose(data.site("A_tcp").xpos, [0.560 + tip, 0.0, 0.815], atol=1e-9)
    assert np.allclose(data.site("B_tcp").xpos, [0.290 - tip, 0.0, 0.815], atol=1e-9)


def test_fixture_heights_and_staging_separation():
    model, data = scene()
    floor_z = -0.610
    table_z = floor_z + 0.930
    assert np.isclose(model.geom("supply_table_top").pos[2] + model.geom("supply_table_top").size[2], table_z)
    assert np.isclose(model.geom("pcb_table_top").pos[2] + model.geom("pcb_table_top").size[2], table_z)
    assert data.site("supply_bin_left_center").xpos[1] > 0
    assert data.site("reorientation_origin").xpos[1] > 0
    assert data.site("pcb_origin").xpos[1] < 0


def test_configured_part_is_loaded_and_grasped():
    from mujoco_sim.simulation.workcell import WorkcellSim
    sim = WorkcellSim()
    mesh_id = sim.model.mesh("active_part_mesh").id
    assert sim.model.mesh_facenum[mesh_id] == 5150
    tcp_pos, tcp_rot = sim.site_pose("A_tcp")
    part_pos, _ = sim.body_pose("part")
    expected_tcp_part = np.array([-0.0127075, 0.0052400, 0.0000550])
    assert np.allclose(part_pos, tcp_pos + tcp_rot @ expected_tcp_part, atol=1e-9)
    assert sim.data.eq_active[sim.model.equality("A_part_grasp").id] == 1
    assert sim.data.eq_active[sim.model.equality("B_part_grasp").id] == 0


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS  {test.__name__}")
    print(f"\n{len(tests)}/{len(tests)} passed")
