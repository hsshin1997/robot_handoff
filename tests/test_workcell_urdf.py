"""Structural and calibration tests for the generated full-workcell URDF."""
from __future__ import annotations

import copy
import math
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_workcell_urdf import (
    DEFAULT_CALIBRATION,
    DEFAULT_OUTPUT,
    build_workcell_urdf,
)


def _joint(root: ET.Element, name: str) -> ET.Element:
    value = root.find(f"joint[@name='{name}']")
    assert value is not None, name
    return value


def _origin_transform(joint: ET.Element) -> np.ndarray:
    origin = joint.find("origin")
    assert origin is not None
    xyz = np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ")
    roll, pitch, yaw = np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ")
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rotation = np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = xyz
    return result


def _rpy_transform(position, rpy) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    result = np.eye(4)
    result[:3, :3] = [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]
    result[:3, 3] = position
    return result


def _enabled_calibration_joint_count() -> int:
    with DEFAULT_CALIBRATION.open(encoding="utf-8") as stream:
        calibration = yaml.safe_load(stream)
    cameras = sum(
        2
        for item in calibration.get("cameras", [])
        if item.get("enabled", False) and item.get("calibrated", False)
    )
    targets = sum(
        1
        for item in calibration.get("calibration_targets", [])
        if item.get("enabled", False) and item.get("calibrated", False)
    )
    additional = sum(
        1
        for item in calibration.get("additional_frames", [])
        if item.get("enabled", True)
    )
    return cameras + targets + additional


def test_inventory_and_authoritative_transforms() -> None:
    root = ET.parse(DEFAULT_OUTPUT).getroot()
    links = {link.attrib["name"] for link in root.findall("link")}
    joints = root.findall("joint")
    assert root.attrib["name"] == "handoff_workcell"
    assert len([joint for joint in joints if joint.get("type") == "revolute"]) == 12
    assert {
        "world",
        "cell",
        "workcell_link",
        "fixtures_link",
        "A_base_link",
        "A_tool0",
        "A_gripper",
        "A_tcp",
        "B_base_link",
        "B_tool0",
        "B_gripper",
        "B_tcp",
        "reorientation_surface_frame",
        "pcb_frame",
        "scanner_region_frame",
        "pcb_slot_0_insertion_frame",
    } <= links
    with DEFAULT_CALIBRATION.open(encoding="utf-8") as stream:
        calibration = yaml.safe_load(stream)
    for camera in calibration.get("cameras", []):
        camera_link = f"camera_{camera['name']}_link"
        optical_link = f"camera_{camera['name']}_optical_frame"
        expected = camera.get("enabled", False) and camera.get("calibrated", False)
        assert (camera_link in links) is expected
        assert (optical_link in links) is expected

    workcell = root.find("link[@name='workcell_link']")
    fixtures = root.find("link[@name='fixtures_link']")
    assert workcell is not None and fixtures is not None
    assert len(workcell.findall("collision")) == 30
    assert len(fixtures.findall("collision")) == 33

    b_mount = _origin_transform(_joint(root, "cell_to_B_base_link"))
    assert np.allclose(b_mount[:3, 3], [0.850, 0.0, 0.0], atol=1e-12)
    assert np.allclose(b_mount[:3, :3], np.diag([-1.0, -1.0, 1.0]), atol=1e-11)
    for prefix in ("A", "B"):
        tcp = _origin_transform(_joint(root, f"{prefix}_gripper_to_tcp"))
        assert np.allclose(tcp[:3, 3], [0.0, 0.0, 0.23292807], atol=1e-12)


def test_meshes_resolve_from_generated_urdf() -> None:
    root = ET.parse(DEFAULT_OUTPUT).getroot()
    for mesh in root.findall(".//mesh"):
        path = Path(mesh.attrib["filename"])
        if not path.is_absolute():
            path = DEFAULT_OUTPUT.parent / path
        assert path.resolve().is_file(), path


def test_checked_in_urdf_is_current() -> None:
    with tempfile.NamedTemporaryFile(
        dir=DEFAULT_OUTPUT.parent, suffix=".urdf", delete=False
    ) as stream:
        candidate = Path(stream.name)
    try:
        build_workcell_urdf(output_path=candidate)
        assert candidate.read_bytes() == DEFAULT_OUTPUT.read_bytes()
    finally:
        candidate.unlink(missing_ok=True)


def test_calibrated_camera_matrix_round_trip() -> None:
    with DEFAULT_CALIBRATION.open(encoding="utf-8") as stream:
        calibration = yaml.safe_load(stream)
    calibration = copy.deepcopy(calibration)
    measured = _rpy_transform([0.41, -0.18, 1.37], [0.22, -0.31, 0.47])
    measured[:3, :3] = np.round(measured[:3, :3], 6)
    u, _, vt = np.linalg.svd(measured[:3, :3])
    expected = measured.copy()
    expected[:3, :3] = u @ vt
    calibration["cameras"] = [
        {
            "name": "overhead",
            "enabled": True,
            "calibrated": True,
            "mode": "eye_to_hand",
            "parent": "cell",
            "parent_T_camera_optical": {"matrix": measured.tolist()},
        }
    ]
    calibration["calibration_targets"] = []
    calibration["additional_frames"] = []

    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        calibration_path = directory / "calibration.yaml"
        output_path = directory / "workcell.urdf"
        calibration_path.write_text(yaml.safe_dump(calibration), encoding="utf-8")
        build_workcell_urdf(
            calibration_path=calibration_path,
            output_path=output_path,
        )
        root = ET.parse(output_path).getroot()
        parent_T_link = _origin_transform(
            _joint(root, "cell_to_camera_overhead_link")
        )
        link_T_optical = _origin_transform(
            _joint(
                root,
                "camera_overhead_link_to_camera_overhead_optical_frame",
            )
        )
        assert np.allclose(parent_T_link @ link_T_optical, expected, atol=1e-10)


def test_pybullet_loads_unified_tree() -> None:
    import pybullet as p

    connection = p.connect(p.DIRECT)
    try:
        flags = p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT
        body = p.loadURDF(str(DEFAULT_OUTPUT), useFixedBase=True, flags=flags)
        assert body >= 0
        info = [p.getJointInfo(body, index) for index in range(p.getNumJoints(body))]
        assert len(info) == 36 + _enabled_calibration_joint_count()
        assert sum(item[2] == p.JOINT_REVOLUTE for item in info) == 12
        link_index = {item[12].decode(): index for index, item in enumerate(info)}
        assert {"workcell_link", "A_tcp", "B_tcp", "pcb_frame"} <= link_index.keys()
        a_tcp = p.getLinkState(body, link_index["A_tcp"], computeForwardKinematics=True)[4]
        b_tcp = p.getLinkState(body, link_index["B_tcp"], computeForwardKinematics=True)[4]
        assert np.allclose(a_tcp, [0.79292807, 0.0, 0.815], atol=2e-7)
        assert np.allclose(b_tcp, [0.05707193, 0.0, 0.815], atol=2e-7)
    finally:
        p.disconnect(connection)


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
