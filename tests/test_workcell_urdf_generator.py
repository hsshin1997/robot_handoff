"""Tests for the vendor-neutral, YAML-driven workcell URDF generator.

Run directly::

    python tests/test_workcell_urdf_generator.py

The same functions are also collected by pytest.  Synthetic URDFs are used so
the generic behavior is tested independently of any one robot vendor or mesh
package.
"""
from __future__ import annotations

import copy
import math
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mujoco_sim.modeling.workcell_urdf_generator import generate_workcell_urdf


IDENTITY_POSE = {
    "position_m": [0.0, 0.0, 0.0],
    "rpy_deg": [0.0, 0.0, 0.0],
}


VENDOR_A_URDF = """\
<?xml version="1.0"?>
<robot name="vendor_a">
  <material name="metal"><color rgba="0.5 0.5 0.55 1"/></material>
  <link name="base">
    <visual name="base_visual">
      <geometry><box size="0.20 0.20 0.10"/></geometry>
      <material name="metal"/>
    </visual>
  </link>
  <link name="carriage">
    <collision><geometry><cylinder radius="0.04" length="0.30"/></geometry></collision>
  </link>
  <link name="tool"/>
  <joint name="turntable" type="revolute">
    <parent link="base"/><child link="carriage"/>
    <origin xyz="0 0 0.10" rpy="0 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-2" upper="2" effort="10" velocity="1"/>
  </joint>
  <joint name="tool_slide" type="prismatic">
    <parent link="carriage"/><child link="tool"/>
    <origin xyz="0 0 0.30" rpy="0 0 0"/><axis xyz="0 0 1"/>
    <limit lower="0" upper="0.15" effort="20" velocity="0.2"/>
  </joint>
</robot>
"""


VENDOR_B_URDF = """\
<?xml version="1.0"?>
<robot name="vendor_b">
  <link name="pedestal">
    <visual><geometry><sphere radius="0.08"/></geometry></visual>
  </link>
  <link name="wrist"/>
  <link name="flange"/>
  <joint name="spin" type="continuous">
    <parent link="pedestal"/><child link="wrist"/>
    <axis xyz="1 0 0"/><limit effort="5" velocity="2"/>
  </joint>
  <joint name="flange_fixed" type="fixed">
    <parent link="wrist"/><child link="flange"/>
    <origin xyz="0.25 0 0" rpy="0 0 0"/>
  </joint>
</robot>
"""


ARTICULATED_GRIPPER_URDF = """\
<?xml version="1.0"?>
<robot name="parallel_hand">
  <link name="mount">
    <visual><geometry><box size="0.08 0.06 0.04"/></geometry></visual>
  </link>
  <link name="left_finger"/>
  <link name="right_finger"/>
  <link name="tcp"/>
  <joint name="left_slide" type="prismatic">
    <parent link="mount"/><child link="left_finger"/>
    <axis xyz="0 1 0"/>
    <limit lower="0" upper="0.03" effort="8" velocity="0.1"/>
  </joint>
  <joint name="right_slide" type="prismatic">
    <parent link="mount"/><child link="right_finger"/>
    <axis xyz="0 -1 0"/>
    <limit lower="0" upper="0.03" effort="8" velocity="0.1"/>
    <mimic joint="left_slide" multiplier="-1" offset="0"/>
  </joint>
  <joint name="tcp_fixed" type="fixed">
    <parent link="mount"/><child link="tcp"/>
    <origin xyz="0 0 0.12" rpy="0 0 0"/>
  </joint>
  <transmission name="parallel_transmission">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="left_slide">
      <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
    </joint>
    <actuator name="finger_motor">
      <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>
</robot>
"""


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _write_yaml(path: Path, value: Any) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _pose(position: list[float], rpy_deg: list[float] | None = None) -> dict[str, Any]:
    return {
        "position_m": position,
        "rpy_deg": [0.0, 0.0, 0.0] if rpy_deg is None else rpy_deg,
    }


def _synthetic_manifest(directory: Path) -> tuple[Path, dict[str, Any]]:
    _write_text(directory / "vendor_a.urdf", VENDOR_A_URDF)
    _write_text(directory / "vendor_b.urdf", VENDOR_B_URDF)
    _write_text(directory / "parallel_hand.urdf", ARTICULATED_GRIPPER_URDF)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "name": "synthetic_cell",
        "root_link": "world",
        "output": {
            "urdf": "generated/cell.urdf",
            "camera_info_dir": "generated/camera_info",
            "report": "generated/cell.report.yaml",
            "mesh_uri_mode": "relative",
            "extension_policy": "reject",
        },
        "frames": [
            {
                "name": "cell",
                "parent": "world",
                "parent_T_child": copy.deepcopy(IDENTITY_POSE),
            }
        ],
        "static_bodies": [
            {
                "name": "table",
                "parent": {"frame": "cell"},
                "parent_T_body": _pose([0.4, 0.0, 0.35]),
                "visuals": [
                    {
                        "geometry": {"box": {"size_m": [1.2, 0.8, 0.05]}},
                        "material": {
                            "name": "table_gray",
                            "rgba": [0.4, 0.4, 0.4, 1.0],
                        },
                    }
                ],
                "collisions": [
                    {"geometry": {"box": {"size_m": [1.2, 0.8, 0.05]}}}
                ],
            }
        ],
        "robots": [
            {
                "name": "arm_a",
                "path": "vendor_a.urdf",
                "parent": {"frame": "cell"},
                "parent_T_root": _pose([-0.3, 0.0, 0.4]),
                "root_link": "base",
                "flange_link": "tool",
            },
            {
                "name": "arm_b",
                "path": "vendor_b.urdf",
                "parent": {"frame": "cell"},
                "parent_T_root": _pose([0.6, 0.0, 0.4], [0.0, 0.0, 180.0]),
                "root_link": "pedestal",
                "flange_link": "flange",
            },
        ],
        "grippers": [
            {
                "name": "hand",
                "type": "urdf",
                "path": "parallel_hand.urdf",
                "parent": {"robot": "arm_a", "link": "tool"},
                "parent_T_mount": copy.deepcopy(IDENTITY_POSE),
                "root_link": "mount",
                "tcp_link": "tcp",
            }
        ],
        "cameras": [
            {
                "name": "overhead",
                "enabled": True,
                "pose_status": "calibrated",
                "parent": {"frame": "cell"},
                "parent_T_camera_optical": _pose([0.4, 0.0, 1.4], [180.0, 0.0, 0.0]),
                "intrinsics": {
                    "image_width": 640,
                    "image_height": 480,
                    "fx": 320.0,
                    "fy": 240.0,
                    "cx": 320.0,
                    "cy": 240.0,
                    "skew": 0.0,
                    "distortion_model": "plumb_bob",
                    "distortion_coefficients": [0.1, -0.05, 0.001, -0.002, 0.01],
                },
                "operating_envelope": {
                    "working_distance_m": {"min": 0.6, "nominal": 1.0, "max": 1.4},
                    "view_depth_m": {"near": 0.2, "far": 1.8},
                    "depth_measurement_range_m": {"min": 0.3, "max": 2.0},
                    "focus": {"depth_of_field_m": {"near": 0.7, "far": 1.3}},
                },
            },
            {
                "name": "wrist",
                "enabled": True,
                "pose_status": "nominal",
                "parent": {"robot": "arm_b", "link": "flange"},
                "parent_T_camera_link": _pose([0.03, 0.0, 0.06]),
                "camera_link_T_optical": _pose([0.0, 0.0, 0.0], [-90.0, 0.0, -90.0]),
            },
        ],
        "attached_frames": [
            {
                "name": "inspection_target",
                "parent": {"body": "table"},
                "parent_T_child": _pose([0.0, 0.0, 0.05]),
            }
        ],
    }
    manifest_path = directory / "workcell.yaml"
    _write_yaml(manifest_path, manifest)
    return manifest_path, manifest


def _joint(root: ET.Element, name: str) -> ET.Element:
    joint = root.find(f"joint[@name='{name}']")
    assert joint is not None, name
    return joint


def _origin_transform(joint: ET.Element) -> np.ndarray:
    origin = joint.find("origin")
    assert origin is not None
    xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
    roll, pitch, yaw = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    result = np.eye(4)
    result[:3, :3] = [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]
    result[:3, 3] = xyz
    return result


def _assert_single_connected_tree(root: ET.Element) -> None:
    links = {link.get("name") for link in root.findall("link")}
    child_to_parent: dict[str, str] = {}
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        assert parent is not None and child is not None
        parent_name, child_name = parent.get("link"), child.get("link")
        assert parent_name in links
        assert child_name in links
        assert child_name not in child_to_parent
        child_to_parent[child_name] = parent_name
    roots = links - child_to_parent.keys()
    assert roots == {"world"}
    for link in links - roots:
        visited: set[str] = set()
        current = link
        while current in child_to_parent:
            assert current not in visited
            visited.add(current)
            current = child_to_parent[current]
        assert current == "world"


def test_heterogeneous_imports_and_namespaced_references() -> None:
    with tempfile.TemporaryDirectory() as raw_directory:
        directory = Path(raw_directory)
        manifest_path, _ = _synthetic_manifest(directory)
        result = generate_workcell_urdf(manifest_path)

        assert result.wrote_files
        assert result.urdf_path.is_file()
        assert result.report_path.is_file()
        root = ET.parse(result.urdf_path).getroot()
        assert root.get("name") == "synthetic_cell"
        _assert_single_connected_tree(root)

        links = {link.get("name") for link in root.findall("link")}
        assert {
            "world",
            "frame__cell",
            "body__table",
            "arm_a__base",
            "arm_a__carriage",
            "arm_a__tool",
            "arm_b__pedestal",
            "arm_b__wrist",
            "arm_b__flange",
            "hand__mount",
            "hand__left_finger",
            "hand__right_finger",
            "hand__tcp",
            "camera__overhead__link",
            "camera__overhead__optical_frame",
            "camera__wrist__link",
            "camera__wrist__optical_frame",
            "frame__inspection_target",
        } <= links

        joint_types = {
            joint.get("name"): joint.get("type") for joint in root.findall("joint")
        }
        assert joint_types["arm_a__turntable"] == "revolute"
        assert joint_types["arm_a__tool_slide"] == "prismatic"
        assert joint_types["arm_b__spin"] == "continuous"
        assert joint_types["arm_b__flange_fixed"] == "fixed"
        assert joint_types["hand__left_slide"] == "prismatic"

        mimic = _joint(root, "hand__right_slide").find("mimic")
        assert mimic is not None
        assert mimic.get("joint") == "hand__left_slide"
        assert float(mimic.get("multiplier", "nan")) == -1.0

        transmission = root.find("transmission[@name='hand__parallel_transmission']")
        assert transmission is not None
        transmission_joint = transmission.find("joint")
        transmission_actuator = transmission.find("actuator")
        assert transmission_joint is not None and transmission_actuator is not None
        assert transmission_joint.get("name") == "hand__left_slide"
        assert transmission_actuator.get("name") == "hand__finger_motor"

        material = root.find("material[@name='arm_a__metal']")
        material_reference = root.find(
            "link[@name='arm_a__base']/visual/material[@name='arm_a__metal']"
        )
        assert material is not None
        assert material_reference is not None

        assert set(result.report["robots"]) == {"arm_a", "arm_b"}
        assert set(result.report["grippers"]) == {"hand"}
        assert result.report["robots"]["arm_a"]["link_count"] == 3
        assert result.report["robots"]["arm_b"]["link_count"] == 3


def test_camera_info_fov_envelope_and_optical_transform() -> None:
    with tempfile.TemporaryDirectory() as raw_directory:
        directory = Path(raw_directory)
        manifest_path, _ = _synthetic_manifest(directory)
        result = generate_workcell_urdf(manifest_path)
        root = ET.parse(result.urdf_path).getroot()

        parent_T_link = _origin_transform(_joint(root, "fixed__camera__overhead__link"))
        link_T_optical = _origin_transform(
            _joint(root, "fixed__camera__overhead__optical_frame")
        )
        expected = np.eye(4)
        expected[:3, :3] = np.diag([1.0, -1.0, -1.0])
        expected[:3, 3] = [0.4, 0.0, 1.4]
        assert np.allclose(parent_T_link @ link_T_optical, expected, atol=1e-10)

        assert len(result.camera_info_paths) == 1
        camera_info_path = result.camera_info_paths[0]
        assert camera_info_path.name == "overhead_camera_info.yaml"
        camera_info = yaml.safe_load(camera_info_path.read_text(encoding="utf-8"))
        assert camera_info["image_width"] == 640
        assert camera_info["image_height"] == 480
        assert camera_info["camera_name"] == "overhead"
        assert camera_info["distortion_model"] == "plumb_bob"
        assert camera_info["camera_matrix"]["data"] == [
            320.0,
            0.0,
            320.0,
            0.0,
            240.0,
            240.0,
            0.0,
            0.0,
            1.0,
        ]
        assert camera_info["distortion_coefficients"]["data"] == [
            0.1,
            -0.05,
            0.001,
            -0.002,
            0.01,
        ]
        assert camera_info["rectification_matrix"]["data"] == [
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ]
        assert camera_info["projection_matrix"]["data"] == [
            320.0,
            0.0,
            320.0,
            0.0,
            0.0,
            240.0,
            240.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
        ]

        camera_report = result.report["cameras"]["overhead"]
        derived = camera_report["derived"]
        assert math.isclose(derived["horizontal_fov_deg"], 90.0, abs_tol=1e-10)
        assert math.isclose(derived["vertical_fov_deg"], 90.0, abs_tol=1e-10)
        assert np.allclose(
            [
                derived["footprint_at_nominal_working_distance_m"]["width"],
                derived["footprint_at_nominal_working_distance_m"]["height"],
            ],
            [2.0, 2.0],
            atol=1e-12,
        )
        envelope = camera_report["operating_envelope"]
        assert envelope["working_distance_m"] == {
            "min": 0.6,
            "nominal": 1.0,
            "max": 1.4,
        }
        assert envelope["view_depth_m"] == {"near": 0.2, "far": 1.8}
        assert envelope["depth_measurement_range_m"] == {"min": 0.3, "max": 2.0}
        assert envelope["focus"]["depth_of_field_m"] == {"near": 0.7, "far": 1.3}


def test_pasteable_camera_info_mappings_round_trip() -> None:
    """Accept calibration values copied from a standard camera-info YAML."""
    with tempfile.TemporaryDirectory() as raw_directory:
        directory = Path(raw_directory)
        manifest_path, manifest = _synthetic_manifest(directory)
        camera_matrix = [
            500.0,
            1.25,
            400.0,
            0.0,
            510.0,
            300.0,
            0.0,
            0.0,
            1.0,
        ]
        distortion = [0.11, -0.07, 0.002, -0.003, 0.015]
        rectification = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        projection = [
            480.0,
            0.0,
            395.0,
            -24.0,
            0.0,
            482.0,
            298.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
        ]
        manifest["cameras"][0]["intrinsics"] = {
            "image_width": 800,
            "image_height": 600,
            "camera_name": "calibration_tool_name",
            "camera_matrix": {"rows": 3, "cols": 3, "data": camera_matrix},
            "distortion_model": "plumb_bob",
            "D": {"rows": 1, "cols": 5, "data": distortion},
            "R": {"rows": 3, "cols": 3, "data": rectification},
            "P": {"rows": 3, "cols": 4, "data": projection},
        }
        _write_yaml(manifest_path, manifest)

        result = generate_workcell_urdf(manifest_path)
        assert len(result.camera_info_paths) == 1
        camera_info = yaml.safe_load(
            result.camera_info_paths[0].read_text(encoding="utf-8")
        )
        assert camera_info["image_width"] == 800
        assert camera_info["image_height"] == 600
        # The manifest instance name remains authoritative for generated TF and
        # output names even when a pasted calibration carries a driver name.
        assert camera_info["camera_name"] == "overhead"
        assert camera_info["camera_matrix"] == {
            "rows": 3,
            "cols": 3,
            "data": camera_matrix,
        }
        assert camera_info["distortion_coefficients"] == {
            "rows": 1,
            "cols": 5,
            "data": distortion,
        }
        assert camera_info["rectification_matrix"] == {
            "rows": 3,
            "cols": 3,
            "data": rectification,
        }
        assert camera_info["projection_matrix"] == {
            "rows": 3,
            "cols": 4,
            "data": projection,
        }


def test_invalid_intrinsics_are_rejected_before_writing() -> None:
    with tempfile.TemporaryDirectory() as raw_directory:
        directory = Path(raw_directory)
        manifest_path, manifest = _synthetic_manifest(directory)
        camera = manifest["cameras"][0]
        camera["intrinsics"]["fx"] = 0.0
        _write_yaml(manifest_path, manifest)

        try:
            generate_workcell_urdf(manifest_path)
        except ValueError as error:
            message = str(error)
            assert "fx" in message
            assert "positive" in message
        else:
            raise AssertionError("zero focal length was accepted")

        assert not (directory / "generated/cell.urdf").exists()
        assert not (directory / "generated/cell.report.yaml").exists()
        assert not (directory / "generated/camera_info").exists()


def test_misspelled_operating_envelope_key_is_rejected_before_writing() -> None:
    with tempfile.TemporaryDirectory() as raw_directory:
        directory = Path(raw_directory)
        manifest_path, manifest = _synthetic_manifest(directory)
        manifest["cameras"][0]["operating_envelope"][
            "depth_measurment_range_m"
        ] = {"min": 0.3, "max": 2.0}
        _write_yaml(manifest_path, manifest)

        try:
            generate_workcell_urdf(manifest_path)
        except ValueError as error:
            message = str(error)
            assert "operating_envelope" in message
            assert "unknown keys" in message
            assert "depth_measurment_range_m" in message
        else:
            raise AssertionError("misspelled operating-envelope key was accepted")

        assert not (directory / "generated/cell.urdf").exists()
        assert not (directory / "generated/cell.report.yaml").exists()
        assert not (directory / "generated/camera_info").exists()


def test_nested_joint_extensions_and_duplicate_root_materials_are_rejected() -> None:
    nested_extension = VENDOR_A_URDF.replace(
        '    <limit lower="-2" upper="2" effort="10" velocity="1"/>\n',
        '    <vendor_setting gain="4"/>\n'
        '    <limit lower="-2" upper="2" effort="10" velocity="1"/>\n',
        1,
    )
    duplicate_material = VENDOR_A_URDF.replace(
        '  <link name="base">\n',
        '  <material name="metal"><color rgba="0 0 0 1"/></material>\n'
        '  <link name="base">\n',
        1,
    )
    cases = [
        (
            nested_extension,
            ("unsupported", "joint/vendor_setting"),
            "nested joint extension",
        ),
        (
            duplicate_material,
            ("root materials", "duplicate name", "metal"),
            "duplicate root material",
        ),
    ]
    for source, expected_fragments, description in cases:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            manifest_path, _ = _synthetic_manifest(directory)
            _write_text(directory / "vendor_a.urdf", source)

            try:
                generate_workcell_urdf(manifest_path)
            except ValueError as error:
                message = str(error)
                assert all(fragment in message for fragment in expected_fragments), (
                    description,
                    message,
                )
            else:
                raise AssertionError(f"{description} was accepted")

            assert not (directory / "generated/cell.urdf").exists()
            assert not (directory / "generated/cell.report.yaml").exists()
            assert not (directory / "generated/camera_info").exists()


def test_validate_only_returns_inventory_without_output_files() -> None:
    with tempfile.TemporaryDirectory() as raw_directory:
        directory = Path(raw_directory)
        manifest_path, _ = _synthetic_manifest(directory)
        result = generate_workcell_urdf(manifest_path, write_files=False)
        assert not result.wrote_files
        assert set(result.report["robots"]) == {"arm_a", "arm_b"}
        assert set(result.report["cameras"]) == {"overhead", "wrist"}
        assert not result.urdf_path.exists()
        assert not result.report_path.exists()
        assert not result.camera_info_paths


def test_repository_manifest_prunes_bundled_gp7_grippers() -> None:
    """Exercise the real manifest when it is present in the repository."""
    manifest_path = ROOT / "config/workcell_generator.yaml"
    if not manifest_path.is_file():
        return

    with tempfile.TemporaryDirectory() as raw_directory:
        directory = Path(raw_directory)
        result = generate_workcell_urdf(
            manifest_path,
            output_override=directory / "workcell.urdf",
            camera_info_dir_override=directory / "camera_info",
            report_override=directory / "workcell.report.yaml",
        )
        root = ET.parse(result.urdf_path).getroot()
        names = {element.get("name") for element in root.findall("joint")}
        for robot in result.report["robots"]:
            assert f"{robot}__tool0-gripper" not in names
            assert f"{robot}__gripper-tcp" not in names
            assert result.report["robots"][robot]["link_count"] == 9
            assert set(result.report["robots"][robot]["pruned_source_links"]) == {
                "gripper",
                "tcp",
            }
        assert not any(
            link.get("name") in {
                f"{robot}__gripper" for robot in result.report["robots"]
            }
            for link in root.findall("link")
        )
        structure = root.find("link[@name='body__workcell_structure']")
        assert structure is not None
        assert len(structure.findall("collision")) == 30
        links = {link.get("name") for link in root.findall("link")}
        assert {
            "gripper__A_gripper__mount",
            "gripper__A_gripper__tcp",
            "gripper__B_gripper__mount",
            "gripper__B_gripper__tcp",
        } <= links
        _assert_single_connected_tree(root)


def test_repository_manifest_loads_in_mujoco_with_fixed_frame_names() -> None:
    try:
        import mujoco
    except ImportError:
        return

    manifest_path = ROOT / "config/workcell_generator.yaml"
    assert manifest_path.is_file()
    with tempfile.TemporaryDirectory() as raw_directory:
        directory = Path(raw_directory)
        result = generate_workcell_urdf(
            manifest_path,
            output_override=directory / "workcell.urdf",
            camera_info_dir_override=directory / "camera_info",
            report_override=directory / "workcell.report.yaml",
        )
        root = ET.parse(result.urdf_path).getroot()
        visual_names = [
            visual.get("name")
            for visual in root.findall(".//visual")
            if visual.get("name") is not None
        ]
        collision_names = [
            collision.get("name")
            for collision in root.findall(".//collision")
            if collision.get("name") is not None
        ]
        assert visual_names
        assert collision_names
        assert len(visual_names) == len(set(visual_names))
        assert len(collision_names) == len(set(collision_names))
        assert not set(visual_names) & set(collision_names)

        model = mujoco.MjModel.from_xml_path(str(result.urdf_path))
        expected_fixed_bodies = {
            "camera__overhead__link",
            "camera__overhead__optical_frame",
            "gripper__A_gripper__tcp",
            "gripper__B_gripper__tcp",
        }
        for name in expected_fixed_bodies:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            assert body_id >= 0, name


if __name__ == "__main__":
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
