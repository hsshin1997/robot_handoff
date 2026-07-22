#!/usr/bin/env python3
"""Build one rooted URDF for the complete two-arm handoff workcell.

The generated URDF is a calibration/TF/visualization artifact.  It expands two
prefixed copies of the GP7 arm, adds the current gripper CAD and authoritative
TCP, the workcell visual and collision geometry, provisional fixture geometry,
task frames, and any calibrated camera/target frames declared by the user.

Do not hand-edit the generated URDF.  Update the source YAML/URDF assets and
run this script again.
"""
from __future__ import annotations

import argparse
import copy
import math
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROJECT = ROOT / "mujoco_sim" / "config" / "project.yaml"
DEFAULT_FIXTURES = (
    ROOT / "mujoco_sim" / "config" / "internal" / "scene_fallback.yaml"
)
DEFAULT_CALIBRATION = ROOT / "config" / "workcell_calibration.yaml"
DEFAULT_OUTPUT = ROOT / "assets" / "workcell" / "handoff_workcell.urdf"

ARM_LINKS = (
    "base_link",
    "link_1_s",
    "link_2_l",
    "link_3_u",
    "link_4_r",
    "link_5_b",
    "link_6_t",
    "tool0",
)

MATERIALS = {
    "workcell_aluminum": "0.72 0.75 0.78 1",
    "floor_gray": "0.19 0.20 0.21 1",
    "wood": "0.25 0.13 0.055 1",
    "black_steel": "0.035 0.04 0.045 1",
    "bin_gray": "0.42 0.45 0.48 1",
    "reorientation": "0.72 0.68 0.55 1",
    "fixture_aluminum": "0.40 0.43 0.46 1",
    "pcb_green": "0.04 0.35 0.13 1",
    "gripper_dark": "0.08 0.09 0.10 1",
}


def _load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def _resolve_asset(value: str, owner: Path) -> Path:
    path = Path(value)
    candidates = (
        [path]
        if path.is_absolute()
        else [ROOT / path, owner.parent / path]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        f"asset {value!r} was not found relative to {ROOT} or {owner.parent}"
    )


def _numbers(values: Iterable[float]) -> str:
    def clean(value: float) -> str:
        result = f"{float(value):.12g}"
        return "0" if result in ("-0", "-0.0") else result

    return " ".join(clean(value) for value in values)


def _rpy_matrix(rpy: Iterable[float]) -> np.ndarray:
    roll, pitch, yaw = (float(value) for value in rpy)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=float,
    )


def _matrix_rpy(rotation: np.ndarray) -> np.ndarray:
    pitch = math.atan2(
        -float(rotation[2, 0]),
        math.hypot(float(rotation[0, 0]), float(rotation[1, 0])),
    )
    if abs(math.cos(pitch)) > 1e-9:
        roll = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
        yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    else:
        # At gimbal lock, choose yaw=0 and retain an equivalent roll.
        roll = math.atan2(-float(rotation[1, 2]), float(rotation[1, 1]))
        yaw = 0.0
    return np.array([roll, pitch, yaw], dtype=float)


def _validate_transform(transform: np.ndarray, label: str) -> np.ndarray:
    transform = np.asarray(transform, dtype=float)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError(f"{label} must be a finite 4x4 matrix")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError(f"{label} must have homogeneous last row [0, 0, 0, 1]")
    rotation = transform[:3, :3]
    # Calibration tools commonly serialize rotations to six decimals. Accept
    # only small rounding drift, then project it to the nearest proper SO(3)
    # matrix so the generated URDF contains a self-consistent rotation.
    tolerance = 1e-5
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=tolerance):
        raise ValueError(f"{label} rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=tolerance):
        raise ValueError(f"{label} rotation must have determinant +1")
    u, _, vt = np.linalg.svd(rotation)
    projected = u @ vt
    if np.linalg.det(projected) < 0.0:
        u[:, -1] *= -1.0
        projected = u @ vt
    result = transform.copy()
    result[:3, :3] = projected
    return result


def _pose_transform(value: dict | None, label: str) -> np.ndarray:
    value = value or {}
    if "matrix" in value:
        return _validate_transform(np.asarray(value["matrix"], dtype=float), label)

    position = np.asarray(value.get("position_m", [0.0, 0.0, 0.0]), dtype=float)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError(f"{label}.position_m must contain three finite values")
    if "rotation_matrix" in value:
        rotation = np.asarray(value["rotation_matrix"], dtype=float)
        if rotation.shape != (3, 3):
            raise ValueError(f"{label}.rotation_matrix must be 3x3")
    else:
        rpy_deg = np.asarray(value.get("rpy_deg", [0.0, 0.0, 0.0]), dtype=float)
        if rpy_deg.shape != (3,) or not np.all(np.isfinite(rpy_deg)):
            raise ValueError(f"{label}.rpy_deg must contain three finite values")
        rotation = _rpy_matrix(np.radians(rpy_deg))
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = position
    return _validate_transform(transform, label)


def _origin(parent: ET.Element, transform: np.ndarray) -> ET.Element:
    return ET.SubElement(
        parent,
        "origin",
        {
            "xyz": _numbers(transform[:3, 3]),
            "rpy": _numbers(_matrix_rpy(transform[:3, :3])),
        },
    )


def _pose_origin(parent: ET.Element, position, rpy=(0.0, 0.0, 0.0)) -> ET.Element:
    return ET.SubElement(
        parent,
        "origin",
        {"xyz": _numbers(position), "rpy": _numbers(rpy)},
    )


def _add_materials(robot: ET.Element) -> None:
    for name, rgba in MATERIALS.items():
        material = ET.SubElement(robot, "material", {"name": name})
        ET.SubElement(material, "color", {"rgba": rgba})


def _add_fixed_link(
    robot: ET.Element,
    known_links: set[str],
    child: str,
    parent: str,
    transform: np.ndarray,
    *,
    comment: str | None = None,
) -> ET.Element:
    if child in known_links:
        raise ValueError(f"duplicate link name {child!r}")
    if parent not in known_links:
        raise ValueError(f"parent link {parent!r} for {child!r} does not exist")
    if comment:
        robot.append(ET.Comment(comment))
    link = ET.SubElement(robot, "link", {"name": child})
    joint = ET.SubElement(
        robot, "joint", {"name": f"{parent}_to_{child}", "type": "fixed"}
    )
    ET.SubElement(joint, "parent", {"link": parent})
    ET.SubElement(joint, "child", {"link": child})
    _origin(joint, transform)
    known_links.add(child)
    return link


def _add_box(
    link: ET.Element,
    name: str,
    center: Iterable[float],
    size: Iterable[float],
    material: str,
    *,
    collision: bool = True,
) -> None:
    center = tuple(float(value) for value in center)
    size = tuple(float(value) for value in size)
    if len(center) != 3 or len(size) != 3 or min(size) <= 0.0:
        raise ValueError(f"invalid box {name}: center={center}, size={size}")
    visual = ET.SubElement(link, "visual", {"name": f"{name}_visual"})
    _pose_origin(visual, center)
    geometry = ET.SubElement(visual, "geometry")
    ET.SubElement(geometry, "box", {"size": _numbers(size)})
    ET.SubElement(visual, "material", {"name": material})
    if collision:
        collision_element = ET.SubElement(
            link, "collision", {"name": f"{name}_collision"}
        )
        _pose_origin(collision_element, center)
        geometry = ET.SubElement(collision_element, "geometry")
        ET.SubElement(geometry, "box", {"size": _numbers(size)})


def _rewrite_robot_link(
    source: ET.Element,
    prefix: str,
    robot_dir: Path,
    output_dir: Path,
) -> ET.Element:
    result = copy.deepcopy(source)
    result.set("name", f"{prefix}_{source.attrib['name']}")
    for mesh in result.findall(".//mesh"):
        filename = mesh.attrib["filename"]
        if filename.startswith("package://"):
            raise ValueError(
                f"package URI {filename!r} is unsupported by this standalone builder"
            )
        source_mesh = Path(filename)
        if not source_mesh.is_absolute():
            source_mesh = robot_dir / source_mesh
        source_mesh = source_mesh.resolve()
        if not source_mesh.exists():
            raise FileNotFoundError(source_mesh)
        mesh.set("filename", os.path.relpath(source_mesh, output_dir))
    for material in result.findall(".//material"):
        if material.get("name"):
            material.set("name", f"{prefix}_{material.get('name')}")
    return result


def _clone_robot(
    robot: ET.Element,
    known_links: set[str],
    prefix: str,
    source_urdf: Path,
    mount_pose: dict,
    gripper_spec: dict,
    project_path: Path,
    output_dir: Path,
) -> None:
    source_root = ET.parse(source_urdf).getroot()
    source_links = {link.attrib["name"]: link for link in source_root.findall("link")}
    missing = sorted(set(ARM_LINKS) - set(source_links))
    if missing:
        raise ValueError(f"{source_urdf} is missing GP7 links: {missing}")

    robot.append(ET.Comment(f"{prefix} GP7 arm expanded from {source_urdf.name}."))
    for source_name in ARM_LINKS:
        copied = _rewrite_robot_link(
            source_links[source_name], prefix, source_urdf.parent, output_dir
        )
        robot.append(copied)
        known_links.add(copied.attrib["name"])

    mount = ET.SubElement(
        robot,
        "joint",
        {"name": f"cell_to_{prefix}_base_link", "type": "fixed"},
    )
    ET.SubElement(mount, "parent", {"link": "cell"})
    ET.SubElement(mount, "child", {"link": f"{prefix}_base_link"})
    _origin(mount, _pose_transform(mount_pose, f"robots.{prefix}.world_base"))

    for source_joint in source_root.findall("joint"):
        parent = source_joint.find("parent")
        child = source_joint.find("child")
        if parent is None or child is None:
            continue
        parent_name = parent.attrib["link"]
        child_name = child.attrib["link"]
        if parent_name not in ARM_LINKS or child_name not in ARM_LINKS:
            continue
        copied = copy.deepcopy(source_joint)
        copied.set("name", f"{prefix}_{source_joint.attrib['name']}")
        copied.find("parent").set("link", f"{prefix}_{parent_name}")
        copied.find("child").set("link", f"{prefix}_{child_name}")
        robot.append(copied)

    # The current project uses the measured gripper CAD and its 232.92807 mm
    # TCP, not the legacy primitive gripper's 200 mm TCP.
    gripper_name = f"{prefix}_gripper"
    robot.append(
        ET.Comment(
            f"{prefix} gripper uses current CAD/TCP. Its fixed-open collision "
            "primitives and inertia remain provisional."
        )
    )
    gripper_link = ET.SubElement(robot, "link", {"name": gripper_name})
    source_gripper = source_links.get("gripper")
    if source_gripper is not None and source_gripper.find("inertial") is not None:
        gripper_link.append(copy.deepcopy(source_gripper.find("inertial")))
    visual = ET.SubElement(
        gripper_link, "visual", {"name": f"{prefix}_gripper_cad_visual"}
    )
    geometry = ET.SubElement(visual, "geometry")
    gripper_mesh = _resolve_asset(gripper_spec["model"], project_path)
    gripper_units = gripper_spec.get("model_units", "m")
    gripper_scale = {"m": 1.0, "mm": 0.001}.get(gripper_units)
    if gripper_scale is None:
        raise ValueError(f"unsupported gripper units {gripper_units!r}")
    ET.SubElement(
        geometry,
        "mesh",
        {
            "filename": os.path.relpath(gripper_mesh, output_dir),
            "scale": _numbers((gripper_scale, gripper_scale, gripper_scale)),
        },
    )
    ET.SubElement(visual, "material", {"name": "gripper_dark"})
    # Retain the source URDF's stable primitive collision approximation.  The
    # CAD is a disconnected static assembly and one mesh hull would close the
    # jaw aperture.  Geometry must be replaced when articulated gripper CAD is
    # supplied.
    if source_gripper is not None:
        for collision in source_gripper.findall("collision"):
            copied_collision = copy.deepcopy(collision)
            if copied_collision.get("name"):
                copied_collision.set(
                    "name", f"{prefix}_{copied_collision.get('name')}"
                )
            gripper_link.append(copied_collision)
    known_links.add(gripper_name)

    source_mount = next(
        (
            joint
            for joint in source_root.findall("joint")
            if joint.attrib.get("name") == "tool0-gripper"
        ),
        None,
    )
    gripper_joint = ET.SubElement(
        robot,
        "joint",
        {"name": f"{prefix}_tool0_to_gripper", "type": "fixed"},
    )
    ET.SubElement(gripper_joint, "parent", {"link": f"{prefix}_tool0"})
    ET.SubElement(gripper_joint, "child", {"link": gripper_name})
    if source_mount is not None and source_mount.find("origin") is not None:
        gripper_joint.append(copy.deepcopy(source_mount.find("origin")))
    else:
        _pose_origin(gripper_joint, (0.0, 0.0, 0.0), (0.0, math.pi / 2.0, 0.0))

    tcp_name = f"{prefix}_tcp"
    ET.SubElement(robot, "link", {"name": tcp_name})
    known_links.add(tcp_name)
    tcp_joint = ET.SubElement(
        robot,
        "joint",
        {"name": f"{prefix}_gripper_to_tcp", "type": "fixed"},
    )
    ET.SubElement(tcp_joint, "parent", {"link": gripper_name})
    ET.SubElement(tcp_joint, "child", {"link": tcp_name})
    _origin(
        tcp_joint,
        _pose_transform(
            gripper_spec["mount_to_tcp"], f"grippers.{prefix}.mount_to_tcp"
        ),
    )


def _add_workcell_geometry(
    robot: ET.Element,
    known_links: set[str],
    project: dict,
    project_path: Path,
    output_dir: Path,
) -> None:
    link = _add_fixed_link(
        robot,
        known_links,
        "workcell_link",
        "cell",
        np.eye(4),
        comment=(
            "Exact workcell visual in millimetres; collision uses the 30 "
            "stable primitive boxes from collision_boxes.yaml."
        ),
    )
    workstation = project["workstation"]
    visual_mesh = _resolve_asset(workstation["visual_cad"], project_path)
    visual = ET.SubElement(link, "visual", {"name": "workcell_visual"})
    geometry = ET.SubElement(visual, "geometry")
    units = workstation.get("visual_cad_units", "m")
    scale = {"m": 1.0, "mm": 0.001}.get(units)
    if scale is None:
        raise ValueError(f"unsupported workcell visual units {units!r}")
    ET.SubElement(
        geometry,
        "mesh",
        {
            "filename": os.path.relpath(visual_mesh, output_dir),
            "scale": _numbers((scale, scale, scale)),
        },
    )
    ET.SubElement(visual, "material", {"name": "workcell_aluminum"})

    collision_path = _resolve_asset(workstation["collision_cad"], project_path)
    collision_data = _load_yaml(collision_path)
    if collision_data.get("units") != "mm":
        raise ValueError("the current collision-box source must declare units: mm")
    boxes = collision_data.get("pedestals", []) + collision_data.get("boxes", [])
    if len(boxes) != 30:
        raise ValueError(f"expected 30 workcell collision boxes, found {len(boxes)}")
    for index, item in enumerate(boxes):
        center = np.asarray(item["center"], dtype=float) * 0.001
        size = np.asarray(item["half_extents"], dtype=float) * 0.002
        collision = ET.SubElement(
            link, "collision", {"name": f"workcell_collision_{index:02d}"}
        )
        _pose_origin(collision, center)
        geometry = ET.SubElement(collision, "geometry")
        ET.SubElement(geometry, "box", {"size": _numbers(size)})

    floor = _add_fixed_link(
        robot,
        known_links,
        "floor_link",
        "cell",
        np.eye(4),
        comment="Finite URDF approximation of the MuJoCo plane; top surface is z=-0.610 m.",
    )
    _add_box(
        floor,
        "floor",
        (0.425, -0.175, -0.620),
        (4.0, 4.0, 0.020),
        "floor_gray",
    )


def _add_fixture_geometry(
    robot: ET.Element,
    known_links: set[str],
    fixture_config: dict,
) -> dict[str, np.ndarray]:
    fixture_link = _add_fixed_link(
        robot,
        known_links,
        "fixtures_link",
        "cell",
        np.eye(4),
        comment=(
            "PROVISIONAL photo-matched fixtures from scene_fallback.yaml; "
            "replace with surveyed CAD before hardware qualification."
        ),
    )
    fixtures = fixture_config["fixtures"]
    floor_z = float(fixtures["floor_z"])
    top_z = floor_z + float(fixtures["table_height"])
    semantic: dict[str, np.ndarray] = {}

    def frame(position, rpy=(0.0, 0.0, 0.0)):
        transform = np.eye(4)
        transform[:3, :3] = _rpy_matrix(rpy)
        transform[:3, 3] = position
        return transform

    def table(name: str, specification: dict) -> None:
        cx, cy = (float(value) for value in specification["center_xy"])
        sx, sy = (float(value) for value in specification["size_xy"])
        thickness = float(specification["top_thickness"])
        leg = float(specification["leg_size"])
        _add_box(
            fixture_link,
            f"{name}_top",
            (cx, cy, top_z - thickness / 2.0),
            (sx, sy, thickness),
            "wood",
        )
        leg_half_z = (top_z - thickness - floor_z) / 2.0
        leg_z = floor_z + leg_half_z
        inset = leg / 2.0
        for ix, x_sign in enumerate((-1.0, 1.0)):
            for iy, y_sign in enumerate((-1.0, 1.0)):
                _add_box(
                    fixture_link,
                    f"{name}_leg_{ix}_{iy}",
                    (
                        cx + x_sign * (sx / 2.0 - inset),
                        cy + y_sign * (sy / 2.0 - inset),
                        leg_z,
                    ),
                    (leg, leg, 2.0 * leg_half_z),
                    "black_steel",
                )
        rail = 0.018
        rail_z = top_z - thickness - rail / 2.0
        for side, y in (("front", cy - sy / 2.0 + rail / 2.0),
                        ("back", cy + sy / 2.0 - rail / 2.0)):
            _add_box(
                fixture_link,
                f"{name}_rail_{side}",
                (cx, y, rail_z),
                (sx, rail, rail),
                "black_steel",
            )
        semantic[f"{name}_frame"] = frame((cx, cy, top_z))

    table("supply_table", fixtures["supply_table"])
    table("pcb_table", fixtures["pcb_table"])

    for bin_spec in fixtures["bins"]:
        name = str(bin_spec["name"])
        cx, cy = (float(value) for value in bin_spec["center_xy"])
        ix, iy = (float(value) for value in bin_spec["interior_xy"])
        wall = float(bin_spec["wall_thickness"])
        height = float(bin_spec["wall_height"])
        bottom = 0.006
        _add_box(
            fixture_link,
            f"{name}_bottom",
            (cx, cy, top_z + bottom / 2.0),
            (ix + 2.0 * wall, iy + 2.0 * wall, bottom),
            "bin_gray",
        )
        wall_z = top_z + bottom + height / 2.0
        walls = (
            ("left", (cx - ix / 2.0 - wall / 2.0, cy, wall_z), (wall, iy + 2.0 * wall, height)),
            ("right", (cx + ix / 2.0 + wall / 2.0, cy, wall_z), (wall, iy + 2.0 * wall, height)),
            ("front", (cx, cy - iy / 2.0 - wall / 2.0, wall_z), (ix, wall, height)),
            ("back", (cx, cy + iy / 2.0 + wall / 2.0, wall_z), (ix, wall, height)),
        )
        for side, center, size in walls:
            _add_box(fixture_link, f"{name}_{side}", center, size, "bin_gray")
        semantic[f"{name}_frame"] = frame((cx, cy, top_z + bottom))

    plate = fixtures["reorientation_surface"]
    cx, cy = (float(value) for value in plate["center_xy"])
    sx, sy = (float(value) for value in plate["size_xy"])
    thickness = float(plate["thickness"])
    _add_box(
        fixture_link,
        "reorientation_surface",
        (cx, cy, top_z + thickness / 2.0),
        (sx, sy, thickness),
        "reorientation",
    )
    semantic["reorientation_surface_frame"] = frame((cx, cy, top_z + thickness))

    pcb = fixtures["pcb_fixture"]
    cx, cy = (float(value) for value in pcb["center_xy"])
    bx, by = (float(value) for value in pcb["base_size_xy"])
    base_t = float(pcb["base_thickness"])
    px, py = (float(value) for value in pcb["board_size_xy"])
    board_t = float(pcb["board_thickness"])
    ax, ay = (float(value) for value in pcb["aperture_size_xy"])

    def aperture_ring(name, sx, sy, z, thickness, material):
        side_x = 0.5 * (sx - ax)
        side_y = 0.5 * (sy - ay)
        x_offset = 0.25 * (sx + ax)
        y_offset = 0.25 * (sy + ay)
        boxes = (
            ("left", (cx - x_offset, cy, z), (side_x, sy, thickness)),
            ("right", (cx + x_offset, cy, z), (side_x, sy, thickness)),
            ("front", (cx, cy - y_offset, z), (ax, side_y, thickness)),
            ("back", (cx, cy + y_offset, z), (ax, side_y, thickness)),
        )
        for side, center, size in boxes:
            _add_box(fixture_link, f"{name}_{side}", center, size, material)

    aperture_ring(
        "pcb_fixture_base", bx, by, top_z + base_t / 2.0, base_t,
        "fixture_aluminum",
    )
    aperture_ring(
        "pcb_board", px, py, top_z + base_t + board_t / 2.0, board_t,
        "pcb_green",
    )
    semantic["pcb_fixture_top_frame"] = frame(
        (cx, cy, top_z + base_t + board_t)
    )
    return semantic


def _safe_name(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")
    if not result:
        raise ValueError(f"invalid empty frame name derived from {value!r}")
    return result


def _add_semantic_frames(
    robot: ET.Element,
    known_links: set[str],
    project: dict,
    fixture_frames: dict[str, np.ndarray],
) -> None:
    for name, transform in fixture_frames.items():
        _add_fixed_link(robot, known_links, name, "cell", transform)

    regions = project.get("regions", {})
    for name in ("handoff", "scanner", "insertion"):
        region = regions.get(name)
        if region and "center_m" in region:
            transform = np.eye(4)
            transform[:3, 3] = np.asarray(region["center_m"], dtype=float)
            _add_fixed_link(
                robot,
                known_links,
                f"{_safe_name(name)}_region_frame",
                "cell",
                _validate_transform(transform, f"regions.{name}"),
            )

    insertion = project.get("insertion", {})
    if "pcb_world_pose" in insertion:
        _add_fixed_link(
            robot,
            known_links,
            "pcb_frame",
            "cell",
            _pose_transform(insertion["pcb_world_pose"], "insertion.pcb_world_pose"),
        )
    for target in insertion.get("targets", []):
        name = _safe_name(str(target["name"]))
        _add_fixed_link(
            robot,
            known_links,
            f"{name}_part_target_frame",
            "cell",
            _pose_transform(
                target["world_part_pose"],
                f"insertion.targets.{name}.world_part_pose",
            ),
        )
        _add_fixed_link(
            robot,
            known_links,
            f"{name}_insertion_frame",
            "cell",
            _pose_transform(
                target["world_insertion_frame"],
                f"insertion.targets.{name}.world_insertion_frame",
            ),
        )


def _add_calibration_frames(
    robot: ET.Element,
    known_links: set[str],
    calibration: dict,
) -> None:
    if int(calibration.get("schema_version", 0)) != 1:
        raise ValueError("workcell calibration schema_version must be 1")

    standard_link_T_optical = _pose_transform(
        {
            "position_m": [0.0, 0.0, 0.0],
            "rpy_deg": [-90.0, 0.0, -90.0],
        },
        "standard camera_link_T_optical",
    )
    for camera in calibration.get("cameras", []):
        if not camera.get("enabled", False):
            continue
        if not camera.get("calibrated", False):
            raise ValueError(
                f"camera {camera.get('name')!r} is enabled but calibrated is false"
            )
        name = _safe_name(str(camera["name"]))
        parent = str(camera["parent"])
        parent_T_optical = _pose_transform(
            camera["parent_T_camera_optical"],
            f"cameras.{name}.parent_T_camera_optical",
        )
        link_T_optical = _pose_transform(
            camera.get("camera_link_T_optical"),
            f"cameras.{name}.camera_link_T_optical",
        ) if camera.get("camera_link_T_optical") else standard_link_T_optical
        parent_T_link = parent_T_optical @ np.linalg.inv(link_T_optical)
        link_name = f"camera_{name}_link"
        optical_name = f"camera_{name}_optical_frame"
        _add_fixed_link(
            robot,
            known_links,
            link_name,
            parent,
            _validate_transform(parent_T_link, f"cameras.{name}.parent_T_camera_link"),
            comment=(
                f"Calibrated {camera.get('mode', 'unspecified')} camera {name}; "
                "the input matrix is parent_T_camera_optical."
            ),
        )
        _add_fixed_link(
            robot,
            known_links,
            optical_name,
            link_name,
            link_T_optical,
        )

    for target in calibration.get("calibration_targets", []):
        if not target.get("enabled", False):
            continue
        if not target.get("calibrated", False):
            raise ValueError(
                f"calibration target {target.get('name')!r} is enabled but uncalibrated"
            )
        name = _safe_name(str(target["name"]))
        _add_fixed_link(
            robot,
            known_links,
            name,
            str(target["parent"]),
            _pose_transform(
                target["parent_T_target"],
                f"calibration_targets.{name}.parent_T_target",
            ),
        )

    for frame in calibration.get("additional_frames", []):
        if not frame.get("enabled", True):
            continue
        name = _safe_name(str(frame["name"]))
        _add_fixed_link(
            robot,
            known_links,
            name,
            str(frame["parent"]),
            _pose_transform(
                frame["parent_T_child"], f"additional_frames.{name}.parent_T_child"
            ),
        )


def _validate_tree(robot: ET.Element, output_dir: Path) -> None:
    links = [link.attrib["name"] for link in robot.findall("link")]
    joints = [joint.attrib["name"] for joint in robot.findall("joint")]
    if len(links) != len(set(links)):
        raise ValueError("generated URDF contains duplicate link names")
    if len(joints) != len(set(joints)):
        raise ValueError("generated URDF contains duplicate joint names")
    link_set = set(links)
    children: dict[str, str] = {}
    adjacency: dict[str, list[str]] = {name: [] for name in links}
    for joint in robot.findall("joint"):
        parent_element = joint.find("parent")
        child_element = joint.find("child")
        if parent_element is None or child_element is None:
            raise ValueError(f"joint {joint.attrib['name']} lacks parent/child")
        parent = parent_element.attrib["link"]
        child = child_element.attrib["link"]
        if parent not in link_set or child not in link_set:
            raise ValueError(f"joint {joint.attrib['name']} references an unknown link")
        if child in children:
            raise ValueError(f"link {child} has more than one parent")
        children[child] = parent
        adjacency[parent].append(child)
    roots = link_set - set(children)
    if roots != {"world"}:
        raise ValueError(f"generated URDF roots must be {{'world'}}, got {roots}")
    visited: set[str] = set()
    stack = ["world"]
    while stack:
        current = stack.pop()
        if current in visited:
            raise ValueError(f"cycle detected at link {current}")
        visited.add(current)
        stack.extend(adjacency[current])
    if visited != link_set:
        raise ValueError(f"disconnected links: {sorted(link_set - visited)}")

    revolute = [joint for joint in robot.findall("joint") if joint.get("type") == "revolute"]
    if len(revolute) != 12:
        raise ValueError(f"expected 12 revolute GP7 joints, found {len(revolute)}")
    for mesh in robot.findall(".//mesh"):
        filename = mesh.attrib["filename"]
        path = Path(filename)
        if not path.is_absolute():
            path = output_dir / path
        if not path.resolve().exists():
            raise FileNotFoundError(path)


def build_workcell_urdf(
    project_path: Path = DEFAULT_PROJECT,
    fixture_path: Path = DEFAULT_FIXTURES,
    calibration_path: Path = DEFAULT_CALIBRATION,
    output_path: Path = DEFAULT_OUTPUT,
) -> Path:
    project_path = project_path.resolve()
    fixture_path = fixture_path.resolve()
    calibration_path = calibration_path.resolve()
    output_path = output_path.resolve()
    project = _load_yaml(project_path)
    fixture_config = _load_yaml(fixture_path)
    calibration = _load_yaml(calibration_path)
    workstation = project["workstation"]
    if not workstation.get("generated_fixture_primitives", False):
        raise NotImplementedError(
            "the full-workcell URDF builder currently requires the generated "
            "fixture primitives; add URDF-compatible surveyed fixture CAD support "
            "before disabling them"
        )
    if workstation.get("additional_collision_cad"):
        raise NotImplementedError(
            "additional workstation collision CAD is not yet emitted into the "
            "full-workcell URDF; refusing to generate an incomplete cell"
        )
    if project.get("insertion", {}).get("collision_cad"):
        raise NotImplementedError(
            "insertion collision CAD is not yet emitted into the full-workcell "
            "URDF; refusing to generate an incomplete cell"
        )

    robot = ET.Element("robot", {"name": "handoff_workcell"})
    robot.append(
        ET.Comment(
            "GENERATED by scripts/build_workcell_urdf.py; do not hand-edit. "
            "Joint initial positions remain in project.yaml because URDF does not store state."
        )
    )
    _add_materials(robot)
    ET.SubElement(robot, "link", {"name": "world"})
    known_links = {"world"}
    _add_fixed_link(
        robot,
        known_links,
        "cell",
        "world",
        np.eye(4),
        comment=(
            "The cell frame is the workcell CAD frame in metres and coincides "
            "with robot A's calibrated base_link."
        ),
    )

    _add_workcell_geometry(robot, known_links, project, project_path, output_path.parent)
    fixture_frames = _add_fixture_geometry(robot, known_links, fixture_config)

    robots = project["robots"]
    for prefix in ("A", "B"):
        robot_spec = robots[prefix]
        gripper_name = robot_spec["gripper"]
        _clone_robot(
            robot,
            known_links,
            prefix,
            _resolve_asset(robot_spec["model"], project_path),
            robot_spec["world_base"],
            project["grippers"][gripper_name],
            project_path,
            output_path.parent,
        )

    _add_semantic_frames(robot, known_links, project, fixture_frames)
    _add_calibration_frames(robot, known_links, calibration)
    _validate_tree(robot, output_path.parent)

    ET.indent(robot, space="  ")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(robot).write(
        output_path, encoding="utf-8", xml_declaration=True, short_empty_elements=True
    )
    return output_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    args = _parser().parse_args()
    output = build_workcell_urdf(
        project_path=args.project,
        fixture_path=args.fixtures,
        calibration_path=args.calibration,
        output_path=args.output,
    )
    root = ET.parse(output).getroot()
    links = root.findall("link")
    joints = root.findall("joint")
    print(
        f"wrote {output} ({len(links)} links, {len(joints)} joints, "
        f"{sum(j.get('type') == 'revolute' for j in joints)} revolute)"
    )


if __name__ == "__main__":
    main()
