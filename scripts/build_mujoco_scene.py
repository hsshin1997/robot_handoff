"""Compile the calibrated MuJoCo scene from the project manifest.

The generated model contains the workcell, two GP7s, gripper geometry, active
part, and either user CAD fixtures or the current calibrated primitive fallback.
Planning and task logic remain separate from this deterministic scene compiler.

Sources of truth:
  - assets/workcell/workcell.stl (visual CAD, millimetres)
  - assets/workcell/collision_boxes.yaml (stable collision approximation)
  - assets/gp7/gp7.urdf (joint chain, limits, mesh references)
  - mujoco_sim/config/project.yaml (user-owned assets, frames, and task regions)
  - mujoco_sim/config/internal/scene_fallback.yaml (current lab primitive-fixture fallback only)
"""
from __future__ import annotations

import argparse
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mujoco_sim.modeling.cad_preprocess import (  # noqa: E402
    prepare_cad, scale_to_metres)
from mujoco_sim.core.paths import (  # noqa: E402
    DEFAULT_PROJECT_PATH, DEFAULT_SCENE_CONFIG_PATH)
from mujoco_sim.core.se3 import (make_transform, transform_from_rpy,
                                 validate_transform)  # noqa: E402

CONFIG = str(DEFAULT_SCENE_CONFIG_PATH)
PROJECT_CONFIG = str(DEFAULT_PROJECT_PATH)
OUTPUT = os.path.join(ROOT, "mujoco_sim", "models", "scene.xml")
JOINT_SHORT = ("s", "l", "u", "r", "b", "t")


def vec(text: str | None, default="0 0 0") -> str:
    return " ".join((text or default).split())


def fixed_pose_xml(pose: dict | None) -> tuple[str, str]:
    """Return MJCF position and orientation attributes for a manifest pose."""
    value = pose or {}
    if "matrix" in value:
        transform = validate_transform(np.asarray(value["matrix"], dtype=float))
    elif "rotation_matrix" in value:
        transform = make_transform(
            np.asarray(value["rotation_matrix"], dtype=float),
            value["position_m"],
        )
    else:
        position = value.get("position_m", [0.0, 0.0, 0.0])
        rpy_deg = value.get("rpy_deg", [0.0, 0.0, 0.0])
        transform = transform_from_rpy(
            position, np.radians(np.asarray(rpy_deg, dtype=float)))
        rpy = " ".join(map(str, np.radians(np.asarray(rpy_deg, dtype=float))))
        return " ".join(map(str, transform[:3, 3])), f'euler="{rpy}"'
    x_axis = " ".join(map(str, transform[:3, 0]))
    y_axis = " ".join(map(str, transform[:3, 1]))
    return (" ".join(map(str, transform[:3, 3])),
            f'xyaxes="{x_axis} {y_axis}"')


def resolve_project_asset(path: str, project_path: str) -> str:
    """Resolve a manifest asset consistently for default or alternate projects.

    Repository projects conventionally use repository-relative paths.  A
    project outside the repository may instead keep assets beside its YAML;
    absolute paths are always accepted.  Existing candidates determine the
    choice so the resulting MJCF never embeds a guessed missing location.
    """
    if os.path.isabs(path):
        return os.path.realpath(path)
    project_path = os.path.realpath(project_path)
    inside_repository = False
    try:
        inside_repository = os.path.commonpath((project_path, ROOT)) == ROOT
    except ValueError:
        pass
    candidates = ([os.path.join(ROOT, path), os.path.join(os.path.dirname(project_path), path)]
                  if inside_repository else
                  [os.path.join(os.path.dirname(project_path), path), os.path.join(ROOT, path)])
    for candidate in candidates:
        if os.path.exists(candidate):
            return os.path.realpath(candidate)
    raise FileNotFoundError(
        f"project asset {path!r} was not found relative to either "
        f"{os.path.dirname(project_path)!r} or repository root {ROOT!r}"
    )


def load_sources(project_path: str = PROJECT_CONFIG):
    project_path = os.path.realpath(project_path)
    with open(CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(project_path, encoding="utf-8") as f:
        project = yaml.safe_load(f)
    # config/internal/scene_fallback.yaml owns only the calibrated
    # primitive-fixture fallback;
    # project-owned mappings are initialized here rather than shadowed and
    # overwritten from stale duplicate values in that internal file.
    cfg["robots"] = {}
    cfg["gripper"] = {}
    cfg["assets"] = {}
    robot_models = {item["model"] for item in project["robots"].values()}
    if len(robot_models) != 1:
        raise ValueError("current scene compiler requires both robots to share one URDF")
    urdf_path = resolve_project_asset(next(iter(robot_models)), project_path)
    urdf = ET.parse(urdf_path).getroot()
    joints = {j.attrib["name"]: j for j in urdf.findall("joint")}
    for name, robot in project["robots"].items():
        pose = robot["world_base"]
        cfg["robots"][name] = {
            "pos": pose["position_m"],
            "rpy": [value * 3.141592653589793 / 180.0
                    for value in pose["rpy_deg"]],
            "qpos": robot["initial_q"],
        }
    gripper_name = project["robots"]["A"]["gripper"]
    if project["robots"]["B"]["gripper"] != gripper_name:
        raise ValueError("current scene compiler requires the same gripper asset on A and B")
    cfg["gripper"]["tcp_pos"] = (
        project["grippers"][gripper_name]["mount_to_tcp"]["position_m"])
    cfg["assets"]["workcell_visual_source"] = resolve_project_asset(
        project["workstation"]["visual_cad"], project_path)
    cfg["assets"]["workcell_collision"] = resolve_project_asset(
        project["workstation"]["collision_cad"], project_path)
    return cfg, project, joints


def gp7_arm(prefix: str, robot_cfg: dict, gripper_cfg: dict, joints: dict,
            gripper_visual_count: int, gripper_component_count: int) -> str:
    lines = [
        f'<body name="{prefix}_base" pos="{" ".join(map(str, robot_cfg["pos"]))}" '
        f'euler="{" ".join(map(str, robot_cfg["rpy"]))}">',
        f'  <geom name="{prefix}_base_visual" type="mesh" mesh="gp7_base_visual" '
        'class="robot_visual"/>',
        f'  <geom name="{prefix}_base_collision" type="mesh" mesh="gp7_base_collision" '
        'class="robot_collision"/>',
    ]
    indent = "  "
    for index, short in enumerate(JOINT_SHORT, start=1):
        uj = joints[f"joint_{index}_{short}"]
        origin = uj.find("origin")
        axis = uj.find("axis")
        limit = uj.find("limit")
        xyz = vec(origin.attrib.get("xyz") if origin is not None else None)
        rpy = vec(origin.attrib.get("rpy") if origin is not None else None)
        lines.append(f'{indent}<body name="{prefix}_link_{index}" pos="{xyz}" euler="{rpy}">')
        indent += "  "
        lines.append(
            f'{indent}<joint name="{prefix}_{short}" type="hinge" '
            f'axis="{vec(axis.attrib.get("xyz"))}" '
            f'range="{limit.attrib["lower"]} {limit.attrib["upper"]}" '
            'damping="2" armature="0.02"/>'
        )
        lines.append(
            f'{indent}<geom name="{prefix}_link_{index}_visual" type="mesh" '
            f'mesh="gp7_link_{index}_visual" class="robot_visual"/>'
        )
        lines.append(
            f'{indent}<geom name="{prefix}_link_{index}_collision" type="mesh" '
            f'mesh="gp7_link_{index}_collision" class="robot_collision"/>'
        )

    # tool0 is exactly the fixed URDF transform from link_6_t.
    tool_joint = joints["joint_6_t-tool0"]
    tool_origin = tool_joint.find("origin")
    lines.append(
        f'{indent}<body name="{prefix}_tool0" pos="{vec(tool_origin.attrib.get("xyz"))}" '
        f'euler="{vec(tool_origin.attrib.get("rpy"))}">'
    )
    indent += "  "
    lines.append(f'{indent}<site name="{prefix}_flange" type="cylinder" size="0.008 0.002" rgba="1 0.2 0.1 1"/>')
    # gp7.urdf defines the supplied gripper frame as +90 deg about tool0 Y;
    # its local +Z is therefore the flange/tool +X direction.
    lines.append(f'{indent}<body name="{prefix}_gripper" euler="0 1.57079632679 0">')
    lines.append(f'{indent}  <geom name="{prefix}_gripper_visual" type="mesh" mesh="gripper_visual" class="gripper_visual"/>')
    for visual in range(1, gripper_visual_count):
        lines.append(
            f'{indent}  <geom name="{prefix}_gripper_visual_{visual:02d}" '
            f'type="mesh" mesh="gripper_visual_{visual:02d}" '
            f'class="gripper_visual"/>'
        )
    for component in range(gripper_component_count):
        lines.append(
            f'{indent}  <geom name="{prefix}_gripper_collision_{component:02d}" '
            f'type="mesh" mesh="gripper_collision_{component:02d}" '
            f'class="gripper_collision"/>'
        )
    tcp = " ".join(str(value) for value in gripper_cfg["tcp_pos"])
    lines.append(f'{indent}  <site name="{prefix}_tcp" pos="{tcp}" size="0.006" rgba="0.1 1 0.2 1"/>')
    lines.append(f'{indent}</body>')
    indent = indent[:-2]
    lines.append(f"{indent}</body>")
    for _ in JOINT_SHORT:
        indent = indent[:-2]
        lines.append(f"{indent}</body>")
    # Fixed base body enclosing the six moving link bodies.
    lines.append("</body>")
    return "\n".join("    " + line for line in lines)


def workcell_collision(cfg: dict) -> str:
    path = cfg["assets"]["workcell_collision"]
    with open(path, encoding="utf-8") as f:
        boxes = yaml.safe_load(f)
    lines = []
    for i, item in enumerate(boxes.get("pedestals", []) + boxes.get("boxes", [])):
        pos = [v * 0.001 for v in item["center"]]
        size = [v * 0.001 for v in item["half_extents"]]
        lines.append(
            f'    <geom name="cell_collision_{i:02d}" type="box" '
            f'pos="{" ".join(f"{v:.6f}" for v in pos)}" '
            f'size="{" ".join(f"{v:.6f}" for v in size)}" class="cell_collision"/>'
        )
    return "\n".join(lines)


def workstation_collision_assets(
    cfg: dict,
    project: dict,
    *,
    output_path: str,
    generated_cad: str,
) -> tuple[list[str], list[str]]:
    """Compile the primary workstation collision source.

    YAML retains the surveyed primitive-box representation. Mesh CAD is
    normalized/chunked by :func:`prepare_cad`; MuJoCo then collides against
    each mesh chunk's convex hull, not its concave visual triangles.
    """
    source = cfg["assets"]["workcell_collision"]
    suffix = os.path.splitext(source)[1].lower()
    if suffix in (".yaml", ".yml"):
        return [], [workcell_collision(cfg)]
    if suffix not in (".stl", ".obj", ".step", ".stp"):
        raise ValueError(
            "workstation.collision_cad must be collision-box YAML or STL/OBJ/STEP CAD"
        )
    workstation = project["workstation"]
    static_assembly = bool(workstation.get(
        "collision_cad_static_assembly", True))
    prepared = prepare_cad(
        source, generated_cad,
        units=workstation.get("collision_cad_units"),
        scale_to_m=workstation.get("collision_cad_scale_to_m"),
        role="collision-source", static_assembly=static_assembly)
    scale = " ".join(str(value) for value in
                     prepared.metadata["source"]["scale_to_m"])
    pose = workstation.get("collision_cad_world_pose", {})
    position, orientation = fixed_pose_xml(pose)
    collision_chunks = [
        chunk
        for component in prepared.metadata.get(
            "static_assembly", {}).get("components", [])
        for chunk in component["chunks"]
    ]
    if not collision_chunks:
        collision_chunks = prepared.metadata["visual"]["chunks"]
    assets, geoms = [], []
    for index, chunk in enumerate(collision_chunks):
        mesh_name = f"workstation_collision_mesh_{index:02d}"
        relative = os.path.relpath(
            prepared.artifact_dir / chunk["path"], os.path.dirname(output_path))
        assets.append(
            f'    <mesh name="{mesh_name}" file="{relative}" scale="{scale}"/>')
        geoms.append(
            f'    <geom name="workstation_collision_{index:02d}" type="mesh" '
            f'mesh="{mesh_name}" pos="{position}" '
            f'{orientation} class="cell_collision"/>')
    print(
        "warning: workstation collision mesh uses MuJoCo convex hull contact; "
        "supply convex-decomposed components when concave clearance matters"
    )
    return assets, geoms


def fixture_xml(cfg: dict, *, include_pcb_placeholder: bool = True) -> str:
    """Photo-matched tables and task staging geometry."""
    f = cfg["fixtures"]
    floor = float(f["floor_z"])
    top_z = floor + float(f["table_height"])
    lines = ['    <body name="fixtures">']

    def table(name, spec):
        cx, cy = spec["center_xy"]
        sx, sy = spec["size_xy"]
        thickness = spec["top_thickness"]
        leg = spec["leg_size"]
        lines.append(
            f'      <geom name="{name}_top" type="box" pos="{cx} {cy} {top_z-thickness/2}" '
            f'size="{sx/2} {sy/2} {thickness/2}" material="wood" class="fixture_collision"/>'
        )
        leg_half_z = (top_z - thickness - floor) / 2
        leg_z = floor + leg_half_z
        inset = leg / 2
        for ix, xsign in enumerate((-1, 1)):
            for iy, ysign in enumerate((-1, 1)):
                x = cx + xsign * (sx / 2 - inset)
                y = cy + ysign * (sy / 2 - inset)
                lines.append(
                    f'      <geom name="{name}_leg_{ix}_{iy}" type="box" pos="{x} {y} {leg_z}" '
                    f'size="{leg/2} {leg/2} {leg_half_z}" material="black_steel" class="fixture_collision"/>'
                )
        # Black perimeter rail visible in the laboratory tables.
        rail = 0.018
        rail_z = top_z - thickness - rail / 2
        lines.extend([
            f'      <geom type="box" pos="{cx} {cy-sy/2+rail/2} {rail_z}" size="{sx/2} {rail/2} {rail/2}" material="black_steel" class="fixture_collision"/>',
            f'      <geom type="box" pos="{cx} {cy+sy/2-rail/2} {rail_z}" size="{sx/2} {rail/2} {rail/2}" material="black_steel" class="fixture_collision"/>',
        ])

    table("supply_table", f["supply_table"])
    table("pcb_table", f["pcb_table"])

    # Open rectangular bins: thin bottom plus four walls. These are stable
    # contact primitives; replace their visual layer when measured bowl CAD is available.
    for bin_spec in f["bins"]:
        name = bin_spec["name"]
        cx, cy = bin_spec["center_xy"]
        ix, iy = bin_spec["interior_xy"]
        wall = bin_spec["wall_thickness"]
        height = bin_spec["wall_height"]
        bottom = 0.006
        lines.append(f'      <geom name="{name}_bottom" type="box" pos="{cx} {cy} {top_z+bottom/2}" size="{ix/2+wall} {iy/2+wall} {bottom/2}" material="bin_gray" class="fixture_collision"/>')
        wall_z = top_z + bottom + height / 2
        lines.extend([
            f'      <geom type="box" pos="{cx-ix/2-wall/2} {cy} {wall_z}" size="{wall/2} {iy/2+wall} {height/2}" material="bin_gray" class="fixture_collision"/>',
            f'      <geom type="box" pos="{cx+ix/2+wall/2} {cy} {wall_z}" size="{wall/2} {iy/2+wall} {height/2}" material="bin_gray" class="fixture_collision"/>',
            f'      <geom type="box" pos="{cx} {cy-iy/2-wall/2} {wall_z}" size="{ix/2} {wall/2} {height/2}" material="bin_gray" class="fixture_collision"/>',
            f'      <geom type="box" pos="{cx} {cy+iy/2+wall/2} {wall_z}" size="{ix/2} {wall/2} {height/2}" material="bin_gray" class="fixture_collision"/>',
        ])
        lines.append(f'      <site name="{name}_center" pos="{cx} {cy} {top_z+bottom}" size="0.008" rgba="0.9 0.5 0.1 1"/>')

    plate = f["reorientation_surface"]
    cx, cy = plate["center_xy"]
    sx, sy = plate["size_xy"]
    thickness = plate["thickness"]
    lines.append(f'      <geom name="reorientation_surface" type="box" pos="{cx} {cy} {top_z+thickness/2}" size="{sx/2} {sy/2} {thickness/2}" material="reorient" class="fixture_collision"/>')
    lines.append(f'      <site name="reorientation_origin" pos="{cx} {cy} {top_z+thickness}" size="0.008" rgba="1 0.2 0.8 1"/>')

    pcb = f["pcb_fixture"]
    cx, cy = pcb["center_xy"]
    bx, by = pcb["base_size_xy"]
    base_t = pcb["base_thickness"]
    px, py = pcb["board_size_xy"]
    pcb_t = pcb["board_thickness"]
    if include_pcb_placeholder:
        aperture = pcb.get("aperture_size_xy")
        if aperture is None:
            lines.append(f'      <geom name="pcb_fixture_base" type="box" pos="{cx} {cy} {top_z+base_t/2}" size="{bx/2} {by/2} {base_t/2}" material="fixture_aluminum" class="fixture_collision"/>')
            lines.append(f'      <geom name="pcb_board" type="box" pos="{cx} {cy} {top_z+base_t+pcb_t/2}" size="{px/2} {py/2} {pcb_t/2}" material="pcb_green" class="fixture_collision"/>')
        else:
            ax, ay = (float(value) for value in aperture)
            if not (0.0 < ax < min(bx, px) and 0.0 < ay < min(by, py)):
                raise ValueError(
                    "fixtures.pcb_fixture.aperture_size_xy must be positive "
                    "and smaller than both the board and fixture base")

            def aperture_ring(name, sx, sy, z, thickness, material):
                side_x = 0.5 * (sx - ax)
                side_y = 0.5 * (sy - ay)
                x_offset = 0.25 * (sx + ax)
                y_offset = 0.25 * (sy + ay)
                lines.extend([
                    f'      <geom name="{name}_left" type="box" pos="{cx-x_offset} {cy} {z}" size="{side_x/2} {sy/2} {thickness/2}" material="{material}" class="fixture_collision"/>',
                    f'      <geom name="{name}_right" type="box" pos="{cx+x_offset} {cy} {z}" size="{side_x/2} {sy/2} {thickness/2}" material="{material}" class="fixture_collision"/>',
                    f'      <geom name="{name}_front" type="box" pos="{cx} {cy-y_offset} {z}" size="{ax/2} {side_y/2} {thickness/2}" material="{material}" class="fixture_collision"/>',
                    f'      <geom name="{name}_back" type="box" pos="{cx} {cy+y_offset} {z}" size="{ax/2} {side_y/2} {thickness/2}" material="{material}" class="fixture_collision"/>',
                ])

            aperture_ring(
                "pcb_fixture_base", bx, by, top_z + base_t / 2,
                base_t, "fixture_aluminum")
            aperture_ring(
                "pcb_board", px, py, top_z + base_t + pcb_t / 2,
                pcb_t, "pcb_green")
    lines.append(f'      <site name="pcb_origin" pos="{cx} {cy} {top_z+base_t+pcb_t}" size="0.008" rgba="0.1 1 0.2 1"/>')
    lines.append("    </body>")
    return "\n".join(lines)


def part_xml(project: dict, visual_count: int, collision_count: int,
             collision_mesh_prefix: str = "active_part_mesh") -> str:
    name = project["active_task"]["part"]
    if name not in project["parts"]:
        raise KeyError(f"active part {name!r} is not in project.yaml parts registry")
    part = project["parts"][name]
    rgba = " ".join(str(v) for v in part.get("rgba", [0.9, 0.8, 0.5, 1]))
    extra_visual = "\n".join(
        f'      <geom name="part_visual_{index:02d}" type="mesh" '
        f'mesh="active_part_mesh_{index:02d}" contype="0" conaffinity="0" '
        f'group="1" rgba="{rgba}" mass="0"/>'
        for index in range(1, visual_count))
    extra_collision = "\n".join(
        f'      <geom name="part_collision_{index:02d}" type="mesh" '
        f'mesh="{collision_mesh_prefix}_{index:02d}" group="2" rgba="0 0 0 0" '
        f'mass="0" friction="0.8 0.01 0.001"/>'
        for index in range(1, collision_count))
    return f'''    <body name="part" pos="0.425 0 0.65">
      <freejoint name="part_free"/>
      <geom name="part_visual" type="mesh" mesh="active_part_mesh" contype="0" conaffinity="0" group="1" rgba="{rgba}" mass="0"/>
      <geom name="part_collision" type="mesh" mesh="{collision_mesh_prefix}" group="2" rgba="0 0 0 0" mass="{part.get("mass_kg", 0.01)}" friction="0.8 0.01 0.001"/>
{extra_visual}
{extra_collision}
      <site name="part_origin" size="0.0025" rgba="1 0.2 0.1 1"/>
    </body>'''


def additional_collision_assets(
    project: dict,
    *,
    project_path: str,
    output_path: str,
    generated_cad: str,
) -> tuple[list[str], list[str]]:
    """Compile optional fixed collision CAD declared by the user manifest."""
    workstation = project["workstation"]
    default_units = workstation.get("additional_collision_cad_units")
    assets, geoms = [], []
    declared = list(workstation.get("additional_collision_cad", []))
    insertion = project.get("insertion", {})
    if insertion.get("collision_cad"):
        collision_pose = insertion.get(
            "collision_cad_world_pose", insertion.get("pcb_world_pose"))
        if collision_pose is None:
            raise ValueError(
                "explicit insertion collision CAD requires "
                "insertion.collision_cad_world_pose"
            )
        declared.append({
            "cad": insertion["collision_cad"],
            "units": insertion.get("collision_cad_units"),
            "scale_to_m": insertion.get("collision_cad_scale_to_m"),
            "world_pose": collision_pose,
            "static_assembly": bool(insertion.get(
                "collision_cad_static_assembly", True)),
            # Semantic names let collision audits and future pin/hole contact
            # policies distinguish the insertion fixture from unrelated cell
            # obstacles without relying on declaration order.
            "_mesh_name": "insertion_collision_mesh",
            "_geom_name": "insertion_collision",
        })
    for index, raw in enumerate(declared):
        item = {"cad": raw} if isinstance(raw, str) else dict(raw)
        path = item.get("cad", item.get("path"))
        if not path:
            raise ValueError(f"additional_collision_cad[{index}] has no cad/path")
        static_assembly = bool(item.get("static_assembly", True))
        prepared = prepare_cad(
            resolve_project_asset(path, project_path), generated_cad,
            units=item.get("units", default_units),
            scale_to_m=item.get("scale_to_m"),
            role="collision-source", static_assembly=static_assembly)
        scale = " ".join(str(value) for value in
                         prepared.metadata["source"]["scale_to_m"])
        pose = item.get("world_pose", {})
        position, orientation = fixed_pose_xml(pose)
        collision_chunks = [
            chunk
            for component in prepared.metadata.get(
                "static_assembly", {}).get("components", [])
            for chunk in component["chunks"]
        ]
        if not collision_chunks:
            collision_chunks = prepared.metadata["visual"]["chunks"]
        mesh_base = item.get(
            "_mesh_name", f"additional_collision_mesh_{index:02d}")
        geom_base = item.get(
            "_geom_name", f"additional_collision_{index:02d}")
        for chunk_index, chunk in enumerate(collision_chunks):
            suffix = "" if chunk_index == 0 else f"_{chunk_index:02d}"
            mesh_name = f"{mesh_base}{suffix}"
            geom_name = f"{geom_base}{suffix}"
            relative = os.path.relpath(
                prepared.artifact_dir / chunk["path"], os.path.dirname(output_path))
            assets.append(
                f'    <mesh name="{mesh_name}" file="{relative}" scale="{scale}"/>')
            geoms.append(
                f'    <geom name="{geom_name}" '
                f'type="mesh" mesh="{mesh_name}" '
                f'pos="{position}" {orientation} '
                f'class="fixture_collision"/>')
    return assets, geoms


def build_scene(
    project_path: str = PROJECT_CONFIG,
    output_path: str = OUTPUT,
) -> str:
    """Build one deterministic MJCF from the selected project and output path."""
    project_path = os.path.realpath(project_path)
    # Canonicalize symlinked parents (notably macOS /var -> /private/var)
    # before computing relative MJCF references.
    output_path = os.path.realpath(output_path)
    generated_cad = os.path.join(os.path.dirname(output_path), "generated_cad")
    cfg, project, joints = load_sources(project_path)
    robot_urdf = resolve_project_asset(
        next(iter({item["model"] for item in project["robots"].values()})), project_path)
    robot_mesh_root = os.path.join(os.path.dirname(robot_urdf), "meshes")
    visual_dir = os.path.relpath(os.path.join(robot_mesh_root, "visual"), os.path.dirname(output_path))
    collision_dir = os.path.relpath(os.path.join(robot_mesh_root, "collision"), os.path.dirname(output_path))
    workcell_source = cfg["assets"]["workcell_visual_source"]
    workcell_prepared = prepare_cad(
        workcell_source, generated_cad,
        units=project["workstation"].get("visual_cad_units"),
        scale_to_m=project["workstation"].get("visual_cad_scale_to_m"),
        role="workstation-visual", static_assembly=False)
    workcell_chunks = [workcell_prepared.artifact_dir / item["path"]
                       for item in workcell_prepared.metadata["visual"]["chunks"]]
    workcell_scale = " ".join(str(v) for v in
                              workcell_prepared.metadata["source"]["scale_to_m"])
    gripper_name = project["robots"]["A"]["gripper"]
    gripper_spec = project["grippers"][gripper_name]
    gripper_source = resolve_project_asset(gripper_spec["model"], project_path)
    gripper_prepared = prepare_cad(
        gripper_source, generated_cad,
        units=gripper_spec.get("model_units"),
        scale_to_m=gripper_spec.get("model_scale_to_m"),
        role="gripper-visual",
        static_assembly=bool(gripper_spec.get("model_static_assembly", False)))
    gripper_visuals = [gripper_prepared.artifact_dir / item["path"]
                       for item in gripper_prepared.metadata["visual"]["chunks"]]
    gripper_components = [
        gripper_prepared.artifact_dir / chunk["path"]
        for component in gripper_prepared.metadata.get(
            "static_assembly", {}).get("components", [])
        for chunk in component["chunks"]]
    if not gripper_components:
        gripper_components = list(gripper_visuals)
    active_name = project["active_task"]["part"]
    active_part = project["parts"][active_name]
    part_prepared = prepare_cad(
        resolve_project_asset(active_part["cad"], project_path), generated_cad,
        units=active_part.get("cad_units"),
        scale_to_m=active_part.get("cad_scale_to_m"),
        role="part-visual", static_assembly=False)
    part_visuals = [part_prepared.artifact_dir / item["path"]
                    for item in part_prepared.metadata["visual"]["chunks"]]
    active_part_scale = " ".join(str(v) for v in
                                 part_prepared.metadata["source"]["scale_to_m"])
    part_collision_source = active_part.get("collision_cad")
    if part_collision_source:
        part_collision_prepared = prepare_cad(
            resolve_project_asset(part_collision_source, project_path),
            generated_cad,
            units=active_part.get("collision_cad_units"),
            scale_to_m=active_part.get("collision_cad_scale_to_m"),
            role="collision-source",
            static_assembly=bool(active_part.get(
                "collision_cad_static_assembly", True)),
        )
        part_collision_chunks = [
            part_collision_prepared.artifact_dir / chunk["path"]
            for component in part_collision_prepared.metadata.get(
                "static_assembly", {}).get("components", [])
            for chunk in component["chunks"]
        ]
        if not part_collision_chunks:
            part_collision_chunks = [
                part_collision_prepared.artifact_dir / chunk["path"]
                for chunk in part_collision_prepared.metadata["visual"]["chunks"]
            ]
        part_collision_scale = " ".join(
            str(value) for value in
            part_collision_prepared.metadata["source"]["scale_to_m"])
        part_collision_prefix = "active_part_collision_mesh"
    else:
        part_collision_chunks = list(part_visuals)
        part_collision_scale = active_part_scale
        part_collision_prefix = "active_part_mesh"
    gripper_scale = " ".join(str(v) for v in
                             gripper_prepared.metadata["source"]["scale_to_m"])

    meshes = [
        f'    <mesh name="gp7_base_visual" file="{visual_dir}/base_link.stl"/>',
        f'    <mesh name="gp7_base_collision" file="{collision_dir}/base_link.stl"/>',
        f'    <mesh name="gripper_visual" file="{os.path.relpath(gripper_visuals[0], os.path.dirname(output_path))}" scale="{gripper_scale}"/>',
        f'    <mesh name="active_part_mesh" file="{os.path.relpath(part_visuals[0], os.path.dirname(output_path))}" scale="{active_part_scale}"/>',
    ]
    for index, path in enumerate(gripper_visuals[1:], start=1):
        meshes.append(
            f'    <mesh name="gripper_visual_{index:02d}" '
            f'file="{os.path.relpath(path, os.path.dirname(output_path))}" '
            f'scale="{gripper_scale}"/>')
    for index, path in enumerate(part_visuals[1:], start=1):
        meshes.append(
            f'    <mesh name="active_part_mesh_{index:02d}" '
            f'file="{os.path.relpath(path, os.path.dirname(output_path))}" '
            f'scale="{active_part_scale}"/>')
    if part_collision_source:
        for index, path in enumerate(part_collision_chunks):
            suffix = "" if index == 0 else f"_{index:02d}"
            meshes.append(
                f'    <mesh name="active_part_collision_mesh{suffix}" '
                f'file="{os.path.relpath(path, os.path.dirname(output_path))}" '
                f'scale="{part_collision_scale}"/>')
    workstation_collision_meshes, workstation_collision_geoms = (
        workstation_collision_assets(
            cfg, project, output_path=output_path, generated_cad=generated_cad)
    )
    meshes.extend(workstation_collision_meshes)
    extra_assets, extra_geoms = additional_collision_assets(
        project, project_path=project_path, output_path=output_path,
        generated_cad=generated_cad)
    meshes.extend(extra_assets)
    for index, path in enumerate(workcell_chunks):
        relative = os.path.relpath(path, os.path.dirname(output_path))
        meshes.append(f'    <mesh name="workcell_visual_{index}" file="{relative}" scale="{workcell_scale}"/>')
    for index, path in enumerate(gripper_components):
        relative = os.path.relpath(path, os.path.dirname(output_path))
        meshes.append(
            f'    <mesh name="gripper_collision_{index:02d}" file="{relative}" '
            f'scale="{gripper_scale}"/>'
        )
    for i, name in enumerate(("link_1_s", "link_2_l", "link_3_u", "link_4_r", "link_5_b", "link_6_t"), start=1):
        meshes.append(f'    <mesh name="gp7_link_{i}_visual" file="{visual_dir}/{name}.stl"/>')
        meshes.append(f'    <mesh name="gp7_link_{i}_collision" file="{collision_dir}/{name}.stl"/>')

    qpos = cfg["robots"]["A"]["qpos"] + cfg["robots"]["B"]["qpos"]
    ctrl_text = " ".join(str(v) for v in qpos)
    # Robot joints followed by the active part free-joint pose.
    qtext = " ".join(str(v) for v in qpos + [0.425, 0.0, 0.65, 1.0, 0.0, 0.0, 0.0])
    actuators = []
    for prefix in ("A", "B"):
        for short in JOINT_SHORT:
            actuators.append(f'    <position name="{prefix}_{short}_act" joint="{prefix}_{short}" kp="250" kv="30"/>')

    fixtures = (fixture_xml(
        cfg,
        # A declared PCB/hole model replaces only the solid PCB placeholder;
        # the photo-matched tables, bins, and reorientation surface can remain.
        include_pcb_placeholder=not bool(
            project.get("insertion", {}).get("collision_cad")),
    ) if project["workstation"].get(
        "generated_fixture_primitives", False) else "")

    xml = f'''<mujoco model="gp7_real_cell_foundation">
  <compiler angle="radian" autolimits="true" balanceinertia="true"/>
  <option timestep="0.001" integrator="implicitfast" gravity="0 0 -9.81"/>
  <visual>
    <global offwidth="1600" offheight="1000"/>
    <quality shadowsize="4096"/>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.75 0.75 0.75" specular="0.15 0.15 0.15"/>
    <rgba haze="0.15 0.18 0.22 1"/>
  </visual>
  <statistic center="0.425 -0.175 0.50" extent="1.65"/>
  <default>
    <default class="robot_visual">
      <geom contype="0" conaffinity="0" group="1" material="yaskawa_blue" mass="0"/>
    </default>
    <default class="robot_collision">
      <geom contype="1" conaffinity="1" group="3" rgba="0.1 0.5 0.9 0" density="500"/>
    </default>
    <default class="gripper_visual">
      <geom contype="0" conaffinity="0" group="1" material="gripper_dark" mass="0"/>
    </default>
    <default class="gripper_collision">
      <geom contype="1" conaffinity="1" group="4" rgba="0.95 0.2 0.1 0" mass="0"/>
    </default>
    <default class="cell_collision">
      <geom contype="1" conaffinity="1" group="3" rgba="0.8 0.2 0.2 0" friction="0.8 0.01 0.001"/>
    </default>
    <default class="fixture_collision">
      <geom contype="1" conaffinity="1" group="2" friction="0.8 0.01 0.001"/>
    </default>
  </default>
  <asset>
{chr(10).join(meshes)}
    <material name="yaskawa_blue" rgba="0.015 0.24 0.68 1" specular="0.35" shininess="0.45"/>
    <material name="gripper_dark" rgba="0.08 0.09 0.10 1" specular="0.35" shininess="0.40"/>
    <material name="aluminum" rgba="0.72 0.75 0.78 1" specular="0.25" shininess="0.35"/>
    <material name="floor" rgba="0.19 0.20 0.21 1"/>
    <material name="wood" rgba="0.25 0.13 0.055 1" specular="0.12" shininess="0.2"/>
    <material name="black_steel" rgba="0.035 0.04 0.045 1" specular="0.25" shininess="0.35"/>
    <material name="bin_gray" rgba="0.42 0.45 0.48 1"/>
    <material name="reorient" rgba="0.72 0.68 0.55 1"/>
    <material name="fixture_aluminum" rgba="0.40 0.43 0.46 1" specular="0.35"/>
    <material name="pcb_green" rgba="0.04 0.35 0.13 1" specular="0.15"/>
  </asset>
  <worldbody>
    <light name="key" pos="0.4 -0.2 2.6" dir="0 0 -1" directional="false" castshadow="true"/>
    <geom name="floor" type="plane" pos="0 0 -0.610" size="4 4 0.05" material="floor"/>
{chr(10).join(f'    <geom name="workcell_visual_{i}" type="mesh" mesh="workcell_visual_{i}" material="aluminum" contype="0" conaffinity="0" group="1"/>' for i in range(len(workcell_chunks)))}
{chr(10).join(extra_geoms)}
{chr(10).join(workstation_collision_geoms)}
{fixtures}
{gp7_arm("A", cfg["robots"]["A"], cfg["gripper"], joints, len(gripper_visuals), len(gripper_components))}
{gp7_arm("B", cfg["robots"]["B"], cfg["gripper"], joints, len(gripper_visuals), len(gripper_components))}
{part_xml(project, len(part_visuals), len(part_collision_chunks), part_collision_prefix)}
  </worldbody>
  <equality>
    <weld name="A_part_grasp" body1="A_gripper" body2="part" active="false"/>
    <weld name="B_part_grasp" body1="B_gripper" body2="part" active="false"/>
  </equality>
  <actuator>
{chr(10).join(actuators)}
  </actuator>
  <keyframe>
    <key name="inspection" qpos="{qtext}" ctrl="{ctrl_text}"/>
  </keyframe>
</mujoco>
'''
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(xml)
    print(f"wrote {output_path}")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=PROJECT_CONFIG,
                        help="project.yaml manifest used to compile the scene")
    parser.add_argument("--output", default=OUTPUT,
                        help="output MJCF; exact generated CAD is stored beside it")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    build_scene(args.project, args.output)


if __name__ == "__main__":
    main()
