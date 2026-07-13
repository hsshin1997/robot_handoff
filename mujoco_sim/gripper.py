"""Reusable gripper articulation and aperture semantics.

Task authors do not tune jaw opening per part.  An articulated gripper asset
provides prismatic joint limits once; a grasp candidate supplies its geometric
contact separation, and :class:`ParallelJawActuation` maps that separation to
joint coordinates.  A single STL has no joints or limits, so it is explicitly
reported as a static, non-certifiable fallback instead of inventing motion.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from string import Formatter
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np
import yaml

from .se3 import make_transform, transform_from_rpy, validate_transform


@dataclass(frozen=True)
class FingerJoint:
    name: str
    lower: float
    upper: float
    multiplier: float = 1.0


@dataclass(frozen=True)
class ParallelJawActuation:
    joints: tuple[FingerJoint, ...]
    closed_aperture_m: float = 0.0

    def __post_init__(self):
        if not self.joints:
            raise ValueError("parallel-jaw actuation needs at least one slide joint")
        if not np.isfinite(self.closed_aperture_m):
            raise ValueError("closed aperture must be finite")
        if any(not np.isfinite((joint.lower, joint.upper, joint.multiplier)).all()
               for joint in self.joints):
            raise ValueError("finger joint limits and multipliers must be finite")
        if any(joint.upper <= joint.lower for joint in self.joints):
            raise ValueError("finger joint limits must have positive range")
        if any(joint.multiplier == 0.0 for joint in self.joints):
            raise ValueError("finger joint multiplier cannot be zero")

    @property
    def aperture_range(self) -> tuple[float, float]:
        # Each positive multiplier contributes linearly to symmetric opening.
        minimum = self.closed_aperture_m + sum(
            min(joint.multiplier * joint.lower,
                joint.multiplier * joint.upper) for joint in self.joints)
        maximum = self.closed_aperture_m + sum(
            max(joint.multiplier * joint.lower,
                joint.multiplier * joint.upper) for joint in self.joints)
        return float(minimum), float(maximum)

    def joint_positions(self, aperture_m: float) -> dict[str, float]:
        """Map a requested aperture to coordinated finger joint positions."""
        aperture = float(aperture_m)
        low, high = self.aperture_range
        if not low - 1e-12 <= aperture <= high + 1e-12:
            raise ValueError(f"aperture {aperture} is outside [{low}, {high}]")
        span = high - low
        u = 0.0 if span <= 1e-15 else (aperture - low) / span
        positions = {}
        for joint in self.joints:
            low_q, high_q = ((joint.lower, joint.upper)
                             if joint.multiplier > 0 else
                             (joint.upper, joint.lower))
            positions[joint.name] = float((1.0 - u) * low_q + u * high_q)
        return positions


@dataclass(frozen=True)
class GripperInspection:
    source: str
    articulated: bool
    actuation: ParallelJawActuation | None
    contact_geometries: tuple[str, ...]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class GripperAssetContract:
    """Validated source-model semantics for one articulated gripper.

    The contract deliberately stops at the scene-adapter boundary.  It proves
    that the source MJCF/URDF contains the declared mechanism, frames, and
    named geometry, and it defines the names that the compiled workcell must
    expose.  It does *not* claim that merely changing a model path imports a
    kinematic subtree into the generated scene.

    ``T_F_M`` maps the gripper model's mount frame into the robot flange frame;
    ``T_M_E`` maps the TCP into the mount frame.  The resulting flange-to-TCP
    transform is ``T_F_M @ T_M_E``.  All source names are mapped into a scene
    namespace using ``scene_name_template`` (normally ``{robot}_{name}``).
    """

    descriptor_path: str
    model_path: str
    model_format: str
    mount_frame: str
    tcp_frame: str
    T_F_M: np.ndarray
    T_M_E: np.ndarray
    actuation: ParallelJawActuation
    pad_geometries: tuple[str, ...]
    collision_geometries: tuple[str, ...]
    visual_geometries: tuple[str, ...]
    scene_name_template: str = "{robot}_{name}"

    @property
    def T_F_E(self) -> np.ndarray:
        return self.T_F_M @ self.T_M_E

    def scene_name(self, robot: str, source_name: str) -> str:
        if not robot or not source_name:
            raise ValueError("robot and source name must be non-empty")
        return self.scene_name_template.format(robot=robot, name=source_name)

    def scene_actuation(self, robot: str) -> ParallelJawActuation:
        return ParallelJawActuation(
            tuple(FingerJoint(
                self.scene_name(robot, joint.name), joint.lower, joint.upper,
                joint.multiplier,
            ) for joint in self.actuation.joints),
            self.actuation.closed_aperture_m,
        )


@dataclass(frozen=True)
class GripperSceneBinding:
    """Names verified in a compiled MuJoCo model for one robot/gripper pair."""

    robot: str
    mount_body: str
    tcp_site: str
    actuation: ParallelJawActuation
    pad_geometries: tuple[str, ...]
    collision_geometries: tuple[str, ...]
    visual_geometries: tuple[str, ...]


def _float_list(text: str | None, expected: int | None = None) -> list[float]:
    values = [] if text is None else [float(item) for item in text.split()]
    if expected is not None and len(values) != expected:
        raise ValueError(f"expected {expected} values, got {text!r}")
    return values


def _descriptor_pose(value: Any, *, label: str) -> np.ndarray:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a pose mapping")
    try:
        if "matrix" in value:
            return validate_transform(np.asarray(value["matrix"], dtype=float))
        if "rotation_matrix" in value:
            return make_transform(
                np.asarray(value["rotation_matrix"], dtype=float),
                value["position_m"],
            )
        return transform_from_rpy(
            value["position_m"], np.radians(value["rpy_deg"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{label} is not a valid SE(3) pose: {error}") from error


def _required_name(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _unique_names(value: Any, *, label: str, minimum: int = 1) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) < minimum:
        raise ValueError(f"{label} must contain at least {minimum} name(s)")
    names = tuple(_required_name(item, label=f"{label}[]") for item in value)
    if len(set(names)) != len(names):
        raise ValueError(f"{label} contains duplicate names")
    return names


def _validate_name_template(template: Any) -> str:
    template = _required_name(template, label="scene_name_template")
    fields = {
        field for _, field, _, _ in Formatter().parse(template)
        if field is not None
    }
    if fields != {"robot", "name"}:
        raise ValueError(
            "scene_name_template must contain exactly {robot} and {name}"
        )
    try:
        first = template.format(robot="A", name="left")
        second = template.format(robot="B", name="left")
    except (KeyError, ValueError) as error:
        raise ValueError(f"invalid scene_name_template: {error}") from error
    if not first or first == second:
        raise ValueError("scene_name_template must namespace different robots")
    return template


def _named_elements(root: ET.Element, tag: str) -> dict[str, ET.Element]:
    result: dict[str, ET.Element] = {}
    for element in root.findall(f".//{tag}"):
        name = element.get("name")
        if not name:
            continue
        if name in result:
            raise ValueError(f"source model has duplicate {tag} name {name!r}")
        result[name] = element
    return result


def _mjcf_body_ownership(
    root: ET.Element,
) -> tuple[dict[str, ET.Element], dict[str, str], dict[str, str]]:
    bodies: dict[str, ET.Element] = {}
    geom_owner: dict[str, str] = {}
    joint_owner: dict[str, str] = {}

    def visit(body: ET.Element) -> None:
        body_name = body.get("name")
        if not body_name:
            raise ValueError("every articulated MJCF body must have a name")
        if body_name in bodies:
            raise ValueError(f"source model has duplicate body name {body_name!r}")
        bodies[body_name] = body
        for tag, destination in (("geom", geom_owner), ("joint", joint_owner)):
            for element in body.findall(tag):
                name = element.get("name")
                if name:
                    destination[name] = body_name
        for child in body.findall("body"):
            visit(child)

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("articulated MJCF requires worldbody")
    for body in worldbody.findall("body"):
        visit(body)
    return bodies, geom_owner, joint_owner


def _mjcf_descendant_bodies(body: ET.Element) -> set[str]:
    return {
        child.get("name") for child in body.iter("body") if child.get("name")
    }


def _source_joint(
    element: ET.Element,
    *,
    model_format: str,
    explicit_range: Any,
    label: str,
) -> tuple[float, float]:
    if model_format == "mjcf":
        if element.get("type", "hinge") != "slide":
            raise ValueError(f"{label} must name an MJCF slide joint")
        source_range = (_float_list(element.get("range"), 2)
                        if element.get("range") is not None else None)
    else:
        if element.get("type") != "prismatic":
            raise ValueError(f"{label} must name a URDF prismatic joint")
        limit = element.find("limit")
        source_range = (None if limit is None else [
            float(limit.get("lower", "nan")), float(limit.get("upper", "nan")),
        ])
    declared = (None if explicit_range is None else
                np.asarray(explicit_range, dtype=float))
    if declared is not None and declared.shape != (2,):
        raise ValueError(f"{label}.range_m must contain [lower, upper]")
    if source_range is None and declared is None:
        raise ValueError(
            f"{label} has no source-model limits; declare range_m explicitly"
        )
    if source_range is not None:
        source = np.asarray(source_range, dtype=float)
        if source.shape != (2,) or not np.all(np.isfinite(source)):
            raise ValueError(f"{label} has invalid source-model limits")
        if declared is not None and not np.allclose(
                declared, source, rtol=0.0, atol=1e-12):
            raise ValueError(
                f"{label}.range_m {declared.tolist()} contradicts source "
                f"limits {source.tolist()}"
            )
        declared = source
    assert declared is not None
    lower, upper = map(float, declared)
    if not np.isfinite([lower, upper]).all() or upper <= lower:
        raise ValueError(f"{label} limits must be finite with upper > lower")
    return lower, upper


def load_gripper_asset_contract(
    descriptor_path: str | os.PathLike[str],
) -> GripperAssetContract:
    """Load and validate an explicit articulated-gripper YAML descriptor.

    Validation is intentionally strict: contact pads must be actual named
    collision geometry under the configured moving-finger subtrees, all named
    geometry must exist, and descriptor joint limits may not contradict the
    source MJCF/URDF.  This catches the integration errors that otherwise show
    up later as false collision-free grasps or motionless fingers.
    """
    descriptor = Path(descriptor_path).resolve()
    with open(descriptor, encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError("gripper descriptor root must be a mapping")
    if raw.get("schema_version") != 1:
        raise ValueError("unsupported gripper descriptor schema_version")
    model = raw.get("model")
    if not isinstance(model, dict):
        raise ValueError("gripper descriptor requires a model mapping")
    model_path_value = _required_name(model.get("path"), label="model.path")
    model_path = Path(model_path_value).expanduser()
    if not model_path.is_absolute():
        model_path = descriptor.parent / model_path
    model_path = model_path.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"gripper source model does not exist: {model_path}")
    model_format = _required_name(model.get("format"), label="model.format").lower()
    if model_format not in ("mjcf", "urdf"):
        raise ValueError("model.format must be 'mjcf' or 'urdf'")

    source_root = ET.parse(model_path).getroot()
    expected_root = "mujoco" if model_format == "mjcf" else "robot"
    if source_root.tag != expected_root:
        raise ValueError(
            f"model.format={model_format!r} expects XML root {expected_root!r}, "
            f"got {source_root.tag!r}"
        )

    frames = raw.get("frames")
    if not isinstance(frames, dict):
        raise ValueError("gripper descriptor requires a frames mapping")
    mount_frame = _required_name(frames.get("mount"), label="frames.mount")
    tcp_frame = _required_name(frames.get("tcp"), label="frames.tcp")
    T_F_M = _descriptor_pose(frames.get("flange_to_mount"),
                             label="frames.flange_to_mount")
    T_M_E = _descriptor_pose(frames.get("mount_to_tcp"),
                             label="frames.mount_to_tcp")

    geometry = raw.get("geometry")
    if not isinstance(geometry, dict):
        raise ValueError("gripper descriptor requires a geometry mapping")
    pads = _unique_names(geometry.get("pad_collisions"),
                         label="geometry.pad_collisions", minimum=2)
    collisions = _unique_names(geometry.get("collisions"),
                               label="geometry.collisions")
    visuals = _unique_names(geometry.get("visuals"),
                            label="geometry.visuals")
    if not set(pads).issubset(collisions):
        missing = sorted(set(pads) - set(collisions))
        raise ValueError(
            f"pad collisions must also appear in geometry.collisions: {missing}"
        )

    actuation = raw.get("actuation")
    if not isinstance(actuation, dict) or actuation.get("type") != "parallel_jaw":
        raise ValueError("actuation.type must be 'parallel_jaw'")
    joint_specs = actuation.get("joints")
    if not isinstance(joint_specs, list) or not joint_specs:
        raise ValueError("actuation.joints must be a non-empty list")

    joints_by_name = _named_elements(source_root, "joint")
    if model_format == "mjcf":
        geoms_by_name = _named_elements(source_root, "geom")
        for name in (*pads, *collisions, *visuals):
            if name not in geoms_by_name:
                raise ValueError(
                    f"declared geometry {name!r} is absent from source model"
                )
    else:
        collision_elements = _named_elements(source_root, "collision")
        visual_elements = _named_elements(source_root, "visual")
        overlap = set(collision_elements) & set(visual_elements)
        if overlap:
            raise ValueError(
                "URDF collision and visual element names must be globally "
                f"distinct: {sorted(overlap)}"
            )
        for name in (*pads, *collisions):
            if name not in collision_elements:
                raise ValueError(
                    f"declared collision geometry {name!r} is absent from "
                    "source model collision elements"
                )
        for name in visuals:
            if name not in visual_elements:
                raise ValueError(
                    f"declared visual geometry {name!r} is absent from "
                    "source model visual elements"
                )

    finger_joints: list[FingerJoint] = []
    seen_joints: set[str] = set()
    for index, spec in enumerate(joint_specs):
        label = f"actuation.joints[{index}]"
        if not isinstance(spec, dict):
            raise ValueError(f"{label} must be a mapping")
        name = _required_name(spec.get("name"), label=f"{label}.name")
        if name in seen_joints:
            raise ValueError(f"duplicate actuation joint {name!r}")
        seen_joints.add(name)
        if name not in joints_by_name:
            raise ValueError(f"{label} names absent source joint {name!r}")
        lower, upper = _source_joint(
            joints_by_name[name], model_format=model_format,
            explicit_range=spec.get("range_m"), label=label,
        )
        multiplier = spec.get("aperture_multiplier")
        if multiplier is None:
            if model_format == "mjcf":
                user = _float_list(joints_by_name[name].get("user"))
                multiplier = user[0] if user else 1.0
            else:
                mimic = joints_by_name[name].find("mimic")
                multiplier = (float(mimic.get("multiplier", "1"))
                              if mimic is not None else 1.0)
        finger_joints.append(FingerJoint(name, lower, upper, float(multiplier)))

    if model_format == "mjcf":
        bodies, geom_owner, joint_owner = _mjcf_body_ownership(source_root)
        sites = _named_elements(source_root, "site")
        if mount_frame not in bodies:
            raise ValueError(f"frames.mount body {mount_frame!r} is absent")
        if tcp_frame not in sites:
            raise ValueError(f"frames.tcp site {tcp_frame!r} is absent")
        allowed_bodies = _mjcf_descendant_bodies(bodies[mount_frame])
        mount_sites = {
            site.get("name") for site in bodies[mount_frame].iter("site")
            if site.get("name")
        }
        if tcp_frame not in mount_sites:
            raise ValueError(
                f"frames.tcp site {tcp_frame!r} is outside mount subtree "
                f"{mount_frame!r}"
            )
        for name in (*pads, *collisions, *visuals):
            if geom_owner.get(name) not in allowed_bodies:
                raise ValueError(
                    f"geometry {name!r} is outside mount subtree {mount_frame!r}"
                )
        for joint in finger_joints:
            owner = joint_owner.get(joint.name)
            subtree = (_mjcf_descendant_bodies(bodies[owner])
                       if owner in bodies else set())
            if not any(geom_owner.get(pad) in subtree for pad in pads):
                raise ValueError(
                    f"moving joint {joint.name!r} has no declared pad collision "
                    "in its moving subtree"
                )
    else:
        links = _named_elements(source_root, "link")
        if mount_frame not in links:
            raise ValueError(f"frames.mount link {mount_frame!r} is absent")
        if tcp_frame not in links:
            raise ValueError(f"frames.tcp link {tcp_frame!r} is absent")
        geom_owner: dict[str, str] = {}
        for link_name, link in links.items():
            for tag in ("collision", "visual"):
                for item in link.findall(tag):
                    if item.get("name"):
                        geom_owner[item.get("name")] = link_name
        children: dict[str, set[str]] = {name: set() for name in links}
        joint_child: dict[str, str] = {}
        for name, joint in joints_by_name.items():
            parent, child = joint.find("parent"), joint.find("child")
            if parent is not None and child is not None:
                parent_name, child_name = parent.get("link"), child.get("link")
                if parent_name in children and child_name:
                    children[parent_name].add(child_name)
                    joint_child[name] = child_name

        def descendants(link: str) -> set[str]:
            result, pending = set(), [link]
            while pending:
                item = pending.pop()
                if item in result:
                    continue
                result.add(item)
                pending.extend(children.get(item, ()))
            return result

        mount_subtree = descendants(mount_frame)
        for name in (*pads, *collisions, *visuals):
            if geom_owner.get(name) not in mount_subtree:
                raise ValueError(
                    f"geometry {name!r} is outside mount subtree {mount_frame!r}"
                )
        for joint in finger_joints:
            child_link = joint_child.get(joint.name)
            subtree = descendants(child_link) if child_link else set()
            if not any(geom_owner.get(pad) in subtree for pad in pads):
                raise ValueError(
                    f"moving joint {joint.name!r} has no declared pad collision "
                    "in its moving subtree"
                )

    contract = GripperAssetContract(
        str(descriptor), str(model_path), model_format, mount_frame, tcp_frame,
        T_F_M, T_M_E,
        ParallelJawActuation(
            tuple(finger_joints),
            float(actuation.get("closed_aperture_m", 0.0)),
        ),
        pads, collisions, visuals,
        _validate_name_template(raw.get(
            "scene_name_template", "{robot}_{name}")),
    )
    # Force evaluation so a nonsensical closed offset is caught here rather
    # than in a planner process much later.
    low, high = contract.actuation.aperture_range
    if low < -1e-12 or high <= low:
        raise ValueError(
            f"resolved aperture range [{low}, {high}] must be non-negative "
            "with positive span"
        )
    return contract


def bind_gripper_scene(
    model: Any,
    robot: str,
    contract: GripperAssetContract,
) -> GripperSceneBinding:
    """Verify the articulated-gripper adapter contract in compiled MuJoCo.

    This is a post-compilation gate.  A future importer must clone/prefix the
    source subtree and assets, attach it at the flange using ``T_F_M``, expose
    the TCP site, preserve every named collision/visual geom, and install
    actuators/equalities as needed.  This function fails immediately when any
    required scene object is missing.
    """
    expected = {
        "body": (contract.scene_name(robot, contract.mount_frame),),
        "site": (contract.scene_name(robot, contract.tcp_frame),),
        "joint": tuple(contract.scene_name(robot, joint.name)
                       for joint in contract.actuation.joints),
        "geom": tuple(contract.scene_name(robot, name) for name in (
            *contract.pad_geometries,
            *contract.collision_geometries,
            *contract.visual_geometries,
        )),
    }
    accessors = {
        "body": model.body,
        "site": model.site,
        "joint": model.joint,
        "geom": model.geom,
    }
    missing: list[str] = []
    for kind, names in expected.items():
        for name in dict.fromkeys(names):
            try:
                accessors[kind](name)
            except KeyError:
                missing.append(f"{kind}:{name}")
    if missing:
        raise RuntimeError(
            "compiled scene does not satisfy articulated gripper contract; "
            f"missing {', '.join(missing)}"
        )
    return GripperSceneBinding(
        robot,
        expected["body"][0],
        expected["site"][0],
        contract.scene_actuation(robot),
        tuple(contract.scene_name(robot, name) for name in contract.pad_geometries),
        tuple(contract.scene_name(robot, name)
              for name in contract.collision_geometries),
        tuple(contract.scene_name(robot, name)
              for name in contract.visual_geometries),
    )


def inspect_gripper_model(path: str) -> GripperInspection:
    """Discover parallel-jaw semantics from MJCF or URDF joint limits.

    MJCF slide joints may declare ``user=\"<aperture multiplier>\"``; absent
    metadata defaults to ``1`` per slide.  URDF mimic multipliers are honored.
    When a model contains more mechanisms than a simple parallel jaw, an asset
    descriptor should explicitly identify the finger joints rather than rely
    on this conservative discovery pass.
    """
    source = os.path.abspath(path)
    extension = os.path.splitext(source)[1].lower()
    if extension in (".stl", ".obj", ".step", ".stp"):
        return GripperInspection(
            source, False, None, (),
            ("surface CAD contains no kinematic joints; provide MJCF/URDF or an asset descriptor",),
        )
    root = ET.parse(source).getroot()
    joints: list[FingerJoint] = []
    contacts: list[str] = []
    if root.tag == "mujoco":
        for element in root.findall(".//joint"):
            if element.get("type", "hinge") != "slide" or not element.get("name"):
                continue
            limits = _float_list(element.get("range"), 2)
            multiplier = _float_list(element.get("user"))
            joints.append(FingerJoint(element.get("name"), limits[0], limits[1],
                                      multiplier[0] if multiplier else 1.0))
        contacts = [geom.get("name") for geom in root.findall(".//geom")
                    if geom.get("name") and any(token in geom.get("name").lower()
                                                for token in ("pad", "finger"))]
    elif root.tag == "robot":
        mimic = {}
        for element in root.findall("joint"):
            child = element.find("mimic")
            if child is not None:
                mimic[element.get("name")] = float(child.get("multiplier", "1"))
        for element in root.findall("joint"):
            if element.get("type") != "prismatic" or not element.get("name"):
                continue
            limit = element.find("limit")
            if limit is None:
                continue
            joints.append(FingerJoint(
                element.get("name"), float(limit.get("lower", "0")),
                float(limit.get("upper", "0")), mimic.get(element.get("name"), 1.0)))
        contacts = [link.get("name") for link in root.findall("link")
                    if link.get("name") and any(token in link.get("name").lower()
                                                for token in ("pad", "finger"))]
    else:
        raise ValueError(f"unsupported articulated gripper XML root {root.tag!r}")
    if not joints:
        return GripperInspection(source, False, None, tuple(contacts),
                                 ("no prismatic/slide finger joints were found",))
    return GripperInspection(source, True, ParallelJawActuation(tuple(joints)),
                             tuple(contacts))


def command_aperture(sim, robot: str, actuation: ParallelJawActuation,
                     aperture_m: float) -> None:
    """Set discovered MuJoCo finger joints; fail if the scene omitted them."""
    positions = actuation.joint_positions(aperture_m)
    for source_name, value in positions.items():
        name = source_name.format(robot=robot)
        try:
            joint = sim.model.joint(name)
        except KeyError as error:
            raise RuntimeError(f"scene is missing articulated gripper joint {name!r}") from error
        sim.data.qpos[int(joint.qposadr[0])] = value
    import mujoco
    mujoco.mj_forward(sim.model, sim.data)


__all__ = [
    "FingerJoint",
    "GripperAssetContract",
    "GripperInspection",
    "GripperSceneBinding",
    "ParallelJawActuation",
    "bind_gripper_scene",
    "command_aperture",
    "inspect_gripper_model",
    "load_gripper_asset_contract",
]
