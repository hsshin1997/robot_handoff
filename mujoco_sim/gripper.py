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
import xml.etree.ElementTree as ET

import numpy as np


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


def _float_list(text: str | None, expected: int | None = None) -> list[float]:
    values = [] if text is None else [float(item) for item in text.split()]
    if expected is not None and len(values) != expected:
        raise ValueError(f"expected {expected} values, got {text!r}")
    return values


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
    "GripperInspection",
    "ParallelJawActuation",
    "command_aperture",
    "inspect_gripper_model",
]
