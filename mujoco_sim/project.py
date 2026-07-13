"""Validated minimal project manifest and feature-frame task compiler.

The manifest contains physical facts and task semantics only.  Search
resolution, collision tolerances, grasp samples, and other solver choices live
in :mod:`solver_defaults.yaml` and may therefore be improved without asking a
cell integrator to retune every part.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import numpy as np
import yaml

from .gripper import inspect_gripper_model
from .se3 import inverse, make_transform, transform_from_rpy, validate_transform

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_PROJECT = os.path.join(HERE, "project.yaml")
DEFAULT_SOLVER = os.path.join(HERE, "solver_defaults.yaml")


def _pose(value: dict) -> np.ndarray:
    if "matrix" in value:
        return validate_transform(np.asarray(value["matrix"], dtype=float))
    if "rotation_matrix" in value:
        return make_transform(np.asarray(value["rotation_matrix"], dtype=float),
                              value["position_m"])
    return transform_from_rpy(value["position_m"], np.radians(value["rpy_deg"]))


@dataclass(frozen=True)
class GripperCapability:
    name: str
    model_path: str
    kind: str
    T_mount_tcp: np.ndarray
    opening_min: float
    opening_max: float
    finger_depth: float
    pad_size: np.ndarray
    articulated: bool


@dataclass(frozen=True)
class BoxRegion:
    center: np.ndarray
    size: np.ndarray

    def contains(self, point) -> bool:
        tolerance = 32.0 * np.finfo(float).eps * max(1.0, float(np.max(self.size)))
        return bool(np.all(np.abs(np.asarray(point) - self.center)
                           <= self.size / 2 + tolerance))

    def deterministic_samples(self, counts=(5, 5, 5)) -> np.ndarray:
        axes = [np.linspace(c - s / 2, c + s / 2, n)
                for c, s, n in zip(self.center, self.size, counts)]
        return np.array(np.meshgrid(*axes, indexing="ij")).reshape(3, -1).T


@dataclass(frozen=True)
class SupportRegion:
    """A rectangular planar support expressed by ``T_W_N`` and local size."""

    T_W_N: np.ndarray
    size: np.ndarray

    def contains_local_xy(self, points, margin: float = 0.0) -> bool:
        points = np.asarray(points, dtype=float)
        if points.shape[-1] != 2:
            raise ValueError("support points must end in an xy pair")
        half = self.size / 2.0 - float(margin)
        return bool(np.all(half >= 0.0) and np.all(np.abs(points) <= half))


@dataclass(frozen=True)
class InsertionTarget:
    """Exact part target plus the frame that defines insertion corrections.

    ``insertion_axis_world`` points *into* the hole.  The pre-insertion part
    pose is offset in the opposite direction, irrespective of world/PCB
    orientation.  ``T_W_I`` is the user-supplied insertion frame in explicit
    mode and the derived hole frame in legacy feature-frame mode.  Its X/Y
    axes define lateral corrections and its +Z axis defines insertion.
    """

    name: str
    T_W_I: np.ndarray
    T_W_P_insert: np.ndarray
    T_W_P_preinsert: np.ndarray
    insertion_axis_world: np.ndarray

    @property
    def T_W_H(self) -> np.ndarray:
        """Backward-compatible alias for the insertion/correction frame."""
        return self.T_W_I


class Project:
    def __init__(self, manifest_path: str = DEFAULT_PROJECT,
                 solver_path: str = DEFAULT_SOLVER):
        self.manifest_path = os.path.realpath(manifest_path)
        with open(self.manifest_path, encoding="utf-8") as f:
            self.manifest = yaml.safe_load(f)
        with open(solver_path, encoding="utf-8") as f:
            self.solver = yaml.safe_load(f)
        if self.manifest.get("schema_version") != 1:
            raise ValueError("unsupported project schema_version")
        if self.solver.get("schema_version") != 1:
            raise ValueError("unsupported solver schema_version")
        self._validate_assets()
        self._validate_qualification_domain()
        self._validate_insertion_schema()
        # Compile immediately so malformed poses and targets outside the
        # declared insertion volume fail at project-load time.
        self.insertion_targets()

    def resolve_asset(self, path: str) -> str:
        """Resolve repository-relative or project-local manifest assets."""
        if os.path.isabs(path):
            return os.path.realpath(path)
        project_dir = os.path.dirname(self.manifest_path)
        try:
            inside_repository = os.path.commonpath(
                (self.manifest_path, ROOT)) == ROOT
        except ValueError:
            inside_repository = False
        candidates = ([os.path.join(ROOT, path), os.path.join(project_dir, path)]
                      if inside_repository else
                      [os.path.join(project_dir, path), os.path.join(ROOT, path)])
        for candidate in candidates:
            if os.path.exists(candidate):
                return os.path.realpath(candidate)
        # Return the preferred candidate so the validation error remains
        # deterministic and names the location that follows project semantics.
        return os.path.realpath(candidates[0])

    def _validate_assets(self):
        paths = []
        paths += [robot["model"] for robot in self.manifest["robots"].values()]
        paths += [gripper["model"] for gripper in self.manifest["grippers"].values()]
        paths += [self.manifest["workstation"]["visual_cad"],
                  self.manifest["workstation"]["collision_cad"]]
        for part in self.manifest["parts"].values():
            paths.append(part["cad"])
            if part.get("collision_cad"):
                paths.append(part["collision_cad"])
        for item in self.manifest["workstation"].get("additional_collision_cad", []):
            paths.append(item if isinstance(item, str)
                         else item.get("cad", item.get("path")))
        insertion_collision = self.manifest.get("insertion", {}).get("collision_cad")
        if insertion_collision:
            paths.append(insertion_collision)
        if any(path is None for path in paths):
            raise ValueError("a declared project CAD asset has no path")
        missing = [path for path in paths
                   if not os.path.exists(self.resolve_asset(path))]
        if missing:
            raise FileNotFoundError(f"project assets not found: {missing}")

    def _validate_qualification_domain(self) -> None:
        declaration = self.manifest.get("qualification", {}).get(
            "initial_grasp_domain")
        if not isinstance(declaration, dict):
            raise ValueError(
                "project must explicitly declare qualification.initial_grasp_domain"
            )
        supported = {
            "known_start",
            "known_start_plus_geometry_library",
        }
        source = declaration.get("source")
        if source not in supported:
            raise ValueError(
                "qualification.initial_grasp_domain.source must be one of "
                f"{sorted(supported)}, got {source!r}"
            )

    @staticmethod
    def _validate_named_entries(entries: Any, *, label: str,
                                required_poses: tuple[str, ...]) -> None:
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"{label} must be a non-empty list")
        names: set[str] = set()
        for index, entry in enumerate(entries):
            context = f"{label}[{index}]"
            if not isinstance(entry, dict):
                raise ValueError(f"{context} must be a mapping")
            name = entry.get("name")
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"{context}.name must be a non-empty string")
            if name in names:
                raise ValueError(f"duplicate insertion target name {name!r}")
            names.add(name)
            for pose_name in required_poses:
                if pose_name not in entry:
                    raise ValueError(f"{context} requires {pose_name}")
                try:
                    _pose(entry[pose_name])
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(
                        f"{context}.{pose_name} is not a valid SE(3) pose: {error}"
                    ) from error

    def _validate_insertion_schema(self) -> None:
        insertion = self.manifest.get("insertion")
        if not isinstance(insertion, dict):
            raise ValueError("project must define an insertion mapping")

        explicit = "targets" in insertion
        feature_frame = "holes" in insertion
        if explicit and feature_frame:
            raise ValueError(
                "insertion is ambiguous: define either explicit targets or "
                "legacy holes, not both"
            )
        if not explicit and not feature_frame:
            raise ValueError(
                "insertion must define either explicit targets or legacy holes"
            )

        if insertion.get("collision_cad"):
            collision_pose = insertion.get(
                "collision_cad_world_pose", insertion.get("pcb_world_pose"))
            if collision_pose is None:
                raise ValueError(
                    "explicit insertion collision CAD requires "
                    "insertion.collision_cad_world_pose"
                )
            try:
                _pose(collision_pose)
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    "insertion.collision_cad_world_pose is not a valid SE(3) "
                    f"pose: {error}"
                ) from error

        if "insertion" not in self.manifest.get("regions", {}):
            raise ValueError(
                "regions.insertion is required to bound insert and pre-insert poses"
            )
        # Validate that the declared region is a well-formed box before target
        # compilation tries to use it.
        insertion_region = self.region("insertion")
        if (insertion_region.center.shape != (3,) or
                insertion_region.size.shape != (3,) or
                not np.all(np.isfinite(insertion_region.center)) or
                not np.all(np.isfinite(insertion_region.size)) or
                np.any(insertion_region.size <= 0.0)):
            raise ValueError(
                "regions.insertion center and size must be finite three-vectors; "
                "size must be positive"
            )

        if explicit:
            self._validate_named_entries(
                insertion["targets"], label="insertion.targets",
                required_poses=("world_part_pose", "world_insertion_frame"),
            )
            return

        if "pcb_world_pose" not in insertion:
            raise ValueError(
                "legacy feature-frame insertion requires insertion.pcb_world_pose"
            )
        try:
            _pose(insertion["pcb_world_pose"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"insertion.pcb_world_pose is not a valid SE(3) pose: {error}"
            ) from error
        if "part_to_pin" not in self.active_part:
            raise ValueError(
                "legacy feature-frame insertion requires parts.<active>.part_to_pin"
            )
        try:
            _pose(self.active_part["part_to_pin"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"active part_to_pin is not a valid SE(3) pose: {error}"
            ) from error
        self._validate_named_entries(
            insertion["holes"], label="insertion.holes",
            required_poses=("pcb_to_hole",),
        )

    @property
    def initial_grasp_domain_source(self) -> str:
        """Return the explicitly declared finite qualification-domain source."""
        return str(self.manifest["qualification"]["initial_grasp_domain"]["source"])

    def gripper(self, robot: str) -> GripperCapability:
        name = self.manifest["robots"][robot]["gripper"]
        item = self.manifest["grippers"][name]
        capability = item["manufacturer_capabilities"]
        model_path = self.resolve_asset(item["model"])
        inspection = inspect_gripper_model(model_path)
        articulated = inspection.articulated
        opening = (inspection.actuation.aperture_range if articulated else
                   capability["opening_range_m"])
        return GripperCapability(
            name, model_path, item["kind"], _pose(item["mount_to_tcp"]),
            float(opening[0]), float(opening[1]), float(capability["finger_depth_m"]),
            np.asarray(capability["pad_size_m"], dtype=float), articulated)

    @property
    def active_part(self) -> dict[str, Any]:
        return self.manifest["parts"][self.manifest["active_task"]["part"]]

    @property
    def active_part_path(self) -> str:
        return self.resolve_asset(self.active_part["cad"])

    def region(self, name: str) -> BoxRegion:
        item = self.manifest["regions"][name]
        if item["type"] != "box":
            raise TypeError(f"region {name} is not a box")
        return BoxRegion(np.asarray(item["center_m"], float),
                         np.asarray(item["size_m"], float))

    def support_region(self, name: str = "reorientation") -> SupportRegion:
        item = self.manifest["regions"][name]
        if item["type"] != "support_rectangle":
            raise TypeError(f"region {name} is not a support rectangle")
        size = np.asarray(item["size_m"], dtype=float)
        if size.shape != (2,) or np.any(size <= 0.0):
            raise ValueError(f"region {name} size must be two positive values")
        return SupportRegion(_pose(item["world_pose"]), size)

    @property
    def reorientation_pose(self) -> np.ndarray:
        return self.support_region().T_W_N

    @property
    def reorientation_size(self) -> np.ndarray:
        return self.support_region().size

    @property
    def T_part_pin(self) -> np.ndarray:
        if "part_to_pin" not in self.active_part:
            raise ValueError(
                "T_part_pin is only defined by legacy feature-frame projects"
            )
        return _pose(self.active_part["part_to_pin"])

    @property
    def T_world_pcb(self) -> np.ndarray:
        if "pcb_world_pose" not in self.manifest["insertion"]:
            raise ValueError(
                "T_world_pcb is only available when insertion.pcb_world_pose is declared"
            )
        return _pose(self.manifest["insertion"]["pcb_world_pose"])

    @property
    def T_tcp_part_start(self) -> np.ndarray:
        """Known startup ``^E T_P`` supplied by the task instance."""
        return _pose(self.manifest["active_task"]["initial_tcp_to_part"])

    def insertion_part_poses(self) -> list[tuple[str, np.ndarray]]:
        """Return configured ``^W T_P`` insertion poses in target order."""
        return [(target.name, target.T_W_P_insert.copy())
                for target in self.insertion_targets()]

    def insertion_targets(self, approach_distance_m: float | None = None) -> list[InsertionTarget]:
        """Compile exact insertion and pre-insertion poses.

        Explicit targets preserve the declared ``world_part_pose`` exactly.
        Legacy targets first derive that pose by pin/hole feature equality.
        In both modes, ``+Z_I`` points into the hole, so pre-insertion is
        translated by ``-d * +Z_I``.  No world-axis assumption is made.
        """
        if approach_distance_m is None:
            approach_distance_m = float(
                self.solver.get("insertion", {}).get("approach_distance_m", 0.04)
            )
        distance = float(approach_distance_m)
        if not np.isfinite(distance) or distance <= 0.0:
            raise ValueError("approach_distance_m must be positive and finite")
        targets = []
        insertion = self.manifest["insertion"]
        if "targets" in insertion:
            compiled = [
                (
                    item["name"],
                    _pose(item["world_part_pose"]),
                    _pose(item["world_insertion_frame"]),
                )
                for item in insertion["targets"]
            ]
        else:
            T_W_C = self.T_world_pcb
            T_P_pin_inv = inverse(self.T_part_pin)
            compiled = []
            for hole in insertion["holes"]:
                T_W_I = T_W_C @ _pose(hole["pcb_to_hole"])
                compiled.append(
                    (hole["name"], T_W_I @ T_P_pin_inv, T_W_I)
                )

        for name, T_W_P, T_W_I in compiled:
            axis = T_W_I[:3, 2].copy()
            T_W_P_pre = T_W_P.copy()
            T_W_P_pre[:3, 3] -= distance * axis
            targets.append(InsertionTarget(
                name, T_W_I, T_W_P, T_W_P_pre, axis
            ))
        region = self.region("insertion")
        for target in targets:
            if not region.contains(target.T_W_P_insert[:3, 3]):
                raise ValueError(
                    f"insertion target {target.name!r} lies outside regions.insertion"
                )
            if not region.contains(target.T_W_P_preinsert[:3, 3]):
                raise ValueError(
                    f"pre-insertion target {target.name!r} lies outside regions.insertion"
                )
        return targets


__all__ = [
    "BoxRegion",
    "DEFAULT_PROJECT",
    "DEFAULT_SOLVER",
    "GripperCapability",
    "InsertionTarget",
    "Project",
    "SupportRegion",
]
