"""Downstream-constrained two-GP7 handoff planner.

Internal frame contract follows the detailed pipeline document exactly:

* part pose ``X`` is ``^W T_P``;
* grasp ``g`` is ``^P T_E``;
* required TCP pose is therefore ``^W T_E = X @ g``.

The project manifest stores the known startup ``^E T_P`` once and it is
inverted at the pipeline boundary. Candidate feasibility is branch-resolved:
every plan stores the actual IK witnesses and checked trajectory samples used
to score it.
"""
from __future__ import annotations

import itertools
import os
import time
from collections import Counter
from dataclasses import dataclass, field, replace

import numpy as np

from .collision import CollisionPolicy, SceneCollisionChecker
from .geometry_grasps import (GraspCandidate, ParallelJawGripper,
                              generate_antipodal_grasps)
from .kinematics import GP7Kinematics
from .offline import (ArtifactCache, ArtifactCategory, fingerprint_file,
                      fingerprint_content, make_artifact_key)
from .motion_planning import MotionPlannerConfig
from .part_mesh import load_project_part_mesh
from .placements import (RectangularStage, generate_stable_placements,
                         instantiate_on_rectangular_stage)
from .project import DEFAULT_PROJECT, Project
from .qualification import physical_prerequisites
from .reachability import ReachabilityMap
from .se3 import (inverse, make_transform, so3_exp, so3_geodesic,
                  transform_from_rpy, validate_transform)
from .sim import WorkcellSim
from .task_graph import (DirectCoGraspEdge, InitialGraspClass,
                         PlacementGraspEdge, TaskGraph)
from .uncertainty import check_axis_aligned_capture, combine_independent

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REORIENTATION_CONTACTS = (
    ("part_collision", "reorientation_surface", 0.00075),
    # Fingers must approach the support plane to reach a resting part. Permit
    # positive margin proximity only; any actual gripper/table penetration is
    # still rejected because max penetration is zero.
    ("A_gripper_collision_*", "reorientation_surface", 0.0),
)


def _allowed_contact_cache_key(values) -> tuple:
    normalized = []
    for entry in values:
        if len(entry) not in (2, 3):
            raise ValueError("allowed contact entries need two or three values")
        normalized.append(tuple(entry))
    return tuple(sorted(normalized, key=repr))


def _normalized_placement_robustness(
    support_margin: float,
    edge_clearance: float,
    part_scale: float,
    stage_scale: float,
) -> tuple[float, float, float]:
    """Return scale-invariant support, stage, and bottleneck robustness.

    A support or stage clearance cannot usefully exceed half its associated
    characteristic length.  Dividing by those upper scales gives dimensionless
    scores in ``[0, 1]`` while preserving comparisons across differently sized
    parts and cells.  The task graph consumes their bottleneck because both
    quasistatic support and stage containment must remain robust.
    """
    values = {
        "support_margin": float(support_margin),
        "edge_clearance": float(edge_clearance),
        "part_scale": float(part_scale),
        "stage_scale": float(stage_scale),
    }
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError("placement robustness inputs must be finite")
    if values["support_margin"] < 0.0 or values["edge_clearance"] < 0.0:
        raise ValueError("placement clearances must be non-negative")
    if values["part_scale"] <= 0.0 or values["stage_scale"] <= 0.0:
        raise ValueError("placement characteristic scales must be positive")
    support = float(np.clip(
        2.0 * values["support_margin"] / values["part_scale"], 0.0, 1.0))
    stage = float(np.clip(
        2.0 * values["edge_clearance"] / values["stage_scale"], 0.0, 1.0))
    return support, stage, min(support, stage)


def _compile_solver_config(solver: dict) -> dict:
    """Compile system-owned defaults into the planner's internal policy shape."""
    planning = solver["planning"]
    uncertainty = dict(planning["uncertainty"])
    return {
        "reachability": {
            "use_cached_maps": True,
            "cache_directory": solver["offline"]["cache_dir"],
            "conservative_fallback_on_miss": True,
        },
        "gates": {
            "joint_limit_margin_deg": planning["joint_limit_margin_deg"],
            "ik_position_tolerance_m": planning["ik_position_tolerance_m"],
            "ik_rotation_tolerance_deg": planning["ik_rotation_tolerance_deg"],
            "ik_restarts": planning["ik_restarts"],
            "ik_max_solutions": planning["ik_max_solutions"],
            "singularity_percentile": planning["singularity_percentile"],
            "minimum_clearance_m": planning["minimum_clearance_m"],
            "calibration_translation_3sigma_m": planning[
                "calibration_translation_3sigma_m"],
            "calibration_rotation_3sigma_deg": planning[
                "calibration_rotation_3sigma_deg"],
            "approach_separation_deg": planning["approach_separation_deg"],
            "uncertainty": uncertainty,
        },
        "downstream": {
            "correction": dict(planning["correction"]),
            "wrist_dither_deg": planning["wrist_dither_deg"],
        },
        "handoff_search": {
            "prehandoff_distance_m": planning["prehandoff_distance_m"],
            "retreat_distance_m": planning["retreat_distance_m"],
            "interpolation_steps": 8,
            "max_candidates": planning["handoff_candidate_limit"],
        },
        "score_weights": dict(planning["score_weights"]),
        "regrasp": {"enabled": bool(planning["reorientation_enabled"])},
        "execution": dict(solver["execution"]),
    }


def _scene_fingerprint(project: Project) -> str:
    """Fingerprint collision semantics, independent of generated XML layout."""
    robot_files = {name: fingerprint_file(project.resolve_asset(item["model"]))
                   for name, item in project.manifest["robots"].items()}
    workstation = project.manifest["workstation"]
    additional = {}
    for index, item in enumerate(workstation.get("additional_collision_cad", [])):
        path = item if isinstance(item, str) else item.get("cad", item.get("path"))
        additional[str(index)] = fingerprint_file(project.resolve_asset(path))
    insertion_path = project.manifest.get("insertion", {}).get("collision_cad")
    return fingerprint_content({
        "compiler_version": "project-scene-v4-exact-gripper-components",
        "robots": project.manifest["robots"],
        "robot_files": robot_files,
        "grippers": project.manifest["grippers"],
        "workstation": workstation,
        "workstation_collision": fingerprint_file(
            project.resolve_asset(workstation["collision_cad"])),
        "additional_collision": additional,
        "insertion_collision": (None if not insertion_path else
                                fingerprint_file(project.resolve_asset(insertion_path))),
        "internal_fixture_fallback": fingerprint_file(
            os.path.join(HERE, "scene_config.yaml")),
        "active_part": project.manifest["active_task"]["part"],
    })


def _planning_manifest(project: Project) -> dict:
    """Return policy-relevant project data, excluding audit-only declarations."""
    return {key: value for key, value in project.manifest.items()
            if key != "qualification"}

@dataclass
class DownstreamWitness:
    grasp_name: str
    grasp: np.ndarray
    q_scanner: np.ndarray
    q_preinsert: list[np.ndarray]
    q_insert: list[np.ndarray]
    correction_solutions: list[list[np.ndarray]]
    trajectories: dict[str, list[np.ndarray]]
    quality: float
    sigma_min: float


@dataclass
class ScoreBreakdown:
    manipulability: float
    joint_margin: float
    clearance: float
    reorientation: float
    cycle: float
    total: float


@dataclass
class DirectHandoffPlan:
    X_handoff: np.ndarray
    g_A: np.ndarray
    grasp_name_B: str
    g_B: np.ndarray
    qA_handoff: np.ndarray
    qB_handoff: np.ndarray
    qA_pre: np.ndarray
    qB_pre: np.ndarray
    qA_retreat: np.ndarray
    downstream: DownstreamWitness
    trajectories: dict[str, list[np.ndarray]]
    score: ScoreBreakdown


@dataclass
class RegraspPlan:
    placement_name: str
    X_place: np.ndarray
    g_A_before: np.ndarray
    g_A_after: np.ndarray
    qA_place: np.ndarray
    qA_repick: np.ndarray
    direct: DirectHandoffPlan
    trajectories: dict[str, list[np.ndarray]]


@dataclass(frozen=True)
class StablePlacementWitness:
    """Cached geometric certificate used by the reorientation task graph."""

    name: str
    T_W_P: np.ndarray
    support_margin: float
    support_area: float
    edge_clearance: float
    probability_proxy: float
    minimum_support_margin: float
    part_scale: float
    stage_scale: float
    support_robustness: float
    stage_robustness: float
    robustness: float

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("stable placement name must be non-empty")
        transform = validate_transform(self.T_W_P)
        transform.setflags(write=False)
        object.__setattr__(self, "T_W_P", transform)

        nonnegative = (
            "support_margin", "support_area", "edge_clearance",
            "probability_proxy", "minimum_support_margin",
        )
        positive = ("part_scale", "stage_scale")
        unit_interval = (
            "support_robustness", "stage_robustness", "robustness",
        )
        for field_name in nonnegative + positive + unit_interval:
            value = float(getattr(self, field_name))
            if not np.isfinite(value):
                raise ValueError(f"{field_name} must be finite")
            if field_name in positive and value <= 0.0:
                raise ValueError(f"{field_name} must be positive")
            if field_name not in positive and value < 0.0:
                raise ValueError(f"{field_name} must be non-negative")
            if field_name in unit_interval and value > 1.0:
                raise ValueError(f"{field_name} must not exceed one")
            object.__setattr__(self, field_name, value)
        if self.probability_proxy > 1.0 + 1e-12:
            raise ValueError("probability_proxy must not exceed one")
        expected = _normalized_placement_robustness(
            self.support_margin, self.edge_clearance,
            self.part_scale, self.stage_scale)
        actual = (self.support_robustness, self.stage_robustness,
                  self.robustness)
        if not np.allclose(actual, expected, atol=1e-12, rtol=0.0):
            raise ValueError("cached placement robustness is inconsistent")


@dataclass
class PlanningReport:
    direct: DirectHandoffPlan | None = None
    regrasp: RegraspPlan | None = None
    stats: Counter = field(default_factory=Counter)
    candidates: int = 0
    downstream_grasps: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0
    mathematical_coverage_certified: bool = False
    physical_certified: bool = False
    # Deprecated compatibility alias for physical_certified.
    certified: bool = False
    limitations: tuple[str, ...] = ()
    coverage: dict = field(default_factory=dict)
    mathematical_corrections: tuple[str, ...] = (
        "G1 queries induced TCP poses, not the part origin",
        "part symmetries left-multiply ^P T_E grasps",
        "reorientation compares part poses rather than TCP and part frames",
    )

    @property
    def feasible(self) -> bool:
        return self.direct is not None or self.regrasp is not None


class HandoffPlanner:
    def __init__(self, sim: WorkcellSim,
                 known_start_pose: np.ndarray | None = None,
                 project_path: str = DEFAULT_PROJECT,
                 cache_dir: str | None = None):
        self.sim = sim
        self.kin = GP7Kinematics(sim)
        self.project = Project(project_path)
        # Reuse the exact prepared-CAD directory belonging to the selected
        # compiled MJCF. This matters for alternate projects and for STEP,
        # where a prior explicit FreeCAD preparation must be reusable online.
        generated_cad = os.path.join(
            os.path.dirname(self.sim.model_path), "generated_cad")
        self.part_geometry = load_project_part_mesh(
            self.project, generated_root=generated_cad)
        # One prepared, combined, SI-unit mesh is authoritative for bounds,
        # contact-grasp generation, and stable placement. Do not independently
        # reopen the user CAD in those stages: it may be ASCII/OBJ/STEP,
        # chunked, or expressed in non-SI source units.
        self.part_mesh = self.part_geometry.mesh
        configured_cache = self.project.solver["offline"]["cache_dir"]
        self.cache_dir = os.path.abspath(
            cache_dir if cache_dir is not None else os.path.join(ROOT, configured_cache))
        edge_step = self.project.solver["planning"]["edge_max_joint_step_rad"]
        path_clearance = self.project.solver["planning"][
            "swept_path_clearance_m"]
        self.collision = SceneCollisionChecker(
            sim, self.kin, edge_step, path_clearance)
        self.cfg = _compile_solver_config(self.project.solver)
        self.rng = np.random.default_rng(20260711)
        self._ik_cache: dict[tuple, list] = {}
        self._seed_ik_cache: dict[tuple, list] = {}
        self._motion_cache: dict[tuple, tuple[bool, list[np.ndarray], str]] = {}
        self.q_start = {robot: self.kin.get_q(robot) for robot in ("A", "B")}
        self.X_start = (sim.part_pose() if known_start_pose is None
                        else np.asarray(known_start_pose, dtype=float))
        if self.X_start.shape != (4, 4):
            raise ValueError("known_start_pose must be a 4x4 ^W T_P transform")
        if known_start_pose is not None:
            sim.set_part_world(self.X_start)
        # Known start pose + current A FK determines the measured direct grasp.
        self.g_A_start = inverse(self.X_start) @ self.kin.fk("A", self.q_start["A"])
        self.part_lo, self.part_hi = self._part_bounds()
        self.part_center = 0.5 * (self.part_lo + self.part_hi)
        self.g_B_candidates = self._receiver_grasps()
        self.reachability_maps = {}
        reach_cfg = self.cfg.get("reachability", {})
        if reach_cfg.get("use_cached_maps", True):
            directory = self.cache_dir
            for robot in ("A", "B"):
                path = os.path.join(directory, f"reachability_{robot}.npz")
                if os.path.exists(path):
                    self.reachability_maps[robot] = ReachabilityMap.load(path)
        gates = self.cfg["gates"]
        self.limit_margin = np.radians(gates["joint_limit_margin_deg"])
        self.pos_tol = gates["ik_position_tolerance_m"]
        self.rot_tol = np.radians(gates["ik_rotation_tolerance_deg"])
        self.restarts = gates["ik_restarts"]
        self.max_solutions = gates["ik_max_solutions"]
        self.w_min = {robot: self.kin.calibrate_manipulability(
            robot, gates["singularity_percentile"], samples=180)
                      for robot in ("A", "B")}
        # Calibration samples mutate the shared MjData. A constructor must not
        # leave the live transaction at its last random sample.
        self.kin.set_q("A", self.q_start["A"])
        self.kin.set_q("B", self.q_start["B"])
        self.sim.set_part_world(self.X_start)
        self.sim.grasp_part("A")
        self.steps = self.cfg["handoff_search"]["interpolation_steps"]
        uncertainty = gates["uncertainty"]
        grasp_sigma = np.array(
            [uncertainty["grasp_translation_sigma_m"]] * 3
            + [np.radians(uncertainty["grasp_rotation_sigma_deg"])] * 3)
        calibration_sigma = np.array(
            [uncertainty["robot_to_robot_translation_sigma_m"]] * 3
            + [np.radians(uncertainty["robot_to_robot_rotation_sigma_deg"])] * 3)
        self.capture_region = check_axis_aligned_capture(
            combine_independent(np.diag(grasp_sigma**2),
                                np.diag(calibration_sigma**2)),
            uncertainty["capture_translation_half_width_m"],
            np.radians(uncertainty["capture_rotation_half_width_deg"]))

    def _motion_config(self) -> MotionPlannerConfig:
        cfg = self.project.solver["planning"]
        return MotionPlannerConfig(
            extension_step=0.18,
            edge_max_step=float(cfg["edge_max_joint_step_rad"]),
            max_nodes=int(cfg["rrt_node_budget"]),
            timeout_s=float(cfg["rrt_time_budget_s"]),
            shortcut_attempts=8,
            seed=20260711,
        )

    def _held_path(self, robot, q_from, q_to, other_q, grasp,
                   holders, stats=None, allowed_geom_pairs=()):
        """Direct adaptive edge first, bounded RRT-Connect on failure."""
        key = ("held", robot, np.round(q_from, 9).tobytes(),
               np.round(q_to, 9).tobytes(), np.round(other_q, 9).tobytes(),
               np.round(grasp, 9).tobytes(), tuple(holders),
               _allowed_contact_cache_key(allowed_geom_pairs))
        if key in self._motion_cache:
            if stats is not None:
                stats["G6_motion_cache_hit"] += 1
            ok, path, reason = self._motion_cache[key]
            return ok, [q.copy() for q in path], reason
        ok, path, reason = self.collision.path(
            robot, q_from, q_to, other_q, grasp, self.steps, holders,
            allowed_geom_pairs)
        if ok:
            self._motion_cache[key] = (ok, [q.copy() for q in path], reason)
            return ok, path, reason
        result = self.collision.plan_motion(
            robot, q_from, q_to, other_q,
            grasp_part_tcp=grasp,
            allowed_part_holders=holders,
            allowed_geom_pairs=allowed_geom_pairs,
            config=self._motion_config(),
        )
        if result.success:
            if stats is not None:
                stats["G6_rrt_connected"] += 1
            path = self.collision.execution_waypoints(result.path)
            self._motion_cache[key] = (True, [q.copy() for q in path],
                                       result.reason.value)
            return True, path, result.reason.value
        if stats is not None:
            stats[f"G6_{result.reason.value}"] += 1
        final_reason = f"{reason};rrt:{result.reason.value}"
        self._motion_cache[key] = (False, [], final_reason)
        return False, [], final_reason

    def _fixed_path(self, robot, q_from, q_to, other_q, X_part,
                    holders, stats=None, allowed_geom_pairs=()):
        key = ("fixed", robot, np.round(q_from, 9).tobytes(),
               np.round(q_to, 9).tobytes(), np.round(other_q, 9).tobytes(),
               np.round(X_part, 9).tobytes(), tuple(holders),
               _allowed_contact_cache_key(allowed_geom_pairs))
        if key in self._motion_cache:
            if stats is not None:
                stats["G6_motion_cache_hit"] += 1
            ok, path, reason = self._motion_cache[key]
            return ok, [q.copy() for q in path], reason
        ok, path, reason = self.collision.path_fixed_part(
            robot, q_from, q_to, other_q, X_part, self.steps, holders,
            allowed_geom_pairs)
        if ok:
            self._motion_cache[key] = (ok, [q.copy() for q in path], reason)
            return ok, path, reason
        result = self.collision.plan_motion(
            robot, q_from, q_to, other_q,
            fixed_part_pose=X_part,
            allowed_part_holders=holders,
            allowed_geom_pairs=allowed_geom_pairs,
            config=self._motion_config(),
        )
        if result.success:
            if stats is not None:
                stats["G6_rrt_connected"] += 1
            path = self.collision.execution_waypoints(result.path)
            self._motion_cache[key] = (True, [q.copy() for q in path],
                                       result.reason.value)
            return True, path, result.reason.value
        if stats is not None:
            stats[f"G6_{result.reason.value}"] += 1
        final_reason = f"{reason};rrt:{result.reason.value}"
        self._motion_cache[key] = (False, [], final_reason)
        return False, [], final_reason

    def _part_bounds(self):
        return self.part_mesh.bounds

    @staticmethod
    def _serialize_grasp(candidate: GraspCandidate) -> dict:
        return {
            "T_P_E": candidate.T_P_E,
            "contact_points": candidate.contact_points,
            "contact_normals": candidate.contact_normals,
            "required_opening": candidate.required_opening,
            "approach_direction": candidate.approach_direction,
            "closing_direction": candidate.closing_direction,
            "quality": candidate.quality,
            "antipodal_quality": candidate.antipodal_quality,
            "support_quality": candidate.support_quality,
            "opening_margin": candidate.opening_margin,
            "palm_clearance": candidate.palm_clearance,
        }

    @staticmethod
    def _deserialize_grasp(value: dict) -> GraspCandidate:
        return GraspCandidate(**{key: np.asarray(item, dtype=float)
                                 if key in {"T_P_E", "contact_points",
                                            "contact_normals", "approach_direction",
                                            "closing_direction"} else item
                                 for key, item in value.items()})

    def _receiver_grasps(self):
        """Load or compute contact-pair grasps from part and gripper geometry.

        There are deliberately no part-specific axes, rolls, finger gaps, or
        center-grasp rules here.  The gripper opening range is reusable asset
        metadata (or, with an articulated model, joint-limit metadata).
        """
        capability = self.project.gripper("B")
        defaults = self.project.solver["geometry"]
        gripper = ParallelJawGripper(
            capability.opening_min,
            capability.opening_max,
            tuple(capability.pad_size),
            capability.finger_depth,
        )
        parameters = {
            "surface_samples": int(defaults.get("surface_samples", 800)),
            "approaches_per_pair": int(defaults.get("approaches_per_pair", 4)),
            "max_candidates": int(defaults.get("max_grasp_candidates", 128)),
            "gripper": {
                "opening": [gripper.min_opening, gripper.max_opening],
                "pad_size": list(gripper.pad_size),
                "pad_depth": gripper.pad_depth,
                "friction": gripper.friction_coefficient,
            },
        }
        key = make_artifact_key(
            ArtifactCategory.GRASP,
            f"{self.project.manifest['active_task']['part']}--{capability.name}",
            artifact_version="antipodal-contact-pairs-v2-prepared-si",
            input_fingerprints={
                "part_cad": fingerprint_file(self.project.active_part_path),
                "part_prepared": self.part_geometry.artifact_fingerprint,
                "gripper_cad": fingerprint_file(capability.model_path),
            },
            parameters=parameters,
        )
        cache = ArtifactCache(self.cache_dir)

        def compute():
            generated = generate_antipodal_grasps(
                self.part_mesh, gripper,
                surface_samples=parameters["surface_samples"],
                approaches_per_pair=parameters["approaches_per_pair"],
                max_candidates=parameters["max_candidates"],
            )
            return [self._serialize_grasp(candidate) for candidate in generated]

        candidates = [self._deserialize_grasp(value)
                      for value in cache.get_or_compute(key, compute)]
        if not candidates:
            raise ValueError("part CAD has no antipodal grasp within gripper capability")
        self.grasp_candidates = {
            f"geom_{index:03d}": candidate
            for index, candidate in enumerate(candidates)
        }
        return [(name, candidate.T_P_E)
                for name, candidate in self.grasp_candidates.items()]

    def _pose(self, entry) -> np.ndarray:
        return transform_from_rpy(entry["position_m"], np.radians(entry["rpy_deg"]))

    @property
    def X_scanner(self):
        # Scanner orientation is task-informed by the first insertion target;
        # its position is selected from the user-defined scanner region.
        X = self.project.insertion_targets()[0].T_W_P_insert.copy()
        X[:3, 3] = self.project.region("scanner").center
        return X

    @property
    def insertion_poses(self):
        return [(target.name, target.T_W_P_insert)
                for target in self.project.insertion_targets()]

    def _preinsert_pose(self, X_insert):
        for target in self.project.insertion_targets():
            if np.allclose(target.T_W_P_insert, X_insert, atol=1e-10):
                return target.T_W_P_preinsert.copy()
        raise ValueError("insertion pose is not derived from a configured hole frame")

    def _solutions(self, robot, target, seed=None):
        kwargs = dict(position_tolerance=self.pos_tol,
                      rotation_tolerance=self.rot_tol)
        if seed is not None:
            key = (robot, np.round(np.asarray(target, float), 10).tobytes(),
                   np.round(np.asarray(seed, float), 9).tobytes())
            if key not in self._seed_ik_cache:
                result = self.kin.solve(robot, target, seed=seed, **kwargs)
                self._seed_ik_cache[key] = [] if result is None else [result]
            return self._seed_ik_cache[key]
        # The same A target is shared by every receiver grasp at one handoff
        # pose. Cache exact target solves to avoid repeating dozens of numeric
        # multi-start IK calls inside the nested pair loop.
        key = (robot, np.round(np.asarray(target, float), 10).tobytes(),
               self.restarts, self.max_solutions)
        if key not in self._ik_cache:
            self._ik_cache[key] = self.kin.solutions(
                robot, target, self.restarts, self.max_solutions, self.rng,
                **kwargs)
        return self._ik_cache[key]

    def _config_ok(self, robot, q):
        return (self.kin.within_limits(robot, q, self.limit_margin)
                and self.kin.manipulability(robot, q) >= self.w_min[robot])

    def _reach_lookup(self, robot: str, tcp_pose: np.ndarray) -> bool:
        """Conservative G1 surrogate queried at the *induced TCP pose*.

        The exact G2 IK remains authoritative. This avoids the document's
        incorrect part-origin lookup while a persisted voxel map is built.
        """
        base = self.sim.data.body(f"{robot}_base").xpos
        distance = np.linalg.norm(tcp_pose[:3, 3] - base)
        sphere_ok = 0.12 <= distance <= 0.97
        mapping = self.reachability_maps.get(robot)
        if mapping is None:
            return sphere_ok
        if mapping.query(tcp_pose) > 0:
            return True
        return (sphere_ok and self.cfg["reachability"].get(
            "conservative_fallback_on_miss", True))

    def _correction_vertices(self, X_insert):
        cfg = self.cfg["downstream"]["correction"]
        target = next((item for item in self.project.insertion_targets()
                       if np.allclose(item.T_W_P_insert, X_insert, atol=1e-10)), None)
        if target is None:
            raise ValueError("correction envelope requires a configured hole frame")
        R_W_H = target.T_W_H[:3, :3]
        axis = target.insertion_axis_world
        for dx, dy, dz, yaw in itertools.product(
                (-cfg["lateral_m"], cfg["lateral_m"]),
                (-cfg["lateral_m"], cfg["lateral_m"]),
                (-cfg["axial_m"], cfg["axial_m"]),
                (-cfg["yaw_deg"], cfg["yaw_deg"])):
            X = X_insert.copy()
            X[:3, 3] += R_W_H @ np.array([dx, dy, dz])
            # Left-multiply a world-frame rotation about the physical hole
            # axis rather than an assumed world-Z yaw.
            X[:3, :3] = so3_exp(axis * np.radians(yaw)) @ X[:3, :3]
            yield X

    def _correction_ok(self, grasp, q_insert, X_insert):
        correction = self.cfg["downstream"]["correction"]
        J = self.kin.jacobian("B", q_insert)
        sigma_min = float(np.linalg.svd(J, compute_uv=False)[-1])
        if sigma_min < correction["min_sigma"]:
            return False, [], sigma_min
        solutions = []
        for X_perturb in self._correction_vertices(X_insert):
            target = X_perturb @ grasp
            error = self.kin.pose_error(self.kin.fk("B", q_insert), target)
            # Weighted norm avoids silently mixing metres and radians.
            weighted = error.copy()
            weighted[3:] *= 0.10  # 10 cm characteristic length
            dq = np.linalg.pinv(J) @ error
            if np.linalg.norm(dq) > correction["max_joint_gain"] * max(np.linalg.norm(weighted), 1e-9):
                return False, [], sigma_min
            branch = self._solutions("B", target, seed=q_insert)
            if not branch or not self._config_ok("B", branch[0].q):
                return False, [], sigma_min
            if np.linalg.norm(branch[0].q - q_insert) > 0.75:
                return False, [], sigma_min
            solutions.append(branch[0].q)
        return True, solutions, sigma_min

    @staticmethod
    def _serialize_downstream(witness: DownstreamWitness) -> dict:
        return {
            "grasp_name": witness.grasp_name,
            "grasp": witness.grasp,
            "q_scanner": witness.q_scanner,
            "q_preinsert": witness.q_preinsert,
            "q_insert": witness.q_insert,
            "correction_solutions": witness.correction_solutions,
            "trajectories": witness.trajectories,
            "quality": witness.quality,
            "sigma_min": witness.sigma_min,
        }

    @staticmethod
    def _deserialize_downstream(value: dict) -> DownstreamWitness:
        return DownstreamWitness(
            value["grasp_name"], np.asarray(value["grasp"], float),
            np.asarray(value["q_scanner"], float),
            [np.asarray(q, float) for q in value["q_preinsert"]],
            [np.asarray(q, float) for q in value["q_insert"]],
            [[np.asarray(q, float) for q in group]
             for group in value["correction_solutions"]],
            {name: [np.asarray(q, float) for q in path]
             for name, path in value["trajectories"].items()},
            float(value["quality"]), float(value["sigma_min"]),
        )

    @classmethod
    def _serialize_direct(cls, plan: DirectHandoffPlan | None):
        if plan is None:
            return None
        return {
            "X_handoff": plan.X_handoff,
            "g_A": plan.g_A,
            "grasp_name_B": plan.grasp_name_B,
            "g_B": plan.g_B,
            "qA_handoff": plan.qA_handoff,
            "qB_handoff": plan.qB_handoff,
            "qA_pre": plan.qA_pre,
            "qB_pre": plan.qB_pre,
            "qA_retreat": plan.qA_retreat,
            "downstream": cls._serialize_downstream(plan.downstream),
            "trajectories": plan.trajectories,
            "score": plan.score.__dict__,
        }

    @classmethod
    def _deserialize_direct(cls, value) -> DirectHandoffPlan | None:
        if value is None:
            return None
        arrays = {name: np.asarray(value[name], float) for name in (
            "X_handoff", "g_A", "g_B", "qA_handoff", "qB_handoff",
            "qA_pre", "qB_pre", "qA_retreat")}
        score = ScoreBreakdown(**{name: float(number)
                                  for name, number in value["score"].items()})
        return DirectHandoffPlan(
            arrays["X_handoff"], arrays["g_A"], value["grasp_name_B"],
            arrays["g_B"], arrays["qA_handoff"], arrays["qB_handoff"],
            arrays["qA_pre"], arrays["qB_pre"], arrays["qA_retreat"],
            cls._deserialize_downstream(value["downstream"]),
            {name: [np.asarray(q, float) for q in path]
            for name, path in value["trajectories"].items()}, score)

    @classmethod
    def _serialize_regrasp(cls, plan: RegraspPlan | None):
        if plan is None:
            return None
        return {
            "placement_name": plan.placement_name,
            "X_place": plan.X_place,
            "g_A_before": plan.g_A_before,
            "g_A_after": plan.g_A_after,
            "qA_place": plan.qA_place,
            "qA_repick": plan.qA_repick,
            "direct": cls._serialize_direct(plan.direct),
            "trajectories": plan.trajectories,
        }

    @classmethod
    def _deserialize_regrasp(cls, value) -> RegraspPlan | None:
        if value is None:
            return None
        return RegraspPlan(
            value["placement_name"], np.asarray(value["X_place"], float),
            np.asarray(value["g_A_before"], float),
            np.asarray(value["g_A_after"], float),
            np.asarray(value["qA_place"], float),
            np.asarray(value["qA_repick"], float),
            cls._deserialize_direct(value["direct"]),
            {name: [np.asarray(q, float) for q in path]
             for name, path in value["trajectories"].items()},
        )

    def _compute_downstream(self, stats: Counter | None = None) -> list[DownstreamWitness]:
        stats = stats if stats is not None else Counter()
        output = []
        qA_park = self.q_start["A"]
        dither = np.radians(self.cfg["downstream"]["wrist_dither_deg"])
        for name, grasp in self.g_B_candidates:
            scanner_solutions = self._solutions("B", self.X_scanner @ grasp)
            for scanner in scanner_solutions:
                if not self._config_ok("B", scanner.q):
                    continue
                q_pre, q_ins, corrections, trajectories = [], [], [], {}
                quality_values = [self.kin.penalized_manipulability("B", scanner.q)]
                sigma_values = []
                previous = scanner.q
                feasible = True
                for placement_name, X_insert in self.insertion_poses:
                    X_pre = self._preinsert_pose(X_insert)
                    pre = self._solutions("B", X_pre @ grasp, seed=previous)
                    ins = self._solutions("B", X_insert @ grasp,
                                          seed=pre[0].q if pre else previous)
                    if not pre or not ins or not self._config_ok("B", pre[0].q) or not self._config_ok("B", ins[0].q):
                        feasible = False
                        break
                    lower, upper = self.kin.lower["B"][5], self.kin.upper["B"][5]
                    if not (lower + self.limit_margin + dither <= ins[0].q[5]
                            <= upper - self.limit_margin - dither):
                        feasible = False
                        break
                    ok_corr, corr, sigma = self._correction_ok(grasp, ins[0].q, X_insert)
                    if not ok_corr:
                        feasible = False
                        break
                    ok, transit, _ = self._held_path(
                        "B", previous, pre[0].q, qA_park, grasp, ("B",), stats)
                    ok2, descent, _ = self._held_path(
                        "B", pre[0].q, ins[0].q, qA_park, grasp, ("B",), stats,
                        (("part_collision", "pcb_board"),))
                    if not ok or not ok2:
                        feasible = False
                        break
                    q_pre.append(pre[0].q); q_ins.append(ins[0].q); corrections.append(corr)
                    trajectories[f"scanner_to_{placement_name}_pre"] = transit
                    trajectories[f"{placement_name}_insert"] = descent
                    previous = ins[0].q
                    quality_values.extend((self.kin.penalized_manipulability("B", pre[0].q),
                                           self.kin.penalized_manipulability("B", ins[0].q)))
                    sigma_values.append(sigma)
                if feasible:
                    output.append(DownstreamWitness(name, grasp, scanner.q, q_pre, q_ins,
                                                    corrections, trajectories,
                                                    min(quality_values), min(sigma_values)))
                    break
            if not any(w.grasp_name == name for w in output):
                stats["downstream_rejected"] += 1
        return output

    def filter_downstream(self, stats: Counter | None = None) -> list[DownstreamWitness]:
        """Return insertion-feasible B grasps from a content-addressed cache.

        This factorization removes scanner/insertion IK, correction-envelope,
        collision, and motion planning from the online handoff loop.  The key
        includes the complete scene, task frames, grasp library, and numerical
        policies, so stale feasibility cannot survive a CAD/calibration edit.
        """
        stats = stats if stats is not None else Counter()
        motion_keys = (
            "collision_margin_mesh_fraction", "direct_first",
            "edge_max_joint_step_rad", "ik_max_solutions", "ik_restarts",
            "joint_limit_margin_fraction", "max_reorientation_hops",
            "reorientation_yaw_samples", "rrt_node_budget",
            "rrt_time_budget_s", "swept_path_clearance_m",
        )
        parameters = {
            "grasps": [{"name": name, "T_P_E": grasp}
                       for name, grasp in self.g_B_candidates],
            "scanner": self.X_scanner,
            "insertions": [{"name": name, "T_W_P": pose,
                            "T_W_P_pre": self._preinsert_pose(pose)}
                           for name, pose in self.insertion_poses],
            "gates": self.cfg["gates"],
            "correction": self.cfg["downstream"]["correction"],
            "motion": {key: self.project.solver["planning"][key]
                       for key in motion_keys},
        }
        key = make_artifact_key(
            ArtifactCategory.TASK_POLICY,
            "receiver-downstream-feasibility",
            artifact_version="exact-scene-downstream-v4-swept-clearance",
            input_fingerprints={
                "scene": _scene_fingerprint(self.project),
                "part": fingerprint_file(self.project.active_part_path),
                "gripper": fingerprint_file(self.project.gripper("B").model_path),
            },
            parameters=parameters,
        )
        cache = ArtifactCache(self.cache_dir)
        was_cached = cache.contains(key)

        def compute():
            self.kin.set_q("A", self.q_start["A"])
            self.kin.set_q("B", self.q_start["B"])
            local = Counter()
            witnesses = self._compute_downstream(local)
            return {
                "witnesses": [self._serialize_downstream(item) for item in witnesses],
                "statistics": dict(local),
            }

        value = cache.get_or_compute(key, compute)
        stats.update(value.get("statistics", {}))
        stats["downstream_cache_hit" if was_cached else "downstream_cache_miss"] += 1
        return [self._deserialize_downstream(item) for item in value["witnesses"]]

    def _gripper_compatibility(self, gA, gB):
        # Reject overlapping occupied contact patches before expensive IK.
        # Exact full-component collision remains authoritative at G4; no palm
        # proxy is permitted to override it.
        capability_A = self.project.gripper("A")
        capability_B = self.project.gripper("B")
        occupied_radius = 0.5 * max(float(capability_A.pad_size[0]),
                                    float(capability_B.pad_size[0]))
        center_distance = float(np.linalg.norm(gA[:3, 3] - gB[:3, 3]))
        contact_regions_separate = center_distance >= 2.0 * occupied_radius
        gates = self.cfg["gates"]
        # Retreat is -z_A; B arrives +z_B. Require angular separation.
        dot = float((-gA[:3, 2]) @ gB[:3, 2])
        angular_ok = dot <= np.cos(np.radians(gates["approach_separation_deg"]))
        # This is an occupied-part-patch separation, not gripper clearance.
        # The latter is measured from exact MuJoCo collision components.
        separation = center_distance - 2.0 * occupied_radius
        return contact_regions_separate and angular_ok, separation

    @staticmethod
    def _cube_rotation_grid() -> list[np.ndarray]:
        """The 24 proper axis-aligned rotations, deterministically ordered."""
        rotations = []
        identity = np.eye(3)
        for permutation in itertools.permutations(range(3)):
            P = identity[:, permutation]
            for signs in itertools.product((-1.0, 1.0), repeat=3):
                R = P @ np.diag(signs)
                if np.linalg.det(R) > 0.5:
                    rotations.append(R)
        return sorted(rotations, key=lambda R: tuple(np.round(R.ravel(), 12)))

    def pose_grid(self):
        cfg = self.cfg["handoff_search"]
        region = self.project.region("handoff")
        # Resolution is a solver policy, scaled to region size, not a task or
        # part knob. Exact IK/collision still certify every retained sample.
        nominal = float(self.project.solver["planning"][
            "handoff_position_resolution_m"])
        counts = tuple(int(np.clip(np.ceil(size / nominal) + 1, 3, 8))
                       for size in region.size)
        positions = region.deterministic_samples(counts)
        poses = []
        insert_R = self.insertion_poses[0][1][:3, :3]
        rotations = [insert_R @ variant for variant in self._cube_rotation_grid()]
        for xyz in positions:
            for rotation in rotations:
                poses.append(make_transform(rotation, xyz))
        center = region.center
        # Task-informed ordering from §3.1: prefer positions near the dual
        # reach center and part orientations requiring little insertion
        # reorientation. (The score later remains branch-resolved.)
        return sorted(poses, key=lambda X: (
            np.linalg.norm(X[:3, 3] - center)
            + 0.12 * so3_geodesic(X[:3, :3], insert_R)))[:cfg["max_candidates"]]

    def _backoff_target(self, target, distance):
        offset = np.eye(4); offset[2, 3] = -distance
        return target @ offset

    def _score(self, X_h, qA, qB, downstream, clearance):
        manip_raw = min(self.kin.penalized_manipulability("A", qA),
                        self.kin.penalized_manipulability("B", qB), downstream.quality)
        reference = max(self.w_min["A"], self.w_min["B"], 1e-12)
        manip = float(np.clip(manip_raw / (5 * reference), 0, 1))
        joint = min(self.kin.normalized_limit_margin("A", qA),
                    self.kin.normalized_limit_margin("B", qB),
                    *(self.kin.normalized_limit_margin("B", q) for q in downstream.q_insert))
        clear = float(np.clip(clearance / 0.05, 0, 1))
        angle = so3_geodesic(X_h[:3, :3], self.insertion_poses[0][1][:3, :3])
        reorient = 1.0 - float(np.clip(angle / np.pi, 0, 1))
        travel = (np.linalg.norm(qA - self.q_start["A"])
                  + np.linalg.norm(qB - self.q_start["B"])
                  + np.linalg.norm(downstream.q_scanner - qB))
        cycle = float(np.exp(-travel / 6.0))
        w = self.cfg["score_weights"]
        total = (w["manipulability"] * manip + w["joint_margin"] * joint
                 + w["clearance"] * clear + w["reorientation"] * reorient
                 + w["cycle"] * cycle)
        return ScoreBreakdown(manip, joint, clear, reorient, cycle, float(total))

    def _candidate(self, X_h, gA, downstream, stats, fast=False,
                   warm_only=False):
        gB = downstream.grasp
        compatible, gripper_clearance = self._gripper_compatibility(gA, gB)
        if not compatible:
            stats["grasp_incompatible"] += 1
            return None
        target_A, target_B = X_h @ gA, X_h @ gB
        if not self._reach_lookup("A", target_A) or not self._reach_lookup("B", target_B):
            stats["G1_reach"] += 1
            return None
        dpre = self.cfg["handoff_search"]["prehandoff_distance_m"]
        dret = self.cfg["handoff_search"]["retreat_distance_m"]
        best = None

        def evaluate(A_solutions, B_solutions, stop_first):
            nonlocal best
            for A, B in itertools.product(A_solutions, B_solutions):
                if not self._config_ok("A", A.q) or not self._config_ok("B", B.q):
                    stats["G3_margin"] += 1
                    continue
                state = self.collision.check(A.q, B.q, X_h, ("A", "B"))
                if not state.free:
                    stats["G4_cograsp_collision"] += 1
                    continue
                A_pre = self._solutions(
                    "A", self._backoff_target(target_A, dpre), seed=A.q)
                B_pre = self._solutions(
                    "B", self._backoff_target(target_B, dpre), seed=B.q)
                A_out = self._solutions(
                    "A", self._backoff_target(target_A, dret), seed=A.q)
                if not A_pre or not B_pre or not A_out:
                    stats["G6_approach_ik"] += 1
                    continue
                trajectories = {}
                ok, trajectories["A_current_to_pre"], _ = self._held_path(
                    "A", self.q_start["A"], A_pre[0].q, self.q_start["B"], gA,
                    ("A",), stats)
                ok2, trajectories["A_approach"], _ = self._held_path(
                    "A", A_pre[0].q, A.q, self.q_start["B"], gA,
                    ("A",), stats)
                ok3, trajectories["B_current_to_pre"], _ = self._fixed_path(
                    "B", self.q_start["B"], B_pre[0].q, A.q, X_h,
                    ("A",), stats)
                ok4, trajectories["B_approach"], _ = self._fixed_path(
                    "B", B_pre[0].q, B.q, A.q, X_h, ("A", "B"), stats)
                ok5, trajectories["A_retreat"], _ = self._fixed_path(
                    "A", A.q, A_out[0].q, B.q, X_h, ("A", "B"), stats)
                ok6, trajectories["B_to_scanner"], _ = self._held_path(
                    "B", B.q, downstream.q_scanner, A_out[0].q, gB,
                    ("B",), stats)
                if not all((ok, ok2, ok3, ok4, ok5, ok6)):
                    stats["G6_path"] += 1
                    continue
                # Every path query mutates the shared MjData and the final
                # query leaves it at B's scanner state.  Clearance is a
                # property of the simultaneous handoff witness, so restore
                # that exact state explicitly before querying distances.
                co_grasp_state = self.collision.check(
                    A.q, B.q, X_h, ("A", "B"))
                if not co_grasp_state.free:
                    stats["G4_cograsp_collision"] += 1
                    continue
                exact_clearance = self.collision.minimum_clearance(
                    policy=CollisionPolicy(part_holders=("A", "B")))
                required_clearance = (
                    self.cfg["gates"]["minimum_clearance_m"]
                    + self.cfg["gates"]["calibration_translation_3sigma_m"])
                if exact_clearance < required_clearance:
                    stats["G4_clearance"] += 1
                    continue
                clearance = min(
                    gripper_clearance,
                    exact_clearance)
                score = self._score(X_h, A.q, B.q, downstream, clearance)
                plan = DirectHandoffPlan(
                    X_h, gA, downstream.grasp_name, gB, A.q, B.q,
                    A_pre[0].q, B_pre[0].q, A_out[0].q, downstream,
                    trajectories, score)
                if best is None or plan.score.total > best.score.total:
                    best = plan
                if stop_first:
                    return True
            return False

        if fast:
            # Branch-continuous warm starts often solve in a few iterations.
            # Only pay for multi-start enumeration when the warm pair fails a
            # downstream gate; correctness is therefore preserved.
            warm_A = self._solutions("A", target_A, seed=self.q_start["A"])
            warm_B = self._solutions("B", target_B, seed=downstream.q_scanner)
            if warm_A and warm_B and evaluate(warm_A, warm_B, True):
                stats["G2_warm_start_success"] += 1
                return best
            if warm_only:
                stats["G2_warm_start_rejected"] += 1
                return None

        A_solutions = self._solutions("A", target_A)
        B_solutions = self._solutions("B", target_B)
        if not A_solutions or not B_solutions:
            stats["G2_ik"] += 1
            return None
        evaluate(A_solutions, B_solutions, fast)
        return best

    def _search_direct_core(self, gA, downstream, stats, return_best):
        best, candidates = None, 0
        poses = self.pose_grid()
        # Latency path: scan branch-continuous warm starts over the candidate
        # grid before paying for exhaustive numeric branch enumeration at any
        # single bad pose. This cannot remove a feasible solution because the
        # complete pass below is retained as fallback.
        if not return_best:
            for X_h in poses:
                for witness in downstream:
                    candidates += 1
                    plan = self._candidate(
                        X_h, gA, witness, stats, fast=True, warm_only=True)
                    if plan is not None:
                        return plan, candidates
        for X_h in poses:
            for witness in downstream:
                candidates += 1
                plan = self._candidate(X_h, gA, witness, stats, fast=False)
                if plan is None:
                    continue
                if not return_best:
                    return plan, candidates
                if best is None or plan.score.total > best.score.total:
                    best = plan
        return best, candidates

    def search_direct(self, gA=None, stats=None, return_best=True):
        stats = stats if stats is not None else Counter()
        if not self.capture_region.accepted:
            stats["capture_uncertainty"] += 1
            return None, 0, []
        gA = self.g_A_start if gA is None else np.asarray(gA)
        downstream = self.filter_downstream(stats)
        parameters = {
            "g_A": np.round(gA, 10),
            "q_start": {name: np.round(value, 10)
                        for name, value in self.q_start.items()},
            "receiver_grasps": [item.grasp_name for item in downstream],
            "project": _planning_manifest(self.project),
            "solver": self.project.solver,
            "compiled_solver_policy": self.cfg,
            "return_best": bool(return_best),
        }
        key = make_artifact_key(
            ArtifactCategory.TASK_POLICY,
            "direct-handoff-policy",
            artifact_version=(
                "direct-first-exact-components-v7-swept-clearance"),
            input_fingerprints={
                "scene": _scene_fingerprint(self.project),
                "part": fingerprint_file(self.project.active_part_path),
                "gripper_A": fingerprint_file(self.project.gripper("A").model_path),
                "gripper_B": fingerprint_file(self.project.gripper("B").model_path),
            },
            parameters=parameters,
        )
        cache = ArtifactCache(self.cache_dir)
        was_cached = cache.contains(key)

        def compute():
            local = Counter()
            plan, candidates = self._search_direct_core(
                gA, downstream, local, return_best)
            return {
                "plan": self._serialize_direct(plan),
                "candidates": candidates,
                "statistics": dict(local),
            }

        value = cache.get_or_compute(key, compute)
        stats.update(value.get("statistics", {}))
        stats["direct_cache_hit" if was_cached else "direct_cache_miss"] += 1
        return (self._deserialize_direct(value["plan"]),
                int(value["candidates"]), downstream)

    def stable_placement_witnesses(self):
        """Yield stable poses with cached, dimensionless robustness metrics."""
        support = self.project.support_region()
        stage_margin = 0.02 * float(np.min(support.size))
        part_scale = max(float(np.linalg.norm(self.part_mesh.extent)), 1e-12)
        stage_scale = float(np.min(support.size))
        margin_fraction = float(self.project.solver["geometry"].get(
            "minimum_support_margin_mesh_fraction", 0.005))
        if not np.isfinite(margin_fraction) or not 0.0 < margin_fraction < 0.5:
            raise ValueError(
                "geometry.minimum_support_margin_mesh_fraction must be in (0, 0.5)")
        minimum_support_margin = margin_fraction * part_scale
        yaw_count = int(self.project.solver["planning"].get(
            "reorientation_yaw_samples", 16))
        key = make_artifact_key(
            ArtifactCategory.STABLE_POSE,
            self.project.manifest["active_task"]["part"],
            artifact_version=(
                "support-facets-com-footprint-v4-prepared-si-robustness"),
            input_fingerprints={
                "part": fingerprint_file(self.project.active_part_path),
                "part_prepared": self.part_geometry.artifact_fingerprint,
            },
            parameters={
                "T_W_stage": support.T_W_N,
                "stage_size": support.size,
                "stage_margin": stage_margin,
                "yaw_count": yaw_count,
                "part_scale": part_scale,
                "stage_scale": stage_scale,
                "minimum_support_margin_mesh_fraction": margin_fraction,
                "minimum_support_margin": minimum_support_margin,
                "robustness_normalization": "twice-clearance-over-scale-v1",
            },
        )
        cache = ArtifactCache(self.cache_dir)

        def compute():
            placements = generate_stable_placements(
                self.part_mesh,
                minimum_support_margin=minimum_support_margin)
            stage = RectangularStage(
                support.T_W_N, tuple(support.size), stage_margin)
            yaws = np.linspace(0.0, 360.0, yaw_count, endpoint=False)
            instances = instantiate_on_rectangular_stage(
                self.part_mesh, placements, stage, yaw_samples_deg=yaws)
            values = []
            for instance in instances:
                support_robustness, stage_robustness, robustness = (
                    _normalized_placement_robustness(
                        instance.placement.support_margin,
                        instance.edge_clearance,
                        part_scale,
                        stage_scale,
                    ))
                values.append({
                    "T_W_P": instance.T_W_P,
                    "edge_clearance": instance.edge_clearance,
                    "support_margin": instance.placement.support_margin,
                    "support_area": instance.placement.support_area,
                    "probability_proxy": instance.placement.probability_proxy,
                    "minimum_support_margin": minimum_support_margin,
                    "part_scale": part_scale,
                    "stage_scale": stage_scale,
                    "support_robustness": support_robustness,
                    "stage_robustness": stage_robustness,
                    "robustness": robustness,
                })
            return values

        for index, value in enumerate(cache.get_or_compute(key, compute)):
            yield StablePlacementWitness(
                name=f"stable_{index:04d}",
                T_W_P=np.asarray(value["T_W_P"], float),
                support_margin=value["support_margin"],
                support_area=value["support_area"],
                edge_clearance=value["edge_clearance"],
                probability_proxy=value["probability_proxy"],
                minimum_support_margin=value["minimum_support_margin"],
                part_scale=value["part_scale"],
                stage_scale=value["stage_scale"],
                support_robustness=value["support_robustness"],
                stage_robustness=value["stage_robustness"],
                robustness=value["robustness"],
            )

    def stable_placements(self):
        """Backward-compatible ``(name, ^W T_P)`` stable-pose iterator."""
        for witness in self.stable_placement_witnesses():
            yield witness.name, witness.T_W_P.copy()

    def _search_regrasp_core(self, stats):
        if not self.cfg["regrasp"]["enabled"]:
            return None
        planning = self.project.solver["planning"]
        # Backward goals: grasps for which a complete B insertion-valid direct
        # handoff already exists. The nominal asset grasp is considered first,
        # followed by geometry-ranked contact grasps; this is not a part rule.
        raw_goals = [("nominal", inverse(self.project.T_tcp_part_start))]
        raw_goals.extend(self.g_B_candidates[:int(
            planning["reorientation_goal_grasp_limit"])])
        goal_templates = []
        seen = []
        for goal_id, grasp in raw_goals:
            if any(np.allclose(grasp, old, atol=1e-9) for old in seen):
                continue
            seen.append(grasp)
            direct, _, _ = self.search_direct(grasp, stats, return_best=False)
            if direct is not None:
                goal_templates.append((str(goal_id), grasp, direct))
            if len(goal_templates) >= int(
                    planning["reorientation_direct_goal_limit"]):
                break
        if not goal_templates:
            stats["regrasp_no_insertion_valid_goal"] += 1
            return None

        def path_cost(path):
            values = np.asarray(path, float)
            return (0.0 if len(values) < 2 else
                    float(np.linalg.norm(np.diff(values, axis=0), axis=1).sum()))

        support_contact = REORIENTATION_CONTACTS
        current_id = "current"
        direct_edges = [DirectCoGraspEdge(
            goal_id, direct.grasp_name_B,
            cost=path_cost(direct.trajectories["A_approach"])
                 + path_cost(direct.trajectories["B_approach"]),
            robustness=max(1e-9, direct.score.clearance),
            edge_id=f"direct:{goal_id}:{direct.grasp_name_B}")
            for goal_id, _, direct in goal_templates]
        placement_edges = []
        options = {}
        for placement in self.stable_placement_witnesses():
            placement_name = placement.name
            X_place = placement.T_W_P
            place = self._solutions(
                "A", X_place @ self.g_A_start, seed=self.q_start["A"])
            if not place or not self._config_ok("A", place[0].q):
                continue
            ok_place, place_path, _ = self._held_path(
                "A", self.q_start["A"], place[0].q, self.q_start["B"],
                self.g_A_start, ("A",), stats, support_contact)
            if not ok_place:
                continue
            placement_edges.append(PlacementGraspEdge(
                placement_name, current_id, path_cost(place_path),
                placement.robustness,
                edge_id=f"place:{placement_name}:current"))
            for goal_id, g_new, template in goal_templates:
                repick = self._solutions(
                    "A", X_place @ g_new, seed=place[0].q)
                if not repick or not self._config_ok("A", repick[0].q):
                    continue
                ok_pick, repick_path, _ = self._fixed_path(
                    "A", place[0].q, repick[0].q, self.q_start["B"],
                    X_place, ("A",), stats, support_contact)
                if not ok_pick:
                    continue
                ok_goal, repick_to_goal, _ = self._held_path(
                    "A", repick[0].q, template.qA_pre, self.q_start["B"],
                    g_new, ("A",), stats, support_contact)
                if not ok_goal:
                    continue
                trajectories = dict(template.trajectories)
                trajectories["A_current_to_pre"] = repick_to_goal
                direct = replace(template, trajectories=trajectories)
                placement_edges.append(PlacementGraspEdge(
                    placement_name, goal_id,
                    path_cost(repick_path) + path_cost(repick_to_goal),
                    placement.robustness,
                    edge_id=f"pick:{placement_name}:{goal_id}"))
                options[(placement_name, goal_id)] = RegraspPlan(
                    placement_name, X_place, self.g_A_start, g_new,
                    place[0].q, repick[0].q, direct,
                    {"A_to_place": place_path,
                     "A_place_to_repick": repick_path})

        graph = TaskGraph(
            [InitialGraspClass.singleton(current_id)],
            sorted({edge.receiver_grasp for edge in direct_edges}),
            direct_edges,
            placement_edges,
        )
        discrete = graph.plan(
            current_id,
            max_reorientation_hops=int(planning["max_reorientation_hops"]))
        if discrete.success:
            placement_id = next(step.target for step in discrete.steps
                                if step.kind == "place")
            goal_id = next(step.target for step in discrete.steps
                           if step.kind == "regrasp")
            stats["regrasp_graph_edges"] += len(placement_edges) + len(direct_edges)
            return options[(placement_id, goal_id)]
        stats["regrasp_failed"] += 1
        return None

    def search_regrasp(self, stats):
        """Load or compute the backward insertion-goal reorientation policy."""
        if not self.cfg["regrasp"]["enabled"]:
            return None
        key = make_artifact_key(
            ArtifactCategory.TASK_POLICY,
            "backward-reorientation-policy",
            artifact_version="stable-placement-task-graph-v7-lift-clearance",
            input_fingerprints={
                "scene": _scene_fingerprint(self.project),
                "part": fingerprint_file(self.project.active_part_path),
                "gripper_A": fingerprint_file(self.project.gripper("A").model_path),
                "gripper_B": fingerprint_file(self.project.gripper("B").model_path),
            },
            parameters={
                "g_A_current": np.round(self.g_A_start, 10),
                "q_start": {name: np.round(value, 10)
                            for name, value in self.q_start.items()},
                "project": _planning_manifest(self.project),
                "solver": self.project.solver,
            },
        )
        cache = ArtifactCache(self.cache_dir)
        was_cached = cache.contains(key)

        def compute():
            local = Counter()
            plan = self._search_regrasp_core(local)
            return {"plan": self._serialize_regrasp(plan),
                    "statistics": dict(local)}

        value = cache.get_or_compute(key, compute)
        stats.update(value.get("statistics", {}))
        stats["regrasp_cache_hit" if was_cached else "regrasp_cache_miss"] += 1
        return self._deserialize_regrasp(value["plan"])

    def plan(self, allow_regrasp=True, return_best=False) -> PlanningReport:
        started = time.perf_counter()
        report = PlanningReport()
        direct, report.candidates, downstream = self.search_direct(
            stats=report.stats, return_best=return_best)
        report.downstream_grasps = [item.grasp_name for item in downstream]
        if direct is not None:
            report.direct = direct
        elif allow_regrasp:
            report.regrasp = self.search_regrasp(report.stats)
        static_grippers = [robot for robot in ("A", "B")
                           if not self.project.gripper(robot).articulated]
        physical_gates = physical_prerequisites(self.project)
        limitations = []
        physical_limitations = []
        if static_grippers:
            physical_limitations.append(
                "static gripper CAD: opening/closing and aperture capture are not physically certified"
            )
        if not physical_gates["articulated_gripper_scene_adapter"]:
            physical_limitations.append(
                "articulated gripper scene/contact adapter is not implemented"
            )
        if not self.project.manifest["insertion"].get("collision_cad"):
            physical_limitations.append(
                "no pin/hole collision CAD: insertion frame/path are checked, seating force is not"
            )
        if not physical_gates["part_pin_collision_cad"]:
            physical_limitations.append(
                "no separate part pin collision CAD: the visual convex hull cannot certify insertion contact"
            )
        if not all(physical_gates[name] for name in (
                "calibrated_gripper_contacts", "calibrated_part_contact",
                "calibrated_pin_hole_materials")):
            physical_limitations.append(
                "contact friction/material parameters are not physically calibrated"
            )
        if not physical_gates["physical_contact_execution_backend"]:
            physical_limitations.append(
                "current executor uses ideal weld ownership and virtual capture/insertion predicates"
            )
        limitations.extend(physical_limitations)
        limitations.append(
            "multi-seed numerical GP7 IK is FK-verified but not analytically branch-complete"
        )
        report.limitations = tuple(limitations)
        # A run proves only its supplied initial state. Production qualification
        # uses task_graph.coverage_report over all declared admissible classes.
        report.coverage = {
            "domain": "known_start_pose_singleton",
            "domain_declaration": "active_task.initial_tcp_to_part",
            "covered": int(report.feasible),
            "required": 1,
            "fraction": 1.0 if report.feasible else 0.0,
            "mathematical_coverage_certified": bool(report.feasible),
            "physical_prerequisites": physical_gates,
            "physical_certified": bool(
                report.feasible and all(physical_gates.values())),
        }
        report.mathematical_coverage_certified = report.coverage[
            "mathematical_coverage_certified"]
        report.physical_certified = report.coverage["physical_certified"]
        report.certified = report.physical_certified
        report.elapsed_s = time.perf_counter() - started
        return report
