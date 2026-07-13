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

import hashlib
import itertools
import os
import time
from collections import Counter
from contextlib import nullcontext

import numpy as np

from .collision import SceneCollisionChecker
from .geometry_grasps import (GraspCandidate, ParallelJawGripper,
                              generate_antipodal_grasps)
from .kinematics import GP7Kinematics
from .offline import (ArtifactCache, ArtifactCategory, fingerprint_file,
                      fingerprint_content, make_artifact_key)
from .motion_planning import MotionPlannerConfig
from .part_mesh import load_project_part_mesh
from .plan_codec import (deserialize_direct, deserialize_downstream,
                         deserialize_grasp, deserialize_regrasp,
                         serialize_direct, serialize_downstream,
                         serialize_grasp, serialize_regrasp)
from .plan_validation import validate_direct_plan, validate_regrasp_plan
from .phase_contacts import (
    EXACT_INSERTION_CONTACTS,
    PLACEHOLDER_INSERTION_CONTACTS,
    REORIENTATION_CONTACTS,
    insertion_contacts,
)
from .planner_stages import (DirectCandidateEvaluator, DirectHandoffSearch,
                             DownstreamCertifier, ReorientationSearch)
from .placements import (RectangularStage, generate_stable_placements,
                         instantiate_on_rectangular_stage)
from .pose_templates import (load_declared_pose_templates,
                             rank_contact_validated_grasps)
from .project import DEFAULT_PROJECT, Project
from .profiling import HierarchicalProfiler
from .planning_types import (
    DirectHandoffPlan,
    DownstreamWitness,
    PlanningReport,
    RegraspPlan,
    ScoreBreakdown,
    StablePlacementWitness,
    normalized_placement_robustness,
)
from .qualification import physical_prerequisites
from .reachability import ReachabilityMap
from .se3 import (inverse, make_transform, so3_exp, so3_geodesic,
                  validate_transform)
from .sim import WorkcellSim
from .task_graph import TaskGraph
from .uncertainty import check_axis_aligned_capture, combine_independent

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def _allowed_contact_cache_key(values) -> tuple:
    normalized = []
    for entry in values:
        if len(entry) not in (2, 3):
            raise ValueError("allowed contact entries need two or three values")
        normalized.append(tuple(entry))
    return tuple(sorted(normalized, key=repr))


_normalized_placement_robustness = normalized_placement_robustness


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
    active_part = project.active_part
    part_collision_path = active_part.get("collision_cad")
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
        "part_collision": (None if not part_collision_path else
                           fingerprint_file(project.resolve_asset(
                               part_collision_path))),
        "internal_fixture_fallback": fingerprint_file(
            os.path.join(HERE, "scene_config.yaml")),
        "active_part": project.manifest["active_task"]["part"],
    })


def _planning_manifest(project: Project) -> dict:
    """Return policy-relevant project data, excluding audit-only declarations."""
    return {key: value for key, value in project.manifest.items()
            if key != "qualification"}

class HandoffPlanner:
    def __init__(self, sim: WorkcellSim,
                 known_start_pose: np.ndarray | None = None,
                 project_path: str = DEFAULT_PROJECT,
                 cache_dir: str | None = None):
        initialization_started = time.perf_counter()
        self.sim = sim
        self.profiler = HierarchicalProfiler("planning")
        with self.profiler.span("initialize.kinematics"):
            self.kin = GP7Kinematics(sim)
        requested_project = os.path.realpath(project_path)
        if sim.project.manifest_path != requested_project:
            raise ValueError(
                "planner project does not match the project loaded by WorkcellSim: "
                f"{requested_project!r} != {sim.project.manifest_path!r}")
        # One validated Project instance is authoritative for scene state and
        # planning.  Previously both layers parsed independent manifests,
        # making an alternate model/project mismatch difficult to diagnose.
        self.project = sim.project
        # Reuse the exact prepared-CAD directory belonging to the selected
        # compiled MJCF. This matters for alternate projects and for STEP,
        # where a prior explicit FreeCAD preparation must be reusable online.
        generated_cad = os.path.join(
            os.path.dirname(self.sim.model_path), "generated_cad")
        with self.profiler.span("initialize.part_geometry"):
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
        with self.profiler.span("initialize.collision_runtime"):
            self.collision = SceneCollisionChecker(
                sim, self.kin, edge_step, path_clearance)
        self.cfg = _compile_solver_config(self.project.solver)
        self._ik_cache: dict[tuple, list] = {}
        self._seed_ik_cache: dict[tuple, list] = {}
        self._motion_cache: dict[tuple, tuple[bool, list[np.ndarray], str]] = {}
        self.q_start = {robot: self.kin.get_q(robot) for robot in ("A", "B")}
        self.X_start = validate_transform(
            sim.part_pose() if known_start_pose is None else known_start_pose)
        if known_start_pose is not None:
            sim.set_part_world(self.X_start)
        # Known start pose + current A FK determines the measured direct grasp.
        self.g_A_start = inverse(self.X_start) @ self.kin.fk("A", self.q_start["A"])
        self.part_lo, self.part_hi = self._part_bounds()
        self.part_center = 0.5 * (self.part_lo + self.part_hi)
        with self.profiler.span("initialize.receiver_grasp_library"):
            self.g_B_candidates = self._receiver_grasps()
        self.reachability_maps = {}
        reach_cfg = self.cfg.get("reachability", {})
        if reach_cfg.get("use_cached_maps", True):
            directory = self.cache_dir
            for robot in ("A", "B"):
                path = os.path.join(directory, f"reachability_{robot}.npz")
                if os.path.exists(path):
                    with self.profiler.span("initialize.reachability_map"):
                        self.reachability_maps[robot] = ReachabilityMap.load(path)
        gates = self.cfg["gates"]
        self.limit_margin = np.radians(gates["joint_limit_margin_deg"])
        self.pos_tol = gates["ik_position_tolerance_m"]
        self.rot_tol = np.radians(gates["ik_rotation_tolerance_deg"])
        self.restarts = gates["ik_restarts"]
        self.max_solutions = gates["ik_max_solutions"]
        with self.profiler.span("initialize.manipulability_calibration"):
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
        self.initialization_profile = self.profiler.report()
        self.initialization_elapsed_s = time.perf_counter() - initialization_started
        self.profiler.reset()

    def _profile(self, name: str):
        """Return a span while keeping ``__new__``-based unit fakes usable."""
        profiler = getattr(self, "profiler", None)
        return profiler.span(name) if profiler is not None else nullcontext()

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
        with self._profile("motion.held_edge_check"):
            ok, path, reason = self.collision.path(
                robot, q_from, q_to, other_q, grasp, self.steps, holders,
                allowed_geom_pairs)
        if ok:
            self._motion_cache[key] = (ok, [q.copy() for q in path], reason)
            return ok, path, reason
        with self._profile("motion.held_rrt_connect"):
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
        with self._profile("motion.fixed_part_edge_check"):
            ok, path, reason = self.collision.path_fixed_part(
                robot, q_from, q_to, other_q, X_part, self.steps, holders,
                allowed_geom_pairs)
        if ok:
            self._motion_cache[key] = (ok, [q.copy() for q in path], reason)
            return ok, path, reason
        with self._profile("motion.fixed_part_rrt_connect"):
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
        return serialize_grasp(candidate)

    @staticmethod
    def _deserialize_grasp(value: dict) -> GraspCandidate:
        return deserialize_grasp(value)

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

        with self._profile("cache.receiver_grasps"):
            cached_candidates = cache.get_or_compute(key, compute)
        candidates = [self._deserialize_grasp(value)
                      for value in cached_candidates]
        if not candidates:
            raise ValueError("part CAD has no antipodal grasp within gripper capability")
        self.grasp_candidates = {
            f"geom_{index:03d}": candidate
            for index, candidate in enumerate(candidates)
        }
        top_level = self.project.manifest.get("proposal_templates")
        part_level = self.project.active_part.get("proposal_templates")
        if top_level is not None and part_level is not None:
            raise ValueError(
                "declare proposal_templates either at project top level or "
                "on the active part, not both")
        declared = top_level if top_level is not None else part_level
        template_limit = int(defaults.get("template_max_proposals", 10000))
        templates = load_declared_pose_templates(
            declared,
            resolve_path=self.project.resolve_asset,
            max_proposals_per_template=template_limit,
        )
        self.grasp_pose_templates = templates
        self.grasp_template_fingerprint = fingerprint_content(
            [template.semantic_fingerprint for template in templates])
        ordered_names, matches = rank_contact_validated_grasps(
            self.grasp_candidates,
            templates,
            part_scale=max(float(np.linalg.norm(self.part_mesh.extent)), 1e-12),
            position_tolerance_fraction=float(defaults.get(
                "template_position_tolerance_mesh_fraction", 0.12)),
            rotation_tolerance_deg=float(defaults.get(
                "template_rotation_tolerance_deg", 20.0)),
            normal_tolerance_deg=float(defaults.get(
                "template_normal_tolerance_deg", 35.0)),
            max_matches_per_proposal=int(defaults.get(
                "template_matches_per_proposal", 2)),
        )
        self.grasp_template_matches = matches
        # Template transforms are not inserted here. This is strictly a
        # permutation of candidates already certified by CAD contact geometry;
        # downstream IK and exact scene collision gates remain unchanged.
        return [(name, self.grasp_candidates[name].T_P_E)
                for name in ordered_names]

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
                with self._profile("ik.seeded_solve"):
                    result = self.kin.solve(robot, target, seed=seed, **kwargs)
                self._seed_ik_cache[key] = [] if result is None else [result]
            return self._seed_ik_cache[key]
        # The same A target is shared by every receiver grasp at one handoff
        # pose. Cache exact target solves to avoid repeating dozens of numeric
        # multi-start IK calls inside the nested pair loop.
        key = (robot, np.round(np.asarray(target, float), 10).tobytes(),
               self.restarts, self.max_solutions)
        if key not in self._ik_cache:
            # Multi-start IK must not depend on which cache entry happened to
            # run before this one. Use the declared robot start as its primary
            # seed and deterministic target-keyed random restarts. Previously
            # both the shared RNG and live MjData q depended on cache history,
            # changing feasibility and CT between cold and warm runs.
            self.kin.set_q(robot, self.q_start[robot])
            digest = hashlib.sha256(
                b"handoff-target-ik-v1\0" + robot.encode("ascii") + key[1]
            ).digest()
            target_rng = np.random.default_rng(
                int.from_bytes(digest[:8], "little", signed=False))
            with self._profile("ik.multistart_solve"):
                self._ik_cache[key] = self.kin.solutions(
                    robot, target, self.restarts, self.max_solutions, target_rng,
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
        return serialize_downstream(witness)

    @staticmethod
    def _deserialize_downstream(value: dict) -> DownstreamWitness:
        return deserialize_downstream(value)

    @classmethod
    def _serialize_direct(cls, plan: DirectHandoffPlan | None):
        return serialize_direct(plan)

    @classmethod
    def _deserialize_direct(cls, value) -> DirectHandoffPlan | None:
        return deserialize_direct(value)

    @classmethod
    def _serialize_regrasp(cls, plan: RegraspPlan | None):
        return serialize_regrasp(plan)

    @classmethod
    def _deserialize_regrasp(cls, value) -> RegraspPlan | None:
        return deserialize_regrasp(value)

    def _compute_downstream(self, stats: Counter | None = None) -> list[DownstreamWitness]:
        certifier = DownstreamCertifier(
            self, insertion_contacts(self.project))
        with self._profile("downstream.certification"):
            return certifier.certify(stats)

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
            artifact_version=(
                "exact-scene-downstream-v6-stateless-target-ik"
                + ("-exact-zero-penetration-v1"
                   if self.project.manifest["insertion"].get("collision_cad")
                   else "")),
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

        with self._profile("cache.downstream_policy"):
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
        evaluator = getattr(self, "_candidate_evaluator", None)
        if evaluator is None:
            evaluator = DirectCandidateEvaluator(self)
            self._candidate_evaluator = evaluator
        return evaluator.evaluate(
            X_h, gA, downstream, stats, fast=fast, warm_only=warm_only)

    def _search_direct_core(self, gA, downstream, stats, return_best):
        stage = DirectHandoffSearch(
            self.pose_grid, self._candidate, self._profile)
        return stage.search(
            gA, downstream, stats, return_best=return_best)

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
            "grasp_template_fingerprint": self.grasp_template_fingerprint,
            "project": _planning_manifest(self.project),
            "solver": self.project.solver,
            "compiled_solver_policy": self.cfg,
            "return_best": bool(return_best),
        }
        key = make_artifact_key(
            ArtifactCategory.TASK_POLICY,
            "direct-handoff-policy",
            artifact_version=(
                "direct-first-exact-components-v10-explicit-sender-park"),
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

        with self._profile("cache.direct_policy"):
            value = cache.get_or_compute(key, compute)
        stats.update(value.get("statistics", {}))
        stats["direct_cache_hit" if was_cached else "direct_cache_miss"] += 1
        plan = self._deserialize_direct(value["plan"])
        if plan is not None:
            validate_direct_plan(plan, q_start=self.q_start)
        return plan, int(value["candidates"]), downstream

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

        with self._profile("cache.stable_placements"):
            cached_placements = cache.get_or_compute(key, compute)
        for index, value in enumerate(cached_placements):
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
        stage = ReorientationSearch(
            self, REORIENTATION_CONTACTS, task_graph_type=TaskGraph)
        with self._profile("reorientation.task_graph"):
            return stage.search(stats)

    def search_regrasp(self, stats):
        """Load or compute the backward insertion-goal reorientation policy."""
        if not self.cfg["regrasp"]["enabled"]:
            return None
        key = make_artifact_key(
            ArtifactCategory.TASK_POLICY,
            "backward-reorientation-policy",
            artifact_version="stable-placement-task-graph-v10-explicit-sender-park",
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
                "receiver_grasp_order": [name for name, _ in self.g_B_candidates],
                "grasp_template_fingerprint": self.grasp_template_fingerprint,
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

        with self._profile("cache.reorientation_policy"):
            value = cache.get_or_compute(key, compute)
        stats.update(value.get("statistics", {}))
        stats["regrasp_cache_hit" if was_cached else "regrasp_cache_miss"] += 1
        plan = self._deserialize_regrasp(value["plan"])
        if plan is not None:
            validate_regrasp_plan(plan, q_start=self.q_start)
        return plan

    def _plan_impl(self, allow_regrasp=True, return_best=False) -> PlanningReport:
        report = PlanningReport()
        with self._profile("direct_search"):
            direct, report.candidates, downstream = self.search_direct(
                stats=report.stats, return_best=return_best)
        report.downstream_grasps = [item.grasp_name for item in downstream]
        if direct is not None:
            report.direct = direct
        elif allow_regrasp:
            with self._profile("reorientation_search"):
                report.regrasp = self.search_regrasp(report.stats)
        with self._profile("qualification"):
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
        if not physical_gates["complete_part_collision_cad"]:
            physical_limitations.append(
                "no complete convex-decomposed part collision CAD: the visual hull cannot certify body/pin insertion contact"
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
        return report

    def plan(self, allow_regrasp=True, return_best=False) -> PlanningReport:
        """Run the direct-first query and attach hierarchical bottleneck data."""
        self.profiler.reset()
        started = time.perf_counter()
        with self.profiler.span("query"):
            report = self._plan_impl(allow_regrasp, return_best)
        report.elapsed_s = time.perf_counter() - started
        report.initialization_timings = self.initialization_profile
        report.stage_timings = self.profiler.report()
        report.bottlenecks = self.profiler.bottlenecks()
        return report
