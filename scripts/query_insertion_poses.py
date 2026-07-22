#!/usr/bin/env python3
"""Compose an insertion grasp library with a runtime world target.

Insertion-purpose queries emit seated and pre-insert targets only for grasps
that passed both library checks.  Explicit pre-insert diagnostics keep the
seated target null for pre-insert-only grasps.  Optional ``--solve-ik`` runs the
current GP7 numerical IK, but never claims collision-free motion or physical
insertion qualification.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mujoco_sim.core.paths import (  # noqa: E402
    DEFAULT_MODEL_PATH,
    DEFAULT_PROJECT_PATH,
)
from mujoco_sim.modeling.insertion_query import (  # noqa: E402
    attach_provisional_gp7_ik,
    bind_pcb_socket_contract,
    compose_insertion_pose_query,
    load_insertion_pose_library,
    resolve_world_part_insert_pose,
)
from mujoco_sim.offline_tools.artifacts import (  # noqa: E402
    atomic_write_json,
    fingerprint_file,
)


def _load_yaml(path: Path, *, label: str) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{label} root must be a YAML mapping")
    return value


def _resolve_path(value: str | Path, owner: Path) -> Path:
    supplied = Path(value).expanduser()
    if supplied.is_absolute():
        return supplied.resolve()
    local = (owner.parent / supplied).resolve()
    if local.exists():
        return local
    return (ROOT / supplied).resolve()


def _resolve_output_path(value: str | Path, owner: Path) -> Path:
    """Resolve a not-yet-existing output relative to its query YAML."""
    supplied = Path(value).expanduser()
    return (supplied if supplied.is_absolute()
            else owner.parent / supplied).resolve()


def _strict_bool(value: Any, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be boolean")
    return value


def _socket_binding(path: Path, library):
    value = _load_yaml(path, label="PCB socket contract")
    return bind_pcb_socket_contract(value, library)


def _target_from_config(
    config: dict[str, Any],
    config_path: Path,
    library,
):
    direct = config.get("world_part_insert_pose")
    board = config.get("board_world_pose")
    socket_value = config.get("pcb_socket")
    socket_path = None if socket_value is None else _resolve_path(
        socket_value, config_path)
    socket_binding = None if socket_path is None else _socket_binding(
        socket_path, library)
    target, source = resolve_world_part_insert_pose(
        world_part_insert_pose=(None if direct is None
                                else np.asarray(direct, dtype=float)),
        board_world_pose=(None if board is None
                          else np.asarray(board, dtype=float)),
        T_B_P_insert=(None if socket_binding is None
                      else socket_binding.T_B_P_insert),
    )
    return target, source, socket_path, socket_binding


def run_query(
    config_path: str | Path,
    *,
    output_path: str | Path | None = None,
    solve_ik: bool = False,
    acknowledge_provisional_tcp: bool = False,
    project_path: str | Path | None = None,
    model_path: str | Path | None = None,
) -> tuple[dict[str, Any], Path]:
    """Run one YAML query and return its JSON-compatible result and path."""
    query_path = Path(config_path).resolve()
    config = _load_yaml(query_path, label="insertion query")
    if int(config.get("schema_version", 0)) != 1:
        raise ValueError("insertion query schema_version must be 1")
    if "pose_library" not in config:
        raise ValueError("insertion query requires pose_library")
    if "robot" not in config:
        raise ValueError("insertion query requires robot A or B")
    if "world_frame" not in config:
        raise ValueError("insertion query requires world_frame metadata")
    if not isinstance(solve_ik, bool):
        raise ValueError("solve_ik must be boolean")
    if not isinstance(acknowledge_provisional_tcp, bool):
        raise ValueError("acknowledge_provisional_tcp must be boolean")

    library_path = _resolve_path(config["pose_library"], query_path)
    library = load_insertion_pose_library(library_path)
    T_W_P_insert, target_source, socket_path, socket_binding = _target_from_config(
        config, query_path, library)
    result = compose_insertion_pose_query(
        library,
        robot=config["robot"],
        T_W_P_insert=T_W_P_insert,
        world_frame=config["world_frame"],
        target_source=target_source,
        preinsert_distance_m=config.get("preinsert_distance_m"),
        correction_bounds=config.get("correction_bounds"),
        selection=config.get("selection"),
    )
    result["query"] = {
        "path": str(query_path),
        "sha256": fingerprint_file(query_path),
        "pcb_socket_path": (None if socket_path is None else str(socket_path)),
        "pcb_socket_sha256": (None if socket_path is None
                               else fingerprint_file(socket_path)),
        "pcb_socket_project_id": (None if socket_binding is None
                                  else socket_binding.project_id),
        "pcb_socket_compatible_library_project_ids": (
            None if socket_binding is None
            else list(socket_binding.compatible_library_project_ids)
        ),
        "pcb_socket_connector_sha256": (
            None if socket_binding is None else socket_binding.connector_sha256
        ),
    }

    configured_solve_ik = _strict_bool(
        config.get("solve_ik", False), label="solve_ik")
    requested_ik = solve_ik or configured_solve_ik
    tcp_config = config.get("tcp_assumption", {})
    if tcp_config is None:
        tcp_config = {}
    if not isinstance(tcp_config, dict):
        raise ValueError("tcp_assumption must be a mapping")
    configured_acknowledgment = _strict_bool(
        tcp_config.get("acknowledge_provisional", False),
        label="tcp_assumption.acknowledge_provisional",
    )
    if requested_ik:
        acknowledged = acknowledge_provisional_tcp or configured_acknowledgment
        if not acknowledged:
            raise ValueError(
                "--solve-ik requires --acknowledge-provisional-tcp or "
                "tcp_assumption.acknowledge_provisional: true; the supplied "
                "gripper has no calibrated flange-to-E transform"
            )
        selected_project = _resolve_path(
            project_path or config.get("project", DEFAULT_PROJECT_PATH),
            query_path,
        )
        selected_model = _resolve_path(
            model_path or config.get("model", DEFAULT_MODEL_PATH),
            query_path,
        )

        # Imports stay lazy so transform-only queries do not need a working
        # MuJoCo runtime or load a 3-D scene.
        from mujoco_sim.simulation.kinematics import GP7Kinematics
        from mujoco_sim.simulation.workcell import WorkcellSim

        sim = WorkcellSim(
            model_path=str(selected_model), project_path=str(selected_project))
        kinematics = GP7Kinematics(sim)
        ik_config = config.get("ik", {})
        if not isinstance(ik_config, dict):
            raise ValueError("ik must be a mapping")
        planning = sim.project.solver["planning"]
        result = attach_provisional_gp7_ik(
            result,
            kinematics,
            acknowledge_provisional_tcp=acknowledged,
            random_restarts=int(ik_config.get(
                "random_restarts", planning["ik_restarts"])),
            max_solutions=int(ik_config.get(
                "max_solutions", planning["ik_max_solutions"])),
            position_tolerance_m=float(ik_config.get(
                "position_tolerance_m", planning["ik_position_tolerance_m"])),
            rotation_tolerance_rad=np.radians(float(ik_config.get(
                "rotation_tolerance_deg", planning["ik_rotation_tolerance_deg"]))),
        )
        result["ik_evaluation"].update({
            "project_path": str(selected_project),
            "project_sha256": fingerprint_file(selected_project),
            "model_path": str(selected_model),
            "model_sha256": fingerprint_file(selected_model),
            "tcp_assumption_note": tcp_config.get(
                "note",
                "No calibrated flange-to-E transform was supplied for this query.",
            ),
        })

    configured_output = config.get("output")
    destination = (
        Path(output_path).expanduser()
        if output_path is not None else
        _resolve_output_path(configured_output, query_path)
        if configured_output is not None else
        query_path.with_name(f"{query_path.stem}_result.json")
    ).resolve()
    atomic_write_json(destination, result)
    return result, destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path,
                        help="runtime insertion-query YAML")
    parser.add_argument("--output", type=Path,
                        help="override output JSON path")
    parser.add_argument("--solve-ik", action="store_true",
                        help="run endpoint-only GP7 IK using the compiled TCP site")
    parser.add_argument(
        "--acknowledge-provisional-tcp", action="store_true",
        help="acknowledge that flange-to-E calibration is absent/unverified",
    )
    parser.add_argument("--project", type=Path,
                        help="project manifest for optional IK")
    parser.add_argument("--model", type=Path,
                        help="compiled MJCF for optional IK")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result, output = run_query(
        args.config,
        output_path=args.output,
        solve_ik=args.solve_ik,
        acknowledge_provisional_tcp=args.acknowledge_provisional_tcp,
        project_path=args.project,
        model_path=args.model,
    )
    print(
        f"wrote {output}: {result['selected_candidate_count']} selected pose records; "
        f"claim={result['claim_level']}; certified={result['certified']}"
    )
    if result["ik_evaluation"]["performed"]:
        print(
            "IK-only endpoint reachable candidates: "
            f"{result['ik_reachable_candidate_count']} "
            "(collision/path not checked)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
