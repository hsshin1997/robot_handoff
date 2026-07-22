#!/usr/bin/env python3
"""Build the GP7-conditioned, provisional insertion path-witness set."""
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

from mujoco_sim.core.paths import DEFAULT_MODEL_PATH, DEFAULT_PROJECT_PATH  # noqa: E402
from mujoco_sim.offline_tools.artifacts import (  # noqa: E402
    atomic_write_json,
    fingerprint_file,
)
from mujoco_sim.planner.robot_insertion_set import (  # noqa: E402
    build_robot_insertion_set,
    load_insertion_task_set,
    load_verified_continuous_robot_certificate,
    resolve_world_part_insert,
)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError("robot insertion-set config root must be a mapping")
    return value


def _resolve_input(value: str | Path, owner: Path) -> Path:
    supplied = Path(value).expanduser()
    if supplied.is_absolute():
        return supplied.resolve()
    local = (owner.parent / supplied).resolve()
    if local.exists():
        return local
    return (ROOT / supplied).resolve()


def _resolve_output(value: str | Path, owner: Path) -> Path:
    supplied = Path(value).expanduser()
    return (supplied if supplied.is_absolute()
            else owner.parent / supplied).resolve()


def _mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return value


def run_build(
    config_path: str | Path,
    *,
    output_path: str | Path | None = None,
    project_path: str | Path | None = None,
    model_path: str | Path | None = None,
) -> tuple[dict[str, Any], Path]:
    """Load the GP7 runtime, evaluate the task set, and write JSON."""
    source = Path(config_path).expanduser().resolve()
    config = _load_yaml(source)
    if int(config.get("schema_version", 0)) != 1:
        raise ValueError("robot insertion-set config schema_version must be 1")
    for key in ("layer1_artifact", "robot", "world_frame"):
        if key not in config:
            raise ValueError(f"robot insertion-set config requires {key}")

    task_path = _resolve_input(config["layer1_artifact"], source)
    task_set = load_insertion_task_set(task_path)
    direct = config.get("world_part_insert_pose", config.get("T_W_P_insert"))
    board = config.get("board_world_pose", config.get("T_W_B"))
    socket = config.get("T_B_P_insert")
    target, target_source = resolve_world_part_insert(
        task_set,
        T_W_P_insert=(None if direct is None
                      else np.asarray(direct, dtype=float)),
        T_W_B=(None if board is None else np.asarray(board, dtype=float)),
        T_B_P_insert=(None if socket is None
                      else np.asarray(socket, dtype=float)),
    )

    selected_project = _resolve_input(
        project_path or config.get("project", DEFAULT_PROJECT_PATH), source)
    selected_model = _resolve_input(
        model_path or config.get("model", DEFAULT_MODEL_PATH), source)
    # Lazy imports keep schema/path inspection usable without loading MuJoCo.
    from mujoco_sim.simulation.kinematics import GP7Kinematics
    from mujoco_sim.simulation.workcell import WorkcellSim

    sim = WorkcellSim(
        model_path=str(selected_model), project_path=str(selected_project))
    kinematics = GP7Kinematics(sim)

    tcp = _mapping(config.get("tcp_contract", {}), label="tcp_contract")
    ik = _mapping(config.get("ik", {}), label="ik")
    selection = _mapping(config.get("selection", {}), label="selection")
    calibrated = tcp.get("calibrated", False)
    acknowledged = tcp.get("acknowledge_provisional", False)
    if not isinstance(calibrated, bool):
        raise ValueError("tcp_contract.calibrated must be boolean")
    if not isinstance(acknowledged, bool):
        raise ValueError("tcp_contract.acknowledge_provisional must be boolean")
    project_sha256 = fingerprint_file(selected_project)
    model_sha256 = fingerprint_file(selected_model)
    certificate_config = _mapping(
        config.get("continuous_robot_cell_certificate", {}),
        label="continuous_robot_cell_certificate",
    )
    certificate_path_value = certificate_config.get("path")
    expected_certificate_sha = certificate_config.get("expected_file_sha256")
    if certificate_path_value is None:
        if expected_certificate_sha is not None:
            raise ValueError(
                "continuous certificate hash requires a certificate path")
        continuous_certificate = None
    else:
        if expected_certificate_sha is None:
            raise ValueError(
                "continuous certificate path requires expected_file_sha256")
        continuous_certificate = load_verified_continuous_robot_certificate(
            _resolve_input(certificate_path_value, source),
            expected_file_sha256=expected_certificate_sha,
            task_set=task_set,
            robot=config["robot"],
            T_W_P_insert=target,
            world_frame=config["world_frame"],
            tcp_calibration_fingerprint=tcp.get("calibration_fingerprint"),
            execution_bindings={
                "project_sha256": project_sha256,
                "model_sha256": model_sha256,
            },
        )

    planning = sim.project.solver["planning"]
    result = build_robot_insertion_set(
        task_set,
        kinematics,
        robot=config["robot"],
        T_W_P_insert=target,
        world_frame=config["world_frame"],
        target_source=target_source,
        source_classifications=selection.get(
            "source_classifications", ["SAFE"]),
        max_cells=selection.get("max_cells"),
        path_sample_count=int(ik.get("path_sample_count", 11)),
        random_restarts=int(ik.get(
            "random_restarts", planning["ik_restarts"])),
        max_solutions=int(ik.get(
            "max_solutions", planning["ik_max_solutions"])),
        position_tolerance_m=float(ik.get(
            "position_tolerance_m", planning["ik_position_tolerance_m"])),
        rotation_tolerance_rad=np.radians(float(ik.get(
            "rotation_tolerance_deg", planning["ik_rotation_tolerance_deg"]))),
        max_joint_step_rad=float(ik.get("max_joint_step_rad", 0.35)),
        minimum_joint_limit_margin_rad=float(ik.get(
            "minimum_joint_limit_margin_rad", 0.03)),
        minimum_normalized_joint_limit_margin=float(ik.get(
            "minimum_normalized_joint_limit_margin", 0.02)),
        minimum_sigma=float(ik.get("minimum_sigma", 1e-4)),
        tcp_calibrated=calibrated,
        tcp_calibration_fingerprint=tcp.get("calibration_fingerprint"),
        acknowledge_provisional_tcp=acknowledged,
        continuous_certificate=continuous_certificate,
    )
    result["build"] = {
        "config_path": str(source),
        "config_sha256": fingerprint_file(source),
        "project_path": str(selected_project),
        "project_sha256": project_sha256,
        "model_path": str(selected_model),
        "model_sha256": model_sha256,
        "tcp_note": tcp.get("note"),
    }

    destination_value = output_path or config.get(
        "output", "../generated/sets/robot_insertion_set.json")
    destination = _resolve_output(destination_value, source)
    atomic_write_json(destination, result)
    return result, destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path,
                        help="robot-set YAML configuration")
    parser.add_argument("--output", type=Path,
                        help="override generated JSON path")
    parser.add_argument("--project", type=Path,
                        help="override MuJoCo project manifest")
    parser.add_argument("--model", type=Path,
                        help="override compiled MuJoCo scene")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result, destination = run_build(
        args.config,
        output_path=args.output,
        project_path=args.project,
        model_path=args.model,
    )
    summary = result["summary"]
    print(
        f"wrote {destination}: "
        f"{summary['provisional_center_path_witness_count']} "
        "provisional center-path witnesses, "
        f"{summary['certified_receiver_cell_count']} certified receiver cells"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
