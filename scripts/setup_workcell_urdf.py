#!/usr/bin/env python3
"""Generate a vendor-neutral workcell URDF from a YAML manifest."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mujoco_sim.modeling.workcell_urdf_generator import generate_workcell_urdf


DEFAULT_MANIFEST = ROOT / "config" / "workcell_generator.yaml"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build one connected workcell URDF plus camera calibration metadata "
            "from YAML."
        )
    )
    parser.add_argument(
        "manifest",
        nargs="?",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"workcell YAML manifest (default: {DEFAULT_MANIFEST})",
    )
    parser.add_argument("--output", type=Path, help="override output URDF path")
    parser.add_argument(
        "--camera-info-dir",
        type=Path,
        help="override the camera-info output directory",
    )
    parser.add_argument("--report", type=Path, help="override report YAML path")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate and build in memory without writing any files",
    )
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    result = generate_workcell_urdf(
        arguments.manifest,
        output_override=arguments.output,
        camera_info_dir_override=arguments.camera_info_dir,
        report_override=arguments.report,
        write_files=not arguments.validate_only,
    )
    inventory = result.report["inventory"]
    action = "Validated" if arguments.validate_only else "Generated"
    print(
        f"{action} {result.report['workcell']}: "
        f"{inventory['link_count']} links, {inventory['joint_count']} joints, "
        f"{inventory['robot_count']} robots, {inventory['gripper_count']} grippers, "
        f"{inventory['camera_count']} cameras"
    )
    if not arguments.validate_only:
        print(f"URDF: {result.urdf_path}")
        for camera_info_path in result.camera_info_paths:
            print(f"Camera info: {camera_info_path}")
        print(f"Report: {result.report_path}")
    for warning in result.report["warnings"]:
        print(f"Warning: {warning}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
