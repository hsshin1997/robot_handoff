#!/usr/bin/env python3
"""Run the current MuJoCo test tiers without requiring pytest.

Legacy ``src/``/PyBullet tests are deliberately excluded: they exercise a
different planner and cannot be used as release evidence for this pipeline.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]

TIERS = {
    "t1": (
        "tests/test_mujoco_se3.py",
        "tests/test_geometry_grasps.py",
        "tests/test_motion_planning.py",
        "tests/test_mujoco_cad_preprocess.py",
        "tests/test_mujoco_part_mesh.py",
        "tests/test_mujoco_placements.py",
        "tests/test_mujoco_pose_templates.py",
        "tests/test_mujoco_offline.py",
        "tests/test_mujoco_reachability.py",
        "tests/test_mujoco_task_graph.py",
        "tests/test_mujoco_qualification.py",
        "tests/test_mujoco_qualification_cli.py",
        "tests/test_mujoco_profiling.py",
        "tests/test_mujoco_planner_stages.py",
        "tests/test_mujoco_timing.py",
        "tests/test_mujoco_plan_validation.py",
        "tests/test_mujoco_learning.py",
    ),
    "t2": (
        "tests/test_mujoco.py",
        "tests/test_mujoco_project.py",
        "tests/test_mujoco_scene.py",
        "tests/test_mujoco_gripper.py",
        "tests/test_mujoco_collision_policy.py",
        "tests/test_mujoco_contact_modes.py",
        "tests/test_mujoco_pipeline.py",
        "tests/test_mujoco_planning_robustness.py",
        "tests/test_mujoco_pipeline_exec.py",
        "tests/test_mujoco_debug_artifacts.py",
        "tests/test_mujoco_cli_paths.py",
        "tests/test_mujoco_experiments.py",
        "tests/test_mujoco_reorientation.py",
    ),
    "t3": (
        "tests/test_mujoco_e2e.py",
    ),
}


def selected_tests(tier: str) -> tuple[str, ...]:
    if tier == "all":
        return tuple(path for name in ("t1", "t2", "t3")
                     for path in TIERS[name])
    return TIERS[tier]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tier", choices=("t1", "t2", "t3", "all"), default="t1",
        help="t1 pure/fast, t2 scene integration, t3 release end-to-end")
    parser.add_argument("--list", action="store_true",
                        help="print selected files without running them")
    parser.add_argument("--continue-on-failure", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tests = selected_tests(args.tier)
    if args.list:
        print("\n".join(tests))
        return 0
    started = time.perf_counter()
    failures = []
    for index, relative in enumerate(tests, start=1):
        path = ROOT / relative
        if not path.is_file():
            failures.append((relative, "missing"))
            if not args.continue_on_failure:
                break
            continue
        print(f"\n[{index}/{len(tests)}] {relative}", flush=True)
        completed = subprocess.run([sys.executable, str(path)], cwd=ROOT)
        if completed.returncode:
            failures.append((relative, f"exit {completed.returncode}"))
            if not args.continue_on_failure:
                break
    elapsed = time.perf_counter() - started
    if failures:
        print(f"\nFAILED {len(failures)} file(s) in {elapsed:.2f} s")
        for path, reason in failures:
            print(f"  {path}: {reason}")
        return 1
    print(f"\nPASS {len(tests)} test files in {elapsed:.2f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
