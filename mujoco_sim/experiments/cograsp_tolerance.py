"""Physical experiment: dual-gripper co-grasp capture tolerance.

The current pipeline transfers ownership with an ideal weld and its supplied
gripper is a static STL.  This command therefore reports missing contact
prerequisites instead of fabricating a capture region from virtual predicates.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mujoco_sim.experiments._common import (  # noqa: E402
    ExperimentBlocked,
    cograsp_preflight,
    future_backend_block,
    load_or_derive_direct_plan,
    load_project,
    parse_positive_csv,
    print_blocked,
)
from mujoco_sim.project import DEFAULT_PROJECT  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=DEFAULT_PROJECT,
                        help="current project.yaml manifest")
    parser.add_argument(
        "--plan", default=None,
        help="optional current pipeline JSON; if omitted, derive a plan after preflight",
    )
    parser.add_argument("--n", type=int, default=20, help="trials per nonzero offset")
    parser.add_argument(
        "--eps", default="0.0,0.0005,0.001,0.002,0.004",
        help="comma-separated lateral offsets in metres",
    )
    parser.add_argument("--seed", type=int, default=0,
                        help="deterministic Monte Carlo seed")
    return parser


def preflight(project_path: str = DEFAULT_PROJECT):
    project = load_project(project_path)
    return cograsp_preflight(project, project_path)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.n <= 0:
        parser.error("--n must be positive")
    try:
        parse_positive_csv(args.eps, label="--eps", allow_zero=True)
    except ValueError as error:
        parser.error(str(error))
    report = preflight(args.project)
    if not report.ready:
        return print_blocked(report)

    try:
        load_or_derive_direct_plan(args.project, args.plan)
        raise future_backend_block(report.experiment)
    except (ExperimentBlocked, FileNotFoundError, ValueError) as error:
        print(f"BLOCKED: {error}", file=sys.stderr)
        print("No simulation trials were run.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
