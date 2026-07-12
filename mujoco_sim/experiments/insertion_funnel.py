"""Physical experiment: pin/hole insertion success versus lateral error.

The semantic insertion frame is available, but the current project has no
pin/hole collision CAD or calibrated contact materials.  This entry point
fails before loading/deriving a plan so a kinematic target cannot be mistaken
for an empirical insertion funnel.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mujoco_sim.experiments._common import (  # noqa: E402
    ExperimentBlocked,
    future_backend_block,
    insertion_preflight,
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
    parser.add_argument("--n", type=int, default=8, help="trials per nonzero offset")
    parser.add_argument(
        "--eps", default="0.0,0.0003,0.0006,0.001,0.002",
        help="comma-separated lateral offsets in metres",
    )
    parser.add_argument("--seed", type=int, default=1,
                        help="deterministic Monte Carlo seed")
    return parser


def preflight(project_path: str = DEFAULT_PROJECT):
    project = load_project(project_path)
    return insertion_preflight(project, project_path)


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
