"""Physical experiment: transport-speed limit for a friction-held part.

This experiment is intentionally blocked for the current static gripper CAD.
It will not reinterpret the pipeline's ideal weld as frictional grasping.  The
entry point remains useful as a repeatable project/physics preflight::

    python -m mujoco_sim.experiments.transport_speed --help
    python -m mujoco_sim.experiments.transport_speed
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
    load_or_derive_direct_plan,
    load_project,
    parse_positive_csv,
    print_blocked,
    transport_preflight,
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
    parser.add_argument(
        "--fracs", default="0.1,0.2,0.3,0.5,0.7,1.0",
        help="comma-separated positive speed fractions",
    )
    return parser


def preflight(project_path: str = DEFAULT_PROJECT):
    project = load_project(project_path)
    return transport_preflight(project, project_path)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        parse_positive_csv(args.fracs, label="--fracs", allow_zero=False)
    except ValueError as error:
        parser.error(str(error))
    report = preflight(args.project)
    if not report.ready:
        return print_blocked(report)

    # Plan loading uses current DirectHandoffPlan/PlanningReport JSON, never
    # the discarded prototype's `segments`/`g` dictionary.
    try:
        load_or_derive_direct_plan(args.project, args.plan)
        raise future_backend_block(report.experiment)
    except (ExperimentBlocked, FileNotFoundError, ValueError) as error:
        print(f"BLOCKED: {error}", file=sys.stderr)
        print("No simulation trials were run.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
