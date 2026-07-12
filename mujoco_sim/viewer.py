"""Launch the calibrated scene in MuJoCo's managed viewer.

On macOS run this with ordinary ``python``, not ``mjpython``. The latter is
only required when an application uses ``launch_passive``.
"""
from __future__ import annotations

import argparse

import mujoco.viewer

from .project import DEFAULT_PROJECT
from .sim import MODEL, WorkcellSim


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--model", default=MODEL)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    sim = WorkcellSim(model_path=args.model, project_path=args.project)
    mujoco.viewer.launch(sim.model, sim.data)


if __name__ == "__main__":
    main()
