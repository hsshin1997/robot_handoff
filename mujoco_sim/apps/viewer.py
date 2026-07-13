"""Launch the calibrated scene in MuJoCo's passive viewer.

On macOS run ``mjpython -m mujoco_sim.viewer``. Linux can use ordinary
``python``. The explicit synchronization loop keeps the native window alive
and uses the same supported viewer path as the animated pipeline modules.
"""
from __future__ import annotations

import argparse
import time

import mujoco.viewer

from ..modeling.project import DEFAULT_PROJECT
from ..simulation.workcell import MODEL, WorkcellSim


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--model", default=MODEL)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    sim = WorkcellSim(model_path=args.model, project_path=args.project)
    print("Opening MuJoCo scene...", flush=True)
    try:
        with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
            with viewer.lock():
                viewer.cam.lookat[:] = [0.425, -0.175, 0.50]
                viewer.cam.distance = 3.15
                viewer.cam.azimuth = 135
                viewer.cam.elevation = -18
            viewer.sync()
            print("Scene loaded. Close the MuJoCo window to exit.", flush=True)
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.02)
    except RuntimeError as error:
        if "requires that the Python script be run under `mjpython`" in str(error):
            raise SystemExit(
                "On macOS, launch this viewer with: "
                "mjpython -m mujoco_sim.viewer"
            ) from None
        raise


if __name__ == "__main__":
    main()
