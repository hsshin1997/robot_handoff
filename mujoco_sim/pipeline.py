"""Stable ``python -m`` launcher for :mod:`mujoco_sim.apps.pipeline`."""
from .apps.pipeline import build_parser, main, plan_and_execute

__all__ = ["build_parser", "main", "plan_and_execute"]

if __name__ == "__main__":
    main()
