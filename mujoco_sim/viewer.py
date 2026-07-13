"""Stable ``python -m`` launcher for :mod:`mujoco_sim.apps.viewer`."""
from .apps.viewer import build_parser, main

__all__ = ["build_parser", "main"]

if __name__ == "__main__":
    main()
