"""Legacy launcher for :mod:`mujoco_sim.apps.visualize_pipeline`."""
from ._compat import export_module

_implementation = export_module(globals(), "mujoco_sim.apps.visualize_pipeline")

if __name__ == "__main__":
    _implementation.main()
