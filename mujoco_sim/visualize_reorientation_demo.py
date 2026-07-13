"""Legacy launcher for :mod:`mujoco_sim.apps.visualize_reorientation_demo`."""
from ._compat import export_module

_implementation = export_module(
    globals(), "mujoco_sim.apps.visualize_reorientation_demo")

if __name__ == "__main__":
    _implementation.main()
