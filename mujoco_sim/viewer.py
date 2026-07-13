"""Legacy launcher for :mod:`mujoco_sim.apps.viewer`."""
from ._compat import export_module

_implementation = export_module(globals(), "mujoco_sim.apps.viewer")

if __name__ == "__main__":
    _implementation.main()
