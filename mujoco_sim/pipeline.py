"""Legacy launcher for :mod:`mujoco_sim.apps.pipeline`."""
from functools import wraps

from ._compat import export_module

_implementation = export_module(globals(), "mujoco_sim.apps.pipeline")
_canonical_plan_and_execute = _implementation.plan_and_execute


@wraps(_canonical_plan_and_execute)
def plan_and_execute(*args, **kwargs):
    """Forward through facade-visible factories for legacy test/integration hooks."""
    kwargs.setdefault("_sim_factory", WorkcellSim)
    kwargs.setdefault("_planner_factory", HandoffPlanner)
    kwargs.setdefault("_executor_factory", PipelineExecutor)
    return _canonical_plan_and_execute(*args, **kwargs)

if __name__ == "__main__":
    _implementation.main()
