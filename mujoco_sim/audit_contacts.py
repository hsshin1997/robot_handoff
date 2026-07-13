"""Legacy launcher for :mod:`mujoco_sim.diagnostics.contact_audit`."""
from functools import wraps

from ._compat import export_module

_implementation = export_module(
    globals(), "mujoco_sim.diagnostics.contact_audit")
_canonical_audit_insertion = _implementation.audit_insertion
_canonical_audit_reorientation = _implementation.audit_reorientation


@wraps(_canonical_audit_insertion)
def audit_insertion(*args, **kwargs):
    kwargs.setdefault("contact_reader", _contacts)
    return _canonical_audit_insertion(*args, **kwargs)


@wraps(_canonical_audit_reorientation)
def audit_reorientation(*args, **kwargs):
    kwargs.setdefault("contact_reader", _contacts)
    return _canonical_audit_reorientation(*args, **kwargs)

if __name__ == "__main__":
    _implementation.main()
