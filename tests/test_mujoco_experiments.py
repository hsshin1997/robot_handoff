"""Import, CLI, and physical-preflight checks for guarded experiments."""
from __future__ import annotations

from contextlib import redirect_stderr
import importlib
import io
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mujoco_sim.experiments._common import (  # noqa: E402
    extract_direct_plan_payload,
    insertion_preflight,
)
from mujoco_sim.modeling.project import DEFAULT_PROJECT, Project  # noqa: E402


MODULES = (
    "transport_speed",
    "cograsp_tolerance",
    "insertion_funnel",
)


def test_experiment_modules_import_without_legacy_exec_or_plan_api():
    for name in MODULES:
        module = importlib.import_module(f"mujoco_sim.experiments.{name}")
        assert callable(module.build_parser)
        assert callable(module.preflight)
        assert callable(module.main)


def test_help_works_as_direct_script_and_mentions_current_project():
    for name in MODULES:
        completed = subprocess.run(
            [sys.executable, str(ROOT / "mujoco_sim" / "experiments" / f"{name}.py"), "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, (name, completed.stderr)
        assert "--project" in completed.stdout
        assert "--plan" in completed.stdout
        assert "Traceback" not in completed.stderr


def test_current_project_preflights_block_before_any_trial_or_plan_search():
    expected = {
        "transport_speed": ("static mesh", "contact_material"),
        "cograsp_tolerance": ("static mesh", "contact_material"),
        "insertion_funnel": (
            "static mesh", "complete collision_cad", "contact_materials",
        ),
    }
    for name in MODULES:
        module = importlib.import_module(f"mujoco_sim.experiments.{name}")
        # If the preflight accidentally passes, this makes an expensive plan
        # search a test failure rather than silently doing work.
        module.load_or_derive_direct_plan = lambda *_: (_ for _ in ()).throw(
            AssertionError("plan search was reached before physical preflight")
        )
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            status = module.main(["--project", DEFAULT_PROJECT])
        message = stderr.getvalue()
        assert status == 2
        assert "BLOCKED:" in message
        assert "No simulation trials were run" in message
        for fragment in expected[name]:
            assert fragment in message, (name, message)


def test_current_project_insertion_preflight_does_not_accept_visual_convex_hull():
    project = Project()
    report = insertion_preflight(project)
    assert not report.ready
    assert any("visual mesh convex hull" in item for item in report.missing)
    assert any("PCB-hole/chamfer" in item for item in report.missing)


def test_current_pipeline_json_envelopes_extract_direct_payload():
    direct = {"X_handoff": [[1, 0, 0, 0]] * 4, "trajectories": {}}
    assert extract_direct_plan_payload(direct) is direct
    assert extract_direct_plan_payload({"direct": direct}) is direct
    assert extract_direct_plan_payload({"planning": {"direct": direct}}) is direct
    assert extract_direct_plan_payload(
        {"planning": {"direct": None, "regrasp": {"direct": direct}}}
    ) is direct
    assert extract_direct_plan_payload({"plan": direct}) is direct
    try:
        extract_direct_plan_payload({"segments": {}, "g": []})
    except ValueError as error:
        assert "current direct plan" in str(error)
    else:
        raise AssertionError("discarded prototype plan JSON was accepted")


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS  {test.__name__}")
    print(f"\n{len(tests)}/{len(tests)} passed")
