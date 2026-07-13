"""Coverage and physical certification are deliberately separate."""
from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from mujoco_sim.offline_tools.qualification import (CoverageOutcome,
                                      PHYSICAL_PREREQUISITE_KEYS,
                                      build_coverage_certificate,
                                      physical_prerequisites)  # noqa: E402
from mujoco_sim.modeling.project import Project  # noqa: E402


def test_full_policy_coverage_does_not_hide_missing_physics():
    report = build_coverage_certificate([
        CoverageOutcome("g0", "direct", "ok"),
        CoverageOutcome("g1", "reorientation", "ok"),
    ], physical_prerequisites={"articulated_gripper_A": False})
    assert report["fraction"] == 1.0
    assert report["mathematical_coverage_certified"]
    assert not report["physical_certified"]
    assert not report["physical_prerequisite_schema_complete"]


def test_one_uncovered_class_fails_a_hundred_percent_target():
    report = build_coverage_certificate([
        CoverageOutcome("g0", "direct", "ok"),
        CoverageOutcome("g1", None, "no_path"),
    ])
    assert report["fraction"] == 0.5
    assert report["uncovered_classes"] == ["g1"]
    assert not report["mathematical_coverage_certified"]


def test_certificate_records_domain_and_uses_minimum_target_semantics():
    report = build_coverage_certificate([
        CoverageOutcome("g0", "direct", "ok"),
        CoverageOutcome("g1", "direct", "ok"),
    ], required_fraction=0.5,
       domain_declaration={"source": "declared_test_domain"})
    assert report["fraction"] == 1.0
    assert report["mathematical_coverage_certified"]
    assert report["domain_declaration"] == {"source": "declared_test_domain"}
    assert report["required_count"] == 1
    assert report["domain_count"] == 2
    assert not report["physical_certified"]


def test_physical_certificate_fails_closed_when_prerequisites_are_omitted():
    for prerequisites in (None, {}, {"articulated_gripper_A": True}):
        report = build_coverage_certificate(
            [CoverageOutcome("g0", "direct", "ok")],
            physical_prerequisites=prerequisites)
        assert report["mathematical_coverage_certified"]
        assert not report["physical_certified"]


def test_physical_prerequisites_require_exact_boolean_schema():
    try:
        build_coverage_certificate(
            [CoverageOutcome("g0", "direct", "ok")],
            physical_prerequisites={"unknown": True})
    except ValueError as error:
        assert "unknown physical prerequisite" in str(error)
    else:
        raise AssertionError("unknown prerequisite key was accepted")

    try:
        build_coverage_certificate(
            [CoverageOutcome("g0", "direct", "ok")],
            physical_prerequisites={
                "articulated_gripper_A": "false",
            })
    except ValueError as error:
        assert "strict boolean" in str(error)
    else:
        raise AssertionError("truthy non-boolean prerequisite was accepted")


def test_physical_certification_always_requires_complete_domain():
    complete = {name: True for name in PHYSICAL_PREREQUISITE_KEYS}
    full = build_coverage_certificate([
        CoverageOutcome("g0", "direct", "ok"),
        CoverageOutcome("g1", "reorientation", "ok"),
    ], required_fraction=0.5, physical_prerequisites=complete)
    partial = build_coverage_certificate([
        CoverageOutcome("g0", "direct", "ok"),
        CoverageOutcome("g1", None, "no path"),
    ], required_fraction=0.5, physical_prerequisites=complete)
    assert full["physical_certified"]
    assert partial["mathematical_coverage_certified"]
    assert not partial["physical_certified"]


def test_coverage_outcome_rejects_invalid_runtime_values():
    for args in (("", "direct", "ok"),
                 ("g0", "unsafe", "ok"),
                 ("g0", "direct", "")):
        try:
            CoverageOutcome(*args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid outcome accepted: {args}")


def test_current_project_cannot_be_promoted_by_manifest_only():
    prerequisites = physical_prerequisites(Project())
    assert not prerequisites["articulated_gripper_scene_adapter"]
    assert not prerequisites["physical_contact_execution_backend"]
    assert not all(prerequisites.values())


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS  {test.__name__}")
    print(f"\n{len(tests)}/{len(tests)} passed")
