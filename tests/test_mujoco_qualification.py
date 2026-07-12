"""Coverage and physical certification are deliberately separate."""
from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from mujoco_sim.qualification import (CoverageOutcome,
                                      build_coverage_certificate,
                                      physical_prerequisites)  # noqa: E402
from mujoco_sim.project import Project  # noqa: E402


def test_full_policy_coverage_does_not_hide_missing_physics():
    report = build_coverage_certificate([
        CoverageOutcome("g0", "direct", "ok"),
        CoverageOutcome("g1", "reorientation", "ok"),
    ], physical_prerequisites={"articulated_gripper": False})
    assert report["fraction"] == 1.0
    assert report["mathematical_coverage_certified"]
    assert not report["physical_certified"]


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
