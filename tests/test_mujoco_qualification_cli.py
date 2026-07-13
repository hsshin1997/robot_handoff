"""Qualification CLI provenance and truncation semantics."""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from mujoco_sim.offline_tools.qualification import PHYSICAL_PREREQUISITE_KEYS  # noqa: E402
from scripts import qualify_pipeline  # noqa: E402


class FakeProject:
    initial_grasp_domain_source = "known_start_plus_geometry_library"
    T_tcp_part_start = np.eye(4)
    manifest_path = __file__
    active_part_path = __file__
    solver = {"schema_version": 1}

    @staticmethod
    def gripper(robot):
        return SimpleNamespace(
            model_path=__file__, opening_min=0.0, opening_max=0.1,
            finger_depth=0.02, pad_size=np.array([0.01, 0.02]))


class FakePlanner:
    def __init__(self, *args, **kwargs):
        self.project = FakeProject()
        first = np.eye(4); first[0, 3] = 0.01
        second = np.eye(4); second[0, 3] = 0.02
        self.g_B_candidates = [("g1", first), ("g2", second)]
        self.g_A_start = np.eye(4)
        self.X_start = np.eye(4)
        self.q_start = {"A": np.zeros(6), "B": np.zeros(6)}
        self.kin = SimpleNamespace(fk=lambda robot, q: np.eye(4))

    @staticmethod
    def search_direct(grasp, return_best=False):
        return object(), 1, []

    @staticmethod
    def search_regrasp(stats):
        raise AssertionError("direct fake should prevent reorientation")


def run_qualification(max_classes):
    prerequisites = {name: False for name in PHYSICAL_PREREQUISITE_KEYS}
    with patch.object(qualify_pipeline, "WorkcellSim", lambda **kwargs: object()), \
            patch.object(qualify_pipeline, "HandoffPlanner", FakePlanner), \
            patch.object(qualify_pipeline, "physical_prerequisites",
                         lambda project: prerequisites):
        return qualify_pipeline.qualify(
            max_classes=max_classes, model=__file__)


def test_truncated_prefix_cannot_be_labeled_full_domain_certificate():
    report = run_qualification(1)
    declaration = report["domain_declaration"]
    assert declaration["generated_class_count"] == 3
    assert declaration["evaluated_class_count"] == 1
    assert declaration["truncated_prefix_smoke"]
    assert report["prefix_smoke_passed"]
    assert not report["evaluated_domain_complete"]
    assert not report["mathematical_coverage_certified"]
    assert not report["physical_certified"]


def test_full_qualification_contains_auditable_provenance():
    report = run_qualification(None)
    assert report["evaluated_domain_complete"]
    assert report["mathematical_coverage_certified"]
    assert report["covered_count"] == report["domain_count"] == 3
    provenance = report["provenance"]
    assert provenance["producer"] == "scripts/qualify_pipeline.py"
    for name in ("project_manifest_sha256", "solver_sha256",
                 "compiled_model_sha256", "active_part_sha256",
                 "gripper_A_sha256", "gripper_B_sha256"):
        assert len(provenance[name]) == 64


def test_qualification_rejects_nonpositive_smoke_limit():
    for value in (0, -1, True):
        try:
            qualify_pipeline.qualify(max_classes=value, model=__file__)
        except ValueError as error:
            assert "positive integer" in str(error)
        else:
            raise AssertionError(f"invalid max_classes accepted: {value!r}")


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"PASS  {test.__name__}")
    print(f"\n{len(tests)}/{len(tests)} passed")
