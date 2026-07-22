from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mujoco_sim.apps.insertion_set_visualization import (
    _validate_robot_partition,
    _validate_handoff_partition,
    build_visualization_data,
    _decorate_robot,
    _decorate_handoff,
    _task_pose_records,
    _validate_artifact_chain,
    render_fragment,
    render_standalone,
)
from mujoco_sim.modeling.insertion_task_set import artifact_sha256
from mujoco_sim.planner.handoff_preimage_set import build_handoff_preimage_set


def test_task_pose_records_preserve_cell_semantics() -> None:
    artifact = {
        "cells": [
            {
                "id": "cell_a",
                "classification": "UNRESOLVED",
                "classification_reason": "needs proof",
                "contact_mode": "housing_opposed_z",
                "bounds": {
                    "u_P_m": [0.001, 0.002],
                    "v_P_m": [0.003, 0.004],
                    "roll_rad": [0.0, 1.5707963267948966],
                },
                "representative": {
                    "seed_grasp_id": "g_a",
                    "T_P_E": [
                        [1, 0, 0, 0.001],
                        [0, 1, 0, 0.002],
                        [0, 0, 1, 0.003],
                        [0, 0, 0, 1],
                    ],
                    "required_aperture_m": 0.005,
                    "quality": 0.7,
                },
            },
            {
                "id": "empty",
                "classification": "REJECTED",
                "contact_mode": "housing_opposed_y",
                "bounds": {},
                "center_pose": {
                    "T_P_E": [
                        [1, 0, 0, 0.004],
                        [0, 1, 0, 0.005],
                        [0, 0, 1, 0.006],
                        [0, 0, 0, 1],
                    ],
                    "required_aperture_m": 0.007,
                },
                "representative": None,
                "witnesses": [],
            },
        ]
    }
    records = _task_pose_records(artifact)
    assert len(records) == 2
    assert records[0]["id"] == "cell_a"
    assert records[0]["T_P_E"][0][3] == 1.0
    assert records[0]["required_aperture_mm"] == 5.0
    assert records[0]["bounds"]["roll_deg"] == [0.0, 90.0]
    assert records[1]["id"] == "empty"
    assert records[1]["T_P_E"][0][3] == 4.0
    assert records[1]["has_sample_witness"] is False


def test_robot_decoration_is_fail_closed() -> None:
    records = [{"id": "a", "robot_classification": "NOT_EVALUATED", "robot_branch_count": 0}]
    _decorate_robot(
        records,
        {
            "cells": [
                {
                    "id": "a",
                    "robot_classification": "PROVISIONAL_PATH_WITNESS",
                    "accepted_provisional_branch_count": 3,
                    "certified": False,
                }
            ]
        },
    )
    assert records[0]["robot_classification"] == "PROVISIONAL_PATH_WITNESS"
    assert records[0]["robot_branch_count"] == 3
    assert records[0]["robot_certified"] is False


def test_handoff_status_is_global_not_joined_to_receiver_cells() -> None:
    records = [
        {"id": "receiver_a", "seed_grasp_id": "donor_a", "handoff_status": "NOT_EVALUATED"}
    ]
    _decorate_handoff(
        records,
        {
            "sets": {
                "direct": [], "reorientation": [], "uncovered": [],
                "unknown": [{"representative_grasp_ids": ["donor_a"]}],
            }
        },
    )
    assert records[0]["handoff_status"] == "GLOBAL_UNKNOWN"


def test_visualization_rejects_stale_cross_layer_hashes() -> None:
    transform = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    task = {
        "schema_version": 1,
        "artifact_type": "robot_independent_insertion_task_set",
        "project_id": "test-project",
        "inputs": {},
        "insertion_trajectory": {"T_B_P_insert": transform},
        "whole_cell_task_certificates": {
            "base_artifact_certificate_binding_sha256": "1" * 64,
        },
    }
    task["semantic_sha256"] = artifact_sha256(task)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        task_path = root / "task.json"
        robot_path = root / "robot.json"
        task_path.write_text(json.dumps(task), encoding="utf-8")
        robot = {
            "schema_version": 1,
            "artifact_type": "robot_conditioned_insertion_path_set",
            "source_task_set": {
                "file_sha256": "0" * 64,
                "artifact_type": task["artifact_type"],
                "semantic_sha256": task["semantic_sha256"],
                "project_id": task["project_id"],
                "task_certificate_binding_sha256": "1" * 64,
            }
        }
        robot_path.write_text(json.dumps(robot), encoding="utf-8")
        try:
            _validate_artifact_chain(
                root,
                task_set=task,
                task_set_path=task_path,
                robot_set=robot,
                robot_set_path=robot_path,
                handoff_set=None,
                socket_config={"T_B_P_insert": transform},
            )
        except ValueError as error:
            assert "not bound" in str(error)
        else:
            raise AssertionError("stale robot/task hash must be rejected")


def test_robot_overlay_rejects_unsubstantiated_certified_cell() -> None:
    center = {
        "T_P_E": [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        "required_aperture_m": 0.005,
    }
    task = {
        "cells": [{
            "id": "cell_a",
            "classification": "UNRESOLVED",
            "center_pose": center,
        }]
    }
    fabricated = {
        "cells": [{
            "id": "cell_a",
            "source_classification": "UNRESOLVED",
            "center_pose": center,
            "robot_classification": "CERTIFIED_SAFE",
            "certified": True,
            "complete_discrete_branch_count": 0,
            "accepted_provisional_branch_count": 0,
            "accepted_provisional_branch_ids": [],
        }],
    }
    with tempfile.TemporaryDirectory() as directory:
        try:
            _validate_robot_partition(
                Path(directory), task_set=task, robot_set=fabricated, checks=[]
            )
        except ValueError as error:
            assert "lacks exact certificate evidence" in str(error)
        else:
            raise AssertionError("an unsubstantiated green cell must be rejected")


def test_handoff_overlay_rejects_positive_without_certified_receiver() -> None:
    fabricated = {
        "task_id": "task",
        "receiver_pose_id": "receiver",
        "evidence_contract_version": 2,
        "current_grasp_domain": {
            "id": "domain",
            "sha256": "d" * 64,
            "complete": True,
            "class_count": 1,
        },
        "receiver_insertion_set": {"certified_receiver_cell_ids": []},
        "sets": {
            "direct": [{
                "task_id": "task",
                "receiver_pose_id": "receiver",
                "class_id": "class_a",
                "representative_grasp_ids": ["g0"],
                "domain": None,
                "current_grasp_domain_sha256": "d" * 64,
                "current_class_sha256": "c" * 64,
                "status": "DIRECT",
                "receiver_cell_id": "not_certified",
                "evidence_ids": [],
                "missing_inputs": [],
            }],
            "reorientation": [],
            "uncovered": [],
            "unknown": [],
        },
        "summary": {
            "class_count": 1,
            "direct_count": 1,
            "reorientation_count": 0,
            "uncovered_count": 0,
            "unknown_count": 0,
            "positive_preimage_count": 1,
        },
        "certification": {
            "positive_membership_sound": True,
            "verified_classification_count": 1,
            "partition_complete": True,
            "coverage_certified": True,
            "missing_inputs": [],
        },
        "edge_evidence_audit": [],
        "current_class_coverage_audit": [],
    }
    try:
        _validate_handoff_partition(
            robot_set={"certified_receiver_cell_ids": []},
            handoff_set=fabricated,
            checks=[],
        )
    except ValueError as error:
        assert "positive context mismatch" in str(error)
    else:
        raise AssertionError("a positive class without a certified receiver must fail")


def test_handoff_overlay_accepts_builder_verified_positive_contract() -> None:
    # Reuse the layer-3 adversarial fixture so the viewer and classifier cannot
    # silently drift to incompatible positive-evidence contracts.
    from tests.test_handoff_preimage_set import (
        RECEIVER_SHA,
        _direct_case,
        _receiver,
    )

    declarations, catalog = _direct_case()
    receiver = _receiver(certified=True)
    result = build_handoff_preimage_set(
        receiver,
        declarations,
        evidence_catalog=catalog,
        receiver_source={"sha256": RECEIVER_SHA, "status": "LOADED"},
    )
    checks: list[str] = []
    _validate_handoff_partition(
        robot_set=receiver,
        handoff_set=result,
        checks=checks,
    )
    assert checks == ["layer3:internal_class_partition"]


def test_real_three_layer_payload_is_current_and_inline_safe() -> None:
    project = ROOT / "projects/connector_header_insertion/generated/sets"
    data = build_visualization_data(
        ROOT,
        task_set_path=project / "insertion_task_set.json",
        robot_set_path=project / "robot_insertion_set.json",
        handoff_set_path=project / "handoff_preimage_set.json",
        point_budget={"board": 8, "connector": 8, "body": 8, "finger": 8},
    )
    assert len(data["poses"]) == 2304
    assert data["task_counts"] == {
        "cells": 2304,
        "cells_with_seed_representative": 1728,
        "pose_library_seed_records": 4798,
        "rejected": 1344,
        "safe": 0,
        "unresolved": 960,
    }
    assert data["robot_counts"]["numerical_center_evaluated_count"] == 24
    assert data["robot_counts"]["provisional_center_path_witness_count"] == 12
    assert data["robot_counts"]["numerically_unresolved_count"] == 12
    assert data["handoff_counts"]["unknown_count"] == 1
    assert data["provenance"]["label"] == "layers 1–3 chain verified"
    fragment = render_fragment(data)
    assert len(fragment.encode("utf-8")) < 2_000_000


def test_fragment_is_inline_and_under_size_limit() -> None:
    data = {
        "task_counts": {}, "robot_counts": {}, "handoff_counts": {},
        "claim_note": "test", "poses": [], "geometry": {},
    }
    fragment = render_fragment(data)
    assert "<html" not in fragment.lower()
    assert "fetch(" not in fragment
    assert "const DATA=" in fragment
    assert len(fragment.encode("utf-8")) < 2_000_000
    standalone = render_standalone(fragment)
    assert standalone.startswith("<!doctype html>")
    assert fragment in standalone


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
    print(f"passed {len(tests)} insertion-set visualization tests")
