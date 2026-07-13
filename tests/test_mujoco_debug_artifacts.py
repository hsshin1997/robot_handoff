"""Focused checks for opt-in per-stage execution artifacts."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from mujoco_sim.collision import SceneCollisionChecker  # noqa: E402
from mujoco_sim.debug_artifacts import DebugArtifactRecorder  # noqa: E402
from mujoco_sim.exec import PipelineExecutor, PipelineState  # noqa: E402
from mujoco_sim.kinematics import GP7Kinematics  # noqa: E402
from mujoco_sim.planning import HandoffPlanner  # noqa: E402
from mujoco_sim.sim import WorkcellSim  # noqa: E402
from mujoco_sim import pipeline, visualize_pipeline, visualize_reorientation_demo  # noqa: E402


def _collision_context(sim):
    kinematics = GP7Kinematics(sim)
    checker = SceneCollisionChecker(sim, kinematics)
    return checker, checker._legacy_policy(("A",), ())


def test_recorder_writes_complete_state_and_png_fallback():
    sim = WorkcellSim()
    checker, policy = _collision_context(sim)
    with tempfile.TemporaryDirectory() as directory:
        recorder = DebugArtifactRecorder(
            directory, width=160, height=120,
            run_name="20260712T120000.000000Z")
        with patch.object(
                recorder, "_render_scene",
                side_effect=RuntimeError("test OpenGL unavailable")):
            first = recorder.record(
                "owned_by_A", sim,
                event={"state": "owned_by_A", "detail": {"test": True}},
                plan_metadata={"branch": "unit_test", "candidate": 4},
                execution_metadata={"owner": "A"},
                collision_checker=checker, policy=policy)
            second = recorder.record(
                "owned_by_A", sim, event={"state": "again"},
                collision_checker=checker, policy=policy)

        assert first is not None and second is not None
        assert Path(first.directory).parent == recorder.run_dir
        assert recorder.run_dir.name == "20260712T120000.000000Z"
        assert Path(first.directory).name == "owned_by_A"
        assert Path(second.directory).name == "owned_by_A__02"
        state = json.loads(Path(first.state_path).read_text(encoding="utf-8"))
        assert state["schema_version"] == 1
        assert state["event"]["detail"]["test"] is True
        assert state["plan"] == {"branch": "unit_test", "candidate": 4}
        assert set(state["q"]) == {"A", "B"}
        assert len(state["q"]["A"]) == len(state["q"]["B"]) == 6
        for name in ("world_tcp_A", "world_tcp_B", "world_part"):
            assert np.asarray(state["transforms"][name]).shape == (4, 4)
        assert state["collision_policy"]["part_holders"] == ["A"]
        assert state["contact_summary"]["count"] == len(state["contacts"])
        for contact in state["contacts"]:
            assert {
                "geom1", "geom2", "signed_distance_m", "penetration_m",
                "position_world_m", "wrench_contact_frame", "force_world_n",
                "allowed",
            } <= set(contact)
            assert isinstance(contact["allowed"], bool)
        assert state["render"]["fallback_image"] is True
        assert state["render"]["mujoco_rendered"] is False
        assert state["render"]["fallback_kind"] == (
            "cpu_top_side_contact_projection")
        assert "test OpenGL unavailable" in state["render"]["error"]
        assert Path(first.image_path).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        recorder.close()


def test_collision_snapshot_preserves_unexpected_pair_and_signed_distance():
    sim = WorkcellSim()
    checker, _ = _collision_context(sim)
    sim.release_part()
    transform = np.eye(4)
    transform[:3, 3] = [0.470, -0.455, 0.346]
    sim.set_part_world(transform)
    policy = checker._legacy_policy((), ())
    with tempfile.TemporaryDirectory() as directory:
        recorder = DebugArtifactRecorder(
            directory, width=96, height=72, run_name="collision-test")
        with patch.object(
                recorder, "_render_scene",
                side_effect=RuntimeError("headless")):
            record = recorder.record(
                "aborted", sim,
                event={"state": "aborted", "reason": "collision"},
                collision_checker=checker, policy=policy)
        state = json.loads(Path(record.state_path).read_text(encoding="utf-8"))
        matching = [
            item for item in state["contacts"]
            if "part_collision" in {item["geom1"], item["geom2"]}
            and any(name.startswith("pcb_board_")
                    for name in (item["geom1"], item["geom2"]))
        ]
        assert matching
        assert all(item["allowed"] is False for item in matching)
        assert min(item["signed_distance_m"] for item in matching) < 0.0
        assert max(item["penetration_m"] for item in matching) > 0.0
        assert state["contact_summary"]["unexpected_count"] >= 1
        recorder.close()


class _FailingRecorder:
    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.errors = []

    def record(self, *args, **kwargs):
        raise RuntimeError("recorder broke")


def test_executor_isolates_recorder_failure_unless_strict_debug():
    sim = WorkcellSim()
    planner = HandoffPlanner(sim)
    with tempfile.TemporaryDirectory() as directory:
        relaxed = PipelineExecutor(
            sim, planner, recorder=_FailingRecorder(directory))
        relaxed._event(PipelineState.OWNED_BY_A)
        result = relaxed._result(False, "test_complete")
        assert result.outcome == "test_complete"
        assert result.debug_run_dir == directory
        assert any("recorder broke" in error for error in result.debug_errors)

        strict = PipelineExecutor(
            sim, planner, recorder=_FailingRecorder(directory),
            strict_debug=True)
        try:
            strict._event(PipelineState.OWNED_BY_A)
        except RuntimeError as error:
            assert "recorder broke" in str(error)
        else:
            raise AssertionError("strict debug did not propagate recorder failure")


class _CollectingRecorder:
    def __init__(self):
        self.steps = []
        self.errors = []
        self.run_dir = None

    def record(self, step_name, sim, **kwargs):
        self.steps.append((step_name, kwargs["event"].state.value))


def test_named_sender_stage_is_independently_executable_and_recorded():
    sim = WorkcellSim()
    planner = HandoffPlanner(sim)
    recorder = _CollectingRecorder()
    executor = PipelineExecutor(sim, planner, recorder=recorder)
    q = sim.arm_qpos("A")
    plan = SimpleNamespace(trajectories={
        "A_current_to_pre": [q.copy(), q.copy()],
        "A_approach": [q.copy(), q.copy()],
    })
    executor._stage_sender_to_handoff(
        plan, speed=0.1, approach=0.1,
        initial_allowed_geom_pairs=())
    assert recorder.steps == [("A_at_handoff", "A_at_handoff")]


def test_default_executor_constructs_no_recorder_or_log_directory():
    sim = WorkcellSim()
    planner = HandoffPlanner(sim)
    executor = PipelineExecutor(sim, planner)
    assert executor.recorder is None
    assert executor._debug_errors == []
    assert executor.playback_speed == 1.0
    assert executor._render_stride == 10


def test_visual_playback_multiplier_reduces_sync_overhead_safely():
    sim = WorkcellSim()
    planner = HandoffPlanner(sim)
    accelerated = PipelineExecutor(sim, planner, playback_speed=4.0)
    assert accelerated.playback_speed == 4.0
    assert accelerated._render_stride == 40
    try:
        PipelineExecutor(sim, planner, playback_speed=0.0)
    except ValueError as error:
        assert "playback_speed" in str(error)
    else:
        raise AssertionError("non-positive playback speed was accepted")

    syncs = []
    accelerated.viewer = SimpleNamespace(sync=lambda: syncs.append(True))
    accelerated.realtime = True
    start = sim.arm_qpos("A")
    trajectory = [start.copy()]
    for index in range(1, 5):
        q = start.copy()
        q[0] += index * 0.0005
        trajectory.append(q)
    with patch("mujoco_sim.exec.time.sleep") as sleeper:
        accelerated._follow("A", trajectory, 0.25, ("A",))

    # Four collision-sampling edges must not inherit the old 4 * 100 ms floor.
    assert 0.0 < accelerated.estimated_robot_time_s < 0.05
    assert len(syncs) == 1  # cumulative pacing, flushed once for this path
    slept = sum(call.args[0] for call in sleeper.call_args_list)
    assert np.isclose(
        slept, accelerated.estimated_robot_time_s / 4.0, atol=1e-12)
    accelerated._event(PipelineState.A_CLEAR)
    result = accelerated._result(True, "timing_test")
    assert result.estimated_cycle_time_s == accelerated.estimated_robot_time_s
    assert result.stage_timings[-1]["completed_state"] == "A_clear"


def test_execution_clis_expose_opt_in_debug_root_and_strict_mode():
    for parser in (
        pipeline.build_parser(),
        visualize_pipeline.build_parser(),
        visualize_reorientation_demo.build_parser(),
    ):
        args = parser.parse_args([
            "--debug-artifacts", "diagnostics", "--strict-debug",
        ])
        assert args.debug_artifacts == "diagnostics"
        assert args.strict_debug is True
        defaults = parser.parse_args([])
        assert defaults.debug_artifacts is None
        assert defaults.strict_debug is False
    for parser in (
        visualize_pipeline.build_parser(),
        visualize_reorientation_demo.build_parser(),
    ):
        assert parser.parse_args([]).playback_speed == 1.0
        assert parser.parse_args(["--playback-speed", "4"]).playback_speed == 4.0
        assert parser.parse_args([]).start_delay == 1.0
        assert parser.parse_args(["--start-delay", "5"]).start_delay == 5.0


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS  {test.__name__}")
    print(f"\n{len(tests)}/{len(tests)} passed")
