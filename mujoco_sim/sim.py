"""State wrapper for the compiled workcell, two GP7s, fixtures, and active part.

The wrapper owns only MuJoCo state and ideal-weld ownership. Geometry-driven
grasp generation, task planning, collision policy, and transactional execution
remain in their dedicated modules.
"""
from __future__ import annotations

import os
import math

import mujoco
import numpy as np

from .project import DEFAULT_PROJECT, Project

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "models", "scene.xml")
ARM_JOINTS = ("s", "l", "u", "r", "b", "t")


def _rpy_matrix(rpy_rad) -> np.ndarray:
    roll, pitch, yaw = rpy_rad
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


class WorkcellSim:
    def __init__(self, model_path: str = MODEL, keyframe: str = "inspection",
                 project_path: str = DEFAULT_PROJECT):
        self.model_path = os.path.abspath(model_path)
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        # Short aliases retained because they mirror MuJoCo's m/d notation.
        self.m = self.model
        self.d = self.data
        self._qpos = {
            (robot, joint): int(self.model.joint(f"{robot}_{joint}").qposadr[0])
            for robot in ("A", "B") for joint in ARM_JOINTS
        }
        self._actuator = {
            (robot, joint): int(self.model.actuator(f"{robot}_{joint}_act").id)
            for robot in ("A", "B") for joint in ARM_JOINTS
        }
        key = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, keyframe)
        if key < 0:
            raise ValueError(f"unknown scene keyframe: {keyframe}")
        mujoco.mj_resetDataKeyframe(self.model, self.data, key)
        mujoco.mj_forward(self.model, self.data)
        self.project = Project(project_path)
        self.apply_active_grasp()

    def arm_qpos(self, robot: str) -> np.ndarray:
        return np.array([self.data.qpos[self._qpos[robot, j]] for j in ARM_JOINTS])

    def set_arm_qpos(self, robot: str, qpos, hold: bool = True) -> None:
        qpos = np.asarray(qpos, dtype=float)
        if qpos.shape != (6,):
            raise ValueError(f"expected six GP7 joint positions, got {qpos.shape}")
        for joint, value in zip(ARM_JOINTS, qpos):
            self.data.qpos[self._qpos[robot, joint]] = value
            if hold:
                self.data.ctrl[self._actuator[robot, joint]] = value
        mujoco.mj_forward(self.model, self.data)

    def body_pose(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        body = self.data.body(name)
        return body.xpos.copy(), body.xmat.reshape(3, 3).copy()

    def site_pose(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        site = self.data.site(name)
        return site.xpos.copy(), site.xmat.reshape(3, 3).copy()

    def set_part_in_tcp(self, robot: str, position_m, rpy_deg) -> None:
        """Place the part using T_world_part = T_world_tcp @ T_tcp_part."""
        tcp_pos, tcp_rot = self.site_pose(f"{robot}_tcp")
        part_in_tcp = np.asarray(position_m, dtype=float)
        part_rot_tcp = _rpy_matrix(np.radians(np.asarray(rpy_deg, dtype=float)))
        world_pos = tcp_pos + tcp_rot @ part_in_tcp
        world_rot = tcp_rot @ part_rot_tcp
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, world_rot.ravel())
        address = int(self.model.joint("part_free").qposadr[0])
        self.data.qpos[address:address + 3] = world_pos
        self.data.qpos[address + 3:address + 7] = quat
        dof = int(self.model.joint("part_free").dofadr[0])
        self.data.qvel[dof:dof + 6] = 0
        mujoco.mj_forward(self.model, self.data)

    def set_part_world(self, transform: np.ndarray) -> None:
        """Set the free part from a homogeneous ``^W T_P`` transform."""
        transform = np.asarray(transform, dtype=float)
        if transform.shape != (4, 4):
            raise ValueError(f"expected a 4x4 transform, got {transform.shape}")
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, transform[:3, :3].ravel())
        address = int(self.model.joint("part_free").qposadr[0])
        self.data.qpos[address:address + 3] = transform[:3, 3]
        self.data.qpos[address + 3:address + 7] = quat
        dof = int(self.model.joint("part_free").dofadr[0])
        self.data.qvel[dof:dof + 6] = 0
        mujoco.mj_forward(self.model, self.data)

    def part_pose(self) -> np.ndarray:
        position, rotation = self.body_pose("part")
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = position
        return transform

    def grasp_part(self, robot: str) -> None:
        """Weld the part at its current pose to the selected gripper."""
        for other in ("A", "B"):
            self.data.eq_active[self.model.equality(f"{other}_part_grasp").id] = 0
        equality = self.model.equality(f"{robot}_part_grasp").id
        gripper = self.model.body(f"{robot}_gripper").id
        part = self.model.body("part").id
        r_gripper = self.data.xmat[gripper].reshape(3, 3)
        r_part = self.data.xmat[part].reshape(3, 3)
        relative_position = r_gripper.T @ (self.data.xpos[part] - self.data.xpos[gripper])
        relative_rotation = r_gripper.T @ r_part
        relative_quat = np.empty(4)
        mujoco.mju_mat2Quat(relative_quat, relative_rotation.ravel())
        self.model.eq_data[equality, 0:3] = 0
        self.model.eq_data[equality, 3:6] = relative_position
        self.model.eq_data[equality, 6:10] = relative_quat
        self.model.eq_data[equality, 10] = 1
        self.data.eq_active[equality] = 1
        mujoco.mj_forward(self.model, self.data)

    def release_part(self) -> None:
        for robot in ("A", "B"):
            self.data.eq_active[self.model.equality(f"{robot}_part_grasp").id] = 0

    def apply_active_grasp(self) -> None:
        task = self.project.manifest["active_task"]
        robot = task["initial_holder"].upper()
        if robot not in ("A", "B"):
            raise ValueError("active_task.initial_holder must be A or B")
        tcp_pos, tcp_rot = self.site_pose(f"{robot}_tcp")
        T_W_E = np.eye(4)
        T_W_E[:3, :3] = tcp_rot
        T_W_E[:3, 3] = tcp_pos
        self.set_part_world(T_W_E @ self.project.T_tcp_part_start)
        self.grasp_part(robot)

    def step(self, count: int = 1) -> None:
        mujoco.mj_step(self.model, self.data, nstep=count)

    def step_for(self, seconds: float) -> None:
        self.step(max(0, round(seconds / self.model.opt.timestep)))


# Transitional name for callers that only need scene/arm state. The previous
# contact-handoff methods were intentionally removed with the old scene.
HandoffSim = WorkcellSim
