"""Scene: PyBullet world with two GP7s, workcell, and (later) the part.

World frame = workcell STL frame in meters (robot A mount flange = origin, z up).
Workcell visual is the raw STL (mm, scaled 0.001); collision is the simplified
box set in assets/workcell/collision_boxes.yaml. The raw mesh is never
collision-checked.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pybullet as p
import yaml

MM = 0.001
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _pose_from_T(T) -> tuple[list[float], list[float]]:
    """4x4 (nested list or ndarray) -> (pos, quat xyzw)."""
    T = np.asarray(T, dtype=float)
    assert T.shape == (4, 4), f"expected 4x4, got {T.shape}"
    pos = T[:3, 3].tolist()
    # rotation matrix -> quaternion via pybullet (expects row-major 3x3)
    m = T[:3, :3]
    tr = np.trace(m)
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        w = 0.25 / s
        x, y, z = (m[2, 1] - m[1, 2]) * s, (m[0, 2] - m[2, 0]) * s, (m[1, 0] - m[0, 1]) * s
    else:
        i = int(np.argmax(np.diag(m)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(max(1e-12, 1.0 + m[i, i] - m[j, j] - m[k, k]))
        q = [0.0, 0.0, 0.0, 0.0]
        q[i] = 0.25 * s
        q[3] = (m[k, j] - m[j, k]) / s
        q[j] = (m[j, i] + m[i, j]) / s
        q[k] = (m[k, i] + m[i, k]) / s
        x, y, z, w = q[0], q[1], q[2], q[3]
    return pos, [x, y, z, w]


def _stl_bbox(path: str) -> tuple[np.ndarray, np.ndarray]:
    """(min, max) corners of an STL's vertices (ascii or binary)."""
    with open(path, "rb") as f:
        head = f.read(5)
    if head == b"solid":  # ascii
        import re
        txt = open(path).read()
        v = np.array(re.findall(r"vertex\s+(\S+)\s+(\S+)\s+(\S+)", txt), dtype=float)
    else:  # binary: 80-byte header, uint32 count, 50-byte records
        import struct
        with open(path, "rb") as f:
            f.seek(80)
            n = struct.unpack("<I", f.read(4))[0]
            data = np.frombuffer(f.read(n * 50), dtype=np.uint8).reshape(n, 50)
        v = data[:, 12:48].copy().view(np.float32).reshape(-1, 3).astype(float)
    return v.min(axis=0), v.max(axis=0)


@dataclass
class Robot:
    body: int
    name: str
    joint_indices: list[int] = field(default_factory=list)  # the 6 revolute joints
    link_index: dict[str, int] = field(default_factory=dict)
    lower: np.ndarray = None
    upper: np.ndarray = None

    @property
    def flange_link(self) -> int:
        return self.link_index["tool0"]

    @property
    def tcp_link(self) -> int:
        return self.link_index["tcp"]

    def set_q(self, q) -> None:
        for idx, qi in zip(self.joint_indices, q):
            p.resetJointState(self.body, idx, float(qi))

    def get_q(self) -> np.ndarray:
        return np.array([p.getJointState(self.body, i)[0] for i in self.joint_indices])


class Scene:
    def __init__(self, config_path: str = "config/cell.yaml", gui: bool = False,
                 load_visuals: bool | None = None):
        """load_visuals: force the workcell visual mesh on/off (default: only
        in GUI mode — headless planning skips the 724k-tri parse)."""
        if not os.path.isabs(config_path):
            config_path = os.path.join(ROOT, config_path)
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self.gui = gui
        self._load_visuals = gui if load_visuals is None else load_visuals
        # a fixed moderate window size avoids full-retina rendering on macOS
        self.client = (p.connect(p.GUI, options="--width=1280 --height=800")
                       if gui else p.connect(p.DIRECT))
        if gui:
            # trim GUI overhead: shadows and the RGB/depth/segmentation preview
            # panes are the main frame-rate killers with a large scene mesh
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0)
        p.setGravity(0, 0, -9.81)

        if gui:  # don't re-render after every body while loading
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        self.workcell_ids = self._load_workcell()
        self.robotA = self._load_robot("A", self.cfg["robotA_base"])
        self.robotB = self._load_robot("B", self.cfg["robotB_base"])
        self.robotA.set_q(self.cfg.get("home_qA", self.cfg.get("home_q", [0.0] * 6)))
        self.robotB.set_q(self.cfg.get("home_qB", self.cfg.get("home_q", [0.0] * 6)))

        self.part_id: int | None = None
        self._part_constraint: int | None = None
        self.fixture_ids: list[int] = []   # every static body beyond the workcell
        self.pcb_id: int | None = None
        self.bin_id: int | None = None
        self.nest_id: int | None = None
        if "T_world_pcb" in self.cfg:
            self.pcb_id = self._load_pcb()
        if "bin" in self.cfg:
            self.bin_id = self._load_bin()
        if "nest" in self.cfg:
            self.nest_id = self._load_nest()
        if "T_flangeA_part" in self.cfg:
            self.spawn_part(half_extents=self.cfg.get("part_half_extents", (0.02, 0.02, 0.02)))
            self.attach_part(self.robotA, self.cfg["T_flangeA_part"])
        if gui:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

    # ---------- loading ----------

    def _asset(self, key: str) -> str:
        return os.path.join(ROOT, self.cfg["assets"][key])

    def _load_workcell(self) -> list[int]:
        """Visual: raw STL (mm->m), GUI only — headless planning skips the
        724k-tri parse entirely. Collision: simplified boxes, one static body."""
        ids = []
        if self._load_visuals:
            vis = p.createVisualShape(
                p.GEOM_MESH,
                fileName=self._asset("workcell_visual"),
                meshScale=[MM] * 3,
                rgbaColor=[0.75, 0.78, 0.82, 1.0],
            )
            ids.append(p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis))

        # collision body from simplified boxes
        with open(self._asset("workcell_collision")) as f:
            cc = yaml.safe_load(f)
        entries = cc.get("pedestals", []) + cc.get("boxes", [])
        cols, viss, poss = [], [], []
        for e in entries:
            half = [h * MM for h in e["half_extents"]]
            cols.append(p.createCollisionShape(p.GEOM_BOX, halfExtents=half))
            # invisible visual so the render shows the STL, not the bounding boxes
            viss.append(p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=[0, 0, 0, 0]))
            poss.append([c * MM for c in e["center"]])
        body = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            linkMasses=[0.0] * len(cols),
            linkCollisionShapeIndices=cols,
            linkVisualShapeIndices=viss,
            linkPositions=poss,
            linkOrientations=[[0, 0, 0, 1]] * len(cols),
            linkInertialFramePositions=[[0, 0, 0]] * len(cols),
            linkInertialFrameOrientations=[[0, 0, 0, 1]] * len(cols),
            linkParentIndices=[0] * len(cols),
            linkJointTypes=[p.JOINT_FIXED] * len(cols),
            linkJointAxis=[[0, 0, 1]] * len(cols),
        )
        ids.append(body)
        self.workcell_collision_id = body

        # floor plane at the cell feet
        floor_z = float(self.cfg.get("floor_z", -0.610))
        plane = p.createCollisionShape(p.GEOM_PLANE)
        ids.append(p.createMultiBody(0, plane, basePosition=[0, 0, floor_z]))
        return ids

    def _load_robot(self, name: str, base_T) -> Robot:
        pos, quat = _pose_from_T(base_T)
        body = p.loadURDF(
            self._asset("gp7_urdf"),
            basePosition=pos,
            baseOrientation=quat,
            useFixedBase=True,
            flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT,
        )
        r = Robot(body=body, name=name)
        lows, ups = [], []
        for i in range(p.getNumJoints(body)):
            info = p.getJointInfo(body, i)
            jname = info[1].decode()
            lname = info[12].decode()
            r.link_index[lname] = i
            if info[2] == p.JOINT_REVOLUTE:
                r.joint_indices.append(i)
                lows.append(info[8])
                ups.append(info[9])
        r.lower, r.upper = np.array(lows), np.array(ups)
        assert len(r.joint_indices) == 6, f"expected 6 revolute joints, got {len(r.joint_indices)}"
        assert "tool0" in r.link_index and "tcp" in r.link_index
        return r

    def _static_boxes(self, entries, rgba) -> int:
        """One static multibody from [(center, half_extents), ...]."""
        cols = [p.createCollisionShape(p.GEOM_BOX, halfExtents=list(h)) for _, h in entries]
        viss = [p.createVisualShape(p.GEOM_BOX, halfExtents=list(h), rgbaColor=list(rgba))
                for _, h in entries]
        n = len(entries)
        body = p.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=-1,
            linkMasses=[0.0] * n, linkCollisionShapeIndices=cols,
            linkVisualShapeIndices=viss, linkPositions=[list(c) for c, _ in entries],
            linkOrientations=[[0, 0, 0, 1]] * n,
            linkInertialFramePositions=[[0, 0, 0]] * n,
            linkInertialFrameOrientations=[[0, 0, 0, 1]] * n,
            linkParentIndices=[0] * n, linkJointTypes=[p.JOINT_FIXED] * n,
            linkJointAxis=[[0, 0, 1]] * n,
        )
        self.fixture_ids.append(body)
        return body

    def _load_pcb(self) -> int:
        """PCB as a static box at T_world_pcb (insertion target), on a stand."""
        half = list(self.cfg.get("pcb_half_extents", (0.05, 0.04, 0.0008)))
        pos, quat = _pose_from_T(self.cfg["T_world_pcb"])
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=[0.1, 0.5, 0.15, 1.0])
        body = p.createMultiBody(0, col, vis, basePosition=pos, baseOrientation=quat)
        self.fixture_ids.append(body)
        floor_z = float(self.cfg.get("floor_z", -0.610))
        top = pos[2] - half[2]
        stand_h = (top - floor_z) / 2.0
        self._static_boxes([([pos[0], pos[1], floor_z + stand_h], [0.03, 0.03, stand_h])],
                           rgba=[0.45, 0.45, 0.48, 1.0])
        return body

    def _load_bin(self) -> int:
        """Open-top bin (floor + 4 walls) on a stand to the cell floor."""
        b = self.cfg["bin"]
        cx, cy = b["center_xy"]
        li, wi = b["interior"]          # interior length x width
        t = b["wall_thickness"]
        h = b["wall_height"]
        z0 = b["interior_floor_z"]      # inside bottom face
        floor_z = float(self.cfg.get("floor_z", -0.610))
        wz = z0 + h / 2.0               # wall center z
        entries = [
            ([cx, cy, z0 - t / 2.0], [li / 2 + t, wi / 2 + t, t / 2.0]),          # bottom
            ([cx - li / 2 - t / 2, cy, wz], [t / 2.0, wi / 2 + t, h / 2.0]),      # -x wall
            ([cx + li / 2 + t / 2, cy, wz], [t / 2.0, wi / 2 + t, h / 2.0]),      # +x wall
            ([cx, cy - wi / 2 - t / 2, wz], [li / 2 + t, t / 2.0, h / 2.0]),      # -y wall
            ([cx, cy + wi / 2 + t / 2, wz], [li / 2 + t, t / 2.0, h / 2.0]),      # +y wall
        ]
        stand_h = (z0 - t - floor_z) / 2.0
        entries.append(([cx, cy, floor_z + stand_h], [0.04, 0.04, stand_h]))      # stand
        return self._static_boxes(entries, rgba=[0.25, 0.35, 0.55, 1.0])

    def _load_nest(self) -> int:
        """Flat reorientation plate on a stand to the cell floor."""
        n = self.cfg["nest"]
        cx, cy = n["center_xy"]
        lx, ly = n["size"]
        top = n["top_z"]
        t = 0.012
        floor_z = float(self.cfg.get("floor_z", -0.610))
        stand_h = (top - t - floor_z) / 2.0
        entries = [
            ([cx, cy, top - t / 2.0], [lx / 2.0, ly / 2.0, t / 2.0]),             # plate
            ([cx, cy, floor_z + stand_h], [0.04, 0.04, stand_h]),                 # stand
        ]
        return self._static_boxes(entries, rgba=[0.75, 0.68, 0.45, 1.0])

    # ---------- part handling ----------

    def spawn_part(self, half_extents=(0.02, 0.02, 0.02), pos=(0, 0, 1.0), rgba=(0.9, 0.3, 0.1, 1)) -> int:
        """Part body. Uses the part_mesh from config if present (convex hull
        collision, part frame = mesh bbox center), else a box."""
        mesh = self.cfg.get("part_mesh")
        if mesh:
            path = os.path.join(ROOT, mesh)
            lo, hi = _stl_bbox(path)
            center = (-0.5 * (lo + hi)).tolist()  # recenter: part frame = bbox center
            col = p.createCollisionShape(p.GEOM_MESH, fileName=path, collisionFramePosition=center)
            vis = p.createVisualShape(p.GEOM_MESH, fileName=path, visualFramePosition=center,
                                      rgbaColor=list(rgba))
        else:
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=list(half_extents))
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=list(half_extents), rgbaColor=list(rgba))
        self.part_id = p.createMultiBody(0.05, col, vis, basePosition=list(pos))
        return self.part_id

    def attach_part(self, robot: Robot, T_flange_part) -> None:
        """Rigidly attach the part to robot's flange (tool0) at T_flange_part."""
        assert self.part_id is not None, "spawn_part first"
        self.detach_part()
        rel_pos, rel_quat = _pose_from_T(T_flange_part)
        # place the part consistently before constraining
        ls = p.getLinkState(robot.body, robot.flange_link)
        wpos, wquat = p.multiplyTransforms(ls[4], ls[5], rel_pos, rel_quat)
        p.resetBasePositionAndOrientation(self.part_id, wpos, wquat)
        self._part_constraint = p.createConstraint(
            robot.body, robot.flange_link, self.part_id, -1,
            p.JOINT_FIXED, [0, 0, 0], rel_pos, [0, 0, 0],
            parentFrameOrientation=rel_quat,
        )

    def detach_part(self) -> None:
        if self._part_constraint is not None:
            p.removeConstraint(self._part_constraint)
            self._part_constraint = None

    # ---------- misc ----------

    def disconnect(self) -> None:
        p.disconnect(self.client)
