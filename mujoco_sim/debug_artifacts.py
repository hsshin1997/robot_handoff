"""Opt-in, per-stage MuJoCo execution diagnostics.

The recorder is intentionally absent from the production fast path.  When
enabled it creates one immutable directory per execution stage containing a
machine-readable state snapshot and an offscreen MuJoCo render with contact
points enabled.  Rendering is best-effort because OpenGL availability differs
between headless Linux, macOS ``mjpython``, and CI.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
import enum
import json
from pathlib import Path
import re
import struct
from typing import Any
import zlib

import mujoco
import numpy as np

from .collision import CollisionPolicy


_SAFE_STEP = re.compile(r"[^A-Za-z0-9_.-]+")


def _jsonable(value: Any) -> Any:
    """Convert common numerical/planner values to strict JSON values."""
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, enum.Enum):
        return _jsonable(value.value)
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if hasattr(value, "__dict__"):
        return _jsonable(vars(value))
    return str(value)


def _transform(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = np.asarray(rotation, dtype=float).reshape(3, 3)
    result[:3, 3] = np.asarray(position, dtype=float)
    return result


def _png_bytes(rgb: np.ndarray) -> bytes:
    """Encode one uint8 RGB array without an optional image dependency."""
    image = np.asarray(rgb, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError(f"PNG image must be HxWx3/4, got {image.shape}")
    if image.shape[2] == 4:
        image = image[:, :, :3]
    height, width, _ = image.shape
    scanlines = b"".join(
        b"\x00" + np.ascontiguousarray(image[row]).tobytes()
        for row in range(height)
    )

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (struct.pack(">I", len(payload)) + kind + payload
                + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF))

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(scanlines, level=6))
            + chunk(b"IEND", b""))


def _draw_disk(image: np.ndarray, x: int, y: int, radius: int, color) -> None:
    height, width = image.shape[:2]
    radius = max(1, int(radius))
    x0, x1 = max(0, x - radius), min(width, x + radius + 1)
    y0, y1 = max(0, y - radius), min(height, y + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.ogrid[y0:y1, x0:x1]
    mask = (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2
    image[y0:y1, x0:x1][mask] = color


def _draw_line(image: np.ndarray, first, second, color, thickness: int = 1) -> None:
    x0, y0 = map(int, first)
    x1, y1 = map(int, second)
    count = max(abs(x1 - x0), abs(y1 - y0), 1) + 1
    for x, y in zip(np.linspace(x0, x1, count),
                    np.linspace(y0, y1, count)):
        _draw_disk(image, int(round(x)), int(round(y)), thickness, color)


def _fallback_image(width: int, height: int, sim, contacts: list[dict]) -> np.ndarray:
    """CPU top/side contact schematic used when OpenGL is unavailable.

    This is deliberately more useful than an error placeholder: both robot
    kinematic chains, collision-geom centers, TCPs, the part, fixtures, and
    allowed/forbidden contact points remain spatially inspectable. Exact CAD
    silhouettes still require the MuJoCo render; ``state.json`` labels this
    image as a projection fallback.
    """
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[:] = [24, 27, 34]
    banner = max(8, height // 28)
    image[:banner] = [210, 126, 24]
    split = width // 2
    image[:, max(0, split - 1):min(width, split + 1)] = [80, 86, 98]

    part = np.asarray(sim.part_pose()[:3, 3], dtype=float)
    # Each panel is centered on the active part so handoff, support, and PCB
    # stages all remain readable. The spans cover the complete reference cell.
    xy_span = 1.9
    xz_span = 1.9
    margin = max(8, min(width, height) // 45)

    def project(point, panel: str):
        point = np.asarray(point, dtype=float)
        panel_left = 0 if panel == "xy" else split
        panel_width = split if panel == "xy" else width - split
        usable_w = max(1, panel_width - 2 * margin)
        usable_h = max(1, height - banner - 2 * margin)
        if panel == "xy":
            u = 0.5 + (point[0] - part[0]) / xy_span
            v = 0.5 - (point[1] - part[1]) / xy_span
        else:
            u = 0.5 + (point[0] - part[0]) / xz_span
            v = 0.72 - (point[2] - part[2]) / xz_span
        return (
            int(round(panel_left + margin + u * usable_w)),
            int(round(banner + margin + v * usable_h)),
        )

    colors = {
        "A": np.array([34, 112, 238], dtype=np.uint8),
        "B": np.array([32, 190, 224], dtype=np.uint8),
        "part": np.array([250, 190, 55], dtype=np.uint8),
        "pcb": np.array([55, 165, 92], dtype=np.uint8),
        "stage": np.array([205, 105, 205], dtype=np.uint8),
        "fixture": np.array([112, 120, 134], dtype=np.uint8),
    }

    # Draw articulated body trees first, so geom/contact marks remain on top.
    for robot in ("A", "B"):
        for body_id in range(1, int(sim.model.nbody)):
            name = sim.model.body(body_id).name or ""
            if not name.startswith(f"{robot}_"):
                continue
            parent_id = int(sim.model.body_parentid[body_id])
            parent_name = (sim.model.body(parent_id).name or ""
                           if parent_id >= 0 else "")
            if not parent_name.startswith(f"{robot}_"):
                continue
            for panel in ("xy", "xz"):
                _draw_line(
                    image,
                    project(sim.data.xpos[parent_id], panel),
                    project(sim.data.xpos[body_id], panel),
                    colors[robot], thickness=max(1, min(width, height) // 260),
                )

    for geom_id in range(int(sim.model.ngeom)):
        if (sim.model.geom_contype[geom_id] == 0
                and sim.model.geom_conaffinity[geom_id] == 0):
            continue
        name = sim.model.geom(geom_id).name or ""
        if name.startswith("A_"):
            color = colors["A"]
        elif name.startswith("B_"):
            color = colors["B"]
        elif name.startswith("part_collision"):
            color = colors["part"]
        elif name.startswith(("pcb_", "insertion_collision")):
            color = colors["pcb"]
        elif name == "reorientation_surface":
            color = colors["stage"]
        else:
            color = colors["fixture"]
        radius = 2 if name.startswith(("A_", "B_")) else 1
        for panel in ("xy", "xz"):
            x, y = project(sim.data.geom_xpos[geom_id], panel)
            _draw_disk(image, x, y, radius, color)

    # TCPs and part origin use distinct cross/circle glyphs.
    for robot in ("A", "B"):
        tcp, _ = sim.site_pose(f"{robot}_tcp")
        for panel in ("xy", "xz"):
            x, y = project(tcp, panel)
            arm = max(4, min(width, height) // 100)
            _draw_line(image, (x - arm, y), (x + arm, y), [240, 240, 245], 1)
            _draw_line(image, (x, y - arm), (x, y + arm), [240, 240, 245], 1)
    for panel in ("xy", "xz"):
        x, y = project(part, panel)
        _draw_disk(image, x, y, max(4, min(width, height) // 90), colors["part"])

    for contact in contacts:
        allowed = contact.get("allowed")
        color = ([70, 225, 110] if allowed is True else
                 [245, 65, 65] if allowed is False else [250, 215, 70])
        for panel in ("xy", "xz"):
            x, y = project(contact["position_world_m"], panel)
            radius = max(4, min(width, height) // 95)
            _draw_disk(image, x, y, radius, color)
            _draw_line(image, (x - radius - 2, y), (x + radius + 2, y),
                       [255, 255, 255], 1)
            _draw_line(image, (x, y - radius - 2), (x, y + radius + 2),
                       [255, 255, 255], 1)
    return image


@dataclass(frozen=True)
class ArtifactRecord:
    step_name: str
    directory: str
    state_path: str
    image_path: str
    rendered: bool
    render_error: str | None


class DebugArtifactRecorder:
    """Write per-stage state/PNG artifacts below one UTC-named run directory."""

    def __init__(self, log_root: str | Path = "logs", *, strict: bool = False,
                 width: int = 960, height: int = 640,
                 run_name: str | None = None):
        self.log_root = Path(log_root)
        self.strict = bool(strict)
        self.width = int(width)
        self.height = int(height)
        if self.width <= 0 or self.height <= 0:
            raise ValueError("debug render width/height must be positive")
        timestamp = run_name or datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%S.%fZ")
        base = self.log_root / timestamp
        self.run_dir = base
        suffix = 2
        while self.run_dir.exists():
            self.run_dir = self.log_root / f"{timestamp}__{suffix:02d}"
            suffix += 1
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.records: list[ArtifactRecord] = []
        self.errors: list[str] = []
        self._step_counts: dict[str, int] = {}
        self._renderer = None
        self._renderer_model = None

    @staticmethod
    def _policy_json(policy: CollisionPolicy | None) -> dict | None:
        if policy is None:
            return None
        return {
            "part_holders": list(policy.part_holders),
            "holder_contact_penetration_m": policy.holder_contact_penetration_m,
            "allowed_contacts": [
                {
                    "geom1": item.geom1,
                    "geom2": item.geom2,
                    "max_penetration_m": item.max_penetration_m,
                }
                for item in policy.allowed_contacts
            ],
        }

    def _stage_directory(self, step_name: str) -> tuple[str, Path]:
        safe = _SAFE_STEP.sub("_", str(step_name)).strip("._") or "step"
        count = self._step_counts.get(safe, 0) + 1
        self._step_counts[safe] = count
        unique = safe if count == 1 else f"{safe}__{count:02d}"
        path = self.run_dir / unique
        path.mkdir(parents=False, exist_ok=False)
        return unique, path

    @staticmethod
    def _geom_name(model, geom_id: int) -> str:
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id))
        return name or f"geom_{int(geom_id)}"

    def _contacts(self, sim, collision_checker, policy) -> list[dict]:
        contacts = []
        for index in range(int(sim.data.ncon)):
            contact = sim.data.contact[index]
            name1 = self._geom_name(sim.model, contact.geom1)
            name2 = self._geom_name(sim.model, contact.geom2)
            wrench = np.zeros(6, dtype=float)
            mujoco.mj_contactForce(sim.model, sim.data, index, wrench)
            frame = np.asarray(contact.frame, dtype=float).reshape(3, 3)
            allowed = None
            if collision_checker is not None and policy is not None:
                allowed = bool(collision_checker._allowed(contact, policy=policy))
            contacts.append({
                "index": index,
                "geom1": name1,
                "geom2": name2,
                "geom_ids": [int(contact.geom1), int(contact.geom2)],
                "signed_distance_m": float(contact.dist),
                "penetration_m": float(max(0.0, -contact.dist)),
                "position_world_m": np.asarray(contact.pos, dtype=float).copy(),
                "frame_world_rows": frame.copy(),
                "wrench_contact_frame": wrench.copy(),
                "force_world_n": frame.T @ wrench[:3],
                "torque_world_nm": frame.T @ wrench[3:],
                "allowed": allowed,
            })
        return contacts

    def _state(self, sim, *, step_name: str, event: Any,
               plan_metadata: Any, execution_metadata: Any,
               collision_checker, policy: CollisionPolicy | None) -> dict:
        mujoco.mj_forward(sim.model, sim.data)
        contacts = self._contacts(sim, collision_checker, policy)
        tcp = {}
        for robot in ("A", "B"):
            position, rotation = sim.site_pose(f"{robot}_tcp")
            tcp[robot] = _transform(position, rotation)
        allowed_count = sum(item["allowed"] is True for item in contacts)
        unexpected_count = sum(item["allowed"] is False for item in contacts)
        return {
            "schema_version": 1,
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "step_name": step_name,
            "model": {
                "path": getattr(sim, "model_path", None),
                "name": getattr(sim.model, "name", None),
                "time_s": float(sim.data.time),
            },
            "event": event,
            "plan": plan_metadata,
            "execution": execution_metadata,
            "collision_policy": self._policy_json(policy),
            "q": {
                "A": sim.arm_qpos("A"),
                "B": sim.arm_qpos("B"),
            },
            "transforms": {
                "world_tcp_A": tcp["A"],
                "world_tcp_B": tcp["B"],
                "world_part": sim.part_pose(),
            },
            "contacts": contacts,
            "contact_summary": {
                "count": len(contacts),
                "allowed_count": allowed_count,
                "unexpected_count": unexpected_count,
                "minimum_signed_distance_m": (
                    min(item["signed_distance_m"] for item in contacts)
                    if contacts else None
                ),
            },
        }

    def _render_scene(self, sim) -> np.ndarray:
        if self._renderer is None or self._renderer_model is not sim.model:
            self.close_renderer()
            self._renderer = mujoco.Renderer(
                sim.model, height=self.height, width=self.width)
            self._renderer_model = sim.model
        option = mujoco.MjvOption()
        option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = sim.part_pose()[:3, 3]
        camera.distance = 1.45
        camera.azimuth = 135.0
        camera.elevation = -22.0
        self._renderer.update_scene(
            sim.data, camera=camera, scene_option=option)
        return np.asarray(self._renderer.render(), dtype=np.uint8).copy()

    def _record(self, step_name: str, sim, *, event: Any = None,
                plan_metadata: Any = None, execution_metadata: Any = None,
                collision_checker=None,
                policy: CollisionPolicy | None = None) -> ArtifactRecord:
        unique_name, directory = self._stage_directory(step_name)
        state = self._state(
            sim, step_name=unique_name, event=event,
            plan_metadata=plan_metadata, execution_metadata=execution_metadata,
            collision_checker=collision_checker, policy=policy)
        image_path = directory / "contacts.png"
        render_error = None
        rendered = False
        try:
            image = self._render_scene(sim)
            rendered = True
        except Exception as error:  # OpenGL may be unavailable in headless CI.
            render_error = f"{type(error).__name__}: {error}"
            image = _fallback_image(
                self.width, self.height, sim, state["contacts"])
        image_path.write_bytes(_png_bytes(image))
        state["render"] = {
            "image": image_path.name,
            "mujoco_rendered": rendered,
            "contact_visualization": rendered,
            "fallback_image": not rendered,
            "fallback_kind": (None if rendered else "cpu_top_side_contact_projection"),
            "error": render_error,
        }
        state_path = directory / "state.json"
        temporary = directory / ".state.json.tmp"
        temporary.write_text(
            json.dumps(_jsonable(state), indent=2, sort_keys=True,
                       allow_nan=False) + "\n",
            encoding="utf-8")
        temporary.replace(state_path)
        record = ArtifactRecord(
            unique_name, str(directory), str(state_path), str(image_path),
            rendered, render_error)
        self.records.append(record)
        if render_error:
            self.errors.append(f"{unique_name}: {render_error}")
        return record

    def record(self, step_name: str, sim, **kwargs) -> ArtifactRecord | None:
        """Record a stage, swallowing failures unless this recorder is strict."""
        try:
            return self._record(step_name, sim, **kwargs)
        except Exception as error:
            message = f"{step_name}: {type(error).__name__}: {error}"
            self.errors.append(message)
            if self.strict:
                raise
            return None

    def close_renderer(self) -> None:
        if self._renderer is not None:
            try:
                self._renderer.close()
            finally:
                self._renderer = None
                self._renderer_model = None

    def close(self) -> None:
        self.close_renderer()

    def __del__(self):  # pragma: no cover - interpreter shutdown is platform-specific.
        try:
            self.close_renderer()
        except Exception:
            pass


__all__ = ["ArtifactRecord", "DebugArtifactRecorder"]
