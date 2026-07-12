"""Persisted TCP reachability maps for corrected G1 lookup.

Voxels index the induced TCP translation and quantized TCP +Z direction. A
map miss is treated conservatively by the online planner unless the map was
built with production density; exact IK remains the authoritative G2 gate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import itertools

import numpy as np

from .kinematics import GP7Kinematics


def direction_bin(direction) -> int:
    direction = np.asarray(direction)
    axis = int(np.argmax(np.abs(direction)))
    return 2 * axis + int(direction[axis] < 0)


@dataclass
class ReachabilityMap:
    voxel_size: float
    keys: np.ndarray
    quality: np.ndarray
    _index: dict = field(init=False, repr=False)

    def __post_init__(self):
        self.voxel_size = float(self.voxel_size)
        self.keys = np.asarray(self.keys, dtype=np.int32)
        self.quality = np.asarray(self.quality, dtype=float)
        if self.voxel_size <= 0.0:
            raise ValueError("voxel_size must be positive")
        if self.keys.ndim != 2 or self.keys.shape[1] != 4:
            raise ValueError("reachability keys must have shape (N, 4)")
        if self.quality.shape != (len(self.keys),):
            raise ValueError("reachability quality must have shape (N,)")
        self._index = {tuple(int(value) for value in key): float(quality)
                       for key, quality in zip(self.keys, self.quality)}

    @classmethod
    def build(cls, kinematics: GP7Kinematics, robot: str, samples: int = 50_000,
              voxel_size: float = 0.04, seed: int = 17):
        rng = np.random.default_rng(seed)
        values = {}
        for _ in range(samples):
            q = rng.uniform(kinematics.lower[robot], kinematics.upper[robot])
            T = kinematics.fk(robot, q)
            voxel = tuple(np.floor(T[:3, 3] / voxel_size).astype(int))
            key = voxel + (direction_bin(T[:3, 2]),)
            quality = kinematics.penalized_manipulability(robot, q)
            values[key] = max(values.get(key, 0.0), quality)
        keys = np.array(list(values), dtype=np.int32)
        quality = np.array([values[tuple(key)] for key in keys])
        return cls(voxel_size, keys, quality)

    def query(self, tcp_pose: np.ndarray, neighborhood: int = 1) -> float:
        voxel = np.floor(tcp_pose[:3, 3] / self.voxel_size).astype(int)
        direction = direction_bin(tcp_pose[:3, 2])
        best = 0.0
        offsets = range(-int(neighborhood), int(neighborhood) + 1)
        for delta in itertools.product(offsets, repeat=3):
            key = tuple(int(value) for value in voxel + np.asarray(delta)) + (direction,)
            best = max(best, self._index.get(key, 0.0))
        return best

    def save(self, path):
        np.savez_compressed(path, voxel_size=self.voxel_size,
                            keys=self.keys, quality=self.quality)

    @classmethod
    def load(cls, path):
        data = np.load(path)
        return cls(float(data["voxel_size"]), data["keys"], data["quality"])
