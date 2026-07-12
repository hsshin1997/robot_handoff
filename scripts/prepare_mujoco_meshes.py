"""Prepare exact CAD meshes for MuJoCo without geometric decimation.

MuJoCo limits one STL asset to 200,000 triangles. Binary STL is a triangle
soup, so the full workcell can be divided into multiple valid STL files while
preserving every input triangle byte-for-byte. The same parser reports and can
export connected components of the single-file gripper CAD.
"""
from __future__ import annotations

import argparse
import os
import struct
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GENERATED = os.path.join(ROOT, "mujoco_sim", "models", "generated")


def read_binary_stl(path: str) -> tuple[bytes, list[bytes]]:
    with open(path, "rb") as f:
        header = f.read(80)
        count_raw = f.read(4)
        if len(count_raw) != 4:
            raise ValueError(f"not a binary STL: {path}")
        count = struct.unpack("<I", count_raw)[0]
        records = [f.read(50) for _ in range(count)]
        if any(len(record) != 50 for record in records) or f.read(1):
            raise ValueError(f"invalid binary STL length/count: {path}")
    return header, records


def write_binary_stl(path: str, header: bytes, records: list[bytes]) -> None:
    with open(path, "wb") as f:
        f.write(header[:80].ljust(80, b" "))
        f.write(struct.pack("<I", len(records)))
        f.writelines(records)


def split_exact(path: str, stem: str, max_faces: int = 190_000) -> list[str]:
    header, records = read_binary_stl(path)
    paths = []
    for index, start in enumerate(range(0, len(records), max_faces)):
        output = os.path.join(GENERATED, f"{stem}_{index:02d}.stl")
        write_binary_stl(output, header, records[start:start + max_faces])
        paths.append(output)
    print(f"{os.path.basename(path)}: {len(records)} exact triangles -> {len(paths)} chunks")
    return paths


def connected_components(path: str) -> list[tuple[list[int], tuple, tuple]]:
    """Return face indices and bounds for components joined by exact vertices."""
    _, records = read_binary_stl(path)
    parent = list(range(len(records)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a

    owner = {}
    face_vertices = []
    for face, record in enumerate(records):
        vertices = struct.unpack("<9f", record[12:48])
        points = tuple(tuple(round(v, 5) for v in vertices[i:i + 3]) for i in (0, 3, 6))
        face_vertices.append(points)
        for point in points:
            if point in owner:
                union(face, owner[point])
            else:
                owner[point] = face

    groups = defaultdict(list)
    for face in range(len(records)):
        groups[find(face)].append(face)
    result = []
    for faces in groups.values():
        points = [p for face in faces for p in face_vertices[face]]
        low = tuple(min(p[axis] for p in points) for axis in range(3))
        high = tuple(max(p[axis] for p in points) for axis in range(3))
        result.append((faces, low, high))
    return sorted(result, key=lambda item: len(item[0]), reverse=True)


def split_connected_components(path: str, stem: str) -> list[str]:
    """Export exact triangle records grouped by connected surface component."""
    header, records = read_binary_stl(path)
    components = connected_components(path)
    outputs = []
    for index, (faces, _, _) in enumerate(components):
        output = os.path.join(GENERATED, f"{stem}_{index:02d}.stl")
        write_binary_stl(output, header, [records[face] for face in faces])
        outputs.append(output)
    print(f"{os.path.basename(path)}: {len(records)} triangles -> "
          f"{len(outputs)} connected collision components")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-gripper", action="store_true")
    args = parser.parse_args()
    os.makedirs(GENERATED, exist_ok=True)
    split_exact(os.path.join(ROOT, "assets", "workcell", "workcell.stl"), "workcell_full")
    gripper_path = os.path.join(ROOT, "assets", "gp7", "meshes", "gripper.STL")
    split_connected_components(gripper_path, "gripper_component")
    if args.report_gripper:
        components = connected_components(gripper_path)
        print(f"gripper.STL: {len(components)} connected components")
        for index, (faces, low, high) in enumerate(components):
            print(f"  {index:02d}: {len(faces):4d} faces, min={low}, max={high}")


if __name__ == "__main__":
    main()
