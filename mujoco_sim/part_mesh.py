"""Project-aware, SI-unit triangle geometry for part planning.

Scene compilation and manipulation planning must consume the same prepared CAD
artifact.  This module calls :func:`mujoco_sim.cad_preprocess.prepare_cad`, then
combines every generated visual chunk in source-face order and applies the
manifest's declared XYZ scale exactly once.

STL and STEP/STP preparations produce deterministic binary STL chunks.  STEP
therefore follows the same path after FreeCAD tessellation.  OBJ faces are
triangulated deterministically without inserting, moving, centering, or welding
vertices.  The resulting :class:`TriangleMesh` is expressed in metres.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .cad_preprocess import CADPreparation, prepare_cad
from .geometry_grasps import TriangleMesh


HERE = Path(__file__).resolve().parent
DEFAULT_GENERATED_CAD = HERE / "models" / "generated_cad"


def _cross_2d(first: np.ndarray, second: np.ndarray, third: np.ndarray) -> float:
    left = second - first
    right = third - first
    return float(left[0] * right[1] - left[1] * right[0])


def _point_in_triangle(
    point: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    third: np.ndarray,
    orientation: float,
    tolerance: float,
) -> bool:
    return all(orientation * value >= -tolerance for value in (
        _cross_2d(first, second, point),
        _cross_2d(second, third, point),
        _cross_2d(third, first, point),
    ))


def _triangulate_face(
    face_indices: list[int],
    vertices: list[np.ndarray],
    *,
    source: Path,
    line_number: int,
) -> list[tuple[int, int, int]]:
    """Deterministically ear-clip one simple planar OBJ polygon."""
    if len(face_indices) == 3:
        return [tuple(face_indices)]  # type: ignore[list-item]
    points = np.vstack([vertices[index] for index in face_indices])
    # Newell normal is stable for polygons with more than three vertices.
    following = np.roll(points, -1, axis=0)
    normal = np.sum(np.cross(points, following), axis=0)
    normal_length = float(np.linalg.norm(normal))
    scale = max(float(np.linalg.norm(np.ptp(points, axis=0))), 1.0)
    if normal_length <= 1e-12 * scale**2:
        raise ValueError(f"degenerate OBJ face at {source}:{line_number}")
    normal /= normal_length
    plane_error = np.abs((points - points[0]) @ normal)
    if float(np.max(plane_error)) > 1e-7 * scale:
        raise ValueError(f"non-planar OBJ face at {source}:{line_number}")
    dropped_axis = int(np.argmax(np.abs(normal)))
    projected = np.delete(points, dropped_axis, axis=1)
    signed_area2 = float(np.sum(
        projected[:, 0] * np.roll(projected[:, 1], -1)
        - projected[:, 1] * np.roll(projected[:, 0], -1)))
    if abs(signed_area2) <= 1e-12 * scale**2:
        raise ValueError(f"zero-area OBJ face at {source}:{line_number}")
    orientation = 1.0 if signed_area2 > 0.0 else -1.0
    tolerance = 1e-12 * scale**2
    remaining = list(range(len(face_indices)))
    triangles: list[tuple[int, int, int]] = []
    while len(remaining) > 3:
        ear_position = None
        for position, current in enumerate(remaining):
            previous = remaining[position - 1]
            following_index = remaining[(position + 1) % len(remaining)]
            turn = orientation * _cross_2d(
                projected[previous], projected[current], projected[following_index])
            if turn <= tolerance:
                continue
            if any(
                _point_in_triangle(
                    projected[other], projected[previous], projected[current],
                    projected[following_index], orientation, tolerance)
                for other in remaining
                if other not in (previous, current, following_index)
            ):
                continue
            ear_position = position
            triangles.append((
                face_indices[previous], face_indices[current],
                face_indices[following_index],
            ))
            break
        if ear_position is None:
            raise ValueError(
                f"OBJ face is self-intersecting or cannot be triangulated at "
                f"{source}:{line_number}"
            )
        del remaining[ear_position]
    triangles.append(tuple(face_indices[index] for index in remaining))
    return triangles


def load_obj_triangles(path: str | Path) -> np.ndarray:
    """Parse OBJ vertices/faces and return deterministic native-frame triangles.

    Texture coordinates, normals, materials, objects, and groups do not alter
    planning geometry and are ignored. Positive and negative OBJ vertex indices
    are supported. Homogeneous vertex weights are accepted only when equal to
    one so input XYZ coordinates are never silently transformed.
    """
    source = Path(path)
    vertices: list[np.ndarray] = []
    triangles: list[tuple[int, int, int]] = []
    pending = ""
    pending_line = 0
    for line_number, raw in enumerate(
            source.read_text(encoding="utf-8-sig").splitlines(), start=1):
        content = raw.split("#", 1)[0].strip()
        if pending:
            content = pending + content
            origin_line = pending_line
        else:
            origin_line = line_number
        if content.endswith("\\"):
            pending = content[:-1] + " "
            pending_line = origin_line
            continue
        pending = ""
        if not content:
            continue
        tokens = content.split()
        keyword, values = tokens[0], tokens[1:]
        if keyword == "v":
            if len(values) not in (3, 4):
                raise ValueError(f"OBJ vertex needs XYZ[W] at {source}:{origin_line}")
            try:
                coordinate = np.asarray([float(value) for value in values], dtype=float)
            except ValueError as error:
                raise ValueError(f"invalid OBJ vertex at {source}:{origin_line}") from error
            if not np.all(np.isfinite(coordinate)):
                raise ValueError(f"non-finite OBJ vertex at {source}:{origin_line}")
            if len(coordinate) == 4 and not np.isclose(
                    coordinate[3], 1.0, atol=0.0, rtol=0.0):
                raise ValueError(
                    f"homogeneous OBJ vertex W must equal 1 at {source}:{origin_line}"
                )
            vertices.append(coordinate[:3].copy())
        elif keyword == "f":
            if len(values) < 3:
                raise ValueError(f"OBJ face needs at least 3 vertices at {source}:{origin_line}")
            indices: list[int] = []
            for value in values:
                index_text = value.split("/", 1)[0]
                try:
                    raw_index = int(index_text)
                except ValueError as error:
                    raise ValueError(
                        f"invalid OBJ face index at {source}:{origin_line}"
                    ) from error
                if raw_index == 0:
                    raise ValueError(f"OBJ indices are one-based at {source}:{origin_line}")
                index = raw_index - 1 if raw_index > 0 else len(vertices) + raw_index
                if not 0 <= index < len(vertices):
                    raise ValueError(f"OBJ index out of range at {source}:{origin_line}")
                indices.append(index)
            if len(set(indices)) != len(indices):
                raise ValueError(f"OBJ face repeats a vertex at {source}:{origin_line}")
            triangles.extend(_triangulate_face(
                indices, vertices, source=source, line_number=origin_line))
        # All other standard OBJ records describe shading/grouping or
        # non-surface primitives and cannot change the polygon coordinates.
    if pending:
        raise ValueError(f"unterminated OBJ line continuation at {source}:{pending_line}")
    if not triangles:
        raise ValueError(f"OBJ contains no polygon faces: {source}")
    vertex_array = np.vstack(vertices)
    return vertex_array[np.asarray(triangles, dtype=int)]


def load_prepared_triangle_mesh(preparation: CADPreparation) -> TriangleMesh:
    """Combine prepared visual chunks and convert source coordinates to SI."""
    if not isinstance(preparation, CADPreparation):
        raise TypeError("preparation must be a CADPreparation")
    metadata = preparation.metadata
    visual = metadata["visual"]
    chunks = visual.get("chunks", [])
    if not chunks:
        raise ValueError("prepared CAD contains no visual chunks")
    geometry: list[np.ndarray] = []
    if visual["format"] == "binary-stl":
        for chunk in chunks:
            path = preparation.artifact_dir / chunk["path"]
            geometry.append(TriangleMesh.from_binary_stl(path).triangles)
    elif visual["format"] == "obj":
        for chunk in chunks:
            geometry.append(load_obj_triangles(
                preparation.artifact_dir / chunk["path"]))
    else:
        raise ValueError(f"unsupported prepared planning format {visual['format']!r}")
    triangles = np.concatenate(geometry, axis=0)
    scale = np.asarray(metadata["source"]["scale_to_m"], dtype=float)
    if scale.shape != (3,) or not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError("prepared CAD has invalid source.scale_to_m")
    triangles_si = triangles * scale[None, None, :]
    return TriangleMesh.from_triangles(
        triangles_si,
        source=f"{preparation.metadata_path}#{metadata['artifact_fingerprint']}",
    )


@dataclass(frozen=True)
class ProjectPartMesh:
    mesh: TriangleMesh
    preparation: CADPreparation

    @property
    def artifact_fingerprint(self) -> str:
        return str(self.preparation.metadata["artifact_fingerprint"])


def load_project_part_mesh(
    project: Any,
    generated_root: str | Path = DEFAULT_GENERATED_CAD,
) -> ProjectPartMesh:
    """Prepare and load the active project's part as one SI-unit mesh."""
    part = project.active_part
    preparation = prepare_cad(
        project.active_part_path,
        generated_root,
        units=part.get("cad_units"),
        scale_to_m=part.get("cad_scale_to_m"),
        role="part-visual",
        # Match the scene compiler. An active moving part is one rigid body;
        # connected-component metadata would not define articulation anyway.
        static_assembly=False,
    )
    return ProjectPartMesh(
        mesh=load_prepared_triangle_mesh(preparation),
        preparation=preparation,
    )


__all__ = [
    "DEFAULT_GENERATED_CAD",
    "ProjectPartMesh",
    "load_obj_triangles",
    "load_prepared_triangle_mesh",
    "load_project_part_mesh",
]
