"""Exact visual-CAD preprocessing for MuJoCo.

No visual triangle is decimated, simplified, welded, or reordered.  STL input
is normalized to deterministic binary STL and split only because MuJoCo limits
one STL asset to 200,000 faces.  OBJ input is copied byte-for-byte.  STEP/STP
input is tessellated by an explicitly detected FreeCAD command-line program,
then the complete tessellation is preserved by the same STL path.

Visual fidelity and collision geometry are deliberately different concerns.
MuJoCo collision against a mesh uses its convex hull, not the rendered
concave triangle surface.  The metadata therefore never labels exact visual
chunks or connected surface components as collision decompositions.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
from typing import Any

from .offline import (
    atomic_write_bytes,
    atomic_write_json,
    fingerprint_bytes,
    fingerprint_content,
    fingerprint_file,
)


PREPROCESSOR_VERSION = "1"
METADATA_SCHEMA_VERSION = 1
MUJOCO_STL_FACE_LIMIT = 200_000
DEFAULT_MAX_FACES_PER_CHUNK = MUJOCO_STL_FACE_LIMIT - 1
CANONICAL_STL_HEADER = b"handoff exact visual STL v1; no decimation".ljust(80, b" ")
_STL_RECORD = struct.Struct("<12fH")
_STL_COUNT = struct.Struct("<I")
_UNIT_TO_METRE = {
    "m": 1.0,
    "meter": 1.0,
    "metre": 1.0,
    "mm": 1e-3,
    "millimeter": 1e-3,
    "millimetre": 1e-3,
    "cm": 1e-2,
    "centimeter": 1e-2,
    "centimetre": 1e-2,
    "in": 0.0254,
    "inch": 0.0254,
}


COLLISION_WARNING = (
    "Exact visual mesh chunks are not collision meshes: MuJoCo uses the convex "
    "hull of a mesh for collision. Use primitives or an optional convex "
    "decomposition for concave contact geometry. Export articulated moving "
    "parts as separate CAD bodies with explicit joints."
)


class CADPreprocessError(RuntimeError):
    """Base class for CAD ingestion/conversion failures."""


class FreeCADUnavailableError(CADPreprocessError):
    """Raised when STEP conversion was requested without FreeCADCmd."""


@dataclass(frozen=True)
class STLData:
    source_format: str
    records: tuple[bytes, ...]

    @property
    def face_count(self) -> int:
        return len(self.records)


@dataclass(frozen=True)
class CADPreparation:
    artifact_dir: Path
    metadata_path: Path
    metadata: dict[str, Any]


def scale_to_metres(
    *,
    units: str | None = None,
    scale_to_m: float | Sequence[float] | None = None,
) -> tuple[float, float, float]:
    """Resolve an explicit source-unit conversion into an XYZ MuJoCo scale.

    STL has no unit metadata, so at least one argument is mandatory.  If both
    are supplied, ``scale_to_m`` is a dimensionless post-scale applied after
    the named unit conversion.
    """
    if units is None and scale_to_m is None:
        raise ValueError(
            "CAD units are ambiguous; specify units (for example 'mm' or 'm') "
            "or an explicit scale_to_m"
        )
    unit_scale = 1.0
    if units is not None:
        normalized = units.strip().lower()
        if normalized not in _UNIT_TO_METRE:
            raise ValueError(f"unsupported CAD units {units!r}; supported: {sorted(_UNIT_TO_METRE)}")
        unit_scale = _UNIT_TO_METRE[normalized]

    if scale_to_m is None:
        factors = (1.0, 1.0, 1.0)
    elif isinstance(scale_to_m, (int, float)) and not isinstance(scale_to_m, bool):
        factors = (float(scale_to_m),) * 3
    else:
        factors = tuple(float(value) for value in scale_to_m)  # type: ignore[arg-type]
        if len(factors) != 3:
            raise ValueError("scale_to_m must be a scalar or a three-vector")
    result = tuple(unit_scale * value for value in factors)
    if any(not math.isfinite(value) or value <= 0.0 for value in result):
        raise ValueError("resolved CAD scale must contain three finite positive values")
    return result


def _validate_record(record: bytes, *, source: str) -> bytes:
    if len(record) != _STL_RECORD.size:
        raise CADPreprocessError(f"invalid {len(record)}-byte STL triangle record in {source}")
    values = _STL_RECORD.unpack(record)
    if not all(math.isfinite(value) for value in values[:12]):
        raise CADPreprocessError(f"non-finite STL normal or vertex in {source}")
    return record


def _read_binary_stl(content: bytes, path: Path) -> STLData | None:
    if len(content) < 84:
        return None
    face_count = _STL_COUNT.unpack_from(content, 80)[0]
    expected = 84 + face_count * _STL_RECORD.size
    if expected != len(content):
        return None
    records = tuple(
        _validate_record(content[offset:offset + _STL_RECORD.size], source=str(path))
        for offset in range(84, expected, _STL_RECORD.size)
    )
    if not records:
        raise CADPreprocessError(f"STL contains no triangles: {path}")
    return STLData("binary-stl", records)


def _ascii_float(tokens: list[str], line_number: int, path: Path) -> tuple[float, float, float]:
    if len(tokens) != 3:
        raise CADPreprocessError(f"expected three coordinates at {path}:{line_number}")
    try:
        values = tuple(float(token) for token in tokens)
    except ValueError as error:
        raise CADPreprocessError(f"invalid STL number at {path}:{line_number}") from error
    if not all(math.isfinite(value) for value in values):
        raise CADPreprocessError(f"non-finite STL number at {path}:{line_number}")
    return values  # type: ignore[return-value]


def _read_ascii_stl(content: bytes, path: Path) -> STLData:
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise CADPreprocessError(
            f"STL is neither length-valid binary STL nor UTF-8 ASCII STL: {path}"
        ) from error

    records: list[bytes] = []
    normal: tuple[float, float, float] | None = None
    vertices: list[tuple[float, float, float]] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        tokens = line.split()
        keyword = tokens[0].lower()
        if keyword in ("solid", "endsolid"):
            if normal is not None:
                raise CADPreprocessError(f"solid boundary inside a facet at {path}:{line_number}")
        elif keyword == "facet":
            if normal is not None or len(tokens) < 2 or tokens[1].lower() != "normal":
                raise CADPreprocessError(f"invalid or nested facet at {path}:{line_number}")
            normal = _ascii_float(tokens[2:], line_number, path)
            vertices = []
        elif keyword == "vertex":
            if normal is None:
                raise CADPreprocessError(f"vertex outside a facet at {path}:{line_number}")
            vertices.append(_ascii_float(tokens[1:], line_number, path))
            if len(vertices) > 3:
                raise CADPreprocessError(f"non-triangular facet at {path}:{line_number}")
        elif keyword == "endfacet":
            if normal is None or len(vertices) != 3:
                raise CADPreprocessError(f"facet does not contain exactly three vertices at {path}:{line_number}")
            try:
                records.append(_STL_RECORD.pack(*(normal + vertices[0] + vertices[1] + vertices[2]), 0))
            except (OverflowError, struct.error) as error:
                raise CADPreprocessError(f"STL coordinate is outside float32 range at {path}:{line_number}") from error
            normal = None
            vertices = []
        elif keyword == "outer":
            if [token.lower() for token in tokens] != ["outer", "loop"]:
                raise CADPreprocessError(f"invalid outer loop at {path}:{line_number}")
        elif keyword in ("endloop",):
            if len(tokens) != 1:
                raise CADPreprocessError(f"invalid endloop at {path}:{line_number}")
        else:
            raise CADPreprocessError(f"unsupported ASCII STL statement {line!r} at {path}:{line_number}")
    if normal is not None:
        raise CADPreprocessError(f"unterminated facet in {path}")
    if not records:
        raise CADPreprocessError(f"ASCII STL contains no triangles: {path}")
    return STLData("ascii-stl", tuple(records))


def read_stl(path: str | os.PathLike[str]) -> STLData:
    """Read binary or ASCII STL and preserve every triangle in source order."""
    source = Path(path)
    content = source.read_bytes()
    binary = _read_binary_stl(content, source)
    return binary if binary is not None else _read_ascii_stl(content, source)


def binary_stl_bytes(records: Iterable[bytes]) -> bytes:
    """Encode triangle records with the deterministic canonical STL header."""
    materialized = tuple(_validate_record(bytes(record), source="records") for record in records)
    if len(materialized) > 0xFFFFFFFF:
        raise ValueError("binary STL face count exceeds uint32")
    return CANONICAL_STL_HEADER + _STL_COUNT.pack(len(materialized)) + b"".join(materialized)


def write_binary_stl(path: str | os.PathLike[str], records: Iterable[bytes]) -> Path:
    """Atomically write a deterministic binary STL."""
    return atomic_write_bytes(path, binary_stl_bytes(records))


def chunk_records(
    records: Sequence[bytes],
    *,
    max_faces: int = DEFAULT_MAX_FACES_PER_CHUNK,
) -> tuple[tuple[bytes, ...], ...]:
    """Split without modifying or dropping triangles; every chunk is < limit."""
    if not isinstance(max_faces, int) or isinstance(max_faces, bool):
        raise TypeError("max_faces must be an integer")
    if max_faces <= 0 or max_faces >= MUJOCO_STL_FACE_LIMIT:
        raise ValueError(
            f"max_faces must be positive and strictly below MuJoCo's "
            f"{MUJOCO_STL_FACE_LIMIT}-face STL limit"
        )
    return tuple(tuple(records[start:start + max_faces])
                 for start in range(0, len(records), max_faces))


def _vertex_key(record: bytes, vertex_index: int) -> bytes:
    offset = 12 + vertex_index * 12
    x, y, z = struct.unpack_from("<3f", record, offset)
    # Positive and negative zero are the same CAD vertex.
    return struct.pack("<3f", 0.0 if x == 0.0 else x, 0.0 if y == 0.0 else y,
                       0.0 if z == 0.0 else z)


def connected_components(records: Sequence[bytes]) -> tuple[tuple[int, ...], ...]:
    """Group faces sharing exact float32 vertices into deterministic components."""
    parent = list(range(len(records)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            if first_root > second_root:
                first_root, second_root = second_root, first_root
            parent[second_root] = first_root

    vertex_owner: dict[bytes, int] = {}
    for face_index, record in enumerate(records):
        _validate_record(record, source="component records")
        for vertex_index in range(3):
            key = _vertex_key(record, vertex_index)
            owner = vertex_owner.setdefault(key, face_index)
            union(face_index, owner)

    groups: dict[int, list[int]] = defaultdict(list)
    for face_index in range(len(records)):
        groups[find(face_index)].append(face_index)
    return tuple(tuple(faces) for faces in sorted(groups.values(), key=lambda value: (-len(value), value[0])))


def _bounds(records: Sequence[bytes], scale: tuple[float, float, float]) -> tuple[list[float], list[float], list[float], list[float]]:
    low = [math.inf, math.inf, math.inf]
    high = [-math.inf, -math.inf, -math.inf]
    for record in records:
        vertices = struct.unpack("<9f", record[12:48])
        for offset in (0, 3, 6):
            for axis in range(3):
                low[axis] = min(low[axis], vertices[offset + axis])
                high[axis] = max(high[axis], vertices[offset + axis])
    return low, high, [low[i] * scale[i] for i in range(3)], [high[i] * scale[i] for i in range(3)]


def _chunk_metadata(path: Path, records: Sequence[bytes], artifact_dir: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(artifact_dir).as_posix(),
        "face_count": len(records),
        "sha256": fingerprint_file(path),
    }


def _write_chunks(
    directory: Path,
    prefix: str,
    records: Sequence[bytes],
    max_faces: int,
    artifact_dir: Path,
) -> list[dict[str, Any]]:
    result = []
    for index, chunk in enumerate(chunk_records(records, max_faces=max_faces)):
        output = directory / f"{prefix}-{index:04d}.stl"
        write_binary_stl(output, chunk)
        result.append(_chunk_metadata(output, chunk, artifact_dir))
    return result


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output, open(source, "rb") as input_stream:
            shutil.copyfileobj(input_stream, output, length=1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def detect_freecad(executable: str | os.PathLike[str] | None = None) -> str:
    """Locate FreeCAD's command-line executable or raise an actionable error."""
    candidates: list[str] = []
    if executable is not None:
        candidates.append(os.fspath(executable))
    else:
        if os.environ.get("FREECADCMD"):
            candidates.append(os.environ["FREECADCMD"])
        candidates.extend(("FreeCADCmd", "freecadcmd"))
    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
        path = Path(candidate).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return str(path.resolve())
    explicit = f" ({executable})" if executable is not None else ""
    raise FreeCADUnavailableError(
        "STEP/STP conversion requires FreeCAD's command-line executable"
        f"{explicit}. Install FreeCAD and ensure FreeCADCmd/freecadcmd is on PATH, "
        "or set FREECADCMD=/absolute/path/to/FreeCADCmd. You may also export the "
        "assembly to STL/OBJ in CAD, keeping a common assembly frame and explicit units."
    )


def convert_step_to_stl(
    source: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    *,
    freecad_executable: str | os.PathLike[str] | None = None,
    converter_script: str | os.PathLike[str] | None = None,
    linear_deflection_mm: float = 0.05,
    angular_deflection_deg: float = 5.0,
) -> Path:
    """Tessellate STEP with FreeCAD; no triangle reduction is performed later."""
    if not math.isfinite(linear_deflection_mm) or linear_deflection_mm <= 0:
        raise ValueError("linear_deflection_mm must be finite and positive")
    if not math.isfinite(angular_deflection_deg) or not 0 < angular_deflection_deg < 180:
        raise ValueError("angular_deflection_deg must lie in (0, 180)")
    executable = detect_freecad(freecad_executable)
    if converter_script is None:
        converter_script = Path(__file__).resolve().parents[1] / "scripts" / "freecad_step_to_stl.py"
    script = Path(converter_script).resolve()
    if not script.is_file():
        raise FileNotFoundError(f"FreeCAD converter script does not exist: {script}")
    source_path = Path(source).resolve()
    destination_path = Path(destination).resolve()
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        executable, str(script), "--input", str(source_path), "--output", str(destination_path),
        "--linear-deflection-mm", format(linear_deflection_mm, ".17g"),
        "--angular-deflection-deg", format(angular_deflection_deg, ".17g"),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0 or not destination_path.is_file():
        details = "\n".join(part.strip() for part in (completed.stdout, completed.stderr) if part.strip())
        raise CADPreprocessError(
            f"FreeCAD STEP conversion failed with exit code {completed.returncode}: "
            f"{' '.join(command)}" + (f"\n{details[-4000:]}" if details else "")
        )
    return destination_path


def _metadata_with_fingerprint(metadata: dict[str, Any]) -> dict[str, Any]:
    result = dict(metadata)
    result["metadata_content_sha256"] = fingerprint_content(result)
    return result


def verify_preparation(path: str | os.PathLike[str]) -> CADPreparation:
    """Read and verify atomic metadata plus every registered output file."""
    metadata_path = Path(path)
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CADPreprocessError(f"cannot read CAD metadata {metadata_path}") from error
    claimed = metadata.pop("metadata_content_sha256", None)
    if not isinstance(claimed, str) or fingerprint_content(metadata) != claimed:
        raise CADPreprocessError(f"CAD metadata fingerprint mismatch: {metadata_path}")
    metadata["metadata_content_sha256"] = claimed
    artifact_dir = metadata_path.parent

    registered = list(metadata["visual"]["chunks"])
    for component in metadata.get("static_assembly", {}).get("components", []):
        registered.extend(component["chunks"])
    for output in registered:
        output_path = artifact_dir / output["path"]
        if not output_path.is_file() or fingerprint_file(output_path) != output["sha256"]:
            raise CADPreprocessError(f"generated CAD output fingerprint mismatch: {output_path}")
    return CADPreparation(artifact_dir, metadata_path, metadata)


def prepare_cad(
    source: str | os.PathLike[str],
    generated_root: str | os.PathLike[str],
    *,
    units: str | None = None,
    scale_to_m: float | Sequence[float] | None = None,
    role: str = "visual",
    static_assembly: bool = False,
    max_faces: int = DEFAULT_MAX_FACES_PER_CHUNK,
    freecad_executable: str | os.PathLike[str] | None = None,
    linear_deflection_mm: float = 0.05,
    angular_deflection_deg: float = 5.0,
) -> CADPreparation:
    """Prepare one STL/OBJ/STEP source in a content-addressed directory."""
    source_path = Path(source).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    suffix = source_path.suffix.lower()
    if suffix not in (".stl", ".obj", ".step", ".stp"):
        raise CADPreprocessError(
            f"unsupported CAD format {suffix!r}: {source_path}; use STL, OBJ, STEP, or STP"
        )
    scale = scale_to_metres(units=units, scale_to_m=scale_to_m)
    if suffix in (".step", ".stp") and units is not None:
        # FreeCAD converts STEP's declared source units into its millimetre
        # internal geometry before MeshPart writes STL coordinates.
        normalized_units = units.strip().lower()
        if normalized_units not in ("mm", "millimeter", "millimetre"):
            raise ValueError(
                "FreeCAD emits STEP tessellation coordinates in millimetres; use "
                "units='mm' (independent of the units declared inside STEP), or "
                "provide the equivalent explicit scale_to_m=0.001"
            )
    # Validate even for OBJ so changing this option has identical semantics.
    if max_faces <= 0 or max_faces >= MUJOCO_STL_FACE_LIMIT:
        raise ValueError(
            f"max_faces must be positive and strictly below {MUJOCO_STL_FACE_LIMIT}"
        )
    source_sha256 = fingerprint_file(source_path)
    artifact_fingerprint = fingerprint_content({
        "preprocessor_version": PREPROCESSOR_VERSION,
        "source_sha256": source_sha256,
        "source_format": suffix,
        "scale_to_m": scale,
        "role": role,
        "static_assembly": bool(static_assembly),
        "max_faces": max_faces,
        "step_tessellation": {
            "linear_deflection_mm": linear_deflection_mm,
            "angular_deflection_deg": angular_deflection_deg,
        } if suffix in (".step", ".stp") else None,
    })
    artifact_dir = Path(generated_root).resolve() / artifact_fingerprint
    metadata_path = artifact_dir / "metadata.json"
    if metadata_path.is_file():
        return verify_preparation(metadata_path)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    warnings = [COLLISION_WARNING]
    metadata: dict[str, Any] = {
        "schema_version": METADATA_SCHEMA_VERSION,
        "preprocessor_version": PREPROCESSOR_VERSION,
        "artifact_fingerprint": artifact_fingerprint,
        "source": {
            "name": source_path.name,
            "format": suffix[1:],
            "sha256": source_sha256,
            "size_bytes": source_path.stat().st_size,
            "units": units,
            "scale_to_m": list(scale),
        },
        "role": role,
        "visual": {
            "exact_triangle_preservation": True,
            "downsampled": False,
            "chunks": [],
        },
        "collision": {
            "representation": "not-generated",
            "visual_chunks_are_collision_decomposition": False,
            "warning": COLLISION_WARNING,
        },
        "warnings": warnings,
    }

    temporary_step_stl: Path | None = None
    try:
        if suffix == ".obj":
            output = artifact_dir / "visual" / "visual.obj"
            _atomic_copy(source_path, output)
            metadata["visual"].update({
                "format": "obj",
                "source_encoding_preserved": True,
                "chunks": [{
                    "path": output.relative_to(artifact_dir).as_posix(),
                    "face_count": None,
                    "sha256": fingerprint_file(output),
                }],
            })
            if static_assembly:
                warning = (
                    "Connected-component splitting is currently defined for STL triangle "
                    "records only; export this static OBJ assembly as STL to request component metadata."
                )
                warnings.append(warning)
                metadata["static_assembly"] = {"requested": True, "components": [], "warning": warning}
        else:
            mesh_source = source_path
            if suffix in (".step", ".stp"):
                descriptor, temporary_name = tempfile.mkstemp(suffix=".stl", dir=artifact_dir)
                os.close(descriptor)
                temporary_step_stl = Path(temporary_name)
                temporary_step_stl.unlink()
                convert_step_to_stl(
                    source_path, temporary_step_stl,
                    freecad_executable=freecad_executable,
                    linear_deflection_mm=linear_deflection_mm,
                    angular_deflection_deg=angular_deflection_deg,
                )
                mesh_source = temporary_step_stl
                warning = (
                    "STEP is a B-rep; FreeCAD tessellated it using the recorded deflection "
                    "settings. Every resulting triangle is preserved without downsampling."
                )
                warnings.append(warning)
                metadata["source"]["tessellation"] = {
                    "engine": "FreeCAD",
                    "linear_deflection_mm": linear_deflection_mm,
                    "angular_deflection_deg": angular_deflection_deg,
                }

            stl = read_stl(mesh_source)
            low, high, low_m, high_m = _bounds(stl.records, scale)
            metadata["visual"].update({
                "format": "binary-stl",
                "input_stl_encoding": stl.source_format,
                "face_count": stl.face_count,
                "bounds_source_units": {"min": low, "max": high},
                "bounds_m": {"min": low_m, "max": high_m},
                "chunks": _write_chunks(
                    artifact_dir / "visual", "visual", stl.records, max_faces, artifact_dir
                ),
            })

            if static_assembly:
                component_items = []
                for component_index, face_indices in enumerate(connected_components(stl.records)):
                    component_records = tuple(stl.records[index] for index in face_indices)
                    component_low, component_high, component_low_m, component_high_m = _bounds(
                        component_records, scale
                    )
                    component_items.append({
                        "index": component_index,
                        "face_count": len(component_records),
                        "first_source_face": face_indices[0],
                        "triangle_records_sha256": fingerprint_bytes(b"".join(component_records)),
                        "bounds_source_units": {"min": component_low, "max": component_high},
                        "bounds_m": {"min": component_low_m, "max": component_high_m},
                        "chunks": _write_chunks(
                            artifact_dir / "components", f"component-{component_index:04d}",
                            component_records, max_faces, artifact_dir,
                        ),
                    })
                metadata["static_assembly"] = {
                    "requested": True,
                    "connectivity": "shared exact float32 vertex",
                    "component_count": len(component_items),
                    "components": component_items,
                    "collision_decomposition": False,
                    "warning": (
                        "Connected surface components preserve visual triangles but are not "
                        "necessarily convex and are not an articulated-body definition."
                    ),
                }
                warnings.append(metadata["static_assembly"]["warning"])
    finally:
        if temporary_step_stl is not None:
            try:
                temporary_step_stl.unlink()
            except FileNotFoundError:
                pass

    final_metadata = _metadata_with_fingerprint(metadata)
    atomic_write_json(metadata_path, final_metadata)
    return verify_preparation(metadata_path)


__all__ = [
    "CANONICAL_STL_HEADER",
    "COLLISION_WARNING",
    "DEFAULT_MAX_FACES_PER_CHUNK",
    "METADATA_SCHEMA_VERSION",
    "MUJOCO_STL_FACE_LIMIT",
    "PREPROCESSOR_VERSION",
    "CADPreparation",
    "CADPreprocessError",
    "FreeCADUnavailableError",
    "STLData",
    "binary_stl_bytes",
    "chunk_records",
    "connected_components",
    "convert_step_to_stl",
    "detect_freecad",
    "prepare_cad",
    "read_stl",
    "scale_to_metres",
    "verify_preparation",
    "write_binary_stl",
]
