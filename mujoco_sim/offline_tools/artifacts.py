"""Deterministic, content-addressed storage for offline pipeline artifacts.

The handoff pipeline has several expensive preprocessing stages whose outputs
must be invalidated when geometry, solver parameters, producer code, or an
upstream artifact changes.  This module provides the small dependency-free
core for doing that.  It intentionally knows nothing about MuJoCo or NumPy.

Artifact values are canonical JSON-compatible data.  ``bytes`` values are
also supported through a tagged base64 representation, which is useful for
opaque payloads owned by a preprocessing hook.
"""
from __future__ import annotations

import base64
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any


CACHE_FORMAT_VERSION = 1
KEY_SCHEMA_VERSION = 1
COVERAGE_SCHEMA_VERSION = 1
PROJECT_METADATA_SCHEMA_VERSION = 1
_BYTES_TAG = "__handoff_offline_bytes_v1__"


class ArtifactCategory(str, Enum):
    """Named artifact tiers in dependency order."""

    MESH = "mesh"
    GRASP = "grasp"
    STABLE_POSE = "stable-pose"
    REACHABILITY = "reachability"
    CO_GRASP = "co-grasp"
    MOTION_ROADMAP = "motion-roadmap"
    TASK_POLICY = "task-policy"
    COVERAGE = "coverage"


ARTIFACT_CATEGORIES = tuple(category.value for category in ArtifactCategory)


class CacheError(RuntimeError):
    """Base class for offline-cache failures."""


class CacheCorruptionError(CacheError):
    """Raised when a cache entry fails its content or key checks."""


class CacheLockTimeout(CacheError):
    """Raised when another process holds a cache key for too long."""


def _canonicalize(value: Any) -> Any:
    """Convert supported values into a deterministic JSON data model."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical content cannot contain NaN or infinity")
        # Avoid distinct hashes for the numerically equal values -0.0 and 0.0.
        return 0.0 if value == 0.0 else value
    if isinstance(value, Enum):
        return _canonicalize(value.value)
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {_BYTES_TAG: base64.b64encode(bytes(value)).decode("ascii")}
    if is_dataclass(value) and not isinstance(value, type):
        return _canonicalize(asdict(value))
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"canonical mapping keys must be strings, got {type(key).__name__}")
            result[key] = _canonicalize(item)
        return result
    if isinstance(value, (set, frozenset)):
        normalized = [_canonicalize(item) for item in value]
        return sorted(normalized, key=canonical_json_bytes)
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]

    # Array-like values can participate without importing an array package.
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _canonicalize(tolist())
    raise TypeError(f"unsupported canonical content type: {type(value).__name__}")


def _decanonicalize(value: Any) -> Any:
    if isinstance(value, list):
        return [_decanonicalize(item) for item in value]
    if isinstance(value, dict):
        if set(value) == {_BYTES_TAG}:
            try:
                return base64.b64decode(value[_BYTES_TAG], validate=True)
            except (ValueError, TypeError) as error:
                raise CacheCorruptionError("invalid base64 byte payload") from error
        return {key: _decanonicalize(item) for key, item in value.items()}
    return value


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize content in the cache's stable canonical JSON form."""
    return json.dumps(
        _canonicalize(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def fingerprint_bytes(content: bytes | bytearray | memoryview) -> str:
    """Return the lowercase SHA-256 hex digest of raw bytes."""
    return hashlib.sha256(bytes(content)).hexdigest()


def fingerprint_content(content: Any) -> str:
    """Fingerprint canonical structured content, independent of map ordering."""
    return fingerprint_bytes(canonical_json_bytes(content))


def fingerprint_file(path: str | os.PathLike[str], *, chunk_size: int = 1024 * 1024) -> str:
    """Stream a file and return its content-only SHA-256 fingerprint."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _category_value(category: str | ArtifactCategory) -> str:
    value = category.value if isinstance(category, ArtifactCategory) else str(category)
    if value not in ARTIFACT_CATEGORIES:
        expected = ", ".join(ARTIFACT_CATEGORIES)
        raise ValueError(f"unknown artifact category {value!r}; expected one of: {expected}")
    return value


def _fingerprint_map(
    values: Mapping[str, str | "ArtifactKey"] | None,
    name: str,
) -> tuple[tuple[str, str], ...]:
    result = []
    for label, value in (values or {}).items():
        if not isinstance(label, str) or not label:
            raise ValueError(f"{name} labels must be non-empty strings")
        fingerprint = value.digest if isinstance(value, ArtifactKey) else value
        if not isinstance(fingerprint, str) or not fingerprint:
            raise ValueError(f"{name}[{label!r}] must be a non-empty fingerprint string")
        result.append((label, fingerprint))
    return tuple(sorted(result))


@dataclass(frozen=True)
class ArtifactKey:
    """A schema-, producer-, input-, parameter-, and dependency-aware key.

    Construct keys with :func:`make_artifact_key`; the tuple fields are sorted
    there so the dataclass remains immutable and its digest reproducible.
    """

    category: str
    name: str
    artifact_version: str
    schema_version: int
    input_fingerprints: tuple[tuple[str, str], ...]
    dependency_fingerprints: tuple[tuple[str, str], ...]
    parameters_json: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "category": self.category,
            "name": self.name,
            "artifact_version": self.artifact_version,
            "inputs": dict(self.input_fingerprints),
            "dependencies": dict(self.dependency_fingerprints),
            "parameters": json.loads(self.parameters_json),
        }

    @property
    def digest(self) -> str:
        return fingerprint_content(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ArtifactKey":
        required = {
            "schema_version", "category", "name", "artifact_version",
            "inputs", "dependencies", "parameters",
        }
        if set(value) != required:
            raise CacheCorruptionError(
                f"artifact key fields differ: expected {sorted(required)}, got {sorted(value)}"
            )
        try:
            return make_artifact_key(
                value["category"],
                value["name"],
                artifact_version=value["artifact_version"],
                schema_version=value["schema_version"],
                input_fingerprints=value["inputs"],
                dependencies=value["dependencies"],
                parameters=value["parameters"],
            )
        except (TypeError, ValueError) as error:
            raise CacheCorruptionError("invalid artifact key") from error


def make_artifact_key(
    category: str | ArtifactCategory,
    name: str,
    *,
    artifact_version: str | int,
    schema_version: int = KEY_SCHEMA_VERSION,
    input_fingerprints: Mapping[str, str | ArtifactKey] | None = None,
    dependencies: Mapping[str, str | ArtifactKey] | None = None,
    parameters: Any = None,
) -> ArtifactKey:
    """Build a deterministic content-addressed artifact key.

    Dependency values may be upstream :class:`ArtifactKey` objects or stored
    digests.  Consequently, changing any transitive producer input changes
    every downstream key when callers pass upstream keys as dependencies.
    """
    category_value = _category_value(category)
    if not isinstance(name, str) or not name.strip():
        raise ValueError("artifact name must be a non-empty string")
    if not isinstance(schema_version, int) or isinstance(schema_version, bool) or schema_version < 1:
        raise ValueError("schema_version must be a positive integer")
    version = str(artifact_version)
    if not version:
        raise ValueError("artifact_version must not be empty")
    parameters_json = canonical_json_bytes({} if parameters is None else parameters).decode("utf-8")
    return ArtifactKey(
        category=category_value,
        name=name,
        artifact_version=version,
        schema_version=schema_version,
        input_fingerprints=_fingerprint_map(input_fingerprints, "input_fingerprints"),
        dependency_fingerprints=_fingerprint_map(dependencies, "dependencies"),
        parameters_json=parameters_json,
    )


def _fsync_directory(directory: Path) -> None:
    """Best-effort persistence of the rename itself on POSIX filesystems."""
    flags = getattr(os, "O_DIRECTORY", 0) | os.O_RDONLY
    try:
        descriptor = os.open(directory, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_bytes(path: str | os.PathLike[str], content: bytes) -> Path:
    """Durably replace a file using a same-directory temporary file."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return destination


def atomic_write_json(path: str | os.PathLike[str], content: Any) -> Path:
    """Atomically write canonical JSON followed by one newline."""
    return atomic_write_bytes(path, canonical_json_bytes(content) + b"\n")


@dataclass(frozen=True)
class ArtifactRecord:
    key: ArtifactKey
    value: Any
    value_fingerprint: str
    path: Path


class _KeyLock:
    def __init__(self, path: Path, timeout_s: float, stale_after_s: float):
        self.path = path
        self.timeout_s = timeout_s
        self.stale_after_s = stale_after_s
        self.acquired = False

    def __enter__(self) -> "_KeyLock":
        start = time.monotonic()
        while True:
            try:
                descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                try:
                    age = time.time() - self.path.stat().st_mtime
                    if age > self.stale_after_s:
                        self.path.unlink()
                        continue
                except FileNotFoundError:
                    continue
                if time.monotonic() - start >= self.timeout_s:
                    raise CacheLockTimeout(f"timed out waiting for cache lock {self.path}")
                time.sleep(0.02)
                continue
            with os.fdopen(descriptor, "w", encoding="ascii") as stream:
                stream.write(f"pid={os.getpid()}\n")
                stream.flush()
                os.fsync(stream.fileno())
            self.acquired = True
            return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.acquired:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass


class ArtifactCache:
    """Atomic content-addressed cache for canonical artifact values."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        lock_timeout_s: float = 60.0,
        stale_lock_s: float = 3600.0,
    ):
        if lock_timeout_s <= 0 or stale_lock_s <= 0:
            raise ValueError("lock timeouts must be positive")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock_timeout_s = float(lock_timeout_s)
        self.stale_lock_s = float(stale_lock_s)

    def path_for(self, key: ArtifactKey) -> Path:
        _category_value(key.category)
        return self.root / key.category / f"{key.digest}.json"

    def contains(self, key: ArtifactKey) -> bool:
        return self.path_for(key).is_file()

    def get_record(self, key: ArtifactKey) -> ArtifactRecord | None:
        path = self.path_for(key)
        try:
            with open(path, encoding="utf-8") as stream:
                envelope = json.load(stream)
        except FileNotFoundError:
            return None
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise CacheCorruptionError(f"cannot decode cache entry {path}") from error

        required = {"cache_format_version", "key_digest", "key", "value_fingerprint", "value"}
        if not isinstance(envelope, dict) or set(envelope) != required:
            raise CacheCorruptionError(f"invalid cache envelope at {path}")
        if envelope["cache_format_version"] != CACHE_FORMAT_VERSION:
            raise CacheCorruptionError(
                f"unsupported cache format {envelope['cache_format_version']!r} at {path}"
            )
        stored_key = ArtifactKey.from_dict(envelope["key"])
        if stored_key.digest != envelope["key_digest"] or stored_key.digest != key.digest:
            raise CacheCorruptionError(f"artifact key digest mismatch at {path}")
        if stored_key.to_dict() != key.to_dict():
            raise CacheCorruptionError(f"artifact key payload mismatch at {path}")
        actual_fingerprint = fingerprint_content(envelope["value"])
        if actual_fingerprint != envelope["value_fingerprint"]:
            raise CacheCorruptionError(f"artifact value fingerprint mismatch at {path}")
        return ArtifactRecord(
            key=stored_key,
            value=_decanonicalize(envelope["value"]),
            value_fingerprint=actual_fingerprint,
            path=path,
        )

    def get(self, key: ArtifactKey, default: Any = None) -> Any:
        record = self.get_record(key)
        return default if record is None else record.value

    def put(self, key: ArtifactKey, value: Any) -> ArtifactRecord:
        canonical_value = _canonicalize(value)
        value_fingerprint = fingerprint_content(canonical_value)
        path = self.path_for(key)
        envelope = {
            "cache_format_version": CACHE_FORMAT_VERSION,
            "key_digest": key.digest,
            "key": key.to_dict(),
            "value_fingerprint": value_fingerprint,
            "value": canonical_value,
        }
        atomic_write_json(path, envelope)
        return ArtifactRecord(key, _decanonicalize(canonical_value), value_fingerprint, path)

    def get_or_compute(self, key: ArtifactKey, compute: Callable[[], Any]) -> Any:
        """Return a verified hit or compute and atomically publish one miss.

        A per-key cross-process lock prevents duplicate expensive work.  The
        entry is checked again after acquiring the lock because another worker
        may have populated it while this worker waited.
        """
        record = self.get_record(key)
        if record is not None:
            return record.value
        path = self.path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        lock = path.with_suffix(path.suffix + ".lock")
        with _KeyLock(lock, self.lock_timeout_s, self.stale_lock_s):
            record = self.get_record(key)
            if record is not None:
                return record.value
            value = compute()
            return self.put(key, value).value


def _coverage_ids(values: Iterable[str | ArtifactKey]) -> list[str]:
    identifiers = set()
    for value in values:
        identifier = value.digest if isinstance(value, ArtifactKey) else value
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("coverage artifact identifiers must be non-empty strings")
        identifiers.add(identifier)
    return sorted(identifiers)


def build_coverage_report(
    required: Mapping[str | ArtifactCategory, Iterable[str | ArtifactKey]],
    available: Mapping[str | ArtifactCategory, Iterable[str | ArtifactKey]],
    *,
    project_fingerprint: str,
) -> dict[str, Any]:
    """Build a deterministic coverage report over every named category.

    Both mapping order and identifier order are intentionally discarded.  The
    report contains no wall-clock timestamp, making it suitable for hashing,
    regression tests, and storage as a ``coverage`` artifact.
    """
    if not isinstance(project_fingerprint, str) or not project_fingerprint:
        raise ValueError("project_fingerprint must be a non-empty string")
    required_normalized = {_category_value(key): _coverage_ids(value)
                           for key, value in required.items()}
    available_normalized = {_category_value(key): _coverage_ids(value)
                            for key, value in available.items()}

    categories: dict[str, Any] = {}
    total_required = 0
    total_covered = 0
    for category in ARTIFACT_CATEGORIES:
        required_ids = required_normalized.get(category, [])
        available_ids = available_normalized.get(category, [])
        required_set = set(required_ids)
        available_set = set(available_ids)
        covered = sorted(required_set & available_set)
        missing = sorted(required_set - available_set)
        unexpected = sorted(available_set - required_set)
        total_required += len(required_ids)
        total_covered += len(covered)
        categories[category] = {
            "required": required_ids,
            "available": available_ids,
            "covered": covered,
            "missing": missing,
            "unexpected": unexpected,
            "required_count": len(required_ids),
            "covered_count": len(covered),
            "fraction": 1.0 if not required_ids else len(covered) / len(required_ids),
            "complete": not missing,
        }

    total_missing = total_required - total_covered
    return {
        "schema_version": COVERAGE_SCHEMA_VERSION,
        "project_fingerprint": project_fingerprint,
        "complete": total_missing == 0,
        "summary": {
            "required_count": total_required,
            "covered_count": total_covered,
            "missing_count": total_missing,
            "fraction": 1.0 if total_required == 0 else total_covered / total_required,
        },
        "categories": categories,
    }


_PROJECT_PATH_FIELDS = {
    "model", "cad", "visual_cad", "collision_cad", "additional_collision_cad"
}


def _project_file_references(value: Any, path: tuple[str, ...] = ()) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    if isinstance(value, Mapping):
        for key in sorted(value):
            item = value[key]
            item_path = path + (str(key),)
            if key in _PROJECT_PATH_FIELDS:
                references = item if isinstance(item, list) else [item]
                for index, reference in enumerate(references):
                    if not isinstance(reference, str):
                        raise TypeError(f"project path {'/'.join(item_path)} must be a string")
                    suffix = (str(index),) if isinstance(item, list) else ()
                    result.append(("/".join(item_path + suffix), reference))
            else:
                result.extend(_project_file_references(item, item_path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            result.extend(_project_file_references(item, path + (str(index),)))
    return result


def build_project_metadata(
    manifest: Mapping[str, Any],
    manifest_path: str | os.PathLike[str],
    *,
    project_root: str | os.PathLike[str],
) -> dict[str, Any]:
    """Fingerprint a parsed project manifest and every declared CAD/model file."""
    source = Path(manifest_path).resolve()
    root = Path(project_root).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    assets = []
    for field, reference in _project_file_references(manifest):
        candidate = Path(reference)
        resolved = candidate if candidate.is_absolute() else root / candidate
        if not resolved.is_file():
            raise FileNotFoundError(f"project asset {field!r} does not exist: {resolved}")
        assets.append({
            "field": field,
            "path": reference,
            "size_bytes": resolved.stat().st_size,
            "sha256": fingerprint_file(resolved),
        })
    assets.sort(key=lambda item: (item["field"], item["path"]))
    canonical_manifest_sha256 = fingerprint_content(manifest)
    project_fingerprint = fingerprint_content({
        "manifest": canonical_manifest_sha256,
        "assets": [{"field": item["field"], "sha256": item["sha256"]} for item in assets],
    })
    try:
        project_file = source.relative_to(root).as_posix()
    except ValueError:
        project_file = source.name
    return {
        "schema_version": PROJECT_METADATA_SCHEMA_VERSION,
        "project_file": project_file,
        "manifest_schema_version": manifest.get("schema_version"),
        "manifest_source_sha256": fingerprint_file(source),
        "manifest_canonical_sha256": canonical_manifest_sha256,
        "project_fingerprint": project_fingerprint,
        "artifact_categories": list(ARTIFACT_CATEGORIES),
        "assets": assets,
    }


@dataclass(frozen=True)
class PrecomputeContext:
    """Inputs exposed to optional preprocessing hooks."""

    project_path: Path
    project_root: Path
    manifest: Mapping[str, Any]
    metadata: Mapping[str, Any]
    cache: ArtifactCache
    model_path: Path | None = None


def run_precompute_hooks(
    context: PrecomputeContext,
    hooks: Mapping[str, Callable[[PrecomputeContext], Any]] | Sequence[tuple[str, Callable]],
) -> dict[str, Any]:
    """Run named hooks in lexical order and canonicalize their summaries."""
    items = hooks.items() if isinstance(hooks, Mapping) else hooks
    by_name: dict[str, Callable[[PrecomputeContext], Any]] = {}
    for name, hook in items:
        if not isinstance(name, str) or not name:
            raise ValueError("hook names must be non-empty strings")
        if name in by_name:
            raise ValueError(f"duplicate preprocessing hook: {name}")
        if not callable(hook):
            raise TypeError(f"preprocessing hook {name!r} is not callable")
        by_name[name] = hook
    return {name: _canonicalize(by_name[name](context)) for name in sorted(by_name)}


def write_project_metadata(
    cache_root: str | os.PathLike[str],
    metadata: Mapping[str, Any],
) -> Path:
    """Atomically publish the deterministic project snapshot."""
    return atomic_write_json(Path(cache_root) / "project-metadata.json", metadata)


__all__ = [
    "ARTIFACT_CATEGORIES",
    "CACHE_FORMAT_VERSION",
    "COVERAGE_SCHEMA_VERSION",
    "KEY_SCHEMA_VERSION",
    "PROJECT_METADATA_SCHEMA_VERSION",
    "ArtifactCache",
    "ArtifactCategory",
    "ArtifactKey",
    "ArtifactRecord",
    "CacheCorruptionError",
    "CacheError",
    "CacheLockTimeout",
    "PrecomputeContext",
    "atomic_write_bytes",
    "atomic_write_json",
    "build_coverage_report",
    "build_project_metadata",
    "canonical_json_bytes",
    "fingerprint_bytes",
    "fingerprint_content",
    "fingerprint_file",
    "make_artifact_key",
    "run_precompute_hooks",
    "write_project_metadata",
]
