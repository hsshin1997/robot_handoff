#!/usr/bin/env python3
"""Fingerprint a handoff project and run optional offline preprocessing hooks.

The built-in pass is intentionally cheap: it snapshots the canonical project
manifest plus every referenced model/CAD file.  Expensive producers are added
without coupling this driver to them::

    python scripts/precompute_pipeline.py \
        --hook my_package.mesh_cache:precompute \
        --hook my_package.grasps:precompute

Each hook receives ``mujoco_sim.offline_tools.artifacts.PrecomputeContext`` and may use its
``ArtifactCache`` to publish dependency-aware artifacts.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mujoco_sim.offline_tools.artifacts import (  # noqa: E402
    ArtifactCache,
    PrecomputeContext,
    build_project_metadata,
    canonical_json_bytes,
    run_precompute_hooks,
    write_project_metadata,
)
from mujoco_sim.core.paths import DEFAULT_PROJECT_PATH  # noqa: E402


DEFAULT_PROJECT = DEFAULT_PROJECT_PATH
DEFAULT_CACHE = ROOT / "mujoco_sim" / "cache"
DEFAULT_MODEL = ROOT / "mujoco_sim" / "models" / "scene.xml"


def load_project(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load a YAML manifest and require a mapping at its root."""
    with open(path, encoding="utf-8") as stream:
        manifest = yaml.safe_load(stream)
    if not isinstance(manifest, dict):
        raise TypeError("project manifest root must be a mapping")
    return manifest


def load_hook(specification: str) -> tuple[str, Callable[[PrecomputeContext], Any]]:
    """Resolve ``module:callable`` and return its stable registration name."""
    module_name, separator, attribute_name = specification.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError(f"hook must use module:callable syntax, got {specification!r}")
    module = importlib.import_module(module_name)
    hook = getattr(module, attribute_name)
    if not callable(hook):
        raise TypeError(f"hook {specification!r} is not callable")
    return specification, hook


def precompute(
    project_path: str | os.PathLike[str] = DEFAULT_PROJECT,
    cache_dir: str | os.PathLike[str] = DEFAULT_CACHE,
    *,
    project_root: str | os.PathLike[str] = ROOT,
    model_path: str | os.PathLike[str] | None = None,
    hooks: tuple[tuple[str, Callable[[PrecomputeContext], Any]], ...] = (),
) -> dict[str, Any]:
    """Create the project snapshot, run hooks, and atomically publish metadata."""
    project_path = Path(project_path).resolve()
    project_root = Path(project_root).resolve()
    manifest = load_project(project_path)
    metadata = build_project_metadata(manifest, project_path, project_root=project_root)
    cache = ArtifactCache(cache_dir)
    resolved_model = None if model_path is None else Path(model_path).resolve()
    context = PrecomputeContext(
        project_path, project_root, manifest, metadata, cache, resolved_model)
    hook_results = run_precompute_hooks(context, hooks)
    output = dict(metadata)
    output["hooks"] = hook_results
    write_project_metadata(cache.root, output)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=str(DEFAULT_PROJECT))
    parser.add_argument("--project-root", default=str(ROOT),
                        help="base directory for relative paths in project.yaml")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    parser.add_argument("--model", default=str(DEFAULT_MODEL),
                        help="MJCF compiled from --project for production hooks")
    parser.add_argument("--hook", action="append", default=[], metavar="MODULE:CALLABLE")
    parser.add_argument("--production", action="store_true",
                        help="also build grasp/stable/downstream/task-policy artifacts")
    parser.add_argument("--json", action="store_true",
                        help="print the complete deterministic metadata document")
    args = parser.parse_args(argv)

    hook_specs = list(args.hook)
    if args.production:
        hook_specs.append(
            "mujoco_sim.offline_tools.precompute:precompute_runtime")
    hooks = tuple(load_hook(specification) for specification in hook_specs)
    metadata = precompute(
        args.project,
        args.cache_dir,
        project_root=args.project_root,
        model_path=args.model,
        hooks=hooks,
    )
    if args.json:
        sys.stdout.buffer.write(canonical_json_bytes(metadata) + b"\n")
    else:
        print(f"project fingerprint: {metadata['project_fingerprint']}")
        print(f"manifest fingerprint: {metadata['manifest_canonical_sha256']}")
        print(f"assets fingerprinted: {len(metadata['assets'])}")
        print(f"hooks completed: {len(metadata['hooks'])}")
        print(f"metadata: {Path(args.cache_dir) / 'project-metadata.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
