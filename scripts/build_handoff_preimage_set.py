#!/usr/bin/env python3
"""Build the evidence-backed direct/transfer handoff preimage artifact."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mujoco_sim.offline_tools.artifacts import (  # noqa: E402
    atomic_write_json,
    fingerprint_file,
)
from mujoco_sim.planner.handoff_preimage_set import (  # noqa: E402
    build_handoff_preimage_set,
)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError("preimage-set config root must be a YAML mapping")
    return value


def _resolve(value: str | Path, owner: Path) -> Path:
    supplied = Path(value).expanduser()
    return (supplied if supplied.is_absolute()
            else owner.parent / supplied).resolve()


def _load_json_mapping(path: Path) -> Mapping[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON root must be a mapping: {path}")
    return value


def _select_record(root: Mapping[str, Any], record_id: str | None) -> Mapping[str, Any]:
    if record_id is None:
        return root
    records = root.get("records")
    if isinstance(records, Mapping):
        value = records.get(record_id)
        if isinstance(value, Mapping):
            return value
    if isinstance(records, list):
        for value in records:
            if (isinstance(value, Mapping)
                    and value.get("id", value.get("evidence_id")) == record_id):
                return value
    raise ValueError(f"evidence record {record_id!r} not found")


def _evidence_catalog(config: Mapping[str, Any], config_path: Path) -> dict[str, Any]:
    declarations = config.get("evidence_artifacts", [])
    if not isinstance(declarations, list):
        raise ValueError("evidence_artifacts must be a list")
    catalog: dict[str, Any] = {}
    for declaration in declarations:
        if not isinstance(declaration, Mapping):
            raise ValueError("each evidence_artifacts entry must be a mapping")
        evidence_id = declaration.get("id")
        if not isinstance(evidence_id, str) or not evidence_id:
            raise ValueError("each evidence artifact requires a non-empty id")
        if evidence_id in catalog:
            raise ValueError(f"duplicate evidence artifact id {evidence_id!r}")
        path_value = declaration.get("path")
        if not isinstance(path_value, (str, Path)):
            catalog[evidence_id] = {
                "expected_sha256_configured": False,
                "provenance_verified": False,
                "sha256": None,
                "payload": None,
                "load_error": "path_missing",
            }
            continue
        path = _resolve(path_value, config_path)
        if not path.is_file():
            catalog[evidence_id] = {
                "path": str(path),
                "expected_sha256_configured": isinstance(
                    declaration.get("expected_sha256"), str),
                "provenance_verified": False,
                "sha256": None,
                "payload": None,
                "load_error": "file_missing",
            }
            continue
        actual_sha = fingerprint_file(path)
        expected_sha = declaration.get("expected_sha256")
        provenance_verified = (
            isinstance(expected_sha, str)
            and len(expected_sha) == 64
            and expected_sha.lower() == actual_sha.lower()
        )
        try:
            payload = _select_record(
                _load_json_mapping(path), declaration.get("record_id"))
            load_error = None
        except (OSError, ValueError, json.JSONDecodeError) as error:
            payload = None
            load_error = f"{type(error).__name__}: {error}"
            provenance_verified = False
        catalog[evidence_id] = {
            "path": str(path),
            "sha256": actual_sha,
            "expected_sha256": expected_sha,
            "expected_sha256_configured": (
                isinstance(expected_sha, str) and len(expected_sha) == 64),
            "provenance_verified": provenance_verified,
            "payload": payload,
            "load_error": load_error,
        }
    return catalog


def run(
    config_path: str | Path,
    *,
    output_path: str | Path | None = None,
) -> tuple[dict[str, Any], Path]:
    path = Path(config_path).resolve()
    config = _load_yaml(path)
    receiver_value = config.get("receiver_insertion_set")
    receiver_path = (
        None if not isinstance(receiver_value, (str, Path))
        else _resolve(receiver_value, path)
    )
    receiver: Mapping[str, Any] | None = None
    receiver_source: dict[str, Any]
    if receiver_path is None:
        receiver_source = {"path": None, "sha256": None, "status": "MISSING"}
    elif not receiver_path.is_file():
        receiver_source = {
            "path": str(receiver_path), "sha256": None, "status": "MISSING"}
    else:
        try:
            receiver = _load_json_mapping(receiver_path)
            receiver_source = {
                "path": str(receiver_path),
                "sha256": fingerprint_file(receiver_path),
                "status": "LOADED",
            }
        except (OSError, ValueError, json.JSONDecodeError) as error:
            receiver_source = {
                "path": str(receiver_path),
                "sha256": fingerprint_file(receiver_path),
                "status": "INVALID",
                "error": f"{type(error).__name__}: {error}",
            }
    catalog = _evidence_catalog(config, path)
    result = build_handoff_preimage_set(
        receiver,
        config,
        evidence_catalog=catalog,
        receiver_source=receiver_source,
    )
    result["config"] = {
        "path": str(path),
        "sha256": fingerprint_file(path),
    }
    configured_output = config.get("output")
    if output_path is not None:
        destination = Path(output_path).expanduser().resolve()
    elif isinstance(configured_output, (str, Path)):
        destination = _resolve(configured_output, path)
    else:
        destination = path.with_name(f"{path.stem}_result.json")
    atomic_write_json(destination, result)
    return result, destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result, destination = run(args.config, output_path=args.output)
    summary = result["summary"]
    print(f"Wrote {destination}")
    print(
        "Classes: {class_count}; DIRECT={direct_count}, TRANSFER="
        "{reorientation_count}, UNCOVERED={uncovered_count}, UNKNOWN="
        "{unknown_count}".format(**summary)
    )
    if result["certification"]["missing_inputs"]:
        print("Missing inputs: " + ", ".join(
            result["certification"]["missing_inputs"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
