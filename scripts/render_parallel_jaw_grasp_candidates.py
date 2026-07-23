#!/usr/bin/env python3
"""Render selected candidates from a sampled parallel-jaw grasp JSON file."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mujoco_sim.modeling.parallel_jaw_grasp_visualization import (  # noqa: E402
    DEFAULT_GENERATED_ROOT,
    render_parallel_jaw_candidate_image,
)
from mujoco_sim.modeling.cad_preprocess import verify_preparation  # noqa: E402
from mujoco_sim.offline_tools.artifacts import atomic_write_bytes  # noqa: E402


def _resolve(path: Path) -> Path:
    return path.expanduser().resolve()


def _default_output(candidate_path: Path) -> Path:
    return candidate_path.with_suffix(".preview.png")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "candidates",
        type=Path,
        help="JSON written by generate_parallel_jaw_grasps.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="PNG output (default: <candidate-json-stem>.preview.png)",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        help="companion JSON (default: PNG path with .json suffix)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=4,
        help="automatic display count, 1-8 (default: 4)",
    )
    parser.add_argument(
        "--selection",
        choices=("pose-diverse", "ranked"),
        default="pose-diverse",
        help=(
            "display-coverage selection or generator output order "
            "(default: pose-diverse)"
        ),
    )
    parser.add_argument(
        "--candidate-id",
        action="append",
        default=[],
        help=(
            "exact candidate ID to display; repeat to choose multiple IDs "
            "in the requested order"
        ),
    )
    parser.add_argument(
        "--generated-root",
        type=Path,
        default=DEFAULT_GENERATED_ROOT,
        help=(
            "prepared-CAD cache used during generation "
            f"(default: {DEFAULT_GENERATED_ROOT})"
        ),
    )
    parser.add_argument(
        "--cad",
        type=Path,
        help="relocated source CAD override; its SHA-256 must match the JSON",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1600,
        help="PNG width in pixels, at least 1200 (default: 1600)",
    )
    parser.add_argument(
        "--row-height",
        type=int,
        default=320,
        help="height per displayed candidate, at least 260 (default: 320)",
    )
    parser.add_argument(
        "--max-render-triangles",
        type=int,
        default=0,
        help=(
            "0 projects every CAD triangle (default); a positive value uses "
            "an area-stratified display-only subset"
        ),
    )
    parser.add_argument(
        "--allow-unreliable-input",
        action="store_true",
        help=(
            "permit a diagnostic preview of candidates generated from an "
            "unreliable mesh and add an explicit warning"
        ),
    )
    arguments = parser.parse_args(argv)
    if arguments.max_render_triangles < 0:
        parser.error("--max-render-triangles must be non-negative")

    candidate_path = _resolve(arguments.candidates)
    output_path = _resolve(
        arguments.output
        if arguments.output is not None
        else _default_output(candidate_path)
    )
    metadata_path = (
        _resolve(arguments.metadata_output)
        if arguments.metadata_output is not None
        else output_path.with_suffix(".json")
    )
    generated_root = _resolve(arguments.generated_root)
    explicit_cad = (
        None if arguments.cad is None else _resolve(arguments.cad)
    )
    try:
        source_document = json.loads(
            candidate_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        parser.error(f"cannot read candidate JSON for path validation: {error}")
    protected_sources = {candidate_path}
    if explicit_cad is not None:
        protected_sources.add(explicit_cad)
    if isinstance(source_document, dict):
        cad_record = source_document.get("cad")
        if isinstance(cad_record, dict):
            recorded_cad = cad_record.get("path")
            if isinstance(recorded_cad, str) and recorded_cad:
                protected_sources.add(Path(recorded_cad).expanduser().resolve())
            fingerprint = cad_record.get("artifact_fingerprint")
            if (
                isinstance(fingerprint, str)
                and len(fingerprint) == 64
                and all(
                    character in "0123456789abcdef"
                    for character in fingerprint
                )
            ):
                prepared_metadata = (
                    generated_root / fingerprint / "metadata.json"
                ).resolve()
                protected_sources.add(prepared_metadata)
                if prepared_metadata.is_file():
                    try:
                        prepared_document = json.loads(
                            prepared_metadata.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError) as error:
                        parser.error(
                            "cannot read prepared-CAD metadata for path "
                            f"validation: {error}"
                        )
                    visual = (
                        prepared_document.get("visual")
                        if isinstance(prepared_document, dict)
                        else None
                    )
                    chunks = (
                        visual.get("chunks")
                        if isinstance(visual, dict)
                        else None
                    )
                    if isinstance(chunks, list):
                        for chunk in chunks:
                            if not isinstance(chunk, dict):
                                continue
                            relative = chunk.get("path")
                            if isinstance(relative, str) and relative:
                                protected_sources.add(
                                    (
                                        prepared_metadata.parent / relative
                                    ).resolve()
                                )
    if output_path == metadata_path:
        parser.error("--output and --metadata-output must be different files")
    for target_name, target_path in (
        ("--output", output_path),
        ("--metadata-output", metadata_path),
    ):
        if target_path in protected_sources:
            parser.error(
                f"{target_name} must not overwrite the candidate JSON, CAD, "
                "or prepared-CAD metadata"
            )

    metadata = render_parallel_jaw_candidate_image(
        candidate_path,
        output_path,
        count=arguments.count,
        selection_mode=arguments.selection,
        candidate_ids=arguments.candidate_id,
        generated_root=generated_root,
        cad_path=explicit_cad,
        width=arguments.width,
        row_height=arguments.row_height,
        max_render_triangles=(
            None
            if arguments.max_render_triangles == 0
            else arguments.max_render_triangles
        ),
        allow_unreliable_input=arguments.allow_unreliable_input,
    )
    prepared_metadata_path = Path(
        metadata["cad"]["prepared_metadata_path"]).resolve()
    preparation = verify_preparation(prepared_metadata_path)
    post_render_protected = {
        candidate_path,
        output_path,
        prepared_metadata_path,
        Path(metadata["cad"]["path"]).resolve(),
    }
    registered = list(
        preparation.metadata.get("visual", {}).get("chunks", []))
    for component in preparation.metadata.get(
        "static_assembly", {}
    ).get("components", []):
        registered.extend(component.get("chunks", []))
    for output_record in registered:
        relative = (
            output_record.get("path")
            if isinstance(output_record, dict)
            else None
        )
        if isinstance(relative, str) and relative:
            post_render_protected.add(
                (preparation.artifact_dir / relative).resolve())
    if metadata_path in post_render_protected:
        parser.error(
            "--metadata-output must not overwrite the candidate JSON, PNG, "
            "CAD, or a prepared-CAD artifact"
        )
    atomic_write_bytes(
        metadata_path,
        (
            json.dumps(
            metadata,
            allow_nan=False,
            indent=2,
            sort_keys=True,
            )
            + "\n"
        ).encode("utf-8"),
    )

    selected = metadata["selection"]["displayed"]
    ids = ", ".join(item["id"] for item in selected)
    print(f"Wrote {len(selected)} candidate previews to {output_path}")
    print(f"Wrote exact preview metadata to {metadata_path}")
    print(f"Displayed IDs: {ids}")
    print(
        "Visualization only: physical gripper collision, approach sweep, "
        "robot reachability, and task feasibility are not certified"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
