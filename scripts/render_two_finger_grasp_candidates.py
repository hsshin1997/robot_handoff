#!/usr/bin/env python3
"""Render representative samples from a continuous two-finger grasp map."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mujoco_sim.modeling.two_finger_grasp_visualization import (  # noqa: E402
    render_top_down_candidate_image,
)


DEFAULT_MAP = (
    ROOT
    / "projects"
    / "two_finger_grasp_map"
    / "generated"
    / "connector_header_grasp_map.json"
)
DEFAULT_SINGLE_OUTPUT = (
    ROOT
    / "projects"
    / "two_finger_grasp_map"
    / "generated"
    / "visualization"
    / "top_down_candidate_poses.png"
)
DEFAULT_ORIENTATION_OUTPUT = (
    ROOT
    / "projects"
    / "two_finger_grasp_map"
    / "generated"
    / "visualization"
    / "top_down_orientation_candidates.png"
)


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--map",
        dest="map_path",
        type=Path,
        default=DEFAULT_MAP,
        help="continuous grasp-map JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="top-down PNG output (default depends on selection mode)",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        help="companion JSON output (default: PNG path with .json suffix)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=6,
        help="representative poses in single mode, 1-10 (default: 6)",
    )
    parser.add_argument(
        "--selection-mode",
        choices=("single", "orientations"),
        default="single",
        help=(
            "single broad family or representatives from each distinct "
            "jaw-closing orientation (default: single)"
        ),
    )
    parser.add_argument(
        "--count-per-orientation",
        type=int,
        default=3,
        help="representatives per orientation in orientations mode, 1-5 (default: 3)",
    )
    parser.add_argument(
        "--orientation-tolerance-deg",
        type=float,
        default=15.0,
        help="axis-clustering angular tolerance (default: 15)",
    )
    parser.add_argument(
        "--maximum-orientation-groups",
        type=int,
        default=3,
        help="maximum distinct orientation groups to draw, 1-4 (default: 3)",
    )
    parser.add_argument("--width", type=int, default=1400)
    parser.add_argument("--height", type=int, default=900)
    arguments = parser.parse_args(argv)

    map_path = _resolve(arguments.map_path)
    default_output = (
        DEFAULT_ORIENTATION_OUTPUT
        if arguments.selection_mode == "orientations"
        else DEFAULT_SINGLE_OUTPUT
    )
    output_path = _resolve(
        arguments.output if arguments.output is not None else default_output)
    metadata_path = (
        _resolve(arguments.metadata_output)
        if arguments.metadata_output is not None
        else output_path.with_suffix(".json")
    )
    metadata = render_top_down_candidate_image(
        map_path,
        output_path,
        count=arguments.count,
        selection_mode=arguments.selection_mode,
        count_per_orientation=arguments.count_per_orientation,
        orientation_tolerance_deg=arguments.orientation_tolerance_deg,
        maximum_orientation_groups=arguments.maximum_orientation_groups,
        width=arguments.width,
        height=arguments.height,
    )
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    selected = metadata["selection"]
    print(f"Wrote top-down candidate image to {output_path}")
    print(f"Wrote exact displayed poses to {metadata_path}")
    if selected["mode"] == "orientation_diverse":
        print(
            f"Displayed {len(metadata['candidates'])} samples from "
            f"{selected['orientation_group_count']} closing orientations; "
            "insertion_safe=false"
        )
    else:
        print(
            f"Displayed {len(metadata['candidates'])} samples from "
            f"{selected['family_id']}; insertion_safe=false"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
