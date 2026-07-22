#!/usr/bin/env python3
"""CLI wrapper for the connector insertion feasible-set visualization."""
from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mujoco_sim.apps.insertion_set_visualization import main


if __name__ == "__main__":
    raise SystemExit(main())
