"""Stable repository paths shared by every MuJoCo subsystem.

Implementation modules are intentionally free to move between subpackages.
All user data and generated artifacts remain rooted at the public
``mujoco_sim`` package directory so a code-only reorganization cannot silently
change model, configuration, cache, or CAD-converter locations.
"""
from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parent
MODELS_ROOT = PACKAGE_ROOT / "models"
GENERATED_CAD_ROOT = MODELS_ROOT / "generated_cad"
CACHE_ROOT = PACKAGE_ROOT / "cache"
CONFIG_ROOT = PACKAGE_ROOT / "config"
DEPRECATED_CONFIG_ROOT = CONFIG_ROOT / "deprecated"

DEFAULT_PROJECT_PATH = CONFIG_ROOT / "project.yaml"
DEFAULT_SOLVER_PATH = CONFIG_ROOT / "solver_defaults.yaml"
DEFAULT_SCENE_CONFIG_PATH = CONFIG_ROOT / "internal" / "scene_fallback.yaml"
DEFAULT_PIPELINE_CONFIG_PATH = DEPRECATED_CONFIG_ROOT / "pipeline_config.yaml"
DEFAULT_GRIPPER_TEMPLATE_PATH = (
    CONFIG_ROOT / "templates" / "gripper_asset.template.yaml")
DEFAULT_MODEL_PATH = MODELS_ROOT / "scene.xml"
FREECAD_CONVERTER_PATH = REPOSITORY_ROOT / "scripts" / "freecad_step_to_stl.py"


__all__ = [
    "CACHE_ROOT",
    "CONFIG_ROOT",
    "DEPRECATED_CONFIG_ROOT",
    "DEFAULT_GRIPPER_TEMPLATE_PATH",
    "DEFAULT_MODEL_PATH",
    "DEFAULT_PIPELINE_CONFIG_PATH",
    "DEFAULT_PROJECT_PATH",
    "DEFAULT_SCENE_CONFIG_PATH",
    "DEFAULT_SOLVER_PATH",
    "FREECAD_CONVERTER_PATH",
    "GENERATED_CAD_ROOT",
    "MODELS_ROOT",
    "PACKAGE_ROOT",
    "REPOSITORY_ROOT",
]
