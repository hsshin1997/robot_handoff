"""Compatibility wrapper for the canonical MuJoCo scene builder.

The previous generator targeted the discarded 0.200 m TCP scene. Keeping two
independent generators for the same output path was unsafe.
"""
from build_mujoco_scene import main


if __name__ == "__main__":
    main()
