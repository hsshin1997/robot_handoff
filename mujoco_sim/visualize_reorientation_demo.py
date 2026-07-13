"""Stable ``python -m`` launcher for the reorientation visualization."""
from .apps.visualize_reorientation_demo import build_demo, build_parser, main

__all__ = ["build_demo", "build_parser", "main"]

if __name__ == "__main__":
    main()
