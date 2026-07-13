"""Stable ``python -m`` launcher for the full-pipeline visualization."""
from .apps.visualize_pipeline import build_parser, main

__all__ = ["build_parser", "main"]

if __name__ == "__main__":
    main()
