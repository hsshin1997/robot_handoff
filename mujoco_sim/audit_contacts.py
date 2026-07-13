"""Stable ``python -m`` launcher for the contact-path audit."""
from .diagnostics.contact_audit import build_parser, main, run_audit

__all__ = ["build_parser", "main", "run_audit"]

if __name__ == "__main__":
    main()
