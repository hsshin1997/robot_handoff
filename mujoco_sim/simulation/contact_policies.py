"""Named collision allowances for each physical pipeline phase."""
from __future__ import annotations

from typing import Protocol


class _ProjectLike(Protocol):
    manifest: dict


REORIENTATION_CONTACTS = (
    ("part_collision", "reorientation_surface", 0.00005),
    ("A_gripper_collision_*", "reorientation_surface", 0.0),
)

PLACEHOLDER_INSERTION_CONTACTS = (
    ("part_collision", "pcb_board*", 0.00001),
)

EXACT_INSERTION_CONTACTS = (
    ("part_collision", "insertion_collision*", 0.0),
)


def insertion_contacts(project: _ProjectLike) -> tuple[tuple, ...]:
    """Choose exact-fixture or bounded-placeholder insertion semantics."""
    if project.manifest["insertion"].get("collision_cad"):
        return EXACT_INSERTION_CONTACTS
    return PLACEHOLDER_INSERTION_CONTACTS


__all__ = [
    "EXACT_INSERTION_CONTACTS",
    "PLACEHOLDER_INSERTION_CONTACTS",
    "REORIENTATION_CONTACTS",
    "insertion_contacts",
]
