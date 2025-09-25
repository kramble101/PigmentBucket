"""Resolve Classic palette utilities for the mock analysis service."""
from __future__ import annotations

from typing import Iterator, List, Tuple

# Classic Resolve palette order.
RESOLVE_CLASSIC_COLORS: List[str] = [
    "Yellow",
    "Cyan",
    "Green",
    "Pink",
    "Blue",
    "Orange",
    "Purple",
    "Tan",
    "Red",
    "Fuchsia",
    "Lime",
    "Teal",
    "Lavender",
    "Brown",
    "Sky",
    "Mint",
]


def cycling_palette() -> Iterator[Tuple[int, str]]:
    """Yield endless pairs of palette index and Resolve color name."""
    while True:
        for index, color in enumerate(RESOLVE_CLASSIC_COLORS):
            yield index, color

