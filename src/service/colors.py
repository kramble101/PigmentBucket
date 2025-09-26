"""Color assignment utilities for Pigment Bucket clusters."""
from __future__ import annotations

from typing import Dict, Iterable, List, Mapping

_STABLE_PALETTE: List[str] = [
    "Cyan",
    "Yellow",
    "Pink",
    "Green",
    "Blue",
    "Orange",
    "Purple",
    "Lime",
    "Red",
    "Teal",
    "Fuchsia",
    "Sky",
]


def _compute_luma(centroid: Mapping[str, float]) -> float:
    r = float(centroid.get("r", centroid.get("red", 0.0)))
    g = float(centroid.get("g", centroid.get("green", 0.0)))
    b = float(centroid.get("b", centroid.get("blue", 0.0)))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _normalize_cluster_id(raw_id) -> int:
    if isinstance(raw_id, int):
        return raw_id
    if isinstance(raw_id, float):
        return int(raw_id)
    try:
        return int(str(raw_id))
    except (TypeError, ValueError):
        raise ValueError(f"Unsupported cluster id: {raw_id!r}")


def assign_cluster_colors(clusters: Iterable[Mapping[str, object]]) -> Dict[int, str]:
    """Deterministically map cluster identifiers to Resolve clip colors."""

    cluster_entries = []
    for cluster in clusters:
        if cluster is None:
            continue
        cid = _normalize_cluster_id(cluster.get("id"))
        centroid = cluster.get("centroid") or {}
        luma = _compute_luma(centroid if isinstance(centroid, Mapping) else {})
        cluster_entries.append((cid, luma))

    if not cluster_entries:
        return {}

    cluster_entries.sort(key=lambda item: (item[1], item[0]))

    palette_size = len(_STABLE_PALETTE)
    mapping: Dict[int, str] = {}
    for index, (cid, _luma) in enumerate(cluster_entries):
        mapping[cid] = _STABLE_PALETTE[index % palette_size]
    return mapping


__all__ = ["assign_cluster_colors"]
