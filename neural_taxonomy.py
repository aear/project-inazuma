"""Shared neural type/network taxonomy for EEG and operational views."""
from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping

NEURAL_TYPES = ("sound", "token", "word", "symbol", "logic", "memory")
NEURAL_NETWORKS = ("emotion", "memory_graph", "meaning_map", "audio", "vision", "prediction", "logic", "instinct")


def normalize_node_type(node: Mapping[str, Any], network: str = "memory_graph") -> str:
    explicit = str(node.get("type") or node.get("node_type") or "").strip().casefold()
    if explicit:
        return explicit
    normalized_network = str(node.get("network_type") or network or "memory_graph").casefold()
    if normalized_network == "logic":
        return "logic"
    if normalized_network == "memory_graph":
        return "memory"
    return normalized_network


def count_node_types(nodes: Any, network: str = "memory_graph") -> Counter[str]:
    if isinstance(nodes, Mapping):
        iterable: Iterable[Any] = nodes.values()
    elif isinstance(nodes, (list, tuple)):
        iterable = nodes
    else:
        iterable = ()
    return Counter(normalize_node_type(node, network) for node in iterable if isinstance(node, Mapping))


__all__ = ["NEURAL_NETWORKS", "NEURAL_TYPES", "count_node_types", "normalize_node_type"]
