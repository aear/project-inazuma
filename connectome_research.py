"""Bounded, provenance-first connectome research for Ina.

Biological graphs are immutable reference witnesses.  This module deliberately
has no API that writes Ina's neural maps: it loads a bounded snapshot, derives
measurements, creates explicit copy mutations, and prepares an EEG overlay.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Iterable, Iterator, Mapping, Sequence


SCHEMA = "ina.connectome_reference/V1"
MAX_EDGES = 6_500_000
MAX_MUTATIONS = 2_048


DATASETS: dict[str, dict[str, Any]] = {
    "celegans": {
        "stage": "graph_primitives",
        "recommended_source": "OpenWorm Connectome Toolbox",
        "source_url": "https://github.com/openworm/ConnectomeToolbox",
        "access": "public",
        "default_scope": "complete derived graph",
        "raw_volume_policy": "not_needed",
    },
    "flywire_fafb_v783": {
        "stage": "large_scale_organisation",
        "recommended_source": "FlyWire Codex static snapshot",
        "source_url": "https://codex.flywire.ai/api/download?dataset=fafb",
        "access": "human_sign_in_required",
        "default_scope": "static edge and annotation CSV files",
        "raw_volume_policy": "forbidden_without_explicit_review",
    },
    "microns": {
        "stage": "structure_activity_comparison",
        "recommended_source": "MICrONS CAVE and DANDI",
        "source_url": "https://tutorial.microns-explorer.org/",
        "access": "public_bounded_query",
        "default_scope": "pinned cell cohort and activity window",
        "raw_volume_policy": "forbidden_without_explicit_review",
    },
}


@dataclass(frozen=True, order=True)
class Edge:
    source: str
    target: str
    weight: float = 1.0
    kind: str = "chemical"

    def __post_init__(self) -> None:
        if not self.source or not self.target:
            raise ValueError("edge endpoints are required")
        if not math.isfinite(float(self.weight)) or float(self.weight) < 0:
            raise ValueError("edge weight must be finite and non-negative")


@dataclass(frozen=True)
class ReferenceSnapshot:
    dataset: str
    version: str
    source_uri: str
    source_sha256: str
    edges: tuple[Edge, ...]
    derived: bool = False
    parent_sha256: str | None = None
    mutation_log: tuple[dict[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if self.dataset not in DATASETS:
            raise ValueError(f"unknown dataset: {self.dataset}")
        if not self.version or not self.source_uri or not self.source_sha256:
            raise ValueError("version, source_uri, and source_sha256 are required")
        if len(self.edges) > MAX_EDGES:
            raise ValueError(f"snapshot exceeds {MAX_EDGES} edges")

    @property
    def snapshot_id(self) -> str:
        content = json.dumps(
            {
                "dataset": self.dataset,
                "version": self.version,
                "source_sha256": self.source_sha256,
                "derived": self.derived,
                "parent_sha256": self.parent_sha256,
                "edges": [asdict(edge) for edge in self.edges],
                "mutation_log": self.mutation_log,
            }, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path | str, *, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _open_csv(path: Path):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8-sig", newline="")
    return path.open("r", encoding="utf-8-sig", newline="")


def load_edge_csv(
    path: Path | str, *, dataset: str, version: str, source_uri: str,
    source_column: str, target_column: str, weight_column: str | None = None,
    kind_column: str | None = None, default_kind: str = "chemical",
    max_edges: int = MAX_EDGES,
) -> ReferenceSnapshot:
    """Stream a CSV/CSV.GZ into a bounded, immutable reference snapshot."""
    file_path = Path(path)
    if not 1 <= int(max_edges) <= MAX_EDGES:
        raise ValueError(f"max_edges must be 1..{MAX_EDGES}")
    edges: list[Edge] = []
    with _open_csv(file_path) as handle:
        reader = csv.DictReader(handle)
        required = {source_column, target_column}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"missing required CSV columns: {sorted(required)}")
        for row_number, row in enumerate(reader, start=2):
            if len(edges) >= max_edges:
                raise ValueError(f"edge limit {max_edges} reached at CSV row {row_number}")
            try:
                weight = float(row[weight_column]) if weight_column else 1.0
                edges.append(Edge(
                    str(row[source_column]).strip(), str(row[target_column]).strip(), weight,
                    str(row.get(kind_column) or default_kind).strip() if kind_column else default_kind,
                ))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid edge at CSV row {row_number}: {exc}") from exc
    return ReferenceSnapshot(
        dataset=dataset, version=version, source_uri=source_uri,
        source_sha256=sha256_file(file_path), edges=tuple(edges),
    )


def graph_signals(snapshot: ReferenceSnapshot, *, top_n: int = 12) -> dict[str, Any]:
    """Return independent, inexpensive graph witnesses without third parties."""
    if not 1 <= int(top_n) <= 100:
        raise ValueError("top_n must be 1..100")
    outgoing: Counter[str] = Counter()
    incoming: Counter[str] = Counter()
    weighted: Counter[str] = Counter()
    neighbours: dict[str, set[str]] = defaultdict(set)
    pairs: set[tuple[str, str]] = set()
    nodes: set[str] = set()
    self_edges = 0
    total_weight = 0.0
    kinds: Counter[str] = Counter()
    for edge in snapshot.edges:
        nodes.update((edge.source, edge.target))
        outgoing[edge.source] += 1
        incoming[edge.target] += 1
        weighted[edge.source] += edge.weight
        weighted[edge.target] += edge.weight
        neighbours[edge.source].add(edge.target)
        neighbours[edge.target].add(edge.source)
        pairs.add((edge.source, edge.target))
        total_weight += edge.weight
        kinds[edge.kind] += 1
        self_edges += edge.source == edge.target
    reciprocal = sum(1 for source, target in pairs if source != target and (target, source) in pairs) // 2
    possible = len(nodes) * max(0, len(nodes) - 1)
    components = _component_sizes(nodes, neighbours)
    return {
        "schema": SCHEMA,
        "snapshot_id": snapshot.snapshot_id,
        "dataset": snapshot.dataset,
        "version": snapshot.version,
        "node_count": len(nodes),
        "edge_count": len(snapshot.edges),
        "total_weight": round(total_weight, 9),
        "directed_density": (len(pairs) - self_edges) / possible if possible else 0.0,
        "self_edges": self_edges,
        "reciprocal_pairs": reciprocal,
        "component_sizes": components[:top_n],
        "connection_kinds": dict(sorted(kinds.items())),
        "top_out_degree": outgoing.most_common(top_n),
        "top_in_degree": incoming.most_common(top_n),
        "top_weighted_hubs": weighted.most_common(top_n),
        "witnesses": ("directed_degree", "weighted_strength", "reciprocity", "components"),
    }


def _component_sizes(nodes: set[str], neighbours: Mapping[str, set[str]]) -> list[int]:
    remaining = set(nodes)
    sizes: list[int] = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        queue = deque((start,))
        size = 0
        while queue:
            current = queue.popleft()
            size += 1
            fresh = neighbours.get(current, set()) & remaining
            remaining.difference_update(fresh)
            queue.extend(sorted(fresh))
        sizes.append(size)
    return sorted(sizes, reverse=True)


def mutate_copy(
    snapshot: ReferenceSnapshot, operations: Sequence[Mapping[str, Any]],
    *, max_mutations: int = MAX_MUTATIONS,
) -> ReferenceSnapshot:
    """Apply bounded operations to a copy; the source snapshot remains untouched."""
    if snapshot.derived:
        raise ValueError("mutate from an immutable source snapshot, not a derived copy")
    if not 1 <= int(max_mutations) <= MAX_MUTATIONS or len(operations) > max_mutations:
        raise ValueError(f"mutations exceed bounded limit {min(max_mutations, MAX_MUTATIONS)}")
    edges = list(snapshot.edges)
    log: list[dict[str, Any]] = []
    for raw in operations:
        operation = str(raw.get("op") or "")
        if operation == "remove":
            index = int(raw["index"])
            removed = edges.pop(index)
            log.append({"op": "remove", "index": index, "edge": asdict(removed)})
        elif operation == "scale":
            index = int(raw["index"])
            factor = float(raw["factor"])
            if not math.isfinite(factor) or factor < 0 or factor > 10:
                raise ValueError("scale factor must be finite and in 0..10")
            old = edges[index]
            edges[index] = Edge(old.source, old.target, old.weight * factor, old.kind)
            log.append({"op": "scale", "index": index, "factor": factor})
        elif operation == "add":
            edge = Edge(str(raw["source"]), str(raw["target"]), float(raw.get("weight", 1.0)),
                        str(raw.get("kind", "experimental")))
            edges.append(edge)
            log.append({"op": "add", "edge": asdict(edge)})
        else:
            raise ValueError(f"unsupported mutation: {operation}")
    return ReferenceSnapshot(
        dataset=snapshot.dataset, version=f"{snapshot.version}+experimental-copy",
        source_uri=snapshot.source_uri, source_sha256=snapshot.source_sha256,
        edges=tuple(edges), derived=True, parent_sha256=snapshot.snapshot_id,
        mutation_log=tuple(log),
    )


def compare_signals(reference: ReferenceSnapshot, candidate: ReferenceSnapshot) -> dict[str, Any]:
    if not candidate.derived or candidate.parent_sha256 != reference.snapshot_id:
        raise ValueError("candidate must be a direct experimental copy of reference")
    before = graph_signals(reference)
    after = graph_signals(candidate)
    keys = ("node_count", "edge_count", "total_weight", "directed_density", "reciprocal_pairs")
    return {
        "schema": "ina.connectome_comparison/V1",
        "reference_snapshot_id": reference.snapshot_id,
        "candidate_snapshot_id": candidate.snapshot_id,
        "mutations": list(candidate.mutation_log),
        "signals": {key: {"before": before[key], "after": after[key],
                           "delta": after[key] - before[key]} for key in keys},
        "promotion_state": "review-required",
        "live_neural_map_modified": False,
    }


def eeg_overlay(snapshot: ReferenceSnapshot, *, max_nodes: int = 4_000,
                max_edges: int = 8_000, seed: int = 0) -> dict[str, Any]:
    """Prepare a visibly separate reference layer compatible with EEG rendering."""
    if not 1 <= max_nodes <= 20_000 or not 0 <= max_edges <= 100_000:
        raise ValueError("EEG overlay bounds exceeded")
    ranked = Counter()
    for edge in snapshot.edges:
        ranked[edge.source] += edge.weight
        ranked[edge.target] += edge.weight
    selected = {node for node, _weight in ranked.most_common(max_nodes)}
    rng = random.Random(f"{snapshot.snapshot_id}:{seed}")
    nodes = []
    for node in sorted(selected):
        longitude = rng.uniform(-math.pi, math.pi)
        latitude = math.asin(rng.uniform(-1.0, 1.0))
        nodes.append({
            "id": f"connectome:{snapshot.snapshot_id[:12]}:{node}", "label": node,
            "pos": (12 * math.cos(latitude) * math.cos(longitude),
                    9 * math.cos(latitude) * math.sin(longitude), 7 * math.sin(latitude)),
            "activation": min(1.0, math.log1p(ranked[node]) / 10.0),
            "network_type": "connectome_reference", "reference_only": True,
        })
    edges = []
    for edge in snapshot.edges:
        if len(edges) >= max_edges:
            break
        if edge.source in selected and edge.target in selected:
            edges.append({
                "source": f"connectome:{snapshot.snapshot_id[:12]}:{edge.source}",
                "target": f"connectome:{snapshot.snapshot_id[:12]}:{edge.target}",
                "weight": min(1.0, math.log1p(edge.weight) / 10.0),
                "network_type": "connectome_reference", "reference_only": True,
            })
    return {
        "schema": "ina.connectome_eeg_overlay/V1", "snapshot_id": snapshot.snapshot_id,
        "mode": "experimental_copy" if snapshot.derived else "immutable_reference",
        "nodes": nodes, "edges": edges, "live_neural_map_modified": False,
    }


__all__ = [
    "DATASETS", "Edge", "ReferenceSnapshot", "compare_signals", "eeg_overlay",
    "graph_signals", "load_edge_csv", "mutate_copy", "sha256_file",
]
