import csv

import pytest

from connectome_research import (
    Edge, ReferenceSnapshot, compare_signals, eeg_overlay, graph_signals,
    load_edge_csv, mutate_copy,
)


def _snapshot():
    return ReferenceSnapshot(
        dataset="celegans", version="fixture-v1", source_uri="fixture://worm",
        source_sha256="a" * 64,
        edges=(Edge("sensory", "inter", 3), Edge("inter", "motor", 2),
               Edge("motor", "inter", 1, "gap")),
    )


def test_csv_is_streamed_hashed_and_bounded(tmp_path):
    path = tmp_path / "edges.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("pre", "post", "weight", "kind"))
        writer.writeheader()
        writer.writerow({"pre": "A", "post": "B", "weight": 2, "kind": "chemical"})
        writer.writerow({"pre": "B", "post": "C", "weight": 1, "kind": "chemical"})
    snapshot = load_edge_csv(
        path, dataset="celegans", version="test", source_uri="fixture://csv",
        source_column="pre", target_column="post", weight_column="weight", kind_column="kind",
    )
    assert snapshot.edges == (Edge("A", "B", 2.0, "chemical"), Edge("B", "C", 1.0, "chemical"))
    assert len(snapshot.source_sha256) == 64
    with pytest.raises(ValueError, match="edge limit"):
        load_edge_csv(path, dataset="celegans", version="test", source_uri="fixture://csv",
                      source_column="pre", target_column="post", max_edges=1)


def test_graph_signals_are_independent_and_inspectable():
    result = graph_signals(_snapshot())
    assert result["node_count"] == 3
    assert result["edge_count"] == 3
    assert result["reciprocal_pairs"] == 1
    assert result["component_sizes"] == [3]
    assert len(result["witnesses"]) == 4


def test_mutations_only_touch_a_derived_copy_and_comparison_requires_parent():
    source = _snapshot()
    candidate = mutate_copy(source, [
        {"op": "scale", "index": 0, "factor": 0.5},
        {"op": "add", "source": "sensory", "target": "motor", "weight": 1},
    ])
    comparison = compare_signals(source, candidate)
    assert source.edges[0].weight == 3
    assert candidate.derived is True
    assert candidate.parent_sha256 == source.snapshot_id
    assert comparison["live_neural_map_modified"] is False
    assert comparison["promotion_state"] == "review-required"
    with pytest.raises(ValueError, match="immutable source"):
        mutate_copy(candidate, [{"op": "remove", "index": 0}])


def test_eeg_overlay_is_bounded_deterministic_and_visibly_reference_only():
    first = eeg_overlay(_snapshot(), max_nodes=2, max_edges=1, seed=4)
    second = eeg_overlay(_snapshot(), max_nodes=2, max_edges=1, seed=4)
    assert first == second
    assert len(first["nodes"]) == 2 and len(first["edges"]) <= 1
    assert all(node["reference_only"] for node in first["nodes"])
    assert first["mode"] == "immutable_reference"
    assert first["live_neural_map_modified"] is False
