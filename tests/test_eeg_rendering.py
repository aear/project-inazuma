from eeg_rendering import buffer_bytes, estimated_gl_vertices, pack_edges, pack_nodes


def _color(_item):
    return (0.25, 0.5, 0.75)


def test_packers_retain_every_valid_neuron_and_synapse():
    nodes = [
        {"id": "a", "pos": (1, 2, 3), "activation": 0.0},
        {"id": "b", "pos": (4, 5, 6), "activation": 1.0},
        {"id": "c", "pos": (7, 8, 9), "activation": 0.5},
    ]
    positions = {node["id"]: node["pos"] for node in nodes}
    edges = [
        {"source": "a", "target": "b", "weight": 0.5},
        {"source": "b", "target": "c", "weight": 1.0},
    ]

    packed_nodes = pack_nodes(nodes, _color)
    packed_edges = pack_edges(edges, positions, _color)

    assert packed_nodes["count"] == len(nodes)
    assert packed_edges["count"] == len(edges)
    assert len(packed_nodes["positions"]) == len(nodes) * 3
    assert len(packed_edges["positions"]) == len(edges) * 2 * 3
    assert buffer_bytes(packed_nodes, packed_edges) > 0


def test_pack_edges_omits_only_edges_with_missing_endpoints():
    packed = pack_edges(
        [
            {"source": "a", "target": "b"},
            {"source": "a", "target": "missing"},
        ],
        {"a": (0, 0, 0), "b": (1, 1, 1)},
        _color,
    )
    assert packed["count"] == 1


def test_profiles_report_quality_glow_overhead_without_dropping_data():
    balanced = estimated_gl_vertices(100, 500, "Balanced")
    throughput = estimated_gl_vertices(100, 500, "Throughput")
    quality = estimated_gl_vertices(100, 500, "Quality")
    assert balanced == throughput == 1100
    assert quality == 1200
