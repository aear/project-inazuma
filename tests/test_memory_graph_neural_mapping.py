import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import memory_graph as mg


def test_neural_policy_includes_memory_caps():
    cfg = {
        "neural_map_policy": {
            "max_fragments_per_neuron": 9,
            "max_tags_per_neuron": 5,
            "vector_round_digits": 4,
            "position_round_digits": 3,
            "edge_direction_enabled": False,
            "gc_every_batches": 2,
            "max_neurons_total": 123,
            "max_edges_per_neuron": 7,
        }
    }
    policy = mg._neural_policy(cfg)
    assert policy["max_fragments_per_neuron"] == 9
    assert policy["max_tags_per_neuron"] == 5
    assert policy["vector_round_digits"] == 4
    assert policy["position_round_digits"] == 3
    assert policy["edge_direction_enabled"] is False
    assert policy["gc_every_batches"] == 2
    assert policy["max_neurons_total"] == 123
    assert policy["max_edges_per_neuron"] == 7


def test_update_neuron_from_candidate_applies_caps_and_rounding():
    neuron = {
        "fragments": ["a", "b", "c", "d"],
        "vector": [0.123456, 0.654321],
        "position": [0.0, 0.0, 0.0],
        "tags": ["t0", "t1"],
        "region": "head",
    }
    candidate = {
        "fragments": ["d", "e", "f"],
        "vector": [1.0, 1.0],
        "position": [1.23456, 2.34567, 3.45678],
        "tags": ["t2", "t3", "t4", "t5"],
        "region": "head",
    }
    policy = {
        "position_blend": 0.5,
        "max_fragments_per_neuron": 4,
        "max_tags_per_neuron": 3,
        "vector_round_digits": 3,
        "position_round_digits": 2,
    }

    mg._update_neuron_from_candidate(neuron, candidate, policy)

    assert neuron["fragments"] == ["c", "d", "e", "f"]
    assert len(neuron["tags"]) == 3
    assert neuron["vector"] == [0.562, 0.827]
    assert neuron["position"] == [0.62, 1.17, 1.73]


def test_apply_neuron_caps_compacts_existing_payload():
    neurons = [
        {
            "id": "node_0001",
            "fragments": ["a", "b", "c", "d", "e"],
            "tags": ["z", "x", "x", "y"],
            "vector": [0.1234567, 0.9876543],
            "position": [1.23456, 2.34567, 3.45678],
        }
    ]
    policy = {
        "max_fragments_per_neuron": 3,
        "max_tags_per_neuron": 2,
        "vector_round_digits": 4,
        "position_round_digits": 2,
    }
    changes = mg._apply_neuron_caps(neurons, policy)
    assert changes > 0
    assert neurons[0]["fragments"] == ["c", "d", "e"]
    assert neurons[0]["tags"] == ["y", "z"]
    assert neurons[0]["vector"] == [0.1235, 0.9877]
    assert neurons[0]["position"] == [1.23, 2.35, 3.46]


def test_build_synaptic_links_direction_toggle():
    neurons = [
        {"id": "n1", "vector": [1.0, 0.0], "position": [0.0, 0.0, 0.0]},
        {"id": "n2", "vector": [1.0, 0.0], "position": [1.0, 0.0, 0.0]},
        {"id": "n3", "vector": [1.0, 0.0], "position": [2.0, 0.0, 0.0]},
    ]
    with_direction = mg.build_synaptic_links(neurons, threshold=0.1, include_direction=True)
    without_direction = mg.build_synaptic_links(neurons, threshold=0.1, include_direction=False, max_edges_per_neuron=1)

    assert with_direction
    assert "direction" in with_direction[0]
    assert without_direction
    assert "direction" not in without_direction[0]
    assert len(without_direction) == 2


def test_enforce_neuron_budget_drops_low_priority_nodes():
    neurons = [
        {"id": "n1", "fragments": ["a"], "last_used": "2024-01-01T00:00:00+00:00"},
        {"id": "n2", "fragments": ["a", "b", "c"], "last_used": "2024-01-02T00:00:00+00:00"},
        {"id": "n3", "fragments": ["a", "b"], "last_used": "2024-01-03T00:00:00+00:00"},
    ]
    dropped = mg._enforce_neuron_budget(neurons, 2)
    assert dropped == 1
    remaining = {node["id"] for node in neurons}
    assert remaining == {"n2", "n3"}


def test_custom_runtime_respects_cooldown(monkeypatch, tmp_path):
    usage_path = tmp_path / "custom_transformer_usage.json"
    usage_path.parent.mkdir(parents=True, exist_ok=True)
    usage_path.write_text(json.dumps({"last_run": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())}))

    monkeypatch.setattr(mg, "_custom_transformer_usage_path", lambda child: usage_path)
    runtime = mg._NeuralCustomTransformerRuntime(
        "tester",
        {"custom_transformer_runtime": {"enabled": True, "cooldown_seconds": 3600, "sample_limit": 2}},
    )
    runtime.observe([{"id": "frag_1", "tags": ["alpha"], "summary": "test"}])

    result = runtime.run()
    assert result["status"] == "cooldown"
