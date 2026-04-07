import json
import os
import sys
import tempfile
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import memory_graph as mg
from self_reflection_core import SelfReflectionCore


def test_neural_policy_includes_memory_caps():
    cfg = {
        "neural_map_policy": {
            "max_fragments_per_neuron": 9,
            "max_tags_per_neuron": 5,
            "vector_round_digits": 4,
            "position_round_digits": 3,
            "fragment_anchor_count": 3,
            "tag_anchor_count": 2,
            "edge_direction_enabled": False,
            "max_connection_degree": 4,
            "min_synapse_weight": 0.55,
            "gc_every_batches": 2,
            "compact_save_enabled": True,
            "synapse_spool_enabled": True,
            "spill_to_disk_enabled": True,
            "max_hot_neurons": 41,
            "spill_after_batches": 3,
            "spill_precision_mode": "adaptive",
            "spill_high_precision_threshold": 0.7,
            "spill_medium_precision_threshold": 0.4,
            "max_neurons_total": 123,
            "max_edges_per_neuron": 7,
            "max_synapses_total": 19,
            "max_pending_dirty_fragments": 11,
        }
    }
    policy = mg._neural_policy(cfg)
    assert policy["max_fragments_per_neuron"] == 9
    assert policy["max_tags_per_neuron"] == 5
    assert policy["vector_round_digits"] == 4
    assert policy["position_round_digits"] == 3
    assert policy["fragment_anchor_count"] == 3
    assert policy["tag_anchor_count"] == 2
    assert policy["edge_direction_enabled"] is False
    assert policy["max_connection_degree"] == 4
    assert policy["min_synapse_weight"] == 0.55
    assert policy["gc_every_batches"] == 2
    assert policy["compact_save_enabled"] is True
    assert policy["synapse_spool_enabled"] is True
    assert policy["spill_to_disk_enabled"] is True
    assert policy["max_hot_neurons"] == 41
    assert policy["spill_after_batches"] == 3
    assert policy["spill_precision_mode"] == "adaptive"
    assert policy["spill_high_precision_threshold"] == 0.7
    assert policy["spill_medium_precision_threshold"] == 0.4
    assert policy["max_neurons_total"] == 123
    assert policy["max_edges_per_neuron"] == 7
    assert policy["max_synapses_total"] == 19
    assert policy["max_pending_dirty_fragments"] == 11


def test_memory_graph_phase_args_support_boot_and_neural_modes():
    assert mg._memory_graph_phase_args([]).phase == "full"
    assert mg._memory_graph_phase_args(["--phase", "boot"]).phase == "boot"
    assert mg._memory_graph_phase_args(["--phase", "neural"]).phase == "neural"



def test_set_memory_graph_deferred_build_updates_inastate():
    child = "TestMemoryGraphPhaseState"
    shutil.rmtree(Path("AI_Children") / child, ignore_errors=True)
    try:
        running = mg._set_memory_graph_deferred_build(child, "running", pid=321, requested_by="test")
        assert running["status"] == "running"
        assert running["pid"] == 321

        queued = mg._set_memory_graph_deferred_build(child, "queued", requested_by="test")
        assert queued["status"] == "queued"
        assert "pid" not in queued

        state = mg._load_inastate(child)
        deferred = state.get("memory_graph_deferred_build")
        assert isinstance(deferred, dict)
        assert deferred.get("status") == "queued"
        assert deferred.get("requested_by") == "test"
    finally:
        shutil.rmtree(Path("AI_Children") / child, ignore_errors=True)


def test_run_memory_neural_phase_requeues_when_builder_needs_resume():
    child = "TestMemoryGraphBurstQueue"
    shutil.rmtree(Path("AI_Children") / child, ignore_errors=True)

    class DummyManager:
        def __init__(self):
            self.memory_map = {}
            self.reindexed = False

        def fast_reindex(self, **kwargs):
            self.reindexed = True

    original_build = mg.build_fractal_memory
    original_experience = mg.build_experience_graph
    try:
        mg.build_fractal_memory = lambda current_child: {"needs_resume": True, "selector_remaining": 5}
        mg.build_experience_graph = lambda current_child, base_path=None: (_ for _ in ()).throw(AssertionError("experience graph should not run"))
        mgr = DummyManager()
        mg._run_memory_neural_phase(child, mgr, launch_source="test")
        state = mg._load_inastate(child).get("memory_graph_deferred_build")
        assert mgr.reindexed is True
        assert isinstance(state, dict)
        assert state.get("status") == "queued"
        assert state.get("last_cycle", {}).get("needs_resume") is True
    finally:
        mg.build_fractal_memory = original_build
        mg.build_experience_graph = original_experience
        shutil.rmtree(Path("AI_Children") / child, ignore_errors=True)




def test_clip_sequence_with_anchors_preserves_head_and_tail():
    values = ["a", "b", "c", "d", "e", "f"]
    assert mg._clip_sequence_with_anchors(values, 4, 2) == ["a", "b", "e", "f"]


def test_apply_synapse_connection_caps_limits_hubs():
    synapses = [
        {"source": "n1", "target": "n2", "weight": 0.99},
        {"source": "n1", "target": "n3", "weight": 0.98},
        {"source": "n1", "target": "n4", "weight": 0.97},
        {"source": "n2", "target": "n3", "weight": 0.96},
    ]
    kept, stats = mg._apply_synapse_connection_caps(
        synapses,
        {"max_connection_degree": 2, "min_synapse_weight": 0.0},
    )

    assert [(mg._synapse_get(s, "source"), mg._synapse_get(s, "target")) for s in kept] == [("n1", "n2"), ("n1", "n3"), ("n2", "n3")]
    assert stats["degree_pruned"] == 1
    assert stats["weight_pruned"] == 0


def test_memory_retention_adjusted_tier_protects_high_importance_and_compacts_low_importance():
    retention = {
        "protect_cold_importance": 0.7,
        "compact_low_importance_threshold": 0.2,
        "compact_low_importance_age_hours": 168,
    }

    assert mg.MemoryManager._retention_adjusted_tier("cold", 0.85, 400.0, retention) == "long"
    assert mg.MemoryManager._retention_adjusted_tier("long", 0.1, 200.0, retention) == "cold"
    assert mg.MemoryManager._retention_adjusted_tier("working", 0.1, 200.0, retention) == "working"


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


def test_enforce_synapse_budget_keeps_strongest_edges():
    synapses = [
        {"source": "n1", "target": "n2", "weight": 0.4},
        {"source": "n1", "target": "n3", "weight": 0.9},
        {"source": "n2", "target": "n3", "weight": 0.7},
    ]

    dropped = mg._enforce_synapse_budget(synapses, 2)

    assert dropped == 1
    assert synapses == [
        {"source": "n1", "target": "n3", "weight": 0.9},
        {"source": "n2", "target": "n3", "weight": 0.7},
    ]


def test_trim_pending_fragment_ids_applies_hard_limit():
    index = {
        "f1": {"timestamp": "2024-01-01T00:00:00+00:00", "importance": 0.1},
        "f2": {"timestamp": "2024-01-03T00:00:00+00:00", "importance": 0.2},
        "f3": {"timestamp": "2024-01-02T00:00:00+00:00", "importance": 0.9},
    }

    pending, dropped = mg._trim_pending_fragment_ids(["f1", "f2", "f2", "f3"], index, 2)

    assert dropped == 1
    assert pending == ["f2", "f3"]


def test_compact_graph_for_storage_drops_default_fields():
    neurons = [{
        "id": "n1",
        "fragments": ["a", "b"],
        "vector": [0.123456, 0.987654],
        "position": [1.23456, 2.34567, 3.45678],
        "region": "head",
        "network_type": "memory_graph",
        "symbolic_density": 0.0,
        "tags": ["beta", "alpha"],
        "activation_history": [],
        "last_used": "2024-01-01T00:00:00+00:00",
    }]
    synapses = [{
        "source": "n1",
        "target": "n2",
        "weight": 0.98765,
        "network_type": "memory_graph",
    }]
    policy = {
        "max_fragments_per_neuron": 8,
        "max_tags_per_neuron": 8,
        "vector_round_digits": 4,
        "position_round_digits": 2,
    }

    compact_neurons, compact_synapses = mg._compact_graph_for_storage(neurons, synapses, policy)

    assert compact_neurons == [{
        "id": "n1",
        "fragments": ["a", "b"],
        "vector": [0.1235, 0.9877],
        "position": [1.23, 2.35, 3.46],
        "region": "head",
        "tags": ["beta", "alpha"],
        "last_used": "2024-01-01T00:00:00+00:00",
    }]
    assert compact_synapses == [{"source": "n1", "target": "n2", "weight": 0.9877}]


def test_cluster_fragments_uses_compact_centroids():
    fragments = [
        {"id": "f1", "tags": ["a"]},
        {"id": "f2", "tags": ["a"]},
    ]
    cache = {"f1": [1.0, 0.0], "f2": [1.0, 0.0]}

    clusters = mg.cluster_fragments(fragments, cache, threshold=0.5, tag_weight=0.0)

    assert len(clusters) == 1
    assert "centroid" in clusters[0]
    assert "vector_sum" not in clusters[0]


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


def test_spill_store_spills_and_reactivates():
    original = mg._neural_spill_path
    with tempfile.TemporaryDirectory() as tmpdir:
        spill_path = Path(tmpdir) / "spill.sqlite"
        mg._neural_spill_path = lambda child: spill_path
        try:
            store = mg._NeuralSpillStore(
                "tester",
                {"spill_to_disk_enabled": True, "max_hot_neurons": 1, "spill_after_batches": 1},
            )
            neurons = [
                {
                    "id": "node_0001",
                    "fragments": ["a", "b"],
                    "vector": [1.0, 0.0],
                    "tags": ["alpha"],
                    "region": "head",
                    "last_used": "2024-01-01T00:00:00+00:00",
                },
                {
                    "id": "node_0002",
                    "fragments": ["c"],
                    "vector": [0.0, 1.0],
                    "tags": ["beta"],
                    "region": "torso",
                    "last_used": "2024-01-02T00:00:00+00:00",
                },
            ]

            spilled = store.spill_overflow(neurons)

            assert spilled == 1
            assert len(neurons) == 1
            assert neurons[0]["id"] == "node_0002"
            assert store.count() == 1

            spill_id, spill_score = store.best_match(
                {"vector": [1.0, 0.0], "tags": ["alpha"], "region": "head", "fragments": ["x"]},
                tag_weight=0.0,
            )

            assert spill_id == "node_0001"
            assert spill_score > 0.99

            activated = store.activate_node(spill_id, neurons)

            assert activated is not None
            assert activated["id"] == "node_0001"
            assert len(neurons) == 2
            assert store.count() == 0

            store.cleanup()
            assert not spill_path.exists()
        finally:
            mg._neural_spill_path = original


def test_enforce_total_neuron_budget_prunes_spilled_nodes_first():
    original = mg._neural_spill_path
    with tempfile.TemporaryDirectory() as tmpdir:
        spill_path = Path(tmpdir) / "spill.sqlite"
        mg._neural_spill_path = lambda child: spill_path
        try:
            store = mg._NeuralSpillStore(
                "tester",
                {"spill_to_disk_enabled": True, "max_hot_neurons": 2, "spill_after_batches": 1},
            )
            hot_neurons = [
                {
                    "id": "node_0003",
                    "fragments": ["h1", "h2", "h3"],
                    "vector": [0.0, 1.0],
                    "tags": ["hot"],
                    "region": "torso",
                    "last_used": "2024-01-04T00:00:00+00:00",
                },
                {
                    "id": "node_0004",
                    "fragments": ["h4", "h5"],
                    "vector": [0.0, 1.0],
                    "tags": ["hot"],
                    "region": "torso",
                    "last_used": "2024-01-05T00:00:00+00:00",
                },
            ]
            store._store_neurons([
                {
                    "id": "node_0001",
                    "fragments": ["s1"],
                    "vector": [1.0, 0.0],
                    "tags": ["spill"],
                    "region": "head",
                    "last_used": "2024-01-01T00:00:00+00:00",
                },
                {
                    "id": "node_0002",
                    "fragments": ["s2"],
                    "vector": [1.0, 0.0],
                    "tags": ["spill"],
                    "region": "head",
                    "last_used": "2024-01-02T00:00:00+00:00",
                },
            ])

            dropped, removed_fragments = mg._enforce_total_neuron_budget(hot_neurons, 2, spill_store=store)

            assert dropped == 2
            assert {node["id"] for node in hot_neurons} == {"node_0003", "node_0004"}
            assert store.count() == 0
            assert removed_fragments == {"s1", "s2"}
        finally:
            mg._neural_spill_path = original


def test_merge_candidates_can_match_spilled_neuron():
    original = mg._neural_spill_path
    with tempfile.TemporaryDirectory() as tmpdir:
        spill_path = Path(tmpdir) / "spill.sqlite"
        mg._neural_spill_path = lambda child: spill_path
        try:
            store = mg._NeuralSpillStore(
                "tester",
                {"spill_to_disk_enabled": True, "max_hot_neurons": 1, "spill_after_batches": 1},
            )
            hot_neurons = [
                {
                    "id": "node_0002",
                    "fragments": ["b"],
                    "vector": [0.0, 1.0],
                    "tags": ["beta"],
                    "region": "torso",
                    "last_used": "2024-01-03T00:00:00+00:00",
                }
            ]
            store._store_neurons([
                {
                    "id": "node_0001",
                    "fragments": ["a"],
                    "vector": [1.0, 0.0],
                    "tags": ["alpha"],
                    "region": "head",
                    "last_used": "2024-01-01T00:00:00+00:00",
                }
            ])
            candidate = {
                "fragments": ["x"],
                "tags": ["alpha"],
                "vector": [1.0, 0.0],
                "region": "head",
                "position": [0.0, 0.0, 0.0],
            }
            policy = {
                "merge_slack": 0.0,
                "max_new_neurons": 1,
                "position_blend": 0.5,
                "max_fragments_per_neuron": 8,
                "max_tags_per_neuron": 8,
                "vector_round_digits": 4,
                "position_round_digits": 2,
            }

            merged, created, skipped = mg._merge_candidates_into_neurons(
                hot_neurons,
                [candidate],
                0.75,
                0.0,
                policy,
                spill_store=store,
            )

            assert merged == 1
            assert created == 0
            assert skipped == 0
            assert store.count() == 0
            matched = next(node for node in hot_neurons if node["id"] == "node_0001")
            assert matched["fragments"] == ["a", "x"]
        finally:
            mg._neural_spill_path = original

def test_encode_spilled_neuron_adapts_precision_by_importance():
    policy = {
        "spill_precision_mode": "adaptive",
        "spill_high_precision_threshold": 0.68,
        "spill_medium_precision_threshold": 0.35,
        "vector_round_digits": 4,
        "position_round_digits": 3,
    }

    low_importance = {
        "id": "node_low",
        "fragments": ["a"],
        "vector": [0.123456, -0.234567],
        "position": [1.1111, 2.2222, 3.3333],
        "tags": ["ambient"],
        "symbolic_density": 0.02,
        "activation_history": [],
    }
    high_importance = {
        "id": "node_high",
        "fragments": ["a", "b", "c", "d"],
        "vector": [0.123456, -0.234567],
        "position": [1.1111, 2.2222, 3.3333],
        "tags": ["care", "memory"],
        "symbolic_density": 0.9,
        "activation_history": [0.95],
        "spill_reuse_count": 3,
    }

    encoded_low = mg._encode_spilled_neuron(low_importance, policy)
    encoded_high = mg._encode_spilled_neuron(high_importance, policy)

    assert encoded_low["spill_precision"] == "int8"
    assert "vector_q" in encoded_low
    assert "vector" not in encoded_low
    assert encoded_high["spill_precision"] == "float"
    assert encoded_high["vector"] == [0.1235, -0.2346]
    assert encoded_high["position"] == [1.111, 2.222, 3.333]

    decoded_low = mg._decode_spilled_neuron(encoded_low)
    assert "spill_precision" not in decoded_low
    assert len(decoded_low["vector"]) == 2
    assert abs(decoded_low["vector"][0] - low_importance["vector"][0]) < 0.01


def test_spill_store_iter_link_neurons_uses_dense_meta_without_db_reads():
    original = mg._neural_spill_path
    with tempfile.TemporaryDirectory() as tmpdir:
        spill_path = Path(tmpdir) / "spill.sqlite"
        mg._neural_spill_path = lambda child: spill_path
        try:
            store = mg._NeuralSpillStore(
                "tester",
                {"spill_to_disk_enabled": True, "max_hot_neurons": 1, "spill_after_batches": 1},
            )
            store._store_neurons([
                {
                    "id": "node_0001",
                    "fragments": ["a"],
                    "vector": [1.0, 0.5],
                    "position": [0.0, 1.0, 2.0],
                    "tags": ["alpha"],
                    "region": "head",
                    "last_used": "2024-01-01T00:00:00+00:00",
                }
            ])

            conn = store.conn
            store.conn = None
            if conn is not None:
                conn.close()

            link_nodes = list(store.iter_link_neurons())

            assert len(link_nodes) == 1
            assert link_nodes[0]["id"] == "node_0001"
            assert getattr(link_nodes[0]["vector"], "typecode", None) == "f"
            assert list(link_nodes[0]["position"]) == [0.0, 1.0, 2.0]

            store.cleanup()
        finally:
            mg._neural_spill_path = original


def test_save_neural_map_streaming_writes_hot_and_spilled_neurons():
    original_map = mg._neural_map_path
    original_spill = mg._neural_spill_path
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        map_path = tmpdir_path / "neural_memory_map.json"
        spill_path = tmpdir_path / "spill.sqlite"
        mg._neural_map_path = lambda child: map_path
        mg._neural_spill_path = lambda child: spill_path
        try:
            store = mg._NeuralSpillStore(
                "tester",
                {
                    "spill_to_disk_enabled": True,
                    "max_hot_neurons": 1,
                    "spill_after_batches": 1,
                    "vector_round_digits": 4,
                    "position_round_digits": 3,
                    "compact_save_enabled": True,
                },
            )
            hot_neurons = [
                {
                    "id": "node_0002",
                    "fragments": ["b"],
                    "vector": [0.0, 1.0],
                    "position": [1.0, 2.0, 3.0],
                    "tags": ["beta"],
                    "region": "torso",
                    "last_used": "2024-01-03T00:00:00+00:00",
                }
            ]
            store._store_neurons([
                {
                    "id": "node_0001",
                    "fragments": ["a"],
                    "vector": [1.0, 0.0],
                    "position": [0.0, 0.0, 0.0],
                    "tags": ["alpha"],
                    "region": "head",
                    "last_used": "2024-01-01T00:00:00+00:00",
                }
            ])
            synapses = [{"source": "node_0001", "target": "node_0002", "weight": 0.98765}]

            mg._save_neural_map_streaming(
                "tester",
                {"converted_from_legacy": False, "updated_at": "2026-04-01T00:00:00+00:00"},
                hot_neurons,
                synapses,
                {
                    "compact_save_enabled": True,
                    "max_fragments_per_neuron": 8,
                    "max_tags_per_neuron": 8,
                    "vector_round_digits": 4,
                    "position_round_digits": 3,
                },
                store,
            )

            payload = json.loads(map_path.read_text())

            assert sorted(node["id"] for node in payload["neurons"]) == ["node_0001", "node_0002"]
            assert payload["synapses"] == [{"source": "node_0001", "target": "node_0002", "weight": 0.9877}]

            store.cleanup()
        finally:
            mg._neural_map_path = original_map
            mg._neural_spill_path = original_spill

def test_build_synaptic_links_batched_keeps_strongest_edges():
    neurons = [
        {"id": "n1", "vector": [1.0, 0.0], "position": [0.0, 0.0, 0.0]},
        {"id": "n2", "vector": [0.95, 0.312249], "position": [1.0, 0.0, 0.0]},
        {"id": "n3", "vector": [0.5, 0.8660254], "position": [2.0, 0.0, 0.0]},
        {"id": "n4", "vector": [0.0, 1.0], "position": [3.0, 0.0, 0.0]},
    ]

    synapses, stats = mg.build_synaptic_links(
        neurons,
        threshold=0.1,
        max_edges=2,
        include_direction=False,
        batch_size=1,
        return_stats=True,
    )

    assert [(syn["source"], syn["target"]) for syn in synapses] == [("n1", "n2"), ("n3", "n4")]
    assert stats["edge_count"] == 2
    assert stats["budget_pruned"] >= 1
    assert stats["truncated"] is True


def test_save_neural_map_streaming_writes_non_spilled_graph():
    original_map = mg._neural_map_path
    with tempfile.TemporaryDirectory() as tmpdir:
        map_path = Path(tmpdir) / "neural_memory_map.json"
        mg._neural_map_path = lambda child: map_path
        try:
            hot_neurons = [
                {
                    "id": "node_0001",
                    "fragments": ["a"],
                    "vector": [1.0, 0.0],
                    "position": [0.0, 0.0, 0.0],
                    "tags": ["alpha"],
                    "region": "head",
                    "last_used": "2024-01-01T00:00:00+00:00",
                },
                {
                    "id": "node_0002",
                    "fragments": ["b"],
                    "vector": [0.0, 1.0],
                    "position": [1.0, 2.0, 3.0],
                    "tags": ["beta"],
                    "region": "torso",
                    "last_used": "2024-01-03T00:00:00+00:00",
                },
            ]
            synapses = [{"source": "node_0001", "target": "node_0002", "weight": 0.98765}]

            mg._save_neural_map_streaming(
                "tester",
                {"converted_from_legacy": False, "updated_at": "2026-04-01T00:00:00+00:00"},
                hot_neurons,
                synapses,
                {
                    "compact_save_enabled": True,
                    "max_fragments_per_neuron": 8,
                    "max_tags_per_neuron": 8,
                    "vector_round_digits": 4,
                    "position_round_digits": 3,
                },
                None,
            )

            payload = json.loads(map_path.read_text())

            assert sorted(node["id"] for node in payload["neurons"]) == ["node_0001", "node_0002"]
            assert payload["synapses"] == [{"source": "node_0001", "target": "node_0002", "weight": 0.9877}]
        finally:
            mg._neural_map_path = original_map

def test_save_sparse_snapshot_by_ids_streams_expected_csr_payload():
    original = mg._neural_snapshot_path
    with tempfile.TemporaryDirectory() as tmpdir:
        snapshot_path = Path(tmpdir) / "neural_memory_snapshot_csr.json"
        mg._neural_snapshot_path = lambda child: snapshot_path
        try:
            synapses = [
                {"source": "node_0002", "target": "node_0003", "weight": 0.75},
                {"source": "node_9999", "target": "node_0003", "weight": 0.2},
                {"source": "node_0001", "target": "node_0003", "weight": 0.4},
                {"source": "node_0001", "target": "node_0002", "weight": 0.91},
            ]

            mg._save_sparse_snapshot_by_ids(
                "tester",
                ["node_0001", "node_0002", "node_0003"],
                synapses,
            )

            payload = json.loads(snapshot_path.read_text())

            assert payload["format"] == "csr_v1"
            assert payload["node_ids"] == ["node_0001", "node_0002", "node_0003"]
            assert payload["indptr"] == [0, 2, 3, 3]
            assert payload["indices"] == [1, 2, 2]
            assert payload["weights"] == [0.91, 0.4, 0.75]
            assert payload["edge_count"] == 3
        finally:
            mg._neural_snapshot_path = original

def test_build_synaptic_links_compact_records_stream_through_save_and_snapshot():
    original_map = mg._neural_map_path
    original_snapshot = mg._neural_snapshot_path
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        map_path = tmpdir_path / "neural_memory_map.json"
        snapshot_path = tmpdir_path / "neural_memory_snapshot_csr.json"
        mg._neural_map_path = lambda child: map_path
        mg._neural_snapshot_path = lambda child: snapshot_path
        try:
            neurons = [
                {"id": "n1", "vector": [1.0, 0.0], "position": [0.0, 0.0, 0.0]},
                {"id": "n2", "vector": [1.0, 0.0], "position": [1.0, 0.0, 0.0]},
            ]
            synapses, stats = mg.build_synaptic_links(
                neurons,
                threshold=0.1,
                include_direction=False,
                compact_records=True,
                return_stats=True,
            )

            assert stats["edge_count"] == 1
            assert isinstance(synapses[0], mg._SynapseRecord)

            mg._save_neural_map_streaming(
                "tester",
                {"converted_from_legacy": False, "updated_at": "2026-04-02T00:00:00+00:00"},
                neurons,
                synapses,
                {
                    "compact_save_enabled": True,
                    "max_fragments_per_neuron": 8,
                    "max_tags_per_neuron": 8,
                    "vector_round_digits": 4,
                    "position_round_digits": 3,
                },
                None,
            )
            mg._save_sparse_snapshot_by_ids("tester", ["n1", "n2"], synapses)

            map_payload = json.loads(map_path.read_text())
            snapshot_payload = json.loads(snapshot_path.read_text())

            assert map_payload["synapses"] == [{"source": "n1", "target": "n2", "weight": 1.0}]
            assert snapshot_payload["indices"] == [1]
            assert snapshot_payload["weights"] == [1.0]
        finally:
            mg._neural_map_path = original_map
            mg._neural_snapshot_path = original_snapshot





def test_memory_index_store_reads_from_sqlite_sidecar():
    child = "TestMemoryIndexStore"
    root = Path("AI_Children") / child / "memory"
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "f1": {"tier": "short", "filename": "frag_1.json", "importance": 0.2, "tags": ["alpha"], "last_seen": "2026-04-01T00:00:00+00:00"},
        "f2": {"tier": "long", "filename": "frag_2.json", "importance": 0.1, "tags": ["beta"], "last_seen": "2026-04-02T00:00:00+00:00"},
    }
    json_path = root / "memory_map.json"
    json_path.write_text(json.dumps(payload))
    store = None
    try:
        store = mg._load_memory_index(child)
        assert len(store) == 2
        assert store.get("f1")["filename"] == "frag_1.json"
        assert set(store.keys()) == {"f1", "f2"}
    finally:
        if store is not None and hasattr(store, "close"):
            store.close()
        for target in [root / "memory_map.sqlite", root / "memory_map.sqlite-wal", root / "memory_map.sqlite-shm", json_path]:
            try:
                target.unlink()
            except OSError:
                pass
        for parent in [root, root.parent, root.parent.parent]:
            try:
                parent.rmdir()
            except OSError:
                pass


def test_select_pregraph_compaction_candidates_skips_anchors():
    child = "TestPregraphCompaction"
    base = Path("AI_Children") / child / "memory" / "fragments" / "long"
    base.mkdir(parents=True, exist_ok=True)
    try:
        old_ts = "2026-03-01T00:00:00+00:00"
        for idx in (1, 2):
            path_obj = base / f"frag_{idx}.json"
            path_obj.write_text(json.dumps({"id": f"f{idx}", "tags": [], "summary": "x" * 200, "emotions": {}}))
        index = {
            "f1": {"tier": "long", "filename": "frag_1.json", "importance": 0.1, "tags": ["anchor"], "last_seen": old_ts, "size_bytes": 120000},
            "f2": {"tier": "long", "filename": "frag_2.json", "importance": 0.1, "tags": ["ordinary"], "last_seen": old_ts, "size_bytes": 120000},
        }
        retention = {
            "protect_cold_importance": 0.72,
            "compact_low_importance_threshold": 0.18,
            "compact_low_importance_age_hours": 168,
            "pre_compact_enabled": True,
            "pre_compact_limit": 8,
            "pre_compact_min_size_bytes": 1024,
        }

        selected = mg._select_pregraph_compaction_candidates(child, index, retention)

        assert [frag_id for frag_id, _, _ in selected] == ["f2"]
    finally:
        for frag_path in base.glob("frag_*.json"):
            try:
                frag_path.unlink()
            except OSError:
                pass
        for parent in [base, base.parent, base.parent.parent, base.parent.parent.parent]:
            try:
                parent.rmdir()
            except OSError:
                pass


def test_human_memory_prune_limit_scales_for_large_index():
    retention = mg.DEFAULT_RETENTION_POLICY.copy()

    normal = mg._human_memory_prune_settings(retention, index_count=42)
    large = mg._human_memory_prune_settings(
        retention,
        index_count=retention["human_prune_large_index_threshold"],
    )

    assert normal["limit"] == retention["human_prune_limit"]
    assert large["limit"] == retention["human_prune_large_index_limit"]


def test_select_human_memory_prune_candidates_uses_indexed_human_bias():
    child = "TestHumanMemoryPruneCandidates"
    root = Path("AI_Children") / child / "memory"
    fragment_root = root / "fragments"
    base = fragment_root / "cold"
    shutil.rmtree(Path("AI_Children") / child, ignore_errors=True)
    base.mkdir(parents=True, exist_ok=True)
    store = None
    try:
        old_ts = "2026-03-01T00:00:00+00:00"
        recent_ts = "2026-04-01T12:00:00+00:00"
        entries = {
            "anchor": {"tier": "cold", "filename": "frag_anchor.json", "importance": 0.05, "tags": ["identity"], "last_seen": old_ts, "size_bytes": 4096},
            "ordinary": {"tier": "cold", "filename": "frag_ordinary.json", "importance": 0.05, "tags": ["ordinary"], "last_seen": old_ts, "size_bytes": 4096},
            "legacy": {"tier": "short", "filename": "frag_legacy.json", "importance": 0.04, "tags": ["ordinary"], "last_seen": old_ts, "size_bytes": 8192},
            "important": {"tier": "cold", "filename": "frag_important.json", "importance": 0.9, "tags": ["ordinary"], "last_seen": old_ts, "size_bytes": 4096},
            "recent": {"tier": "cold", "filename": "frag_recent.json", "importance": 0.05, "tags": ["ordinary"], "last_seen": recent_ts, "size_bytes": 4096},
        }
        for frag_id, meta in entries.items():
            payload = {"id": frag_id, "tags": meta["tags"], "importance": meta["importance"], "timestamp": meta["last_seen"], "summary": "memory"}
            target_dir = base if meta["tier"] == "cold" else fragment_root
            (target_dir / meta["filename"]).write_text(json.dumps(payload))
        (root / "memory_map.json").write_text(json.dumps(entries))
        store = mg._MemoryIndexStore(child)
        retention = mg.DEFAULT_RETENTION_POLICY.copy()
        retention.update({
            "human_prune_limit": 8,
            "human_prune_min_age_hours": 24.0 * 7.0,
            "human_prune_max_importance": 0.18,
            "human_prune_min_size_bytes": 1,
        })

        selected = mg._select_human_memory_prune_candidates(
            child,
            store,
            retention,
            index_count=len(store),
            now=datetime(2026, 4, 2, tzinfo=timezone.utc),
        )

        assert [frag_id for frag_id, _, _ in selected] == ["legacy", "ordinary"]
    finally:
        if store is not None:
            store.close()
        shutil.rmtree(Path("AI_Children") / child, ignore_errors=True)


def test_human_memory_prune_pass_compacts_low_value_cold_fragment():
    child = "TestHumanMemoryPrunePass"
    root = Path("AI_Children") / child / "memory"
    fragment_root = root / "fragments"
    cold_dir = fragment_root / "cold"
    shutil.rmtree(Path("AI_Children") / child, ignore_errors=True)
    fragment_root.mkdir(parents=True, exist_ok=True)
    try:
        old_ts = "2026-03-01T00:00:00+00:00"
        path_obj = fragment_root / "frag_low.json"
        payload = {
            "id": "low",
            "tags": ["ordinary"],
            "importance": 0.05,
            "timestamp": old_ts,
            "summary": "low value detail " * 128,
            "emotions": {"intensity": 0.1, "trust": 0.2},
        }
        path_obj.write_text(json.dumps(payload))
        st = path_obj.stat()
        retention = mg.DEFAULT_RETENTION_POLICY.copy()
        retention.update({
            "human_prune_limit": 4,
            "human_prune_min_age_hours": 24.0,
            "human_prune_max_importance": 0.18,
            "human_prune_min_size_bytes": 1,
            "human_prune_cooldown_seconds": 0.0,
        })
        manager = mg.MemoryManager(child=child, tier_policy={"retention": retention}, autoload=False)
        manager.memory_map = {
            "low": {
                "tier": "short",
                "filename": path_obj.name,
                "importance": 0.05,
                "tags": ["ordinary"],
                "last_seen": old_ts,
                "mtime_ns": int(st.st_mtime_ns),
                "size_bytes": int(st.st_size),
            }
        }
        manager._map_loaded = True
        manager.cold_storage_policy.update({
            "enabled": True,
            "auto_compact": True,
            "quarantine_days": 1,
            "purge_pending_delete": True,
            "retain_full_fragment": False,
        })
        manager.save_map()

        stats = manager.human_memory_prune_pass(force=True, now=datetime(2026, 4, 2, tzinfo=timezone.utc))
        stub_path = cold_dir / path_obj.name
        stub = json.loads(stub_path.read_text())
        pending_path = root / "fragments" / "pending_delete" / "cold" / path_obj.name

        assert stats["compacted"] == 1
        assert stats["moved_to_cold"] == 1
        assert isinstance(stub.get("cold_core"), dict)
        assert not path_obj.exists()
        assert pending_path.exists()
    finally:
        shutil.rmtree(Path("AI_Children") / child, ignore_errors=True)


def test_build_synaptic_links_spooled_stream_through_save_and_snapshot():
    original_map = mg._neural_map_path
    original_snapshot = mg._neural_snapshot_path
    synapse_store = None
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        map_path = tmpdir_path / "neural_memory_map.json"
        snapshot_path = tmpdir_path / "neural_memory_snapshot_csr.json"
        spool_path = tmpdir_path / "neural_synapse_spool.sqlite"
        mg._neural_map_path = lambda child: map_path
        mg._neural_snapshot_path = lambda child: snapshot_path
        try:
            neurons = [
                {"id": "n1", "vector": [1.0, 0.0], "position": [0.0, 0.0, 0.0]},
                {"id": "n2", "vector": [1.0, 0.0], "position": [1.0, 0.0, 0.0]},
                {"id": "n3", "vector": [0.99, 0.01], "position": [2.0, 0.0, 0.0]},
            ]
            synapse_store, stats = mg.build_synaptic_links(
                neurons,
                threshold=0.1,
                include_direction=False,
                compact_records=True,
                max_edges=2,
                spool_path=spool_path,
                return_stats=True,
            )

            assert stats["edge_count"] == 2
            assert isinstance(synapse_store, mg._BoundedSynapseSpool)
            assert len(synapse_store) == 2

            mg._save_neural_map_streaming(
                "tester",
                {"converted_from_legacy": False, "updated_at": "2026-04-03T00:00:00+00:00"},
                neurons,
                synapse_store,
                {
                    "compact_save_enabled": True,
                    "max_fragments_per_neuron": 8,
                    "max_tags_per_neuron": 8,
                    "vector_round_digits": 4,
                    "position_round_digits": 3,
                },
                None,
            )
            mg._save_sparse_snapshot_by_ids("tester", ["n1", "n2", "n3"], synapse_store)

            map_payload = json.loads(map_path.read_text())
            snapshot_payload = json.loads(snapshot_path.read_text())

            assert len(map_payload["synapses"]) == 2
            assert sorted((edge["source"], edge["target"]) for edge in map_payload["synapses"]) == [("n1", "n2"), ("n2", "n3")]
            assert snapshot_payload["edge_count"] == 2
            assert snapshot_payload["indices"] == [1, 2]
        finally:
            if synapse_store is not None:
                synapse_store.cleanup()
            mg._neural_map_path = original_map
            mg._neural_snapshot_path = original_snapshot


def test_candidate_pool_from_index_keeps_latest_entries():
    child = "TestSelectorHeap"
    base = Path("AI_Children") / child / "memory" / "fragments"
    base.mkdir(parents=True, exist_ok=True)
    try:
        index = {}
        for idx, ts in enumerate([100.0, 200.0, 300.0, 150.0], start=1):
            frag_id = f"f{idx}"
            frag_path = base / f"frag_{idx}.json"
            frag_path.write_text(json.dumps({"id": frag_id, "tags": [], "emotions": {}}))
            os.utime(frag_path, (ts, ts))
            index[frag_id] = {"filename": frag_path.name, "tier": None, "importance": float(idx) / 10.0}

        selected = mg._candidate_pool_from_index(
            child,
            index,
            pool_max=2,
            blocked_tags=[],
            recent_ids=set(),
            known_fragments=None,
            cost_max_bytes=None,
        )

        assert selected == ["f3", "f2"]
    finally:
        for frag_path in base.glob("frag_*.json"):
            try:
                frag_path.unlink()
            except OSError:
                pass
        for parent in [base, base.parent, base.parent.parent]:
            try:
                parent.rmdir()
            except OSError:
                pass


def test_memory_manager_autoload_can_be_disabled():
    manager = mg.MemoryManager(child="TestNoAutoload", autoload=False)
    assert manager.memory_map == {}
    assert manager._map_loaded is False


def test_self_reflection_core_peek_recent_memory_accepts_iterable_ids():
    core = SelfReflectionCore("test")
    sample = core._peek_recent_memory(["f1", "f2", "f3"])
    assert len(sample) == 3
    assert set(sample) == {"f1", "f2", "f3"}
