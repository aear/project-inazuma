import os
import sys
import types

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

_state = {}
model_manager_stub = types.ModuleType("model_manager")
model_manager_stub.load_config = lambda: {}
model_manager_stub.seed_self_question = lambda prompt: None
model_manager_stub.get_inastate = lambda key=None: _state.get(key) if key is not None else dict(_state)
model_manager_stub.update_inastate = lambda key, value: _state.__setitem__(key, value)
sys.modules["model_manager"] = model_manager_stub

text_memory_stub = types.ModuleType("text_memory")
text_memory_stub.build_text_symbol_links = lambda child: None
sys.modules["text_memory"] = text_memory_stub

import meaning_map as mm


def test_meaning_policy_includes_new_caps():
    cfg = {
        "meaning_map_policy": {
            "max_tags_per_word": 12,
            "vector_round_digits": 4,
            "gc_every_batches": 2,
            "max_words_total": 999,
        }
    }
    policy = mm._meaning_policy(cfg)
    assert policy["max_tags_per_word"] == 12
    assert policy["vector_round_digits"] == 4
    assert policy["gc_every_batches"] == 2
    assert policy["max_words_total"] == 999


def test_apply_word_caps_trims_components_tags_and_rounds_vector():
    word = {
        "components": ["a", "b", "c", "d"],
        "tags": ["z", "x", "x", "y"],
        "vector": [0.123456, 0.987654],
    }
    policy = {
        "max_components_per_word": 2,
        "max_tags_per_word": 2,
        "vector_round_digits": 3,
    }
    mm._apply_word_caps(word, policy)
    assert word["components"] == ["c", "d"]
    assert word["tags"] == ["y", "z"]
    assert word["vector"] == [0.123, 0.988]


def test_enforce_word_budget_prunes_low_usage_words():
    words = [
        {"symbol_word_id": "sym_word_0001", "usage_count": 0, "count": 1, "updated_at": "2024-01-01T00:00:00+00:00"},
        {"symbol_word_id": "sym_word_0002", "usage_count": 3, "count": 5, "updated_at": "2024-01-02T00:00:00+00:00"},
        {"symbol_word_id": "sym_word_0003", "usage_count": 1, "count": 3, "updated_at": "2024-01-03T00:00:00+00:00"},
    ]
    dropped = mm._enforce_word_budget(words, 2)
    assert dropped == 1
    kept = {word["symbol_word_id"] for word in words}
    assert kept == {"sym_word_0002", "sym_word_0003"}


def test_cluster_encoded_keeps_light_members():
    encoded = [
        {"id": "f1", "vector": [1.0, 0.0], "tags": ["symbolic"], "summary": "one"},
        {"id": "f2", "vector": [1.0, 0.0], "tags": ["symbolic"], "summary": "two"},
    ]
    clusters = mm._cluster_encoded(encoded, threshold=0.5)
    assert len(clusters) == 1
    members = clusters[0]["members"]
    assert len(members) == 2
    assert "vector" not in members[0]


def test_promote_symbol_pairs_updates_proto_and_multi_word_state():
    preserved = {}
    cluster_members = [{"id": "frag1"}, {"id": "frag2"}]
    fragments_by_id = {
        "frag1": {
            "id": "frag1",
            "summary": "soft trust pulse",
            "tags": ["symbolic", "comm"],
            "symbols_spoken": ["snd_a", "snd_b"],
        },
        "frag2": {
            "id": "frag2",
            "summary": "soft trust pulse",
            "tags": ["symbolic", "comm"],
            "symbols_spoken": ["snd_a", "snd_b"],
        },
    }
    policy = {"max_tags_per_word": 4, "vector_round_digits": 3}

    proto_updates, multi_updates = mm._promote_symbol_pairs(
        preserved,
        cluster_members,
        fragments_by_id,
        [0.9, 0.1],
        ["symbolic", "comm"],
        "soft trust cluster",
        policy,
    )

    assert proto_updates == 1
    assert multi_updates == 1
    proto_entry = preserved["proto_words"]["snd_a_snd_b"]
    multi_entry = preserved["multi_symbol_words"]["pair:snd_a_snd_b"]
    assert proto_entry["uses"] == 2
    assert proto_entry["summary"] == "soft trust pulse"
    assert proto_entry["vector"] == [0.9, 0.1]
    assert multi_entry["uses"] == 2
    assert multi_entry["source"] == "meaning_cluster"
