from benchmarks.benchmark_priority_foundations import run


def test_vocab_cap_raise_is_versioned_and_does_not_claim_quality():
    result = run()
    assert [item["version"] for item in result["vocab_capacity"]] == ["V1", "V2"]
    assert result["vocab_capacity"][1]["retained"] == 50_000
    assert result["claims"]["live_ina_language_quality"] == "unavailable"
