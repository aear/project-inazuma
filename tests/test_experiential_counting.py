from experiential_counting import ExperientialCounter, count_payload, verify_by_recount


class NoLength:
    def __init__(self, values):
        self.values = values

    def __len__(self):
        raise AssertionError("counting must not use a declared/container length")

    def __iter__(self):
        yield from self.values


def test_beats_are_counted_as_observed_occurrences_even_when_identical():
    beats = NoLength(["kick", "kick", "snare", "kick"])
    result = ExperientialCounter("beat", rule="one detected onset is one beat").observe_many(beats)
    assert result["value"] == 4
    assert result["status"] == "exact"
    assert result["counted_by_observation"] is True


def test_pixels_are_enumerated_and_grouped_not_calculated_from_dimensions():
    pixels = NoLength([
        {"position": (0, 0), "colour": "dark"},
        {"position": (1, 0), "colour": "light"},
        {"position": (0, 1), "colour": "dark"},
        {"position": (1, 1), "colour": "dark"},
    ])
    counter = ExperientialCounter(
        "pixel", rule="one yielded raster position is one pixel",
        mode="unique", identity_of=lambda pixel: pixel["position"],
        group_of=lambda pixel: pixel["colour"],
    )
    result = counter.observe_many(pixels)
    assert result["value"] == 4
    assert result["groups"] == {"dark": 3, "light": 1}


def test_spiral_wraps_use_an_explicit_crossing_rule():
    samples = [-.2, .1, .8, .95, .05, .4, .9, .02]
    prior = None
    crossings = []
    for phase in samples:
        if prior is not None and prior >= .75 and phase <= .25:
            crossings.append({"from": prior, "to": phase})
        prior = phase
    result = ExperientialCounter(
        "spiral wrap", rule="phase crosses from at least .75 to at most .25",
    ).observe_many(crossings)
    assert result["value"] == 2


def test_versions_count_unique_observed_identities_and_report_duplicates():
    versions = [{"id": "v1"}, {"id": "v2"}, {"id": "v2"}, {"id": "v3"}]
    result = ExperientialCounter(
        "version", rule="one distinct immutable version id", mode="unique",
        identity_of=lambda version: version["id"],
    ).observe_many(versions)
    assert result["value"] == 3
    assert result["duplicates"] == 1
    assert result["status"] == "exact"


def test_budget_stop_is_an_honest_lower_bound_and_can_resume():
    counter = ExperientialCounter("pulse", rule="one observed pulse")
    partial = counter.observe_many(iter(range(5)), budget=3)
    assert partial["status"] == "incomplete"
    assert partial["value"] == partial["lower_bound"] == 3
    counter.observe(3)
    counter.observe(4)
    complete = counter.complete_boundary()
    assert complete["status"] == "exact"
    assert complete["value"] == 5


def test_unique_count_can_resume_from_checkpoint_without_double_counting():
    first = ExperientialCounter(
        "mark", rule="one distinct mark id", mode="unique", identity_of=lambda item: item["id"],
    )
    first.observe_many([{"id": "a"}, {"id": "b"}], budget=2)
    second = ExperientialCounter(
        "mark", rule="one distinct mark id", mode="unique", identity_of=lambda item: item["id"],
    )
    second.restore(first.checkpoint())
    result = second.observe_many([{"id": "b"}, {"id": "c"}])
    assert result["value"] == 3
    assert result["duplicates"] == 1


def test_checkpoint_cannot_be_used_as_a_disguised_declared_total():
    original = ExperientialCounter(
        "version", rule="one distinct id", mode="unique", identity_of=lambda item: item["id"],
    )
    original.observe_many([{"id": "v1"}], budget=1)
    forged = original.checkpoint()
    forged["count"] = 999
    resumed = ExperientialCounter(
        "version", rule="one distinct id", mode="unique", identity_of=lambda item: item["id"],
    )
    try:
        resumed.restore(forged)
    except ValueError as exc:
        assert "integrity" in str(exc)
    else:
        raise AssertionError("forged checkpoint count was accepted")


def test_admission_failure_prevents_false_exact_claim():
    def admit(value):
        if value == "unresolved":
            raise ValueError("cannot classify")
        return value == "beat"

    result = ExperientialCounter("beat", rule="classified beat onset", admit=admit).observe_many(
        ["beat", "noise", "unresolved"]
    )
    assert result["value"] == 1
    assert result["status"] == "complete_with_unresolved"
    assert result["lower_bound"] == 1


def test_verification_requires_two_separate_complete_enumerations():
    calls = []
    def source():
        calls.append(object())
        return NoLength([1, 2, 3, 4])
    result = verify_by_recount(source, unit="item", rule="one yielded item")
    assert len(calls) == 2
    assert result["status"] == "verified"
    assert result["value"] == 4
    assert result["independent_enumerations"] == 2


def test_runtime_payload_counts_observations_and_rejects_declared_totals():
    result = count_payload({
        "unit": "version", "rule": "one distinct version id", "mode": "unique",
        "identity_key": "id", "observations": [{"id": "v1"}, {"id": "v2"}, {"id": "v2"}],
        "observation_budget": 10,
    })
    assert result["value"] == 2
    assert result["status"] == "exact"
    try:
        count_payload({"unit": "pixel", "rule": "one pixel", "declared_total": 1920 * 1080,
                       "observations": []})
    except ValueError as exc:
        assert "not observations" in str(exc)
    else:
        raise AssertionError("declared total bypassed observation counting")
