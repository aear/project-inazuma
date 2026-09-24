from counting_cadence import CadenceLearner, compare_linear_and_grouped, count_with_cadence


def test_grouped_counting_observes_every_unit_and_handles_remainder():
    result = count_with_cadence(iter(range(12)), unit="beat", rule="one onset", stride=5)
    assert result["value"] == result["observed"] == 12
    assert [group["size"] for group in result["verified_groups"]] == [5, 5, 2]
    assert result["verified_groups"][-1]["remainder"] is True
    assert result["approximate"] is False


def test_unreliable_grouping_forces_linear_counting():
    learner = CadenceLearner()
    assert learner.choose(grouping_reliable=False) == 1


def test_learning_prefers_verified_efficient_cadence_not_failed_shortcut():
    learner = CadenceLearner()
    learner.learn(1, exact=True, recount_agreed=True, steps=20, observations=20)
    learner.learn(2, exact=True, recount_agreed=True, steps=10, observations=20)
    learner.learn(5, exact=True, recount_agreed=True, steps=4, observations=20)
    learner.learn(10, exact=False, recount_agreed=False, steps=2, observations=20)
    assert learner.choose(grouping_reliable=True) == 5


def test_task_can_offer_any_bounded_sensible_group_sizes():
    learner = CadenceLearner(candidates=(1, 3, 7, 12))
    for stride in learner.candidates:
        learner.learn(stride, exact=True, recount_agreed=True,
                      steps=(84 + stride - 1) // stride, observations=84)
    assert learner.choose(grouping_reliable=True, maximum_stride=8) == 7
    result = count_with_cadence(iter(range(17)), unit="step", rule="one observed step", stride=7)
    assert result["value"] == 17
    assert [group["size"] for group in result["verified_groups"]] == [7, 7, 3]


def test_stride_is_not_a_declared_total_shortcut():
    result = count_with_cadence(iter(["a", "b", "c"]), unit="mark", rule="one mark", stride=10)
    assert result["value"] == 3
    assert result["accumulator_steps"] == 1


def test_linear_and_grouped_accumulators_compare_concurrently():
    result = compare_linear_and_grouped(
        iter(range(23)), unit="triangle", rule="one observed triangular face", group_size=5,
    )
    assert result["status"] == "agreed"
    assert result["linear_value"] == result["grouped_value"] == 23
    assert result["closed_group_total"] == 20
    assert result["remainder"] == 3
    assert result["concurrent_accumulators"] == 2


def test_concurrent_comparison_does_not_claim_independent_recount():
    result = compare_linear_and_grouped(iter(range(4)), unit="beat", rule="one onset", group_size=2)
    assert result["shared_observation_stream"] is True
    assert result["independent_enumerations"] == 1
    assert result["does_not_validate"] == "observation_completeness"
