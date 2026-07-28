from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from cognitive_benchmarks.audit import audit_surface_cues, surface_cues_pass
from cognitive_benchmarks.core import BenchmarkCase, load_cases, run_benchmark
from cognitive_benchmarks.procedural import generate_cases
from cognitive_benchmarks.schedule import MonthlyCadence


class FixedScorer:
    name = "fixed"

    def score_choices(self, prompt, choices):
        expected = int(prompt)
        return [1.0 if index == expected else 0.0 for index in range(len(choices))]


def test_run_benchmark_reports_aggregate_and_categories():
    cases = [
        BenchmarkCase("a", "memory", "1", ("x", "y"), 1),
        BenchmarkCase("b", "memory", "0", ("x", "y"), 0),
        BenchmarkCase("c", "reasoning", "1", ("x", "y"), 1),
    ]
    result = run_benchmark(
        cases, FixedScorer(), now=datetime(2026, 1, 1, tzinfo=timezone.utc)
    )
    assert result.accuracy == 1.0
    assert result.correct == result.total == 3
    assert result.mean_margin == 1.0
    assert result.categories["memory"]["accuracy"] == 1.0


def test_monthly_cadence_uses_calendar_month_and_persists(tmp_path):
    cadence = MonthlyCadence(tmp_path / "cadence.json")
    january = datetime(2026, 1, 31, 12, tzinfo=timezone.utc)
    before_due = datetime(2026, 2, 27, 12, tzinfo=timezone.utc)
    due = datetime(2026, 2, 28, 12, tzinfo=timezone.utc)
    assert cadence.is_due("persistent-cognition", "gpt2", now=january)
    cadence.mark_completed("persistent-cognition", "gpt2", completed_at=january)
    reloaded = MonthlyCadence(tmp_path / "cadence.json")
    assert not reloaded.is_due("persistent-cognition", "gpt2", now=before_due)
    assert reloaded.is_due("persistent-cognition", "gpt2", now=due)


def test_frozen_suite_is_valid_and_has_balanced_answer_positions():
    suite = Path(__file__).resolve().parents[1] / "benchmarks" / "persistent_cognition_v1.jsonl"
    cases = load_cases(suite)
    assert len(cases) == 10
    assert len({case.category for case in cases}) == 5
    assert {case.answer for case in cases} == {0, 1, 2}


def test_blind_suite_keeps_answers_in_separate_key(tmp_path):
    questions = tmp_path / "questions.jsonl"
    key = tmp_path / "answers.json"
    questions.write_text(
        '{"id":"secret-1","category":"heldout","prompt":"P",'
        '"choices":[" a"," b"]}\n', encoding="utf-8"
    )
    key.write_text('{"secret-1":1}', encoding="utf-8")

    cases = load_cases(questions, answer_key_path=key)

    assert cases[0].answer == 1


def test_procedural_suite_is_reproducible_but_varies_by_seed():
    first = generate_cases(count_per_category=3, seed=17)
    repeated = generate_cases(count_per_category=3, seed=17)
    different = generate_cases(count_per_category=3, seed=18)

    assert first == repeated
    assert first != different
    assert len(first) == 15
    assert len({case.category for case in first}) == 5
    assert all(0 <= case.answer < len(case.choices) for case in first)


def test_procedural_suite_passes_surface_cue_preflight():
    cases = generate_cases(count_per_category=200, seed=0x1A2B3C)
    audit = audit_surface_cues(cases)

    assert audit["total"] == 1000
    assert surface_cues_pass(audit)
    assert audit["heuristics"]["shortest-choice"]["accuracy"] < 0.45
    assert audit["heuristics"]["token-overlap"]["accuracy"] < 0.45
