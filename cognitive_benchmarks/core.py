from __future__ import annotations

import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    category: str
    prompt: str
    choices: tuple[str, ...]
    answer: int


class ChoiceScorer(Protocol):
    """Backend contract: larger scores mean a more likely continuation."""

    name: str

    def score_choices(self, prompt: str, choices: Sequence[str]) -> Sequence[float]:
        ...


@dataclass(frozen=True)
class BenchmarkResult:
    benchmark: str
    benchmark_version: str
    model: str
    run_id: str
    started_at: str
    elapsed_seconds: float
    accuracy: float
    mean_margin: float
    correct: int
    total: int
    categories: dict[str, dict[str, float | int]]
    cases: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_cases(path: Path, *, answer_key_path: Path | None = None) -> list[BenchmarkCase]:
    answer_key: dict[str, int] = {}
    if answer_key_path is not None:
        raw_key = json.loads(answer_key_path.read_text(encoding="utf-8"))
        if not isinstance(raw_key, dict):
            raise ValueError("answer key must map case ids to answer indexes")
        answer_key = {str(key): int(value) for key, value in raw_key.items()}
    cases: list[BenchmarkCase] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            choices = tuple(str(value) for value in raw["choices"])
            case_id = str(raw["id"])
            if "answer" in raw and answer_key_path is not None:
                raise ValueError("blind question files must not embed answers")
            try:
                value = raw["answer"] if answer_key_path is None else answer_key[case_id]
                answer = int(value)
            except KeyError as exc:
                raise ValueError(f"missing answer for {case_id!r}") from exc
            if case_id in seen:
                raise ValueError(f"duplicate case id {case_id!r} on line {line_number}")
            if len(choices) < 2 or not 0 <= answer < len(choices):
                raise ValueError(f"invalid choices/answer on line {line_number}")
            seen.add(case_id)
            cases.append(
                BenchmarkCase(
                    case_id=case_id,
                    category=str(raw["category"]),
                    prompt=str(raw["prompt"]),
                    choices=choices,
                    answer=answer,
                )
            )
    if not cases:
        raise ValueError(f"benchmark has no cases: {path}")
    return cases


def _safe_margin(scores: Sequence[float], answer: int) -> float:
    alternatives = [value for index, value in enumerate(scores) if index != answer]
    return float(scores[answer] - max(alternatives))


def run_benchmark(
    cases: Iterable[BenchmarkCase],
    scorer: ChoiceScorer,
    *,
    benchmark: str = "persistent-cognition",
    benchmark_version: str = "1",
    now: datetime | None = None,
) -> BenchmarkResult:
    started = now or datetime.now(timezone.utc)
    clock_started = time.perf_counter()
    details: list[dict[str, Any]] = []
    category_rows: dict[str, list[tuple[bool, float]]] = {}

    for case in cases:
        scores = [float(value) for value in scorer.score_choices(case.prompt, case.choices)]
        if len(scores) != len(case.choices) or any(not math.isfinite(value) for value in scores):
            raise ValueError(f"{scorer.name} returned invalid scores for {case.case_id}")
        predicted = max(range(len(scores)), key=scores.__getitem__)
        margin = _safe_margin(scores, case.answer)
        correct = predicted == case.answer
        category_rows.setdefault(case.category, []).append((correct, margin))
        details.append(
            {
                "id": case.case_id,
                "category": case.category,
                "correct": correct,
                "expected": case.answer,
                "predicted": predicted,
                "margin": round(margin, 6),
            }
        )

    correct_count = sum(bool(row["correct"]) for row in details)
    margins = [float(row["margin"]) for row in details]
    categories = {
        name: {
            "correct": sum(correct for correct, _ in rows),
            "total": len(rows),
            "accuracy": round(sum(correct for correct, _ in rows) / len(rows), 6),
            "mean_margin": round(statistics.fmean(margin for _, margin in rows), 6),
        }
        for name, rows in sorted(category_rows.items())
    }
    return BenchmarkResult(
        benchmark=benchmark,
        benchmark_version=benchmark_version,
        model=scorer.name,
        run_id=f"{started.strftime('%Y%m%dT%H%M%SZ')}-{scorer.name.replace('/', '_')}",
        started_at=started.isoformat(),
        elapsed_seconds=round(time.perf_counter() - clock_started, 6),
        accuracy=round(correct_count / len(details), 6),
        mean_margin=round(statistics.fmean(margins), 6),
        correct=correct_count,
        total=len(details),
        categories=categories,
        cases=tuple(details),
    )
