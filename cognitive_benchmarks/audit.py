"""Cheap preflight checks for surface cues in generated multiple-choice cases."""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Callable, Iterable, Sequence

from .core import BenchmarkCase

Selector = Callable[[BenchmarkCase], int]


def _tokens(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", value.lower()))


def _token_overlap(case: BenchmarkCase) -> int:
    prompt = _tokens(case.prompt)
    scores = [len(prompt & _tokens(choice)) for choice in case.choices]
    return max(range(len(scores)), key=scores.__getitem__)


def audit_surface_cues(cases: Iterable[BenchmarkCase]) -> dict:
    rows = list(cases)
    if not rows:
        raise ValueError("surface audit needs at least one case")
    selectors: dict[str, Selector] = {
        "first-position": lambda case: 0,
        "second-position": lambda case: min(1, len(case.choices) - 1),
        "shortest-choice": lambda case: min(range(len(case.choices)),
                                               key=lambda i: len(case.choices[i].strip())),
        "longest-choice": lambda case: max(range(len(case.choices)),
                                              key=lambda i: len(case.choices[i].strip())),
        "token-overlap": _token_overlap,
    }
    expected_chance = sum(1.0 / len(case.choices) for case in rows) / len(rows)
    category_sizes = defaultdict(list)
    for case in rows:
        category_sizes[case.category].append(len(case.choices))
    category_chance = {
        category: round(sum(1.0 / size for size in sizes) / len(sizes), 6)
        for category, sizes in sorted(category_sizes.items())
    }
    results = {}
    for name, selector in selectors.items():
        category_counts = defaultdict(lambda: [0, 0])
        correct = 0
        for case in rows:
            hit = selector(case) == case.answer
            correct += int(hit)
            category_counts[case.category][0] += int(hit)
            category_counts[case.category][1] += 1
        results[name] = {
            "accuracy": round(correct / len(rows), 6),
            "categories": {
                category: round(hits / total, 6)
                for category, (hits, total) in sorted(category_counts.items())
            },
        }
    return {
        "total": len(rows),
        "expected_chance": round(expected_chance, 6),
        "category_expected_chance": category_chance,
        "heuristics": results,
    }


def surface_cues_pass(audit: dict, *, tolerance: float = 0.08) -> bool:
    overall_limit = float(audit["expected_chance"]) + tolerance
    for row in audit["heuristics"].values():
        if row["accuracy"] > overall_limit:
            return False
        for category, accuracy in row["categories"].items():
            if accuracy > float(audit["category_expected_chance"][category]) + tolerance:
                return False
    return True
