"""Fresh, inspectable benchmark instances: learning the rules is allowed."""
from __future__ import annotations

import random
from typing import Sequence

from .core import BenchmarkCase

PROCEDURAL_VERSION = "2"
NAMES = ("Ari", "Bo", "Cy", "Dee", "Em", "Fox", "Gia", "Hal")
ITEMS = ("key", "book", "coin", "ring", "note", "badge", "card", "shell")
PLACES = ("drawer", "shelf", "basket", "pocket", "locker", "cabinet")
ACTIONS = (
    ("pressed the switch", "the lamp lit"),
    ("tilted the bottle", "the water spilled"),
    ("pulled the cord", "the bell rang"),
    ("turned the handle", "the door opened"),
    ("tapped the screen", "the display woke"),
    ("closed the valve", "the flow stopped"),
)
EVENTS = ("washed the cup", "opened the door", "wrote the note",
          "rang the bell", "watered the plant", "moved the chair")


def _choices(correct: str, wrong: Sequence[str], answer_position: int, rng: random.Random):
    distractors = list(wrong)
    rng.shuffle(distractors)
    values = distractors
    values.insert(answer_position % (len(distractors) + 1), correct)
    return tuple(" " + value for value in values), values.index(correct)


def _position(number: int, category_offset: int, choice_count: int) -> int:
    return (number + category_offset) % choice_count


def generate_cases(*, count_per_category: int = 4, seed: int) -> list[BenchmarkCase]:
    """Generate balanced categories deterministically from an evaluator seed."""
    if count_per_category < 1:
        raise ValueError("count_per_category must be positive")
    rng = random.Random(seed)
    cases: list[BenchmarkCase] = []
    for number in range(count_per_category):
        name, item = rng.choice(NAMES), rng.choice(ITEMS)
        place, other, third = rng.sample(PLACES, 3)
        choices, answer = _choices(f"the {place}.", (f"the {other}.", f"the {third}."),
                                   _position(number, 0, 3), rng)
        continuity_prompt = rng.choice((
            f"The possible places were the {place}, {other}, and {third}. {name} put the {item} in the {place}. Nobody moved it. Later, it was in",
            f"{name} chose the {place}, not the {other} or {third}, for the {item}. It stayed there. Its later location was",
        ))
        cases.append(BenchmarkCase(f"continuity-{number}", "continuity",
                                   continuity_prompt, choices, answer))

        first, second = rng.sample(EVENTS, 2)
        choices, answer = _choices(f"{first}.", (f"{second}.",),
                                   _position(number, 1, 2), rng)
        temporal_prompt = rng.choice((
            f"First {name} {first}. Afterwards {name} {second}. What happened earlier?",
            f"{name} {second}, but only after {name} had {first}. Which event came first?",
        ))
        cases.append(BenchmarkCase(f"temporal-{number}", "temporal-order",
                                   temporal_prompt, choices, answer))

        color = rng.choice(("red", "blue", "green", "gold"))
        choices, answer = _choices("contradictory.", ("complementary.", "nonassociated."),
                                   _position(number, 2, 3), rng)
        contradiction_prompt = rng.choice((
            f"Every {item} is {color}. This {item} is not {color}. Their relation is",
            f"One claim calls this {item} {color}; another denies it is {color}. Their relation is",
        ))
        cases.append(BenchmarkCase(f"contradiction-{number}", "contradiction",
                                   contradiction_prompt, choices, answer))

        old, new, unrelated = rng.sample((2, 3, 4, 5, 6, 7, 8), 3)
        choices, answer = _choices(f"{new}.", (f"{old}.", f"{unrelated}."),
                                   _position(number, 0, 3), rng)
        revision_prompt = rng.choice((
            f"The candidate values were {old}, {new}, and {unrelated}. {name} expected {old}, but reliable evidence corrected it to {new}. Use",
            f"Of {old}, {new}, and {unrelated}, the estimate was {old}, then a trusted correction changed it to {new}. Use",
        ))
        cases.append(BenchmarkCase(f"revision-{number}", "belief-revision",
                                   revision_prompt, choices, answer))

        selected = rng.sample(ACTIONS, 3)
        (cause, effect), (wrong_a, _), (wrong_b, _) = selected
        choices, answer = _choices(f"{name} {cause}.",
                                   (f"{name} {wrong_a}.", f"{name} {wrong_b}."),
                                   _position(number, 1, 3), rng)
        causal_prompt = rng.choice((
            f"{name} {cause}, not {wrong_a} or {wrong_b}; this caused {effect}. Identify the cause:",
            f"The alternatives were: {cause}, {wrong_a}, or {wrong_b}. {effect} happened because {name} {cause}. The cause was:",
        ))
        cases.append(BenchmarkCase(f"causal-{number}", "causal-tracking",
                                   causal_prompt, choices, answer))
    return cases
