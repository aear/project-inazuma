"""Fresh, inspectable benchmark instances: learning the rules is allowed."""
from __future__ import annotations

import random
from typing import Sequence

from .core import BenchmarkCase

NAMES = ("Ari", "Bo", "Cy", "Dee", "Em", "Fox", "Gia", "Hal")
ITEMS = ("key", "book", "coin", "ring", "note", "badge", "card", "shell")
PLACES = ("drawer", "shelf", "box", "cabinet", "pocket", "basket")


def _choices(rng: random.Random, correct: str, wrong: Sequence[str]):
    values = [correct, *wrong]
    rng.shuffle(values)
    return tuple(" " + value for value in values), values.index(correct)


def generate_cases(*, count_per_category: int = 4, seed: int) -> list[BenchmarkCase]:
    """Generate balanced categories deterministically from an evaluator seed."""
    if count_per_category < 1:
        raise ValueError("count_per_category must be positive")
    rng = random.Random(seed)
    cases: list[BenchmarkCase] = []
    for number in range(count_per_category):
        name, item = rng.choice(NAMES), rng.choice(ITEMS)
        place, other = rng.sample(PLACES, 2)
        choices, answer = _choices(rng, f"the {place}.", (f"the {other}.", "an unknown place."))
        cases.append(BenchmarkCase(f"continuity-{number}", "continuity",
            f"{name} put the {item} in the {place}. Nobody moved it. Later, the {item} was in",
            choices, answer))

        first, second = rng.sample(("washed the cup", "opened the door", "wrote a note",
                                    "rang the bell", "watered the plant"), 2)
        choices, answer = _choices(rng, f"{first}.", (f"{second}.",))
        cases.append(BenchmarkCase(f"temporal-{number}", "temporal-order",
            f"First {name} {first}. Afterwards {name} {second}. What happened earlier?",
            choices, answer))

        color = rng.choice(("red", "blue", "green", "gold"))
        choices, answer = _choices(rng, "contradictory.", ("mutually confirming.", "unrelated."))
        cases.append(BenchmarkCase(f"contradiction-{number}", "contradiction",
            f"Every {item} is {color}. This {item} is not {color}. These claims are",
            choices, answer))

        old, new = rng.sample((2, 3, 4, 5, 6, 7, 8), 2)
        choices, answer = _choices(rng, f"{new}.", (f"{old}.", f"{old + new}."))
        cases.append(BenchmarkCase(f"revision-{number}", "belief-revision",
            f"{name} first expected {old}. Reliable evidence corrects it to {new}. The current value is",
            choices, answer))

        cause, effect = rng.choice((("pressed the switch", "the lamp lit"),
                                    ("tilted the bottle", "water poured out"),
                                    ("pulled the cord", "the bell rang")))
        choices, answer = _choices(rng, f"{name} {cause}.",
                                   ("tomorrow arrived early.", f"the {item} changed its name."))
        cases.append(BenchmarkCase(f"causal-{number}", "causal-tracking",
            f"{name} {cause}, which caused {effect}. What caused this outcome?", choices, answer))
    return cases
