"""Explicit bounded V1/V2 benchmark for preserving pre-intent creative material."""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emergence_capture import capture_emergence, propose_interpretation


def main() -> int:
    capture = capture_emergence("We Had Fun on The Deathstar", modality="text")
    original = dict(capture)
    interpretation = propose_interpretation(capture, "one possible reading", confidence=0.2)
    results = {
        "V1_intent_first": {
            "pre_intent_capture": 0, "unresolved_meaning_preserved": 0,
            "linked_non_destructive_interpretation": 0,
        },
        "V2_emergence_first": {
            "pre_intent_capture": int("intention" not in capture),
            "unresolved_meaning_preserved": int(capture.get("meaning_status") == "unresolved"),
            "linked_non_destructive_interpretation": int(
                capture == original and interpretation.get("capture_id") == capture.get("capture_id")
            ),
        },
    }
    print(results)
    return 0 if all(results["V2_emergence_first"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
