"""V1/V2 comparison for multidimensional creativity evidence."""
from __future__ import annotations

from creativity_observatory import build_profile, make_evidence


def main() -> int:
    rows = [
        make_evidence("image", "originality", .8, artifact_id="image-1", origin="blind-review", method="review", evaluator_type="human", material_mode="emergent", reference_set_id="heldout-v1"),
        make_evidence("image", "originality", .7, artifact_id="image-1", origin="similarity-audit", method="audit", evaluator_type="objective", material_mode="emergent", reference_set_id="heldout-v1"),
    ]
    report = build_profile(rows)
    originality = report["domains"]["image"]["dimensions"]["originality"]
    v2 = {
        "multidimensional": int("score" not in report and len(report["domains"]["image"]["dimensions"]) == 6),
        "corroborated_originality": int(originality["evidenced"] and len(originality["origins"]) == 2),
        "emergence_preserved": int(not report["policy"]["intent_required"]),
        "no_thought_surveillance": int(not report["policy"]["private_rationale_required"]),
        "review_only": int(not report["policy"]["automatic_promotion"]),
    }
    print({"V1_vague_creativity_score": {key: 0 for key in v2}, "V2_creativity_profile": v2})
    return 0 if all(v2.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
