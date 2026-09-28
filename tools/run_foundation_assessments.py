"""Run bounded current-implementation assessments and retain one report."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from capability_assessments import (
    assess_cyber_responses, assess_english_foundation, assess_research_foundation,
)
from external_boundary_audit import audit_external_boundaries
from io_utils import atomic_write_json


def main() -> int:
    report = {
        "schema": "ina.foundation_assessment_bundle/V1",
        "assessed_at": datetime.now(timezone.utc).isoformat(),
        "assessments": {
            "language_english": assess_english_foundation(),
            "research": assess_research_foundation(),
            "cyber_defence": assess_cyber_responses(None),
        },
        "external_boundaries": audit_external_boundaries(),
        "promotion_authorized": False,
    }
    path = Path("benchmark_results/capability_assessments/current.json")
    atomic_write_json(path, report, indent=2, ensure_ascii=False)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
