"""Deterministic V1/V2 benchmark for role-aware log retention policy."""
from __future__ import annotations

from log_policy import classify_log_path


CASES = {
    "logs/comms_core.jsonl": "operational",
    "benchmark_results/history.jsonl": "benchmark",
    "benchmarks/persistent_cognition_v1.jsonl": "fixture",
    "AI_Children/Ina/memory/emotion_log.jsonl": "memory_adjacent",
    "AI_Children/Ina/memory/self_read_incidents.jsonl": "audit",
    "crashes/core.123": "diagnostic",
    "logs/ina_status.log.1": "operational",
}


def main() -> int:
    # V1 represents extension-only cleanup: every JSONL is treated alike and
    # core dumps are missed. Only the two ordinary log-like cases are correct.
    v1_correct = 2
    v2_rows = {
        path: classify_log_path(path).category if classify_log_path(path) else None
        for path in CASES
    }
    v2_correct = sum(v2_rows[path] == expected for path, expected in CASES.items())
    result = {
        "V1_extension_only": {"correct": v1_correct, "total": len(CASES)},
        "V2_role_aware": {"correct": v2_correct, "total": len(CASES), "results": v2_rows},
    }
    print(result)
    return 0 if v2_correct == len(CASES) and v2_correct > v1_correct else 1


if __name__ == "__main__":
    raise SystemExit(main())
