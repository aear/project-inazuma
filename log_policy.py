"""Bounded, explicitly invoked classification of Project Inazuma log evidence."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class RetentionPolicy:
    category: str
    purpose: str
    strategy: str
    max_bytes: int | None
    generations: int | None
    max_age_days: int | None
    compress: bool
    automatic_cleanup: bool


POLICIES = {
    "operational": RetentionPolicy(
        "operational", "Live operator and transport visibility",
        "small size-based rotation", 16 * 1024 * 1024, 6, 14, False, True,
    ),
    "diagnostic": RetentionPolicy(
        "diagnostic", "Debugging failures and unexpected behaviour",
        "short rotating generations", 8 * 1024 * 1024, 4, 30, True, True,
    ),
    "audit": RetentionPolicy(
        "audit", "Delivery, security, migration, and incident evidence",
        "compressed long-lived generations with explicit review", 64 * 1024 * 1024, 12, 365, True, False,
    ),
    "benchmark": RetentionPolicy(
        "benchmark", "Version-comparable measurement history",
        "bounded structured history; retain reports, not unlimited raw runs", 32 * 1024 * 1024, 8, None, True, False,
    ),
    "memory_adjacent": RetentionPolicy(
        "memory_adjacent", "Structured witness, learning input, or durable queue",
        "do not clean as a log; promote, compact, or reconcile through its owning subsystem", None, None, None, False, False,
    ),
    "fixture": RetentionPolicy(
        "fixture", "Checked-in benchmark or test input",
        "source-controlled data; never apply log retention", None, None, None, False, False,
    ),
}

AUDIT_NAMES = {
    "github_outbox_history.jsonl", "github_outbox_archive.jsonl",
    "github_issue_feedback.jsonl", "self_read_incidents.jsonl",
    "storage_migration_history.jsonl", "fragment_repair_log.jsonl",
    "heal_tickets.jsonl", "auth_health.jsonl", "security.log",
}
MEMORY_NAMES = {
    "emotion_log.jsonl", "reflection_journal.jsonl", "reflection_public_report.jsonl",
    "precision_memory_map.jsonl", "language_evidence.jsonl", "decision_panic_log.jsonl",
    "neural_selector_log.jsonl", "trauma_processor_log.jsonl", "cold_core.jsonl",
    "github_outbox.jsonl", "typed_outbox.jsonl", "candidate_queue.jsonl",
}
SKIP_DIRECTORIES = {".git", "AI_Children", "venv", ".venv", "env", ".env", "__pycache__"}
LOG_SUFFIXES = {".log", ".jsonl", ".dump", ".crash"}


def classify_log_path(path: Path | str) -> RetentionPolicy | None:
    path = Path(path)
    name = path.name.lower()
    logical_name = name[:-3] if name.endswith(".gz") else name
    stem, separator, generation = logical_name.rpartition(".")
    if separator and generation.isdigit():
        logical_name = stem
    logical_path = path.with_name(logical_name)
    parts = {part.lower() for part in path.parts}
    if "benchmarks" in parts and "benchmark_results" not in parts:
        return POLICIES["fixture"]
    if "benchmark_results" in parts or name in {"module_versions.jsonl", "history.jsonl"}:
        return POLICIES["benchmark"]
    is_log_artifact = logical_path.suffix.lower() in LOG_SUFFIXES
    if logical_name in AUDIT_NAMES or (is_log_artifact and any(
        token in logical_name for token in ("incident", "audit", "migration_history")
    )):
        return POLICIES["audit"]
    if logical_name in MEMORY_NAMES or ("ai_children" in parts and is_log_artifact):
        return POLICIES["memory_adjacent"]
    if logical_name in {"ina_status.log", "comms_core.jsonl", "ina_status_fallback.log"}:
        return POLICIES["operational"]
    is_core_dump = logical_name == "core" or (
        logical_name.startswith("core.") and logical_path.suffix.lower() not in {".py", ".json", ".md"}
    )
    if is_log_artifact or is_core_dump:
        return POLICIES["diagnostic"]
    return None


def inventory_logs(root: Path | str, *, max_files: int = 10_000) -> dict[str, object]:
    """Inspect a bounded workspace surface without entering Ina's memory tree."""
    root = Path(root).resolve()
    limit = max(1, min(100_000, int(max_files)))
    rows: list[dict[str, object]] = []
    scanned = 0
    truncated = False
    for directory, names, files in os.walk(root):
        names[:] = sorted(name for name in names if name not in SKIP_DIRECTORIES)
        for filename in sorted(files):
            scanned += 1
            if scanned > limit:
                truncated = True
                break
            path = Path(directory) / filename
            policy = classify_log_path(path.relative_to(root))
            if policy is None:
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            rows.append({
                "path": str(path.relative_to(root)),
                "bytes": stat.st_size,
                "category": policy.category,
                "strategy": policy.strategy,
                "over_size_policy": policy.max_bytes is not None and stat.st_size > policy.max_bytes,
            })
        if truncated:
            break
    totals: dict[str, dict[str, int]] = {}
    for row in rows:
        bucket = totals.setdefault(str(row["category"]), {"files": 0, "bytes": 0})
        bucket["files"] += 1
        bucket["bytes"] += int(row["bytes"])
    return {
        "root": str(root), "scanned_files": min(scanned, limit), "truncated": truncated,
        "excluded_directories": sorted(SKIP_DIRECTORIES), "totals": totals, "files": rows,
    }


def policy_catalog() -> Iterable[dict[str, object]]:
    return (asdict(policy) for policy in POLICIES.values())


def main() -> int:
    parser = argparse.ArgumentParser(description="Dry-run Project Inazuma log policy sweep")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--max-files", type=int, default=10_000)
    parser.add_argument("--policies", action="store_true", help="Include the policy catalog")
    args = parser.parse_args()
    report = inventory_logs(args.root, max_files=args.max_files)
    if args.policies:
        report["policies"] = list(policy_catalog())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
