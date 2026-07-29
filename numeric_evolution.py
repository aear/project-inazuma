"""Gated, provenance-preserving optimization lane for Homo Silicus numerics.

Ina (or another local model) may generate complete candidate modules. Nothing
is installed automatically: candidates must pass the full test suite and a
paired statistical performance gate before a review-only patch is queued.
"""
from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import shlex
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

STABLE_PATH = Path("homo_silicus_numeric.py")
SAFE_IMPORTS = {"__future__", "builtins", "math", "operator", "array", "typing"}
FORBIDDEN_CALLS = {"open", "eval", "exec", "compile", "input", "__import__", "getattr",
                   "setattr", "delattr", "globals", "locals", "vars"}
DEFAULT_WORKLOADS = ((10, 128), (100, 128), (500, 128))


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_candidate_source(source):
    """Reject capabilities irrelevant to a pure numerical implementation."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"syntax error: {exc}"]
    errors = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in SAFE_IMPORTS:
                    errors.append(f"import not allowed: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in SAFE_IMPORTS:
                errors.append(f"import not allowed: {node.module}")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
            errors.append(f"call not allowed: {node.func.id}")
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
              and node.func.attr in FORBIDDEN_CALLS):
            errors.append(f"call not allowed: {node.func.attr}")
    return sorted(set(errors))


def generate_candidate(command, stable_source, brief, timeout=180):
    payload = json.dumps({
        "task": "Optimize this numerical module without changing its public behavior.",
        "rules": [
            "Return a complete module, not a patch.",
            "Use only the Python standard library imports already permitted by the source.",
            "Correctness is non-negotiable; preserve all tested semantics.",
            "Prefer measurable algorithmic or allocation improvements over cosmetic edits.",
        ],
        "brief": brief,
        "filename": STABLE_PATH.name,
        "source": stable_source,
    })
    completed = subprocess.run(shlex.split(command), input=payload, text=True, capture_output=True, timeout=timeout)
    if completed.returncode:
        raise RuntimeError(f"generator exited {completed.returncode}: {completed.stderr[-1000:]}")
    try:
        response = json.loads(completed.stdout)
        source = response["source"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise RuntimeError("generator must return JSON containing a string 'source'") from exc
    if not isinstance(source, str) or not source.strip():
        raise RuntimeError("generator returned an empty candidate")
    return source, {"generator_response": {key: value for key, value in response.items() if key != "source"}}


def _limit_resources(cpu_seconds, memory_mb):
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
        limit = memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except (ImportError, OSError, ValueError):
        pass


def run_full_suite(candidate_source, timeout=900, memory_mb=4096):
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="ina_numeric_candidate_") as temp:
        candidate_path = Path(temp) / STABLE_PATH.name
        candidate_path.write_text(candidate_source, encoding="utf-8")
        launcher = (
            "import sys;"
            f"sys.path.insert(0,{temp!r});"
            "import pytest;"
            "raise SystemExit(pytest.main(['-q','tests']))"
        )
        try:
            completed = subprocess.run(
                [sys.executable, "-c", launcher], text=True, capture_output=True, timeout=timeout,
                preexec_fn=(lambda: _limit_resources(max(60, timeout), memory_mb)) if os.name == "posix" else None,
            )
        except subprocess.TimeoutExpired as exc:
            return {"passed": False, "reason": "timeout", "seconds": timeout,
                    "output_tail": str(exc.stderr or exc.stdout or "")[-4000:]}
    output = (completed.stdout + "\n" + completed.stderr).strip()
    return {"passed": completed.returncode == 0, "returncode": completed.returncode,
            "seconds": round(time.perf_counter() - started, 3),
            "output_sha256": _sha(output), "output_tail": output[-4000:]}


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _workload_data(items, dimensions, seed):
    rng = random.Random(seed)
    rows = [[rng.uniform(-1.0, 1.0) for _ in range(dimensions)] for _ in range(items)]
    query = [rng.uniform(-1.0, 1.0) for _ in range(dimensions)]
    return rows, query


def _timed(call):
    started = time.perf_counter_ns()
    value = call()
    elapsed = time.perf_counter_ns() - started
    # Consume a result so an implementation cannot optimize the work away.
    checksum = sum(value.tolist()) if hasattr(value, "tolist") else float(value)
    return elapsed / 1_000_000.0, round(float(checksum), 12)


def _bootstrap_ci(changes, seed=1729, samples=3000):
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        draw = [changes[rng.randrange(len(changes))] for _ in changes]
        estimates.append(statistics.median(draw))
    estimates.sort()
    return estimates[int(samples * 0.025)], estimates[min(samples - 1, int(samples * 0.975))]


def benchmark_pair(stable_path, candidate_path, workloads=DEFAULT_WORKLOADS, trials=15,
                   min_improvement=0.05, max_regression=0.02):
    stable = _load_module(stable_path, "hs_stable_benchmark")
    candidate = _load_module(candidate_path, "hs_candidate_benchmark")
    results, all_changes = [], []
    for case_index, (items, dimensions) in enumerate(workloads):
        rows, query = _workload_data(items, dimensions, 9000 + case_index)
        stable_matrix, stable_query = stable.array(rows), stable.array(query)
        candidate_matrix, candidate_query = candidate.array(rows), candidate.array(query)
        stable_call = lambda: stable.cosine_rows(stable_matrix, stable_query)
        candidate_call = lambda: candidate.cosine_rows(candidate_matrix, candidate_query)
        stable_call(); candidate_call()  # warm both implementations
        stable_ms, candidate_ms, changes = [], [], []
        for trial in range(trials):
            calls = ((stable_call, candidate_call) if trial % 2 == 0 else (candidate_call, stable_call))
            measured = [_timed(call) for call in calls]
            if trial % 2:
                measured.reverse()
            (stable_time, stable_sum), (candidate_time, candidate_sum) = measured
            if not math.isclose(stable_sum, candidate_sum, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError(f"benchmark checksum mismatch at {items}x{dimensions}")
            stable_ms.append(stable_time); candidate_ms.append(candidate_time)
            changes.append((candidate_time - stable_time) / stable_time)
        low, high = _bootstrap_ci(changes, seed=1729 + case_index)
        median_change = statistics.median(changes)
        all_changes.extend(changes)
        results.append({"items": items, "dimensions": dimensions, "trials": trials,
                        "stable_ms": stable_ms, "candidate_ms": candidate_ms,
                        "median_relative_change": median_change,
                        "bootstrap_95_ci": [low, high], "regressed": median_change > max_regression})
    low, high = _bootstrap_ci(all_changes)
    median_change = statistics.median(all_changes)
    accepted = median_change <= -min_improvement and high < 0 and not any(row["regressed"] for row in results)
    return {"accepted": accepted, "paired_median_relative_change": median_change,
            "paired_bootstrap_95_ci": [low, high], "minimum_improvement": min_improvement,
            "maximum_workload_regression": max_regression, "workloads": results}


def hardware_provenance():
    cpu = platform.processor()
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                cpu = line.split(":", 1)[1].strip(); break
    except OSError:
        pass
    return {"platform": platform.platform(), "machine": platform.machine(), "processor": cpu,
            "python": platform.python_version(), "logical_cpus": os.cpu_count()}


def append_jsonl(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def queue_review(child, stable_source, candidate_source, report, candidate_path):
    from github_submission import append_github_issue_entry
    improvement = -100.0 * report["benchmark"]["paired_median_relative_change"]
    patch = "".join(difflib.unified_diff(stable_source.splitlines(True), candidate_source.splitlines(True),
                                         fromfile=f"a/{STABLE_PATH}", tofile=f"b/{STABLE_PATH}"))
    body = (f"Ina's numerical candidate passed the full test suite and paired performance gate.\n\n"
            f"Median improvement: {improvement:.2f}%\n"
            f"95% bootstrap interval (relative change): {report['benchmark']['paired_bootstrap_95_ci']}\n"
            f"Candidate artifact: `{candidate_path}`\n\n"
            "This is a review artifact only. Do not merge or execute without human review.")
    return append_github_issue_entry(
        child, f"Review Ina numerical optimization {report['candidate_sha256'][:12]}", body,
        kind="optimization_patch", labels=["ina-suggestion", "optimization", "needs-review"],
        patch_text=patch, metadata={"source": "numeric_evolution", "provenance": report,
                                   "review_notes": ["Correctness gate passed.", "Statistical performance gate passed.",
                                                    "Stable module was not overwritten."]})


def evaluate(candidate_source, child, trials=15, submit=False, full_test_timeout=900,
             generation_metadata=None):
    stable_source = STABLE_PATH.read_text(encoding="utf-8")
    errors = validate_candidate_source(candidate_source)
    candidate_sha = _sha(candidate_source)
    root = Path("AI_Children") / child / "memory" / "numeric_evolution"
    candidate_path = root / "candidates" / f"{candidate_sha}.py"
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_path.write_text(candidate_source, encoding="utf-8")
    report = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
              "stable_path": str(STABLE_PATH), "stable_sha256": _sha(stable_source),
              "candidate_path": str(candidate_path), "candidate_sha256": candidate_sha,
              "hardware": hardware_provenance(), "static_validation": {"passed": not errors, "errors": errors}}
    if generation_metadata:
        report["generation"] = generation_metadata
    if not errors:
        report["stable_baseline_suite"] = run_full_suite(stable_source, timeout=full_test_timeout)
        if report["stable_baseline_suite"].get("passed"):
            report["full_test_suite"] = run_full_suite(candidate_source, timeout=full_test_timeout)
        else:
            report["full_test_suite"] = {
                "passed": False,
                "reason": "baseline_environment_unhealthy",
                "detail": "The stable module does not pass the full suite in this environment; candidate evaluation is blocked.",
            }
    if report.get("stable_baseline_suite", {}).get("passed") and report.get("full_test_suite", {}).get("passed"):
        report["benchmark"] = benchmark_pair(STABLE_PATH, candidate_path, trials=trials)
    report["accepted"] = bool(report.get("full_test_suite", {}).get("passed") and
                              report.get("benchmark", {}).get("accepted"))
    if submit and report["accepted"]:
        report["review_entry_id"] = queue_review(child, stable_source, candidate_source, report, candidate_path)
    append_jsonl(root / "provenance.jsonl", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--candidate", type=Path)
    source.add_argument("--generator-command")
    parser.add_argument("--brief", default="Reduce allocations and loop overhead in cosine_rows and dot.")
    parser.add_argument("--child", default="Inazuma_Yagami")
    parser.add_argument("--trials", type=int, default=15)
    parser.add_argument("--full-test-timeout", type=int, default=900)
    parser.add_argument("--submit-for-review", action="store_true")
    args = parser.parse_args(argv)
    metadata = {}
    if args.candidate:
        candidate_source = args.candidate.read_text(encoding="utf-8")
    else:
        candidate_source, metadata = generate_candidate(
            args.generator_command, STABLE_PATH.read_text(encoding="utf-8"), args.brief)
    report = evaluate(candidate_source, args.child, max(5, args.trials), args.submit_for_review,
                      max(60, args.full_test_timeout), metadata)
    print(json.dumps(report, indent=2))
    return 0 if report["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
