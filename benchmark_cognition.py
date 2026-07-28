"""Run Project Inazuma's persistent cognitive benchmark.

This is event-driven: ``--monthly`` checks persisted due-state when this
command is explicitly invoked; it does not install a timer or daemon.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import shlex
from pathlib import Path

from cognitive_benchmarks.audit import audit_surface_cues, surface_cues_pass
from cognitive_benchmarks.backends import CommandScorer, HuggingFaceCausalScorer
from cognitive_benchmarks.core import load_cases, run_benchmark
from cognitive_benchmarks.schedule import MonthlyCadence
from cognitive_benchmarks.procedural import PROCEDURAL_VERSION, generate_cases

ROOT = Path(__file__).resolve().parent
DEFAULT_CASES = ROOT / "benchmarks" / "persistent_cognition_v1.jsonl"
DEFAULT_OUTPUT = ROOT / "benchmark_results"
STANDARD_SUITES = ("hellaswag", "piqa", "winogrande", "boolq", "lambada")


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("huggingface", "command"), default="huggingface")
    parser.add_argument("--model", default="gpt2", help="model id or label recorded in results")
    parser.add_argument("--command", help="JSON scorer command (required for command backend)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--procedural", action="store_true", help="generate fresh in-memory cases")
    parser.add_argument("--procedural-count", type=int, default=4, help="cases per category")
    parser.add_argument("--seed", type=int, help="debug/reproduction seed; random by default")
    parser.add_argument("--audit-only", action="store_true", help="audit procedural surface cues without scoring")
    parser.add_argument("--answer-key", type=Path, help="separate held-out JSON answer key")
    parser.add_argument("--allow-public-suite", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--monthly", action="store_true", help="skip unless one calendar month is due")
    parser.add_argument("--force", action="store_true", help="run even when monthly state is not due")
    parser.add_argument("--list-suites", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.list_suites:
        print(json.dumps({
            "available": ["persistent-cognition", "procedural-cognition"],
            "planned": list(STANDARD_SUITES),
        }, indent=2))
        return 0

    procedural = args.procedural
    public_suite = not procedural and args.cases.resolve() == DEFAULT_CASES.resolve()
    if args.backend == "command" and not procedural and not public_suite and args.answer_key is None:
        raise SystemExit("Blind command-model scoring requires --answer-key")
    if args.backend == "command" and public_suite and not args.allow_public_suite:
        raise SystemExit(
            "Refusing command-model scoring on the readable public suite. "
            "Use held-out --cases with --answer-key, or explicitly use "
            "--allow-public-suite for a non-evidentiary smoke test."
        )

    seed = args.seed if args.seed is not None else secrets.randbits(128)
    if args.audit_only:
        audit = audit_surface_cues(
            generate_cases(count_per_category=200, seed=seed)
        )
        print(json.dumps({**audit, "passed": surface_cues_pass(audit)}, indent=2))
        return 0 if surface_cues_pass(audit) else 2

    cadence = MonthlyCadence(args.output_dir / "cadence.json")
    suite = "procedural-cognition" if procedural else "persistent-cognition"
    if args.monthly and not args.force and not cadence.is_due(suite, args.model):
        last = cadence.last_completed(suite, args.model)
        print(json.dumps({"status": "not_due", "suite": suite, "model": args.model,
                          "last_completed": last.isoformat() if last else None}, indent=2))
        return 0

    if procedural:
        preflight = audit_surface_cues(
            generate_cases(count_per_category=100, seed=seed ^ 0x51A110)
        )
        if not surface_cues_pass(preflight):
            raise SystemExit("Procedural suite failed shallow-cue preflight")

    if args.backend == "command":
        if not args.command:
            raise SystemExit("--command is required for the command backend")
        scorer = CommandScorer(args.model, shlex.split(args.command))
    else:
        scorer = HuggingFaceCausalScorer(args.model, device=args.device)

    cases = (
        generate_cases(count_per_category=max(1, args.procedural_count), seed=seed)
        if procedural
        else load_cases(args.cases, answer_key_path=args.answer_key)
    )
    result = run_benchmark(
        cases, scorer, benchmark=suite,
        benchmark_version=PROCEDURAL_VERSION if procedural else "1",
    )
    payload = result.to_dict()
    payload["evaluation_protocol"] = (
        "procedural-generative" if procedural else
        "public-smoke" if public_suite else "blind-held-out"
    )
    if procedural:
        payload["seed_fingerprint"] = hashlib.sha256(str(seed).encode()).hexdigest()[:16]
    if not public_suite:
        payload["cases"] = []
        payload["case_details_withheld"] = True
    _append_jsonl(args.output_dir / "history.jsonl", payload)
    cadence.mark_completed(suite, args.model)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
