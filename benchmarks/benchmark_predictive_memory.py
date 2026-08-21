"""Explicit bounded V1/V2 predictive candidate-memory comparison."""
import argparse
import json
import resource
import time
from pathlib import Path

from symbol_word_utils import load_compact_symbol_words


RESULTS_START = "<!-- benchmark-results:start -->"
RESULTS_END = "<!-- benchmark-results:end -->"


def measure(path: Path, version: str) -> dict:
    started = time.perf_counter()
    if version == "V1":
        with path.open(encoding="utf-8") as handle:
            count = len(json.load(handle).get("words", []))
    else:
        count = len(load_compact_symbol_words(path))
    compact_path = path.with_name("symbol_words.logic_index.json")
    return {
        "version": version,
        "candidates": count,
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "source_bytes": path.stat().st_size,
        "index_bytes": compact_path.stat().st_size if compact_path.exists() else None,
    }


def _human_bytes(value: int | None) -> str:
    if value is None:
        return "unavailable"
    amount = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if amount < 1000.0 or unit == "TB":
            digits = 0 if unit == "B" else (2 if amount < 10 else 1)
            return f"{amount:.{digits}f} {unit}"
        amount /= 1000.0
    return f"{value} B"


def render_results(result: dict) -> str:
    """Render public-safe results; source paths are deliberately excluded."""
    peak_mb = float(result["peak_rss_kib"]) / 1024.0
    elapsed = float(result["elapsed_seconds"])
    elapsed_text = f"{elapsed * 1000.0:.2f} ms" if elapsed < 0.1 else f"{elapsed:.3f} seconds"
    return "\n".join([
        RESULTS_START,
        "| Measurement | Historical V1 | Measured V2 |",
        "|---|---:|---:|",
        f'| Symbol store size | {_human_bytes(result["source_bytes"])} | '
        f'{_human_bytes(result["source_bytes"])} source, {_human_bytes(result["index_bytes"])} cached index |',
        "| Observed prediction memory | approximately 2.75 GB | not applicable to isolated lookup |",
        f'| Isolated indexed lookup peak RSS | not rerun | {peak_mb:.1f} MB |',
        f'| Isolated indexed lookup elapsed time | not rerun | {elapsed_text} |',
        f'| Indexed candidates | not recorded | {int(result["candidates"])} |',
        RESULTS_END,
    ])


def update_markdown_report(report_path: Path, result: dict) -> None:
    text = report_path.read_text(encoding="utf-8")
    start = text.find(RESULTS_START)
    end = text.find(RESULTS_END)
    if start < 0 or end < start:
        raise ValueError(f"benchmark result markers missing from {report_path}")
    end += len(RESULTS_END)
    report_path.write_text(
        text[:start] + render_results(result) + text[end:], encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--version", choices=("V1", "V2"), required=True)
    parser.add_argument(
        "--update-report", type=Path,
        help="replace the marked results block in a Markdown benchmark report",
    )
    args = parser.parse_args()
    result = measure(args.path, args.version)
    if args.update_report:
        if args.version != "V2":
            parser.error("--update-report is limited to the bounded V2 benchmark")
        update_markdown_report(args.update_report, result)
    # Machine-readable stdout remains useful to local automation without
    # requiring generated JSON artifacts in version control.
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
