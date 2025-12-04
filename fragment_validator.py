#!/usr/bin/env python3
"""
fragment_validator.py

Soft validator / normaliser for Ina's memory fragments.

- Walks a root directory (e.g. AI_Children/*/memory/fragments)
- Loads all .json fragments
- Ensures core fields exist and have sane types
- Optionally auto-fixes and writes changes back
- Logs everything to a JSONL report so Sakura & Ina can inspect drift

This is deliberately *lenient*. It does not enforce a hard schema,
it just keeps the basics consistent enough for training and introspection.
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple, Optional

ReportIssue = Dict[str, Any]
Fragment = Dict[str, Any]


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def ensure_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    # if it's a single item, wrap it
    return [value]


def load_fragment(path: str) -> Tuple[Optional[Fragment], List[str]]:
    issues: List[str] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            issues.append("Top-level JSON is not an object")
            return None, issues
        return data, issues
    except json.JSONDecodeError as e:
        issues.append(f"JSONDecodeError: {e}")
    except OSError as e:
        issues.append(f"OSError while reading: {e}")
    return None, issues


def validate_fragment_structure(
    fragment: Fragment,
    path: str,
    autofix: bool = False
) -> Tuple[Fragment, List[str]]:
    """
    Validate and optionally normalise core fields of a fragment.

    Returns (possibly modified_fragment, issues).
    """
    issues: List[str] = []

    # --- id ---
    if "id" not in fragment:
        issues.append("Missing 'id' field")
        if autofix:
            # Use filename (without extension) as fallback
            fallback_id = os.path.splitext(os.path.basename(path))[0]
            fragment["id"] = fallback_id
            issues.append(f"Auto-set 'id' to '{fallback_id}'")
    else:
        if not isinstance(fragment["id"], str):
            issues.append("Field 'id' is not a string")
            if autofix:
                fragment["id"] = str(fragment["id"])
                issues.append("Auto-cast 'id' to string")

    # --- timestamp ---
    if "timestamp" not in fragment:
        issues.append("Missing 'timestamp' field")
        if autofix:
            try:
                ts = os.path.getmtime(path)
            except OSError:
                ts = time.time()
            fragment["timestamp"] = ts
            issues.append(f"Auto-set 'timestamp' to file mtime ({ts})")
    else:
        if not is_number(fragment["timestamp"]):
            issues.append("Field 'timestamp' is not a number")
            if autofix:
                try:
                    fragment["timestamp"] = float(fragment["timestamp"])
                    issues.append("Auto-cast 'timestamp' to float")
                except (ValueError, TypeError):
                    ts = time.time()
                    fragment["timestamp"] = ts
                    issues.append(
                        f"Failed to cast 'timestamp', reset to current time ({ts})"
                    )

    # --- modality ---
    if "modality" not in fragment:
        issues.append("Missing 'modality' field")
        if autofix:
            fragment["modality"] = []
            issues.append("Auto-set 'modality' to []")
    else:
        if isinstance(fragment["modality"], str):
            issues.append("Field 'modality' is string, expected list")
            if autofix:
                fragment["modality"] = [fragment["modality"]]
                issues.append("Auto-wrapped 'modality' in list")
        elif not isinstance(fragment["modality"], list):
            issues.append("Field 'modality' is not list or string")
            if autofix:
                fragment["modality"] = [str(fragment["modality"])]
                issues.append("Auto-coerced 'modality' to list[str]")

    # --- emotions ---
    if "emotions" not in fragment:
        issues.append("Missing 'emotions' field")
        if autofix:
            fragment["emotions"] = {}
            issues.append("Auto-set 'emotions' to {}")
    else:
        if not isinstance(fragment["emotions"], dict):
            issues.append("Field 'emotions' is not an object")
            if autofix:
                fragment["emotions"] = {}
                issues.append("Reset 'emotions' to {}")

    # --- symbols ---
    if "symbols" not in fragment:
        issues.append("Missing 'symbols' field")
        if autofix:
            fragment["symbols"] = []
            issues.append("Auto-set 'symbols' to []")
    else:
        if not isinstance(fragment["symbols"], list):
            issues.append("Field 'symbols' is not a list")
            if autofix:
                fragment["symbols"] = ensure_list(fragment["symbols"])
                issues.append("Auto-coerced 'symbols' to list")

    # --- tags ---
    if "tags" not in fragment:
        issues.append("Missing 'tags' field")
        if autofix:
            fragment["tags"] = []
            issues.append("Auto-set 'tags' to []")
    else:
        if not isinstance(fragment["tags"], list):
            issues.append("Field 'tags' is not a list")
            if autofix:
                fragment["tags"] = ensure_list(fragment["tags"])
                issues.append("Auto-coerced 'tags' to list")

    # --- importance ---
    if "importance" not in fragment:
        issues.append("Missing 'importance' field")
        if autofix:
            fragment["importance"] = 0.0
            issues.append("Auto-set 'importance' to 0.0")
    else:
        if not is_number(fragment["importance"]):
            issues.append("Field 'importance' is not a number")
            if autofix:
                try:
                    fragment["importance"] = float(fragment["importance"])
                    issues.append("Auto-cast 'importance' to float")
                except (ValueError, TypeError):
                    fragment["importance"] = 0.0
                    issues.append(
                        "Failed to cast 'importance', reset to 0.0"
                    )
        # clamp to [0,1] if needed
        if is_number(fragment["importance"]):
            orig = fragment["importance"]
            clamped = max(0.0, min(1.0, float(orig)))
            if clamped != orig:
                issues.append(
                    f"'importance' out of range ({orig}), clamped to {clamped}"
                )
                if autofix:
                    fragment["importance"] = clamped

    # --- meta ---
    if "meta" not in fragment:
        issues.append("Missing 'meta' field")
        if autofix:
            fragment["meta"] = {}
            issues.append("Auto-set 'meta' to {}")
    else:
        if not isinstance(fragment["meta"], dict):
            issues.append("Field 'meta' is not an object")
            if autofix:
                fragment["meta"] = {}
                issues.append("Reset 'meta' to {}")

    # --- soft STT / text hints (warnings only for now) ---
    # If we have a 'text' object, but 'modality' doesn't include 'text', note it.
    text_obj = fragment.get("text")
    if text_obj is not None and isinstance(text_obj, dict):
        if "modality" in fragment and "text" not in fragment["modality"]:
            issues.append(
                "Fragment has 'text' field but 'text' not in 'modality' (warning)"
            )

    source_type = None
    if isinstance(text_obj, dict):
        source_type = text_obj.get("source_type")

    # If STT sources are used, gently remind about having both audio+text.
    if isinstance(source_type, str) and source_type.endswith("_stt"):
        mods = set(fragment.get("modality") or [])
        if "text" not in mods or "audio" not in mods:
            issues.append(
                "STT source_type suggests both 'audio' and 'text' modality, "
                "but one or both are missing (warning only)."
            )

    return fragment, issues


def walk_json_files(root: str) -> List[str]:
    paths: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        for name in filenames:
            if name.endswith(".json"):
                paths.append(os.path.join(dirpath, name))
    return paths


def write_fragment(path: str, fragment: Fragment) -> None:
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(fragment, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate & optionally normalise Ina's memory fragments."
    )
    parser.add_argument(
        "--root",
        required=True,
        help="Root directory to scan (e.g. AI_Children or AI_Children/Ina/memory/fragments)",
    )
    parser.add_argument(
        "--autofix",
        action="store_true",
        help="Automatically fix basic issues and write updated fragments.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write any changes, only report issues (overrides --autofix).",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional maximum number of JSON files to process (for quick tests).",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="fragment_validator_report.jsonl",
        help="Path to JSONL report file.",
    )

    args = parser.parse_args()

    root = os.path.abspath(args.root)
    paths = walk_json_files(root)
    if args.max_files is not None:
        paths = paths[: args.max_files]

    if not paths:
        print(f"No JSON fragments found under: {root}")
        sys.exit(0)

    print(f"Found {len(paths)} JSON files under {root}")

    report_path = os.path.abspath(args.report)
    report_f = open(report_path, "w", encoding="utf-8")

    total_issues = 0
    total_fixed = 0

    for i, path in enumerate(sorted(paths), start=1):
        frag, load_issues = load_fragment(path)
        if frag is None:
            # Just log the load error and continue
            report_obj: ReportIssue = {
                "path": path,
                "load_error": load_issues,
                "valid": False,
                "fixed": False,
                "issues": load_issues,
            }
            report_f.write(json.dumps(report_obj, ensure_ascii=False) + "\n")
            total_issues += len(load_issues)
            continue

        fixed_here = False
        frag_before = json.dumps(frag, sort_keys=True, ensure_ascii=False)

        frag_after, issues = validate_fragment_structure(
            frag, path, autofix=args.autofix and not args.dry_run
        )

        if args.autofix and not args.dry_run:
            # Only write if fragment actually changed
            frag_after_str = json.dumps(
                frag_after, sort_keys=True, ensure_ascii=False
            )
            if frag_after_str != frag_before:
                write_fragment(path, frag_after)
                fixed_here = True

        report_obj = {
            "path": path,
            "valid": True if not issues and not load_issues else False,
            "fixed": fixed_here,
            "issues": issues,
        }
        report_f.write(json.dumps(report_obj, ensure_ascii=False) + "\n")

        total_issues += len(issues)
        if fixed_here:
            total_fixed += 1

        if i % 100 == 0:
            print(f"Processed {i}/{len(paths)} files...")

    report_f.close()

    print(f"Done. Processed {len(paths)} fragments.")
    print(f"Total issues noted: {total_issues}")
    print(f"Fragments modified (autofix): {total_fixed}")
    print(f"Report written to: {report_path}")


if __name__ == "__main__":
    main()
